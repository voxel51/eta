"""Tests for the ffmpeg -vsync -> -fps_mode option translation.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import pytest

import eta.core.video as etav


@pytest.fixture(autouse=True)
def _reset_probe_cache():
    etav._ffmpeg_has_fps_mode = None
    yield
    etav._ffmpeg_has_fps_mode = None


class _FakePopen(object):
    calls = 0

    def __init__(self, version_line):
        self._version_line = version_line

    def __call__(self, args, **kwargs):
        type(self).calls += 1
        self._args = args
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def communicate(self):
        out = "ffmpeg version %s Copyright (c) 2000-2026" % self._version_line
        return out.encode("utf-8"), b""


def _probe_with_version(monkeypatch, version_line):
    monkeypatch.setattr(etav, "Popen", _FakePopen(version_line))
    return etav._ffmpeg_supports_fps_mode()


class TestFFmpegVersionProbe(object):
    @pytest.mark.parametrize(
        "version_line,expected",
        [
            ("7.1.1", True),
            ("9.0-essentials_build-www.gyan.dev", True),
            ("5.1.2", True),
            ("n5.1.2", True),
            ("5.0.3", False),
            ("4.4.2-0ubuntu0.22.04.1", False),
            # git/master builds have no parseable version and are assumed
            # to support -fps_mode
            ("N-113445-g0f8a2b1c2d", True),
            ("2023-12-21-git-1e42a48e37", True),
        ],
    )
    def test_version_parsing(self, monkeypatch, version_line, expected):
        assert _probe_with_version(monkeypatch, version_line) is expected

    def test_result_is_cached(self, monkeypatch):
        fake = _FakePopen("7.1.1")
        _FakePopen.calls = 0
        monkeypatch.setattr(etav, "Popen", fake)

        assert etav._ffmpeg_supports_fps_mode() is True
        assert etav._ffmpeg_supports_fps_mode() is True
        assert _FakePopen.calls == 1

    def test_missing_binary_is_not_cached(self, monkeypatch):
        def raise_enoent(args, **kwargs):
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr(etav, "Popen", raise_enoent)
        assert etav._ffmpeg_supports_fps_mode() is False

        # a binary that appears later is probed fresh
        monkeypatch.setattr(etav, "Popen", _FakePopen("7.1.1"))
        assert etav._ffmpeg_supports_fps_mode() is True


class TestTranslateVsyncOpts(object):
    @pytest.fixture
    def supported(self, monkeypatch):
        monkeypatch.setattr(
            etav, "_ffmpeg_supports_fps_mode", lambda: True
        )

    @pytest.fixture
    def unsupported(self, monkeypatch):
        monkeypatch.setattr(
            etav, "_ffmpeg_supports_fps_mode", lambda: False
        )

    def test_out_opts_translated(self, supported):
        in_opts, out_opts = etav.FFmpeg._translate_vsync_opts(
            [], ["-vsync", "0"]
        )
        assert in_opts == []
        assert out_opts == ["-fps_mode", "passthrough"]

    def test_default_video_out_opts(self, supported):
        in_opts, out_opts = etav.FFmpeg._translate_vsync_opts(
            list(etav.FFmpeg.DEFAULT_IN_OPTS),
            list(etav.FFmpeg.DEFAULT_VIDEO_OUT_OPTS),
        )
        assert in_opts == []
        assert "-vsync" not in out_opts
        assert out_opts[-2:] == ["-fps_mode", "passthrough"]

        # all other options are preserved in order
        expected = [
            o for o in etav.FFmpeg.DEFAULT_VIDEO_OUT_OPTS if o != "-vsync"
        ]
        expected.remove("0")
        assert out_opts[:-2] == expected

    def test_input_vsync_moves_to_output(self, supported):
        in_opts, out_opts = etav.FFmpeg._translate_vsync_opts(
            ["-vsync", "0", "-ss", "1.5"], ["-vframes", "1"]
        )
        assert in_opts == ["-ss", "1.5"]
        assert out_opts == ["-vframes", "1", "-fps_mode", "passthrough"]

    def test_output_value_wins_over_input(self, supported):
        in_opts, out_opts = etav.FFmpeg._translate_vsync_opts(
            ["-vsync", "0"], ["-vsync", "2"]
        )
        assert in_opts == []
        assert out_opts == ["-fps_mode", "vfr"]

    @pytest.mark.parametrize(
        "vsync_val,fps_mode_val",
        [
            ("0", "passthrough"),
            ("1", "cfr"),
            ("2", "vfr"),
            ("-1", "auto"),
            ("passthrough", "passthrough"),
            ("vfr", "vfr"),
        ],
    )
    def test_value_mapping(self, supported, vsync_val, fps_mode_val):
        _, out_opts = etav.FFmpeg._translate_vsync_opts(
            [], ["-vsync", vsync_val]
        )
        assert out_opts == ["-fps_mode", fps_mode_val]

    def test_existing_fps_mode_not_duplicated(self, supported):
        in_opts, out_opts = etav.FFmpeg._translate_vsync_opts(
            ["-vsync", "0"], ["-fps_mode", "vfr"]
        )
        assert in_opts == []
        assert out_opts == ["-fps_mode", "vfr"]

    def test_unsupported_ffmpeg_keeps_vsync(self, unsupported):
        in_opts, out_opts = etav.FFmpeg._translate_vsync_opts(
            ["-vsync", "0"], ["-vsync", "0", "-an"]
        )
        assert in_opts == ["-vsync", "0"]
        assert out_opts == ["-vsync", "0", "-an"]

    def test_no_vsync_is_untouched_without_probing(self, monkeypatch):
        def fail():
            raise AssertionError("probe should not run")

        monkeypatch.setattr(etav, "_ffmpeg_supports_fps_mode", fail)

        in_opts, out_opts = etav.FFmpeg._translate_vsync_opts(
            ["-ss", "1.5"], ["-vframes", "1"]
        )
        assert in_opts == ["-ss", "1.5"]
        assert out_opts == ["-vframes", "1"]
