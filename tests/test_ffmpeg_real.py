"""Integration tests that exercise eta's ffmpeg invocation paths against the
real ffmpeg/ffprobe binaries on PATH.

These tests are skipped when ffmpeg is not available. They pass regardless of
ffmpeg version: on ffmpeg >= 5.1 the ``-vsync`` options are translated to
``-fps_mode`` (ffmpeg 9 removed ``-vsync``), and on older binaries they are
passed through unchanged.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import shutil
import subprocess

import pytest

import eta.core.video as etav

FFMPEG = shutil.which("ffmpeg")

pytestmark = pytest.mark.skipif(
    FFMPEG is None, reason="ffmpeg is not installed"
)


@pytest.fixture(scope="session")
def video(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("video") / "test.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=320x240:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "5",
            path,
        ],
        check=True,
    )
    return path


def test_reencode(video, tmp_path):
    out = str(tmp_path / "reencode.mp4")
    etav.FFmpeg().run(video, out)
    assert os.path.getsize(out) > 0


def test_video_to_frames(video, tmp_path):
    patt = str(tmp_path / "frames" / "%05d.jpg")
    etav.FFmpeg(out_opts=etav.FFmpeg.DEFAULT_IMAGES_OUT_OPTS).run(video, patt)
    assert len(os.listdir(tmp_path / "frames")) > 0


def test_stream_all_frames(video):
    with etav.FFmpegVideoReader(video) as reader:
        num_frames = sum(1 for _ in reader)

    assert num_frames == 20


def test_stream_keyframes(video):
    with etav.FFmpegVideoReader(video, keyframes_only=True) as reader:
        num_frames = sum(1 for _ in reader)

    assert num_frames > 0


def test_extract_frame(video, tmp_path):
    out = str(tmp_path / "frame.jpg")
    etav.extract_frame(video, out, start_time=0.5)
    assert os.path.getsize(out) > 0


@pytest.mark.parametrize("fast", [False, True])
def test_extract_clip(video, tmp_path, fast):
    out = str(tmp_path / "clip.mp4")
    etav.extract_clip(video, out, start_time=0.2, duration=1.0, fast=fast)
    assert os.path.getsize(out) > 0


@pytest.mark.parametrize("fast", [False, True])
def test_sample_select_frames(video, tmp_path, fast):
    patt = str(tmp_path / "select" / "%05d.jpg")
    etav.sample_select_frames(
        video, [1, 3, 5], output_patt=patt, fast=fast
    )
    assert len(os.listdir(tmp_path / "select")) == 3
