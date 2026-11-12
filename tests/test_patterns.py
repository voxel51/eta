"""Tests for numeric file patterns.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

import pytest

import eta.core.utils as etau


@pytest.mark.parametrize(
    "dirname", ["frames", "clip (1)", "a+b", "v1.0", "[raw]", "50%"]
)
def test_parse_pattern_literal_path(tmp_path, dirname):
    # The path part of a pattern is literal text, not a regular expression
    frames_dir = tmp_path / dirname
    frames_dir.mkdir()
    for idx in (1, 2, 10):
        (frames_dir / ("frame-%05d.jpg" % idx)).write_bytes(b"")

    (frames_dir / "frame-abc.jpg").write_bytes(b"")

    patt = os.path.join(str(frames_dir), "frame-%05d.jpg")

    assert etau.parse_pattern(patt) == [1, 2, 10]
    assert etau.parse_bounds_from_pattern(patt) == (1, 10)
    assert [os.path.basename(p) for p in etau.get_pattern_matches(patt)] == [
        "frame-00001.jpg",
        "frame-00002.jpg",
        "frame-00010.jpg",
    ]


def test_parse_pattern_multiple_sequences(tmp_path):
    for clip, idx in ((1, 1), (1, 2), (2, 1)):
        (tmp_path / ("%02d-%d.mp4" % (clip, idx))).write_bytes(b"")

    patt = os.path.join(str(tmp_path), "%02d-%d.mp4")

    assert etau.parse_pattern(patt) == [(1, 1), (1, 2), (2, 1)]
