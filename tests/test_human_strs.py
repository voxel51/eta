"""Tests for human-readable number strings.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import pytest

import eta.core.utils as etau


@pytest.mark.parametrize(
    "num,expected",
    [
        (65, "65.0"),
        (123456, "123.5K"),
        (1e7, "10.0M"),
        (2.5e12, "2.5T"),
        (999e12, "999.0T"),
        (5.3e15, "5300.0T"),
        (1e18, "1000000.0T"),
    ],
)
def test_to_human_decimal_str(num, expected):
    assert etau.to_human_decimal_str(num) == expected


@pytest.mark.parametrize("num", [1, 999, 12345, 5.3e12, 5.3e15, 7e20])
def test_human_decimal_str_round_trip(num):
    num_str = etau.to_human_decimal_str(num, decimals=3)

    assert etau.from_human_decimal_str(num_str) == pytest.approx(num, rel=1e-3)


def test_to_human_bytes_str_largest_unit():
    assert etau.to_human_bytes_str(1024**8) == "1.0YB"
    assert etau.to_human_bytes_str(5 * 1024**9) == "5120.0YB"
    assert etau.to_human_bits_str(5 * 1024**9) == "5120.0Yb"
