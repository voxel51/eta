"""Tests for reading and writing JSON files.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import json

import eta.core.serial as etas


NON_ASCII = {"latin": "café", "cjk": "猫"}


def test_read_json_utf8(tmp_path):
    path = str(tmp_path / "labels.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(NON_ASCII, f, ensure_ascii=False)

    assert etas.read_json(path) == NON_ASCII


def test_write_json_utf8(tmp_path):
    path = str(tmp_path / "labels.json")
    etas.write_json(NON_ASCII, path)

    with open(path, "r", encoding="utf-8") as f:
        assert json.load(f) == NON_ASCII

    assert etas.read_json(path) == NON_ASCII


def test_ndjson_utf8(tmp_path):
    path = str(tmp_path / "labels.ndjson")
    etas.write_ndjson([NON_ASCII], path)
    etas.write_ndjson([NON_ASCII], path, append=True)

    assert etas.read_ndjson(path) == [NON_ASCII, NON_ASCII]
