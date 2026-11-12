"""Tests for serializing labels that contain numpy arrays.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

import eta.core.frames as etaf
import eta.core.geometry as etag
import eta.core.objects as etao
import eta.core.serial as etas


def _make_mask():
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True
    return mask


def test_detected_object_mask_round_trip(tmp_path):
    mask = _make_mask()
    bounding_box = etag.BoundingBox.from_coords(0.1, 0.2, 0.5, 0.6)
    obj = etao.DetectedObject(
        label="cat", bounding_box=bounding_box, mask=mask
    )

    path = str(tmp_path / "object.json")
    obj.write_json(path)
    obj2 = etao.DetectedObject.from_json(path)

    assert obj2.label == "cat"
    assert np.array_equal(obj2.mask, mask)


def test_frame_labels_mask_round_trip(tmp_path):
    mask = np.arange(24, dtype=np.uint8).reshape(4, 6)
    frame_labels = etaf.FrameLabels(mask=mask)

    path = str(tmp_path / "frame.json")
    frame_labels.write_json(path)
    frame_labels2 = etaf.FrameLabels.from_json(path)

    assert np.array_equal(frame_labels2.mask, mask)


def test_deserialize_numpy_array():
    mask = _make_mask()

    # the string format is still supported
    numpy_str = etas.serialize_numpy_array(mask)
    assert np.array_equal(etas.deserialize_numpy_array(numpy_str), mask)

    assert np.array_equal(etas.deserialize_numpy_array(mask.tolist()), mask)
