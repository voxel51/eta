"""
Detectors package declaration.

Copyright 2017-2020, Voxel51, Inc.
voxel51.com
"""
# Import all detectors into the `eta.detectors` namespace
from .efficientdet import EfficientDet, EfficientDetConfig
from .tfmodels_detectors import (
    TFModelsDetector,
    TFModelsDetectorConfig,
    TFModelsInstanceSegmenter,
    TFModelsInstanceSegmenterConfig,
)
from .yolo import (
    YOLODetector,
    YOLODetectorConfig,
)
