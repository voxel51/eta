"""
Detectors package declaration.

Copyright 2017-2022, Voxel51, Inc.
voxel51.com
"""
# Import all detectors into the `eta.detectors` namespace
from .efficientdet import EfficientDet, EfficientDetConfig
from .tfmodels_detectors import (
    TFModelsDetector,
    TFModelsDetectorConfig,
    TF2ModelsDetector,
    TF2ModelsDetectorConfig,
    TFModelsInstanceSegmenter,
    TFModelsInstanceSegmenterConfig,
)
from .yolo import (
    YOLODetector,
    YOLODetectorConfig,
)
