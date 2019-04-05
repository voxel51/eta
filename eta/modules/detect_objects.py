#!/usr/bin/env python
'''
A module that uses an `eta.core.learning.ObjectDetector` to detect objects in
video(s).

Info:
    type: eta.core.types.Module
    version: 0.1.0

Copyright 2017-2019, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging
import sys

from eta.core.config import Config
from eta.core.learning import ObjectDetectorConfig
import eta.core.module as etam
import eta.core.video as etav


logger = logging.getLogger(__name__)


class DetectObjectsConfig(etam.BaseModuleConfig):
    '''Module configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(DetectObjectsConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        video_path (eta.core.types.Video): the input video
        input_labels_path (eta.core.types.VideoLabels): [None] an optional
            input VideoLabels file to which to add the detections

    Outputs:
        output_labels_path (eta.core.types.VideoLabels): a VideoLabels file
            containing the detections
    '''

    def __init__(self, d):
        self.video_path = self.parse_string(d, "video_path")
        self.input_labels_path = self.parse_string(
            d, "input_labels_path", default=None)
        self.output_labels_path = self.parse_string(d, "output_labels_path")


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        detector (eta.core.types.Object): an
            `eta.core.learning.ObjectDetectorConfig` describing the object
            detector to use
        objects (eta.core.types.ObjectArray): [None] an array of objects
            describing the labels and confidence thresholds of objects to
            detect. If omitted, all detections emitted by the detector are
            used
    '''

    def __init__(self, d):
        self.detector = self.parse_object(d, "detector", ObjectDetectorConfig)
        self.objects = self.parse_object_array(
            d, "objects", ObjectsConfig, default=None)


class ObjectsConfig(Config):
    '''Objects configuration settings.'''

    def __init__(self, d):
        self.labels = self.parse_array(d, "labels", default=None)
        self.threshold = self.parse_number(d, "threshold", default=None)


def _build_object_filter(labels, threshold):
    if threshold is not None:
        threshold = float(threshold)

    if labels is None:
        if threshold is None:
            logger.info("Detecting all objects")
            filter_fcn = lambda obj: True
        else:
            logger.info(
                "Detecting all objects with confidence >= %g", threshold)
            filter_fcn = lambda obj: obj.confidence >= threshold
    else:
        if threshold is None:
            logger.info("Detecting %s", labels)
            filter_fcn = lambda obj: obj.label in labels
        else:
            logger.info(
                "Detecting %s with confidence >= %g", labels, threshold)
            filter_fcn = (
                lambda obj: obj.label in labels and obj.confidence >= threshold
            )

    return filter_fcn


def _build_detection_filter(objects_config):
    if objects_config is None:
        # Return all detections
        return lambda objs: objs

    # Parse object filter
    obj_filters = [
        _build_object_filter(oc.labels, oc.threshold) for oc in objects_config]
    return lambda objs: objs.get_matches(obj_filters)


def _detect_objects(config):
    # Build detector
    detector = config.parameters.detector.build()
    object_filter = _build_detection_filter(config.parameters.objects)

    with detector:
        for data in config.data:
            _process_video(data, detector, object_filter)


def _process_video(data, detector, object_filter):
    # Load labels
    if data.input_labels_path:
        logger.info(
            "Reading existing labels from '%s'", data.input_labels_path)
        labels = etav.VideoLabels.from_json(data.input_labels_path)
    else:
        labels = etav.VideoLabels()

    logger.info("Processing video '%s'", data.video_path)
    with etav.FFmpegVideoReader(data.video_path) as vr:
        for img in vr:
            logger.debug("Processing frame %d", vr.frame_number)

            objects = object_filter(detector.detect(img))
            for obj in objects:
                obj.frame_number = vr.frame_number
                labels.add_object(obj, vr.frame_number)

    logger.info("Writing labels to '%s'", data.output_labels_path)
    labels.write_json(data.output_labels_path)


def run(config_path, pipeline_config_path=None):
    '''Run the detect_objects module.

    Args:
        config_path: path to a DetectObjectsConfig file
        pipeline_config_path: optional path to a PipelineConfig file
   '''
    config = DetectObjectsConfig.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)
    _detect_objects(config)


if __name__ == "__main__":
    run(*sys.argv[1:])
