#!/usr/bin/env python
'''
A module that uses an `eta.core.learning.ImageClassifier` to classify the
detected objects in a video.

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
import eta.core.learning as etal
import eta.core.module as etam
import eta.core.video as etav


logger = logging.getLogger(__name__)


class ClassifyObjectsConfig(etam.BaseModuleConfig):
    '''Module configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(ClassifyObjectsConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        video_path (eta.core.types.Video): the input video
        input_labels_path (eta.core.types.VideoLabels): a VideoLabels file
            containing detected objects

    Outputs:
        output_labels_path (eta.core.types.VideoLabels): a VideoLabels file
            containing the predictions
    '''

    def __init__(self, d):
        self.video_path = self.parse_string(d, "video_path")
        self.input_labels_path = self.parse_string(d, "input_labels_path")
        self.output_labels_path = self.parse_string(d, "output_labels_path")


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        classifier (eta.core.types.ImageClassifier): an
            `eta.core.learning.ImageClassifierConfig` JSON describing the
            `eta.core.learning.ImageClassifier` to use
        labels (eta.core.types.Array): [None] an optional list of object
            labels to classify. By default, all objects are classified
        bb_padding (eta.core.types.Number): [None] the padding to apply to each
            dimension of the bounding box before classification
        force_square (eta.core.types.Boolean): [False] whether to minimally
            manipulate the object bounding boxes into squares before extraction
        min_height_pixels (eta.core.types.Number): [None] a minimum height,
            in pixels, for a bounding box to be classified
        confidence_threshold (eta.core.types.Number): [None] the minimum
            confidence required for a label to be saved
    '''

    def __init__(self, d):
        self.classifier = self.parse_object(
            d, "classifier", etal.ImageClassifierConfig)
        self.labels = self.parse_array(d, "labels", default=None)
        self.bb_padding = self.parse_number(d, "bb_padding", default=None)
        self.force_square = self.parse_bool(d, "force_square", default=False)
        self.min_height_pixels = self.parse_number(
            d, "min_height_pixels", default=None)
        self.confidence_threshold = self.parse_number(
            d, "confidence_threshold", default=None)


def _build_object_filter(labels):
    if labels is None:
        logger.info("Classifying all objects")
        filter_fcn = lambda objs: objs
    else:
        logger.info("Classifying %s", labels)
        obj_filters = [lambda obj: obj.label in labels]
        filter_fcn = lambda objs: objs.get_matches(obj_filters)

    return filter_fcn


def _build_attribute_filter(threshold):
    if threshold is None:
        logger.info("Predicting all attributes")
        filter_fcn = lambda attrs: attrs
    else:
        logger.info("Returning predictions with confidence >= %f", threshold)
        attr_filters = [
            lambda attr: attr.confidence is None
            or attr.confidence > float(threshold)
        ]
        filter_fcn = lambda attrs: attrs.get_matches(attr_filters)

    return filter_fcn


def _classify_objects(config):
    # Build classifier
    classifier = config.parameters.classifier.build()
    logger.info("Loaded classifier %s", type(classifier))

    # Process videos
    with classifier:
        for data in config.data:
            _process_video(data, classifier, config.parameters)


def _process_video(data, classifier, parameters):
    # Load labels
    labels = etav.VideoLabels.from_json(data.input_labels_path)

    # Build filters
    object_filter = _build_object_filter(parameters.labels)
    attr_filter = _build_attribute_filter(parameters.confidence_threshold)

    logger.info("Processing video '%s'", data.video_path)
    with etav.FFmpegVideoReader(data.video_path) as vr:
        for img in vr:
            logger.debug("Processing frame %d", vr.frame_number)

            frame_labels = labels.get_frame(vr.frame_number)
            _process_frame(
                classifier, img, frame_labels, object_filter, attr_filter,
                parameters)

    logger.info("Writing labels to '%s'", data.output_labels_path)
    labels.write_json(data.output_labels_path)


def _process_frame(
        classifier, img, frame_labels, object_filter, attr_filter, parameters):
    # Parse parameters
    bb_padding = parameters.bb_padding
    force_square = parameters.force_square
    min_height = parameters.min_height_pixels

    # Get objects
    objects = object_filter(frame_labels.objects)

    # Classify objects
    for obj in objects:
        # Extract object chip
        bbox = obj.bounding_box
        if bb_padding:
            bbox = bbox.pad_relative(bb_padding)
        obj_img = bbox.extract_from(img, force_square=force_square)

        # Skip small objects, if requested
        obj_height = obj_img.shape[0]
        if min_height is not None and obj_height < min_height:
            logger.debug(
                "Skipping object with height %d < %d", obj_height, min_height)
            continue

        # Classify object
        attrs = attr_filter(classifier.predict(obj_img))
        obj.add_attributes(attrs)


def run(config_path, pipeline_config_path=None):
    '''Run the classify_objects module.

    Args:
        config_path: path to a ClassifyObjectsConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    config = ClassifyObjectsConfig.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)
    _classify_objects(config)


if __name__ == "__main__":
    run(*sys.argv[1:])
