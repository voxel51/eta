#!/usr/bin/env python
'''
A module that uses an `eta.core.learning.VideoClassifier` to classify a video.

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


class ClassifyVideoConfig(etam.BaseModuleConfig):
    '''Module configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(ClassifyVideoConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        video_path (eta.core.types.Video): the input video
        input_labels_path (eta.core.types.VideoLabels): [None] an optional
            input VideoLabels file to which to add the predictions

    Outputs:
        output_labels_path (eta.core.types.VideoLabels): a VideoLabels file
            containing the predictions
    '''

    def __init__(self, d):
        self.video_path = self.parse_string(d, "video_path")
        self.input_labels_path = self.parse_string(
            d, "input_labels_path", default=None)
        self.output_labels_path = self.parse_string(d, "output_labels_path")


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        classifier (eta.core.types.VideoClassifier): an
            `eta.core.learning.ClassifierConfig` describing the
            `eta.core.learning.VideoClassifier` to use
        confidence_threshold (eta.core.types.Number): [None] a confidence
            threshold to use when assigning labels
    '''

    def __init__(self, d):
        self.classifier = self.parse_object(
            d, "classifier", etal.ClassifierConfig)
        self.confidence_threshold = self.parse_number(
            d, "confidence_threshold", default=None)


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


def _classify_video(config):
    # Build classifier
    classifier = config.parameters.classifier.build()
    logger.info("Loaded classifier %s", type(classifier))
    if not isinstance(classifier, etal.VideoClassifier):
        raise ValueError("Classifier must be a %s" % etal.VideoClassifier)

    # Process videos
    with classifier:
        for data in config.data:
            _process_video(data, classifier, config.parameters)


def _process_video(data, classifier, parameters):
    # Load labels
    if data.input_labels_path:
        logger.info(
            "Reading existing labels from '%s'", data.input_labels_path)
        labels = etav.VideoLabels.from_json(data.input_labels_path)
    else:
        labels = etav.VideoLabels()

    # Build filter
    attr_filter = _build_attribute_filter(parameters.confidence_threshold)

    logger.info("Classifying video '%s'", data.video_path)
    attrs = attr_filter(classifier.predict(data.video_path))
    labels.add_video_attributes(attrs)

    logger.info("Writing labels to '%s'", data.output_labels_path)
    labels.write_json(data.output_labels_path)


def run(config_path, pipeline_config_path=None):
    '''Run the classify_video module.

    Args:
        config_path: path to a ClassifyVideoConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    config = ClassifyVideoConfig.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)
    _classify_video(config)


if __name__ == "__main__":
    run(*sys.argv[1:])
