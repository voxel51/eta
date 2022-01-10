#!/usr/bin/env python
"""
A module that uses an `eta.core.learning.VideoFramesClassifier` to classify the
frames of a video using a sliding window strategy.

Info:
    type: eta.core.types.Module
    version: 0.1.0

Copyright 2017-2022, Voxel51, Inc.
voxel51.com
"""
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from future.utils import iteritems

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict, deque
import logging
import sys

from eta.core.config import Config
import eta.core.data as etad
import eta.core.learning as etal
import eta.core.module as etam
import eta.core.video as etav


logger = logging.getLogger(__name__)


class ModuleConfig(etam.BaseModuleConfig):
    """Module configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    """

    def __init__(self, d):
        super(ModuleConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    """Data configuration settings.

    Inputs:
        video_path (eta.core.types.Video): the input video
        input_labels_path (eta.core.types.VideoLabels): [None] an optional
            input VideoLabels file to which to add the predictions

    Outputs:
        output_labels_path (eta.core.types.VideoLabels): a VideoLabels file
            containing the predictions
    """

    def __init__(self, d):
        self.video_path = self.parse_string(d, "video_path")
        self.input_labels_path = self.parse_string(
            d, "input_labels_path", default=None
        )
        self.output_labels_path = self.parse_string(d, "output_labels_path")


class ParametersConfig(Config):
    """Parameter configuration settings.

    Parameters:
        classifier (eta.core.types.VideoFramesClassifier): an
            `eta.core.learning.VideoFramesClassifierConfig` describing the
            `eta.core.learning.VideoFramesClassifier` to use
        window_size (eta.core.types.Number): the size of the sliding window in
            which to perform classification
        stride (eta.core.types.Number): the stride of the sliding window
        confidence_threshold (eta.core.types.Number): [None] a confidence
            threshold to use when assigning labels
        confidence_weighted_vote (eta.core.types.Boolean): [False] whether to
            weight any per-frame-attribute votes by confidence
    """

    def __init__(self, d):
        self.classifier = self.parse_object(
            d, "classifier", etal.VideoFramesClassifierConfig
        )
        self.window_size = self.parse_number(d, "window_size")
        self.stride = self.parse_number(d, "stride")
        self.confidence_threshold = self.parse_number(
            d, "confidence_threshold", default=None
        )
        self.confidence_weighted_vote = self.parse_bool(
            d, "confidence_weighted_vote", default=False
        )


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


def _apply_video_frames_classifier(config):
    # Build classifier
    classifier = config.parameters.classifier.build()
    logger.info("Loaded classifier %s", type(classifier))

    # Process videos
    with classifier:
        for data in config.data:
            _process_video(data, classifier, config.parameters)


def _process_video(data, classifier, parameters):
    # Load labels
    if data.input_labels_path:
        logger.info(
            "Reading existing labels from '%s'", data.input_labels_path
        )
        labels = etav.VideoLabels.from_json(data.input_labels_path)
    else:
        labels = etav.VideoLabels()

    # Process frames directly
    logger.info("Processing video '%s'", data.video_path)
    with etav.FFmpegVideoReader(data.video_path) as vr:
        _classify_windows(classifier, vr, labels, parameters)

    logger.info("Writing labels to '%s'", data.output_labels_path)
    labels.write_json(data.output_labels_path)


def _classify_windows(classifier, video_reader, labels, parameters):
    # Parameters
    wsize = parameters.window_size
    wstride = parameters.stride
    cthresh = parameters.confidence_threshold
    cweighted = parameters.confidence_weighted_vote

    # Build filter
    attr_filter = _build_attribute_filter(cthresh)

    #
    # Sliding window classification
    #
    # FIFO queue of length `wsize` to hold the window of images
    imgs = deque([], wsize)
    # Containers to store all of the attributes that are generated
    attrs_map = defaultdict(lambda: etad.AttributeContainer())
    # The next frame at which to run the classifier
    next_classify_frame = wsize
    for frame_number, img in enumerate(video_reader, 1):
        # Ingest next frame
        imgs.append(img)

        if frame_number < next_classify_frame:
            # Not time to classify yet
            continue

        # Set next classification frame
        next_classify_frame += wstride

        # Classify window
        attrs = attr_filter(classifier.predict(imgs))
        for attr in attrs:
            for idx in range(frame_number - wsize + 1, frame_number + 1):
                attrs_map[idx].add(attr)

    # Finalize attributes
    for frame_number, attrs in iteritems(attrs_map):
        # Majority vote over frame
        final_attrs = etad.majority_vote_categorical_attrs(
            attrs, confidence_weighted=cweighted
        )

        labels.add_frame_attributes(final_attrs, frame_number)


def run(config_path, pipeline_config_path=None):
    """Run the apply_video_frames_classifier module.

    Args:
        config_path: path to a ModuleConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    """
    config = ModuleConfig.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)
    _apply_video_frames_classifier(config)


if __name__ == "__main__":
    run(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
