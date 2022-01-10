#!/usr/bin/env python
"""
A module that applies an `eta.core.learning.VideoModel` on a video.

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

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging
import sys

from eta.core.config import Config, ConfigError
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
        video_path (eta.core.types.Video): [None] the input video
        video_frames_dir (eta.core.types.ImageSequenceDirectory): [None] a
            directory containing the frames of the video
        input_labels_path (eta.core.types.VideoLabels): [None] an optional
            input VideoLabels file to which to add the predictions

    Outputs:
        output_labels_path (eta.core.types.VideoLabels): a VideoLabels file
            containing the predictions
    """

    def __init__(self, d):
        self.video_path = self.parse_string(d, "video_path", default=None)
        self.video_frames_dir = self.parse_string(
            d, "video_frames_dir", default=None
        )
        self.input_labels_path = self.parse_string(
            d, "input_labels_path", default=None
        )
        self.output_labels_path = self.parse_string(d, "output_labels_path")


class ParametersConfig(Config):
    """Parameter configuration settings.

    Parameters:
        model (eta.core.types.VideoModel): an
            `eta.core.learning.VideoModelConfig` describing the
            `eta.core.learning.VideoModel` to use
    """

    def __init__(self, d):
        self.model = self.parse_object(d, "model", etal.VideoModelConfig)


def _apply_video_model(config):
    # Build model
    model = config.parameters.model.build()
    logger.info("Loaded model %s", type(model))

    # Process videos
    with model:
        for data in config.data:
            _process_video(data, model)


def _process_video(data, model):
    # Load labels
    if data.input_labels_path:
        logger.info(
            "Reading existing labels from '%s'", data.input_labels_path
        )
        labels = etav.VideoLabels.from_json(data.input_labels_path)
    else:
        labels = etav.VideoLabels()

    # Construct VideoReader
    if data.video_path:
        logger.info("Applying model to video '%s'", data.video_path)
        video_reader = etav.FFmpegVideoReader(data.video_path)
    elif data.video_frames_dir:
        logger.info("Applying model to frames in '%s'", data.video_frames_dir)
        video_reader = etav.SampledFramesVideoReader(data.video_frames_dir)
    else:
        raise ConfigError(
            "Either `video_path` or `video_frames_dir` must be provided"
        )

    with video_reader:
        new_labels = model.process(video_reader)

    labels.merge_labels(new_labels)

    logger.info("Writing labels to '%s'", data.output_labels_path)
    labels.write_json(data.output_labels_path)


def run(config_path, pipeline_config_path=None):
    """Run the apply_video_model module.

    Args:
        config_path: path to a ModuleConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    """
    config = ModuleConfig.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)
    _apply_video_model(config)


if __name__ == "__main__":
    run(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
