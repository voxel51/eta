#!/usr/bin/env python
'''
A module that applies an `eta.core.learning.VideoModel` on a video.

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


class ApplyVideoModelConfig(etam.BaseModuleConfig):
    '''Module configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(ApplyVideoModelConfig, self).__init__(d)
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
        model (eta.core.types.VideoModel): an
            `eta.core.learning.VideoModelConfig` describing the
            `eta.core.learning.VideoModel` to use
    '''

    def __init__(self, d):
        self.model = self.parse_object(d, "model", etal.VideoModelConfig)


def _apply_video_model(config):
    # Build model
    model = config.parameters.model.build()
    logger.info("Loaded model %s", type(model))
    if not isinstance(model, etal.VideoModel):
        raise ValueError("Model must be a %s" % etal.VideoModel)

    # Process videos
    with model:
        for data in config.data:
            _process_video(data, model)


def _process_video(data, model):
    # Load labels
    if data.input_labels_path:
        logger.info(
            "Reading existing labels from '%s'", data.input_labels_path)
        labels = etav.VideoLabels.from_json(data.input_labels_path)
    else:
        labels = etav.VideoLabels()

    logger.info("Applying model to video '%s'", data.video_path)
    labels.merge_video_labels(model.process(data.video_path))

    logger.info("Writing labels to '%s'", data.output_labels_path)
    labels.write_json(data.output_labels_path)


def run(config_path, pipeline_config_path=None):
    '''Run the apply_video_model module.

    Args:
        config_path: path to a ApplyVideoModelConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    config = ApplyVideoModelConfig.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)
    _apply_video_model(config)


if __name__ == "__main__":
    run(*sys.argv[1:])
