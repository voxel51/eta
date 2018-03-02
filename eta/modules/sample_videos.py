#!/usr/bin/env python
'''
Sample video frames.

Copyright 2017-2018, Voxel51, LLC
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
import eta.core.events as ev
import eta.core.module as mo
import eta.core.video as vd


logger = logging.getLogger(__name__)


def run(config_path, pipeline_config_path=None):
    '''Run the sample_videos module.

    Args:
        config_path: path to a SampleConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    sample_config = SampleConfig.from_json(config_path)
    mo.setup(sample_config, pipeline_config_path=pipeline_config_path)
    _sample_videos(sample_config)


def _sample_videos(sample_config):
    parameters = sample_config.parameters
    for data_config in sample_config.data:
        if data_config.fps != -1:
            _sample_video_by_fps(data_config, parameters)
        else:
            _sample_video_by_clips(data_config, parameters)


def _sample_video_by_fps(data_config, parameters):
    assert parameters.fps != -1, "Must provide 'fps'"
    logger.info(
        "Sampling video '%s' at %s fps",
        data_config.input_path, parameters.fps)

    vd.FFmpegVideoSampler(fps=parameters.fps).run(
        data_config.input_path,
        data_config.output_path,
    )


def _sample_video_by_clips(data_config, parameters):
    assert parameters.clips_path is not None, "Must provide 'clips_path'"
    logger.info(
        "Sampling video '%s' by clips '%s'",
        data_config.input_path, parameters.clips_path)

    detections = ev.EventDetection.from_json(parameters.clips_path)
    frames = detections.to_series().to_str()

    processor = vd.VideoProcessor(
        data_config.input_path,
        frames=frames,
        out_impath=data_config.output_path,
    )
    with processor:
        for img in processor:
            processor.write(img)


class SampleConfig(mo.BaseModuleConfig):
    '''Sampler configuration settings.'''

    def __init__(self, d):
        super(SampleConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.'''

    def __init__(self, d):
        self.input_path = self.parse_string(d, "input_path")
        self.output_path = self.parse_string(d, "output_path")


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Either `fps` or `clips_path` must be specified.

    @todo add a fps/clips module keyword to handle each case separately.
    '''

    def __init__(self, d):
        self.fps = self.parse_number(d, "fps", default=-1)
        self.clips_path = self.parse_string(d, "clips_path", default=None)


if __name__ == "__main__":
    run(*sys.argv[1:])
