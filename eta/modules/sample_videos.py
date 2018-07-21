#!/usr/bin/env python
'''
Module for (re)sampling videos.

Info:
    type: eta.core.types.Module
    version: 0.1.0

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
import os
import sys

import eta
from eta.core.config import Config
import eta.core.events as etae
import eta.core.image as etai
import eta.core.module as etam
import eta.core.utils as etau
import eta.core.video as etav


logger = logging.getLogger(__name__)


class SampleConfig(etam.BaseModuleConfig):
    '''Sampler configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(SampleConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Exactly one output path must be provided.

    Inputs:
        input_path (eta.core.types.Video): The input video

    Outputs:
        output_video_path (eta.core.types.VideoFile): [None] The output sampled
            video
        output_frames_dir (eta.core.types.ImageSequenceDirectory): [None] A
            directory in which to write the sampled frames
        output_frames_path (eta.core.types.ImageSequence): [None] The output
            sampled frames pattern
    '''

    def __init__(self, d):
        self.input_path = self.parse_string(d, "input_path")
        self.output_video_path = self.parse_string(
            d, "output_video_path", default=None)
        self.output_frames_dir = self.parse_string(
            d, "output_frames_dir", default=None)
        self.output_frames_path = self.parse_string(
            d, "output_frames_path", default=None)

        self._output_field = None
        self._output_val = None
        self._parse_outputs()

    @property
    def output_field(self):
        return self._output_field

    @property
    def output_path(self):
        return self._output_val

    def _parse_outputs(self):
        field, val = Config.parse_mutually_exclusive_fields({
            "output_video_path": self.output_video_path,
            "output_frames_dir": self.output_frames_dir,
            "output_frames_path": self.output_frames_path,
        })
        if field == "output_frames_dir":
            val = etai.make_image_sequence_patt(val)
        self._output_field = field
        self._output_val = val


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        fps (eta.core.types.Number): [None] The output frame rate
    '''

    def __init__(self, d):
        self.fps = self.parse_number(d, "fps", default=None)


def _sample_videos(sample_config):
    # if no fps is provided, then setting to 0 will maintain native fps
    fps = sample_config.parameters.fps or 0
    for data in sample_config.data:
        _sample_video(data.input_path, data.output_path, fps)


def _sample_video(input_path, output_path, fps):
    if fps > 0:
        logger.info(
            "Sampling video %s at %s fps", input_path, fps)
    else:
        logger.info(
            "Retaining the native frame rate of '%s'", input_path)
        if etav.is_same_video_format(input_path, output_path):
            logger.info(
                "Same video format detected, so no computation is required. "
                "Just sylimking %s to %s" % (output_path, input_path))
            etau.symlink_file(input_path, output_path)
            return

    etav.FFmpegVideoSampler(fps=fps).run(input_path, output_path)


def run(config_path, pipeline_config_path=None):
    '''Run the sample_videos module.

    Args:
        config_path: path to a SampleConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    sample_config = SampleConfig.from_json(config_path)
    etam.setup(sample_config, pipeline_config_path=pipeline_config_path)
    _sample_videos(sample_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
