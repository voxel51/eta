#!/usr/bin/env python
'''
Resize videos.

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
import eta.core.module as etam
import eta.core.video as etav


logger = logging.getLogger(__name__)


class ResizeConfig(etam.BaseModuleConfig):
    '''Resize configuration settings.'''

    def __init__(self, d):
        super(ResizeConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.'''

    def __init__(self, d):
        self.input_path = self.parse_string(d, "input_path")
        self.output_path = self.parse_string(d, "output_path")


class ParametersConfig(Config):
    '''Parameter configuration settings.'''

    def __init__(self, d):
        self.size = self.parse_array(d, "size", default=None)
        self.scale = self.parse_number(d, "scale", default=None)
        self.scale_str = self.parse_string(d, "scale_str", default=None)
        self.ffmpeg_out_opts = self.parse_array(
            d, "ffmpeg_out_opts", default=None)


def _resize_videos(resize_config):
    parameters = resize_config.parameters
    for data_config in resize_config.data:
        logger.info("Resizing video '%s'", data_config.input_path)
        etav.FFmpegVideoResizer(
            size=parameters.size,
            scale=parameters.scale,
            scale_str=parameters.scale_str,
            out_opts=parameters.ffmpeg_out_opts,
        ).run(
            data_config.input_path,
            data_config.output_path,
        )


def run(config_path, pipeline_config_path=None):
    '''Run the resize_videos module.

    Args:
        config_path: path to a ResizeConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    resize_config = ResizeConfig.from_json(config_path)
    etam.setup(resize_config, pipeline_config_path=pipeline_config_path)
    _resize_videos(resize_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
