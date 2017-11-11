#!/usr/bin/env python
'''
Resize videos.

Copyright 2017, Voxel51, LLC
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
import eta.core.module as mo
import eta.core.video as vd


logger = logging.getLogger(__name__)


def run(config_path):
    '''Run the resize_videos module.

    Args:
        config_path: path to a ResizeConfig file
    '''
    resize_config = ResizeConfig.from_json(config_path)
    mo.setup(resize_config)
    _resize_videos(resize_config)


def _resize_videos(resize_config):
    for data_config in resize_config.data:
        logger.info("Resizing video '%s'", data_config.input_path)
        vd.FFmpegVideoResizer(
            size=data_config.size,
            scale=data_config.scale,
            scale_str=data_config.scale_str,
            out_opts=data_config.ffmpeg_out_opts,
        ).run(
            data_config.input_path,
            data_config.output_path,
        )


class ResizeConfig(mo.BaseModuleConfig):
    '''Resize configuration settings.'''

    def __init__(self, d):
        super(ResizeConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    '''Data configuration settings.'''

    def __init__(self, d):
        self.input_path = self.parse_string(d, "input_path")
        self.output_path = self.parse_string(d, "output_path")
        self.size = self.parse_array(d, "size", default=None)
        self.scale = self.parse_number(d, "scale", default=None)
        self.scale_str = self.parse_string(d, "scale_str", default=None)
        self.ffmpeg_out_opts = self.parse_array(
            d, "ffmpeg_out_opts", default=None)


if __name__ == "__main__":
    run(sys.argv[1])
