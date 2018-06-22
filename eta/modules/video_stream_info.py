#!/usr/bin/env python
'''
Module that gets stream info for a video.

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
import sys

from eta.core.config import Config
import eta.core.module as etam
import eta.core.video as etav


logger = logging.getLogger(__name__)


class VideoStreamInfoConfig(etam.BaseModuleConfig):
    '''Video stream info configuration settings.

    Attributes:
        data (DataConfig)
    '''

    def __init__(self, d):
        super(VideoStreamInfoConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        video (eta.core.types.Video): The input video

    Outputs:
        video_stream_info (eta.core.types.VideoStreamInfo): The video stream
            info
    '''

    def __init__(self, d):
        self.video = self.parse_string(d, "video")
        self.video_stream_info = self.parse_string(d, "video_stream_info")


def _get_stream_info(stream_info_config):
    for data_config in stream_info_config.data:
        logger.info("Reading stream info for %s", data_config.video)
        vsi = etav.VideoStreamInfo.build_for(data_config.video)
        vsi.write_json(data_config.video_stream_info)


def run(config_path, pipeline_config_path=None):
    '''Run the video_stream_info module.

    Args:
        config_path: path to a VideoStreamInfoConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    stream_info_config = VideoStreamInfoConfig.from_json(config_path)
    etam.setup(stream_info_config, pipeline_config_path=pipeline_config_path)
    _get_stream_info(stream_info_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
