#!/usr/bin/env python
"""
A module for getting the stream info for a video.

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

from eta.core.config import Config
import eta.core.module as etam
import eta.core.video as etav


logger = logging.getLogger(__name__)


class ModuleConfig(etam.BaseModuleConfig):
    """Module configuration settings.

    Attributes:
        data (DataConfig)
    """

    def __init__(self, d):
        super(ModuleConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    """Data configuration settings.

    Inputs:
        video (eta.core.types.Video): The input video

    Outputs:
        stream_info (eta.core.types.VideoStreamInfo): The video stream info
    """

    def __init__(self, d):
        self.video = self.parse_string(d, "video")
        self.stream_info = self.parse_string(d, "stream_info")


def _video_stream_info(config):
    for data in config.data:
        logger.info("Reading stream info for %s", data.video)
        vsi = etav.VideoStreamInfo.build_for(data.video)
        vsi.write_json(data.stream_info)


def run(config_path, pipeline_config_path=None):
    """Run the video_stream_info module.

    Args:
        config_path: path to a ModuleConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    """
    config = ModuleConfig.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)
    _video_stream_info(config)


if __name__ == "__main__":
    run(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
