'''
A module for visualizing labeled objects in videos.

Info:
    type: eta.core.types.Module
    version: 0.1.0

Copyright 2018-2019, Voxel51, Inc.
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

import eta.core.annotations as etaa
from eta.core.config import Config
from eta.core.logo import Logo, LogoConfig
import eta.core.module as etam
import eta.core.objects as etao
import eta.core.video as etav


logger = logging.getLogger(__name__)


class VisualizeLabelsConfig(etam.BaseModuleConfig):
    '''Label visualization configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(VisualizeLabelsConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        video_path (eta.core.types.Video): [None] A video
        video_labels_path (eta.core.types.VideoLabels): [None] A JSON file
            containing the video labels
        objects_path (eta.core.types.DetectedObjects): [None] A JSON file
            containing the detected objects

    Outputs:
        output_path (eta.core.types.VideoFile): The labeled video
    '''

    def __init__(self, d):
        self.video_path = self.parse_string(d, "video_path")
        self.video_labels_path = self.parse_string(
            d, "video_labels_path", default=None)
        self.objects_path = self.parse_string(d, "objects_path", default=None)
        self.output_path = self.parse_string(d, "output_path")


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        add_logo (eta.core.types.Boolean): [True] whether to render a logo on
            the video
        logo (eta.core.types.Object): [None] a LogoConfig defining a logo to
            render on the video. If omitted, the default logo is used
    '''

    def __init__(self, d):
        self.add_logo = self.parse_bool(d, "add_logo", default=True)
        self.logo = self.parse_object(d, "logo", LogoConfig, default=None)


def _visualize_labels(config):
    # Parse logo
    add_logo = config.parameters.add_logo
    if add_logo and config.parameters.logo is not None:
        logo = Logo.from_config(config.parameters.logo)
    else:
        logo = None

    for data in config.data:
        _process_video(data, add_logo, logo)


def _process_video(data, add_logo, logo):
    # Load labels
    if data.video_labels_path:
        labels = etav.VideoLabels.from_json(data.video_labels_path)
    elif data.objects_path:
        objects = etao.DetectedObjectContainer.from_json(data.objects_path)
        labels = etav.VideoLabels.from_detected_objects(objects)
    else:
        raise ValueError(
            "Must supply one of 'video_labels_path' or 'objects_path'")

    # Annotate video
    etaa.annotate_video(
        data.video_path, labels, data.output_path, add_logo=add_logo,
        logo=logo)


def run(config_path, pipeline_config_path=None):
    '''Run the visualize_labels module.

    Args:
        config_path: path to a VisualizeLabelsConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    config = VisualizeLabelsConfig.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)
    _visualize_labels(config)


if __name__ == "__main__":
    run(*sys.argv[1:])
