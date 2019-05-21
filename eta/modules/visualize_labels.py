'''
A module for visualizing labeled videos.

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

import eta.core.annotations as etaa
from eta.core.config import Config
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
        video_path (eta.core.types.Video): A video
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
        annotation_config (eta.core.types.Config): [None] an
            `eta.core.annotations.AnnotationConfig` describing how to render
            the annotations on the video. If omitted, the default settings are
            used
    '''

    def __init__(self, d):
        self.annotation_config = self.parse_object(
            d, "annotation_config", etaa.AnnotationConfig, default=None)


def _visualize_labels(config):
    annotation_config = config.parameters.annotation_config
    for data in config.data:
        _process_video(data, annotation_config)


def _process_video(data, annotation_config):
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
        data.video_path, labels, data.output_path,
        annotation_config=annotation_config)


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
