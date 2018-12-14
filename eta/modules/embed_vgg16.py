#!/usr/bin/env python
'''
A module for embedding videos into the VGG-16 feature space.

Info:
    type: eta.core.types.Module
    version: 0.1.0

Copyright 2017-2018, Voxel51, Inc.
voxel51.com

Jason Corso, jjc@voxel51.com
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

import tensorflow as tf

from eta.core.config import Config
import eta.core.features as etaf
import eta.core.module as etam
import eta.core.serial as etas
import eta.core.vgg16 as etav


logger = logging.getLogger(__name__)


class EmbedVGG16Config(etam.BaseModuleConfig):
    '''VGG-16 embedding configuration settings.

    This is basically the VGG16FeaturizerConfig except that it allows for an
    array of videos to be processed/featurized.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(EmbedVGG16Config, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        video_path (eta.core.types.Video): The input video

    Outputs:
        backing_path (eta.core.types.Directory): The directory in which to
            write the embeddings
    '''

    def __init__(self, d):
        self.video_path = self.parse_string(d, "video_path")
        self.backing_path = self.parse_string(d, "backing_path")


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        crop_box (eta.core.types.Object): [None] A region of interest of
            each frame to extract before embedding
    '''

    def __init__(self, d):
        self.vgg16 = self.parse_object(
                d, "vgg16", etav.VGG16Config, default=None)
        self.crop_box = self.parse_object(
                d, "crop_box", RectangleConfig, default=None)


class Point2Config(Config):
    '''A simple 2D point.'''

    def __init__(self, d):
        self.x = self.parse_number(d, "x")
        self.y = self.parse_number(d, "y")


class RectangleConfig(Config):
    '''A rectangle defined by two Point2Configs.'''

    def __init__(self, d):
        self.top_left = self.parse_object(d, "top_left", Point2Config)
        self.bottom_right = self.parse_object(d, "bottom_right", Point2Config)


def _featurize_driver(config, d):
    '''For each of the featurizers in the config, creates a VGG16Featurizer and
    processes the video.

    @todo Note that I need to manually create the configs for the featurizer as
    I loop through the set of them from this config. This is probably not the
    cleanest way of doing it, but alas, it is doing it... A better way?

    @todo This creates a whole new VGG net for each video, which is not
    necessary. This could somehow reuse the vgg network instance for each
    featurizer.
    '''
    parameters = config.parameters
    for data in config.data:
        # Needed to avoid running out of memory.
        # The proper fix is reuse the vgg-net across featurizers, but this
        # works and keeps the VGG16Featurizer easy to understand.
        tf.reset_default_graph()

        vffcd_ = {"type": "eta.core.vgg16.VGG16Featurizer"}
        if parameters.vgg16 is None:
            vffcd_["config"] = {}
        else:
            vffcd_["config"] = parameters.vgg16

        vffcd = {
            "backing_path": data.backing_path,
            "frame_featurizer": vffcd_,
        }

        vffc = etaf.VideoFramesFeaturizerConfig(vffcd)
        vf = etaf.VideoFramesFeaturizer(vffc)
        if parameters.crop_box is not None:
            vf.frame_preprocessor = _crop(parameters.crop_box)

        # @todo should frames be a part of the config?
        vf.featurize(data.video_path)


def _crop(crop_box):
    '''This is a curried function to allow for a crop to be parameterized
    without making it into a complex class or what have you and still sent to
    the VideoFeaturizer as a hook.
    '''

    def crop_image(img):
        tl = crop_box.top_left
        br = crop_box.bottom_right
        xs = img.shape[1]
        ys = img.shape[0]
        return img[
            int(tl.y * ys):int(br.y * ys),
            int(tl.x * xs):int(br.x * xs),
        ]

    return crop_image


def run(config_path, pipeline_config_path=None):
    '''Run the embed_vgg16 module.

    Args:
        config_path: path to a config file containing the fields to define
            both an EmbedVGG16Config and a VGG16FeaturizerConfig
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    d = etas.read_json(config_path)
    embed_vgg16_config = EmbedVGG16Config(d)
    etam.setup(embed_vgg16_config, pipeline_config_path=pipeline_config_path)
    _featurize_driver(embed_vgg16_config, d)


if __name__ == '__main__':
    run(*sys.argv[1:])
