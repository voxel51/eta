#!/usr/bin/env python
'''
Module for embedding videos into the fc6 vgg16 vector space.

Copyright 2017, Voxel51, LLC
voxel51.com

Jason Corso, jjc@voxel51.com
Brian Moore, brian@voxel51.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import argparse

import tensorflow as tf

from eta.core.config import Config
import eta.core.serial as se
import eta.core.video as vd
import eta.core.vgg16 as vgg


class EmbedVGG16Config(Config):
    '''VGG16 embedding configuration settings.

    This is basically the VGG16FeaturizerConfig except that it allows for an
    array of videos to be processed/featurized.
    '''

    def __init__(self, d):
        self.video_featurizers = self.parse_object_array(
            d, "video_featurizers", vd.VideoFeaturizerConfig)
        self.vgg16 = self.parse_object(
            d, "vgg16", vgg.VGG16Config, default=None)
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


def crop(crop_box):
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


def featurize_driver(config):
    '''For each of the featurizers in the config, creates a VGG16Featurizer and
    processes the video.

    Args:
        config: a config file containing the fields to define both an
            EmbedVGG16Config and a VGG16FeaturizerConfig

    @todo Note that I need to manually create the configs for the featurizer as
    I loop through the set of them from this config. This is probably not the
    cleanest way of doing it, but alas, it is doing it... A better way?

    @todo This creates a whole new VGG net for each video, which is not
    necessary. This could somehow reuse the vgg network instance for each
    featurizer.
    '''
    d = se.read_json(args.config)
    config = EmbedVGG16Config(d)

    for avfc in config.video_featurizers:
        # Needed to avoid running out of memory.
        # The proper fix is reuse the vgg-net across featurizers, but this
        # works and keeps the VGG16Featurizer easy to understand.
        tf.reset_default_graph()

        vfc = vgg.VGG16FeaturizerConfig(d, vfconfig=avfc)
        vf = vgg.VGG16Featurizer(vfc)
        if config.crop_box is not None:
            vf.frame_preprocessor = crop(config.crop_box)

        # @todo should frames be a part of the config?
        vf.featurize(frames="*")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='embed_vgg16', add_help=True)
    parser.add_argument(
        'config', help='Name of the config file needed to run the program')
    args = parser.parse_args()

    featurize_driver(args.config)
