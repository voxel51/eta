#!/usr/bin/env python
'''
Module for embedding videos into the fc6 vgg16 vector space.

Copyright 2017, Voxel51, LLC
voxel51.com

Jason Corso, jjc@voxel51.com
'''
import argparse
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from eta.core.config import Config
import eta.core.image    as im
import eta.core.utils    as ut
import eta.core.video    as vd
import eta.core.vgg16    as vgg


class EmbedVGG16Config(Config):
    '''Configuration settings.

    This is basically the VGG16FeaturizerConfig except that it
    allows for an array of videos to be processed/featurized.
    '''

    def __init__(self, d):
        self.video_featurizers = self.parse_object_array(d, "video_featurizers", vd.VideoFeaturizerConfig)
        self.vgg16config = self.parse_object(d, "vgg16",vgg.VGG16Config,default=None)
        self.crop_box    = self.parse_object(d, "crop_box",RectangleConfig, default=None)


class Point2Config(Config):
    ''' Simple 2D Coordinate
        @todo move to core
    '''
    def __init__(self,d):
        self.y = self.parse_number(d,"y")
        self.x = self.parse_number(d,"x")


class RectangleConfig(Config):
    ''' Uses the Config infrastructure to define a rectangular box
        @todo move to core
    '''

    def __init__(self, d):
        self.top_left = self.parse_object(d, "top_left", Point2Config)
        self.bottom_right = self.parse_object(d, "bottom_right", Point2Config)


# debug code to check image crop
#c = 1


def crop(cropbox):
    ''' This is a curried function to allow for a crop to be parameterized
        without making it into a complex class or what have you and still
        sent to the VideoFeaturizer as a hook.
    '''
    def crop_image(image):
        ys = image.shape[0]
        xs = image.shape[1]
        return  image[int(cropbox.top_left.y*ys):int(cropbox.bottom_right.y*ys),
                      int(cropbox.top_left.x*xs):int(cropbox.bottom_right.x*xs),:]
        # debug code to check image crop
        #print "%d:%d %d:%d\n" % (int(cropbox.top_left.y*ys),int(cropbox.bottom_right.y*ys), int(cropbox.top_left.x*xs),int(cropbox.bottom_right.x*xs))
        #global c
        #a =  image[int(cropbox.top_left.y*ys):int(cropbox.bottom_right.y*ys),
        #           int(cropbox.top_left.x*xs):int(cropbox.bottom_right.x*xs),:]
        #imsave('/tmp/testvf/%05d.png'%c,a)
        #c = c + 1
        #return a
    return crop_image



def featurize_driver(args):
    '''
        For each of the featurizers in the config setting, creates a VGG16Featurizer
        and processes the video.

        @param args.config Path to the config file for the VGG16 Featurizer, which
        contains information for the network, video, and the backing location
        for the featurized video

        Note that I need to manually create the configs for the featurizer as I
        loop through the set of them from this config.  This is probably not the
        cleanest way of doing it, but alas, it is doing it...  A better way?

        @todo This also creates a whole new VGG net for each video, which is not
        necessary.  This could somehow reuse the vgg network instance for each
        featurizer.
    '''

    d = ut.read_json(args.config)
    config = EmbedVGG16Config(d)

    for avfc in config.video_featurizers:
        tf.reset_default_graph()  # needed to avoid going over memory
                                  # proper fix is reuse the vgg-net across featurizers
                                  # but this works and keeps the VGG16Featurizer easy
                                  #  to read and understand
        vfc = vgg.VGG16FeaturizerConfig(d,vfconfig=avfc)
        vf = vgg.VGG16Featurizer(vfc)
        if config.crop_box is not None:
            vf.frame_preprocessor = crop(config.crop_box)
        vf.featurize(frames="*")  # should frames be a part of the video_featurizer config?


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='embed_vgg16', add_help=True)
    parser.add_argument('config',help='Name of the config file needed to run the program')
    args = parser.parse_args()

    featurize_driver(args)

