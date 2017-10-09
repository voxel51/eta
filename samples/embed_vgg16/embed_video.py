#!/usr/bin/env python
'''
Example Code: Embed an video (frame-by-frame) via VGG16.

Also shows the use of the frame_preprocessor functionality in the vgg16 featurizer

Copyright 2017, Voxel51, LLC
voxel51.com

Jason Corso, jjc@voxel51.com
'''
import errno
import os
import sys

import tensorflow as tf
import numpy as np

from eta.core.config import Config
import eta.core.video as vd
import eta.core.vgg16 as vgg
import eta.core.image as im


def crop(img):
    ''' a quick function to show the use of frame_preprocessor in 
        VideoFeaturizer.
    '''
    return img[10:100,10:100,:]


def embed_video(config):
    '''
        Uses the default weights specified in the default config.
        Embed each
        frame of the video using the VGG16-net.  Store the embedded vectors as
        a npz.  Uses a VideoFeaturizer to handle IO and storage on disk

        @param config Path to the config file for the VGG16 Featurizer, which
        contains information for the network, video, and the backing location
        for the featurized video
    '''

    vf = vgg.VGG16Featurizer(config)
    vf.frame_preprocessor = crop
    vf.featurize(frames="1-12")


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"embed_video-config.json")
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
    embed_video(vgg.VGG16FeaturizerConfig.from_json(config_path))

