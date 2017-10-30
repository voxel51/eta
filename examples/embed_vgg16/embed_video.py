#!/usr/bin/env python
'''
ETA example (frame-by-frame) video embbeding via VGG16.

Note: must be run from this directory!

Also shows the use of the frame_preprocessor functionality in VGG16Featurizer.

Copyright 2017, Voxel51, LLC
voxel51.com

Jason Corso, jjc@voxel51.com
'''
import os
import sys

from eta.core.vgg16 import VGG16FeaturizerConfig, VGG16Featurizer


def embed_video(config):
    '''Embeds each frame of the video using the VGG16-net. Uses the default
    weights specified in the default ETA config. Stores the embedded vector as
    .npz, using VideoFeaturizer to handle I/O.

    Args:
        config: a VGG16FeaturizerConfig instance
    '''
    def _crop(img):
        return img[10:100, 10:100, :]

    vf = VGG16Featurizer(config)
    vf.frame_preprocessor = _crop
    vf.featurize(frames="1-12")


def _abspath(path):
    return os.path.realpath(os.path.join(os.path.dirname(__file__), path))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
    else:
        config_path = _abspath("embed_video-config.json")

    embed_video(VGG16FeaturizerConfig.from_json(config_path))
