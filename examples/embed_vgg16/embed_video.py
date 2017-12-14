#!/usr/bin/env python
'''
ETA example (frame-by-frame) video embbeding via VGG16.

Note: must be run from this directory!

Also shows the use of the frame_preprocessor functionality in VGG16Featurizer.

Copyright 2017, Voxel51, LLC
voxel51.com

Jason Corso, jjc@voxel51.com
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

from eta.core.config import Config
from eta.core.features import VideoFramesFeaturizer, \
                              VideoFramesFeaturizerConfig
import eta.core.log as log


logger = logging.getLogger(__name__)
log.basic_setup(level=logging.DEBUG)


class EmbedVideoConfig(Config):
    def __init__(self, d):
        self.video_path = self.parse_string(d, "video_path")
        self.video_frames_featurizer = self.parse_object(
                d, "video_frames_featurizer", VideoFramesFeaturizerConfig)


def embed_video(config):
    '''Embeds each frame of the video using the VGG16-net. Uses the default
    weights specified in the default ETA config. Stores the embedded vector as
    .npz, using VideoFeaturizer to handle I/O.

    Args:
        config: an EmbedConfig instance
    '''
    def _crop(img):
        return img[10:100, 10:100, :]


    vff = VideoFramesFeaturizer(config.video_frames_featurizer)
    vff.frame_preprocessor = _crop
    # the following call is not needed in most cases (or this one); it is just
    # here to force the refeaturization of the frames.
    vff.flush_backing()
    vff.featurize(config.video_path, frames="1-6")

    # Note that after the above call frames 1-6 are featurized.  Here, we will
    # featurize only from 7-9, even tho we say 4-9, because the other ones were
    # already computed and cached.
    vff.featurize(config.video_path, frames="4-9")

    logger.info(
        "features stored in '%s'", config.video_frames_featurizer.backing_path)


def _abspath(path):
    return os.path.realpath(os.path.join(os.path.dirname(__file__), path))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
    else:
        config_path = _abspath("embed_video-config.json")

    embed_video(EmbedVideoConfig.from_json(config_path))
