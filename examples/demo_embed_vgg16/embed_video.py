#!/usr/bin/env python
'''
Example of embedding the frames of a video in the VGG-16 feature space using
`VideoFramesFeaturizer`.

Also shows the use of the `VGG16Featurizer.frame_preprocessor` functionality
to embed a cropped version of each frame.

Copyright 2017-2018, Voxel51, LLC
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
import os
import sys

from eta.core.config import Config
import eta.core.features as etaf
import eta.core.log as etal


logger = logging.getLogger(__name__)
etal.basic_setup(level=logging.DEBUG)


class EmbedVideoConfig(Config):
    '''Embedding configuration settings.'''

    def __init__(self, d):
        self.video_path = self.parse_string(d, "video_path")
        self.video_frames_featurizer = self.parse_object(
            d, "video_frames_featurizer", etaf.VideoFramesFeaturizerConfig)


def embed_video(config):
    '''Embeds each frame of the video using VGG-16 and stores the embeddedings
    as .npz files on disk, using VideoFeaturizer to handle I/O.

    Args:
        config: an EmbedConfig instance
    '''
    def _crop(img):
        return img[10:100, 10:100, :]

    # Invoke the VideoFramesFeaturizer using the with syntax so that start()
    # and stop() are automatically called
    with etaf.VideoFramesFeaturizer(config.video_frames_featurizer) as vff:
        # Use a preprocessor on each frame
        vff.frame_preprocessor = _crop

        # This call is not needed in general. We do it here to force
        # refeaturization of the frames for demonstration purposes
        vff.flush_backing()

        # Featurize frames 1-6
        vff.featurize(config.video_path, frames="1-6")

        # Featurize frames 4-9. Since frames 4-6 have already been featurized,
        # they are skipped this time around
        vff.featurize(config.video_path, frames="4-9")

    logger.info(
        "Features stored in '%s'", config.video_frames_featurizer.backing_path)


def _abspath(path):
    return os.path.realpath(os.path.join(os.path.dirname(__file__), path))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
    else:
        config_path = _abspath("embed_video-config.json")

    config = EmbedVideoConfig.from_json(config_path)
    embed_video(config)
