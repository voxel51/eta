#!/usr/bin/env python
"""
Example of embedding the frames of a video in the VGG-16 feature space using
`CachingVideoFeaturizer`.

Also shows the use of the `frame_preprocessor` functionality to embed a
cropped version of each frame.

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
import os
import sys

from eta.core.config import Config
import eta.core.features as etaf


logger = logging.getLogger(__name__)


class EmbedVideoConfig(Config):
    """Embedding configuration settings."""

    def __init__(self, d):
        self.video_path = self.parse_string(d, "video_path")
        self.caching_video_featurizer = self.parse_object(
            d, "caching_video_featurizer", etaf.CachingVideoFeaturizerConfig
        )


def embed_video(config):
    """Embeds each frame of the video using VGG-16 and stores the embeddings
    on disk using `eta.core.features.CachingVideoFeaturizer`

    Args:
        config: an EmbedConfig instance
    """

    def _crop(img):
        return img[10:100, 10:100, :]

    # Build the featurizer
    cvf = etaf.CachingVideoFeaturizer(config.caching_video_featurizer)

    #
    # Invoke the featurizer using the `with` syntax so that `start()` and
    # `stop()` are automatically called
    #
    with cvf:
        # Use a preprocessor on each frame
        cvf.frame_preprocessor = _crop

        logger.info("Featurizing frames 1-6")
        cvf.featurize(config.video_path, frames="1-6")

        logger.info("Loading feature for frame 1")
        logger.info(cvf.load_feature_for_frame(1))

        logger.info("Loading feature for frames 2-6")
        logger.info(cvf.load_features_for_frames((2, 6)))

    logger.info("Inspect '%s' to see features on disk", cvf.backing_dir)


def _abspath(path):
    return os.path.realpath(os.path.join(os.path.dirname(__file__), path))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
    else:
        config_path = _abspath("embed_video-config.json")

    config = EmbedVideoConfig.from_json(config_path)
    embed_video(config)
