#!/usr/bin/env python
"""
Example of embedding an image via a `VGG16Featurizer`.

Note that the `embed_video.py` example shows the use of the video featurization
infrastructure, which is the preferred approach for ETA modules since they
maintain the on-disk backing store for the class which is used to communicate
between modules.

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

import numpy as np

import eta.core.image as etai
import eta.core.utils as etau
from eta.core.vgg16 import VGG16Featurizer


logger = logging.getLogger(__name__)


def embed_image(impath):
    """Embeds the image using VGG-16 and stores the embeddeding as an .npy file
    on disk, using VideoFeaturizer to handle I/O.

    Args:
        impath: path to an image to embed
    """
    img = etai.read(impath)

    # Invoke the Featurizer using the "with" syntax to automatically handle
    # calling the start() and stop() methods
    with VGG16Featurizer() as featurizer:
        embedding = featurizer.featurize(img)

    logger.info("Image embedded to vector of length %d", len(embedding))
    logger.info("%s", embedding)

    outpath = _abspath("out/result_embed_image.npy")
    etau.ensure_basedir(outpath)
    np.save(outpath, embedding)
    logger.info("Result saved to '%s'", outpath)


def _abspath(path):
    return os.path.realpath(os.path.join(os.path.dirname(__file__), path))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        impath = sys.argv[1]
    else:
        impath = _abspath("../data/water.jpg")

    embed_image(impath)
