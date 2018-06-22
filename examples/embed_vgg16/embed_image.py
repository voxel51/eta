#!/usr/bin/env python
'''
ETA example image embedding via VGG16.

This example has the same effect as embed_image.py except that it directly uses
the low-level functionality rather than the higher level Featurization
functionality.  It is included here for pedagogical reasons with ETA.

Copyright 2017, Voxel51, LLC
voxel51.com
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

import tensorflow as tf
import numpy as np

import eta.core.image as etai
import eta.core.utils as etau
from eta.core.vgg16 import VGG16Featurizer


logger = logging.getLogger(__name__)


def embed_image(impath):
    '''Embeds the image using VGG16. Uses the default weights specified in the
    default ETA config. Stores the embedded vector as .npz, using
    VideoFeaturizer to handle I/O.

    Args:
        impath: path to an image to embed
    '''
    img = etai.read(impath)

    vfeaturizer = VGG16Featurizer()

    embedded_vector = vfeaturizer.featurize(img)

    logger.info("image embedded to vector of length %d", len(embedded_vector))
    logger.info("%s", embedded_vector)

    outpath = _abspath("out/result_embed_image.npz")
    etau.ensure_basedir(outpath)
    np.savez_compressed(outpath, v=embedded_vector)
    logger.info("result saved to '%s'", outpath)


def _abspath(path):
    return os.path.realpath(os.path.join(os.path.dirname(__file__), path))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        impath = sys.argv[1]
    else:
        impath = _abspath("../data/water.jpg")

    embed_image(impath)
