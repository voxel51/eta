#!/usr/bin/env python
'''
ETA example image embedding via VGG16.

Note that the embed_video.py example shows the use of the VideoFeaturization
classes, which is the preferred approach for ETA modules since they maintain
the on-disk backing store for the class which is used to communication between
modules.

Also shows an example of starting a TensorFlow session.

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

import tensorflow as tf
import numpy as np

import eta.core.image as im
from eta.core import utils
from eta.core.vgg16 import VGG16


logger = logging.getLogger(__name__)


def embed_image(impath):
    '''Embeds the image using VGG16. Uses the default weights specified in the
    default ETA config. Stores the embedded vector as .npz, using
    VideoFeaturizer to handle I/O.

    Args:
        impath: path to an image to embed
    '''
    img = im.read(impath)

    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    sess = tf.Session()
    vggn = VGG16(imgs, sess)

    rimg = im.resize(img, 224, 224)

    embedded_vector = sess.run(vggn.fc2l, feed_dict={vggn.imgs: [rimg]})[0]

    logger.info("image embedded to vector of length %d", len(embedded_vector))
    logger.info("%s", embedded_vector)

    outpath = _abspath("out/result_embed_image.npz")
    utils.ensure_basedir(outpath)
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
