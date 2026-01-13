#!/usr/bin/env python
"""
Example of embedding an image manually via `VGG16`.

This example has the same effect as `embed_image.py` except that it directly
uses the low-level functionality rather than the higher level `Featurizer`
functionality. It is included here for pedagogical reasons with ETA.

Copyright 2017-2026, Voxel51, Inc.
voxel51.com
"""

import logging
import os
import sys

import numpy as np

import eta.core.image as etai
import eta.core.utils as etau
import eta.core.vgg16 as etav


logger = logging.getLogger(__name__)


def embed_image(impath):
    """Embeds the image using VGG-16 and stores the embeddeding as an .npy file
    on disk.

    Args:
        impath: path to an image to embed
    """
    img = etai.read(impath)
    rimg = etai.resize(img, 224, 224)

    vgg16 = etav.VGG16()
    embedding = vgg16.evaluate([rimg], [vgg16.fc2l])[0][0]

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
