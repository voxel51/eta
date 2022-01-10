#!/usr/bin/env python
"""
A module for embedding videos into the VGG-16 feature space.

Info:
    type: eta.core.types.Module
    version: 0.1.0

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
import sys

from eta.core.config import Config
import eta.core.features as etaf
import eta.core.module as etam
import eta.core.utils as etau
import eta.core.vgg16 as etav


logger = logging.getLogger(__name__)


class ModuleConfig(etam.BaseModuleConfig):
    """Module configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    """

    def __init__(self, d):
        super(ModuleConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    """Data configuration settings.

    Inputs:
        video_path (eta.core.types.Video): the input video

    Outputs:
        backing_dir (eta.core.types.Directory): the directory to write the
            embeddings
    """

    def __init__(self, d):
        self.video_path = self.parse_string(d, "video_path")
        self.backing_dir = self.parse_string(d, "backing_dir")


class ParametersConfig(Config):
    """Parameter configuration settings.

    Parameters:
        vgg16 (eta.core.types.Config): [None] an optional VGG16FeaturizerConfig
            describing the VGG16Featurizer to use
        crop_box (eta.core.types.Config): [None] an optional region of interest
            to extract from each frame before embedding
    """

    def __init__(self, d):
        self.vgg16 = self.parse_object(
            d, "vgg16", etav.VGG16FeaturizerConfig, default=None
        )
        self.crop_box = self.parse_object(
            d, "crop_box", RectangleConfig, default=None
        )


class Point2Config(Config):
    """A simple 2D point."""

    def __init__(self, d):
        self.x = self.parse_number(d, "x")
        self.y = self.parse_number(d, "y")


class RectangleConfig(Config):
    """A rectangle defined by two Point2Configs."""

    def __init__(self, d):
        self.top_left = self.parse_object(d, "top_left", Point2Config)
        self.bottom_right = self.parse_object(d, "bottom_right", Point2Config)


def _embed_vgg16(config):
    # Build featurizer
    frame_featurizer = (
        etaf.ImageFeaturizerConfig.builder()
        .set(type=etau.get_class_name(etav.VGG16Featurizer))
        .set(config=config.parameters.vgg16)
        .validate()
    )
    featurizer = etaf.CachingVideoFeaturizer(
        etaf.CachingVideoFeaturizerConfig.builder()
        .set(frame_featurizer=frame_featurizer)
        .set(delete_backing_directory=False)
        .build()
    )

    # Set crop box, if provided
    if config.parameters.crop_box is not None:
        featurizer.frame_preprocessor = _crop(config.parameters.crop_box)

    with featurizer:
        for data in config.data:
            # Manually set backing directory for each video
            featurizer.set_manual_backing_dir(data.backing_dir)

            logger.info("Featurizing video '%s'", data.video_path)
            featurizer.featurize(data.video_path)


def _crop(crop_box):
    def crop_image(img):
        tl = crop_box.top_left
        br = crop_box.bottom_right
        xs = img.shape[1]
        ys = img.shape[0]
        return img[
            int(tl.y * ys) : int(br.y * ys), int(tl.x * xs) : int(br.x * xs),
        ]

    return crop_image


def run(config_path, pipeline_config_path=None):
    """Run the embed_vgg16 module.

    Args:
        config_path: path to a config file containing the fields to define
            both an ModuleConfig and a VGG16FeaturizerConfig
        pipeline_config_path: optional path to a PipelineConfig file
    """
    config = ModuleConfig.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)
    _embed_vgg16(config)


if __name__ == "__main__":
    run(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
