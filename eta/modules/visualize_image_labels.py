#!/usr/bin/env python
"""
A module for visualizing labeled images.

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
import os
import sys

import eta.core.annotations as etaa
from eta.core.config import Config, ConfigError
import eta.core.image as etai
import eta.core.module as etam
import eta.core.utils as etau


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
        image_path (eta.core.types.Image): [None] An image
        images_dir (eta.core.types.ImageFileDirectory): [None] A directory of
            images
        image_labels_path (eta.core.types.ImageLabels): [None] A JSON file
            containing the labels for the image specified by `image_path`
        image_set_labels_path (eta.core.types.ImageSetLabels): [None] A JSON
            file containing the labels for each image in the directory
            specified by `images_dir`

    Outputs:
        output_path (eta.core.types.ImageFile): [None] The labeled version of
            the image specified by `image_path`
        output_dir (eta.core.types.ImageFileDirectory): [None] The labeled
            versions of the images specified by `images_dir`
    """

    def __init__(self, d):
        self.image_path = self.parse_string(d, "image_path", default=None)
        self.images_dir = self.parse_string(d, "images_dir", default=None)

        self.image_labels_path = self.parse_string(
            d, "image_labels_path", default=None
        )
        self.image_set_labels_path = self.parse_string(
            d, "image_set_labels_path", default=None
        )

        self.output_path = self.parse_string(d, "output_path", default=None)
        self.output_dir = self.parse_string(d, "output_dir", default=None)

        self._validate()

    def _validate(self):
        if self.image_path:
            if not self.image_labels_path:
                raise ConfigError(
                    "`image_labels_path` is required when `image_path` is set"
                )
            if not self.output_path:
                raise ConfigError(
                    "`output_path` is required when `image_path` is set"
                )

        if self.images_dir:
            if not self.image_set_labels_path:
                raise ConfigError(
                    "`image_set_labels_path` is required when `images_dir` is "
                    "set"
                )
            if not self.output_dir:
                raise ConfigError(
                    "`output_dir` is required when `images_dir` is set"
                )


class ParametersConfig(Config):
    """Parameter configuration settings.

    Parameters:
        annotation_config (eta.core.types.Config): [None] an
            `eta.core.annotations.AnnotationConfig` describing how to render
            the annotations on the images. If omitted, the default settings are
            used
    """

    def __init__(self, d):
        self.annotation_config = self.parse_object(
            d, "annotation_config", etaa.AnnotationConfig, default=None
        )


def _visualize_image_labels(config):
    annotation_config = config.parameters.annotation_config
    for data in config.data:
        if data.image_path:
            logger.info("Processing single image")
            _process_image(data, annotation_config)

        if data.images_dir:
            logger.info("Processing directory of images")
            _process_images_dir(data, annotation_config)


def _process_image(data, annotation_config):
    logger.info("Loading ImageLabels from '%s'", data.image_labels_path)
    image_labels = etai.ImageLabels.from_json(data.image_labels_path)

    logger.info("Annotating image '%s'", data.image_path)
    img = etai.read(data.image_path)
    img_anno = etaa.annotate_image(
        img, image_labels, annotation_config=annotation_config
    )

    logger.info("Writing annotated image to '%s'", data.output_path)
    etai.write(img_anno, data.output_path)


def _process_images_dir(data, annotation_config):
    # Load labels
    logger.info("Loading ImageSetLabels from '%s'", data.image_set_labels_path)
    image_set_labels = etai.ImageSetLabels.from_json(
        data.image_set_labels_path
    )

    # Label images
    image_filenames = set(etau.list_files(data.images_dir))
    for image_labels in image_set_labels:
        filename = image_labels.filename
        if not filename:
            logger.warning("Skipping ImageLabels with no `filename`")
            continue
        if filename not in image_filenames:
            logger.warning(
                "Skipping '%s'; not found in '%s'", filename, data.images_dir
            )
            continue

        inpath = os.path.join(data.images_dir, filename)
        outpath = os.path.join(data.output_dir, filename)

        logger.info("Annotating image '%s'", inpath)
        img = etai.read(inpath)
        img_anno = etaa.annotate_image(
            img, image_labels, annotation_config=annotation_config
        )

        logger.info("Writing annotated image to '%s'", outpath)
        etai.write(img_anno, outpath)


def run(config_path, pipeline_config_path=None):
    """Run the visualize_image_labels module.

    Args:
        config_path: path to a ModuleConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    """
    config = ModuleConfig.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)
    _visualize_image_labels(config)


if __name__ == "__main__":
    run(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
