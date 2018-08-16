'''
ETA package initialization.

Copyright 2017-2018, Voxel51, LLC
voxel51.com

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
from future.utils import iteritems
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging
import os
import sys

import eta.constants as etac
from eta.core.config import Config, EnvConfig
import eta.core.log as etal
import eta.core.utils as etau


logger = logging.getLogger(__name__)


class ETAConfig(EnvConfig):
    '''Sytem-wide ETA configuration settings.'''

    def __init__(self, d):
        self.config_dir = self.parse_string(
            d, "config_dir", env_var="ETA_CONFIG_DIR", default="")
        self.output_dir = self.parse_string(
            d, "output_dir", env_var="ETA_OUTPUT_DIR", default="")
        self.module_dirs = self.parse_string_array(
            d, "module_dirs", env_var="ETA_MODULE_DIRS", default=[])
        self.pipeline_dirs = self.parse_string_array(
            d, "pipeline_dirs", env_var="ETA_PIPELINE_DIRS", default=[])
        self.models_dirs = self.parse_string_array(
            d, "models_dirs", env_var="ETA_MODELS_DIRS", default=[])
        self.pythonpath_dirs = self.parse_string_array(
            d, "pythonpath_dirs", env_var="ETA_PYTHONPATH_DIRS", default=[])
        self.environment_vars = self.parse_dict(
            d, "environment_vars", default={})
        self.tf_config = self.parse_dict(d, "tf_config", default={})
        self.max_model_versions_to_keep = int(self.parse_number(
            d, "max_model_versions_to_keep",
            env_var="ETA_MAX_MODEL_VERSIONS_TO_KEEP", default=-1))
        self.allow_model_downloads = self.parse_bool(
            d, "allow_model_downloads", env_var="ETA_ALLOW_MODEL_DOWNLOADS",
            default=True)
        self.default_sequence_idx = self.parse_string(
            d, "default_sequence_idx", env_var="ETA_DEFAULT_SEQUENCE_IDX",
            default="%05d")
        self.default_image_ext = self.parse_string(
            d, "default_image_ext", env_var="ETA_DEFAULT_IMAGE_EXT",
            default=".png")
        self.default_video_ext = self.parse_string(
            d, "default_video_ext", env_var="ETA_DEFAULT_VIDEO_EXT",
            default=".mp4")


def set_config_settings(**kwargs):
    '''Sets the given ETA config settings.

    The settings are validated by constructing an ETAConfig from them before
    applying them.

    Args:
        **kwargs: keyword arguments defining valid ETA config fields and values

    Raises:
        EnvConfigError: if the settings were invalid
    '''
    # Validiate settings
    _config = ETAConfig.from_dict(kwargs)

    # Apply settings
    for field in kwargs:
        if not hasattr(config, field):
            logger.warning("Skipping unknown config setting '%s'", field)
            continue
        val = getattr(_config, field)
        logger.debug("Setting ETA config field %s = %s", field, str(val))
        setattr(config, field, val)


def startup_message():
    '''Logs ETA startup message.'''
    logger.info("Starting...\n" + etac.ASCII_ART)
    logger.info(version)
    logger.info("Revision %s", etau.get_eta_rev())


# Version string
version = "%s %s, %s" % (etac.NAME, etac.VERSION, etac.AUTHOR)

# Default logging behavior
etal.basic_setup()

# Load global ETA config
config = ETAConfig.from_json(etac.CONFIG_JSON_PATH)

# Augment system path
sys.path = sys.path[:1] + config.pythonpath_dirs + sys.path[1:]

# Set any environment variables
for var, val in iteritems(config.environment_vars):
    os.environ[var] = val
