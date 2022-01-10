"""
ETA package initialization.

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
from future.utils import iteritems

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging
import os
import sys

import eta.constants as etac
from eta.core.config import Config, EnvConfig
import eta.core.logging as etal
import eta.core.utils as etau


logger = logging.getLogger(__name__)


class ETAConfig(EnvConfig):
    """Sytem-wide ETA configuration settings.

    When an ETAConfig is loaded, any `{{eta}}` patterns are replaced with
    `eta.constants.ETA_DIR`.
    """

    def __init__(self, d):
        self.config_dir = self.parse_string(
            d, "config_dir", env_var="ETA_CONFIG_DIR", default=None
        )
        self.output_dir = self.parse_string(
            d, "output_dir", env_var="ETA_OUTPUT_DIR", default=None
        )
        self.module_dirs = self.parse_string_array(
            d, "module_dirs", env_var="ETA_MODULE_DIRS", default=[]
        )
        self.pipeline_dirs = self.parse_string_array(
            d, "pipeline_dirs", env_var="ETA_PIPELINE_DIRS", default=[]
        )
        self.models_dirs = self.parse_string_array(
            d, "models_dirs", env_var="ETA_MODELS_DIRS", default=[]
        )
        self.pythonpath_dirs = self.parse_string_array(
            d, "pythonpath_dirs", env_var="ETA_PYTHONPATH_DIRS", default=[]
        )
        self.environment_vars = self.parse_dict(
            d, "environment_vars", default={}
        )
        self.tf_config = self.parse_dict(d, "tf_config", default={})
        self.patterns = self.parse_dict(d, "patterns", default={})
        self.max_model_versions_to_keep = self.parse_int(
            d,
            "max_model_versions_to_keep",
            env_var="ETA_MAX_MODEL_VERSIONS_TO_KEEP",
            default=-1,
        )
        self.allow_model_downloads = self.parse_bool(
            d,
            "allow_model_downloads",
            env_var="ETA_ALLOW_MODEL_DOWNLOADS",
            default=True,
        )
        self.default_sequence_idx = self.parse_string(
            d,
            "default_sequence_idx",
            env_var="ETA_DEFAULT_SEQUENCE_IDX",
            default="%05d",
        )
        self.default_image_ext = self.parse_string(
            d,
            "default_image_ext",
            env_var="ETA_DEFAULT_IMAGE_EXT",
            default=".png",
        )
        self.default_video_ext = self.parse_string(
            d,
            "default_video_ext",
            env_var="ETA_DEFAULT_VIDEO_EXT",
            default=".mp4",
        )
        self.default_figure_ext = self.parse_string(
            d,
            "default_figure_ext",
            env_var="ETA_DEFAULT_FIGURE_EXT",
            default=".pdf",
        )
        self.show_progress_bars = self.parse_bool(
            d,
            "show_progress_bars",
            env_var="ETA_SHOW_PROGRESS_BARS",
            default=True,
        )

        self._fill_defaults()
        self._parse_patterns()
        self._fill_patterns()

    def _fill_defaults(self):
        if not self.config_dir:
            self.config_dir = os.path.join(etac.ETA_CONFIG_DIR, "configs")

        if not self.output_dir:
            self.output_dir = os.path.join(etac.ETA_CONFIG_DIR, "out")

    def _parse_patterns(self):
        #
        # Add default patterns to dict
        #
        if "{{eta}}" in self.patterns:
            logger.warning("Overwriting existing {{eta}} pattern")

        self.patterns["{{eta}}"] = etac.ETA_DIR

        if "{{eta-resources}}" in self.patterns:
            logger.warning("Overwriting existing {{eta-resources}} pattern")

        self.patterns["{{eta-resources}}"] = etac.RESOURCES_DIR

        #
        # Resolve user-provided patterns by replacing any patterns and
        # converting to realpaths, if necessary
        #
        for patt in self.patterns:
            self.patterns[patt] = os.path.realpath(
                etau.fill_patterns(self.patterns[patt], self.patterns)
            )

    def _fill_patterns(self):
        self.config_dir = etau.fill_patterns(self.config_dir, self.patterns)
        self.output_dir = etau.fill_patterns(self.output_dir, self.patterns)
        self.module_dirs = [
            etau.fill_patterns(m, self.patterns) for m in self.module_dirs
        ]
        self.pipeline_dirs = [
            etau.fill_patterns(p, self.patterns) for p in self.pipeline_dirs
        ]
        self.models_dirs = [
            etau.fill_patterns(m, self.patterns) for m in self.models_dirs
        ]
        self.pythonpath_dirs = [
            etau.fill_patterns(m, self.patterns) for m in self.pythonpath_dirs
        ]


def set_config_settings(**kwargs):
    """Sets the given ETA config settings.

    The settings are validated by constructing an ETAConfig from them before
    applying them.

    Args:
        **kwargs: keyword arguments defining valid ETA config fields and values

    Raises:
        EnvConfigError: if the settings were invalid
    """
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
    """Logs ETA startup message."""
    logger.info("Starting...\n%s", _load_ascii_art())
    logger.info(etac.VERSION_LONG)
    logger.info("Revision %s", etau.get_eta_rev())


def _load_ascii_art():
    with open(etac.ASCII_ART_PATH, "rt") as f:
        return f.read()


def is_python2():
    """Returns True/False whether the Python version running is 2.X."""
    return sys.version_info[0] == 2


def is_python3():
    """Returns True/False whether the Python version running is 3.X."""
    return sys.version_info[0] == 3


# Default logging behavior
etal.basic_setup()

# Load global ETA config
config = ETAConfig.from_json(etac.CONFIG_JSON_PATH)

# Augment system path
sys.path = sys.path[:1] + config.pythonpath_dirs + sys.path[1:]

# Set any environment variables
for var, val in iteritems(config.environment_vars):
    os.environ[var] = val
