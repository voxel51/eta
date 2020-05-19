"""
ETA package-wide constants.

Copyright 2017-2020, Voxel51, Inc.
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

import os

try:
    from importlib.metadata import metadata  # Python 3.8
except ImportError:
    from importlib_metadata import metadata  # Python < 3.8


# Directories
ETA_DIR = os.path.abspath(os.path.dirname(__file__))
ETA_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".eta")
BASE_DIR = os.path.dirname(ETA_DIR)
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")
RESOURCES_DIR = os.path.join(ETA_DIR, "resources")
TF_MODELS_DIR = os.path.join(BASE_DIR, "tensorflow/models")
TF_RESEARCH_DIR = os.path.join(TF_MODELS_DIR, "research")
TF_OBJECT_DETECTION_DIR = os.path.join(TF_RESEARCH_DIR, "object_detection")
TF_SLIM_DIR = os.path.join(TF_RESEARCH_DIR, "slim")


# Paths
CONFIG_JSON_PATH = os.path.join(BASE_DIR, "config.json")
ASCII_ART_PATH = os.path.join(RESOURCES_DIR, "eta-ascii.txt")
DEFAULT_FONT_PATH = os.path.join(RESOURCES_DIR, "lato-regular.ttf")
DEFAULT_LOGO_CONFIG_PATH = os.path.join(
    RESOURCES_DIR, "default-logo-config.json"
)


# Package metadata
_META = metadata("voxel51-eta")
NAME = _META["name"]
VERSION = _META["version"]
DESCRIPTION = _META["summary"]
AUTHOR = _META["author"]
AUTHOR_EMAIL = _META["author-email"]
URL = _META["home-page"]
LICENSE = _META["license"]
VERSION_LONG = "%s v%s, %s" % (NAME, VERSION, AUTHOR)
