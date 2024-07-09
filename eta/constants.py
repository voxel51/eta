"""
ETA package-wide constants.

Copyright 2017-2024, Voxel51, Inc.
voxel51.com
"""
import os

import importlib.metadata as metadata

try:
    _META = metadata.metadata("voxel51-eta")
except metadata.PackageNotFoundError as e:
    try:
        # Old installs may be under `eta`
        _META = metadata.metadata("eta")
    except metadata.PackageNotFoundError:
        raise e


# Directories
ETA_DIR = os.path.abspath(os.path.dirname(__file__))
ETA_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".eta")
EXAMPLES_DIR = os.path.join(ETA_DIR, "examples")
RESOURCES_DIR = os.path.join(ETA_DIR, "resources")
TENSORFLOW_DIR = os.path.join(ETA_DIR, "tensorflow")

# Submodules
DARKFLOW_DIR = os.path.join(TENSORFLOW_DIR, "darkflow")
AUTOML_DIR = os.path.join(TENSORFLOW_DIR, "automl")
EFFICIENTDET_DIR = os.path.join(AUTOML_DIR, "efficientdet")
TF_MODELS_DIR = os.path.join(TENSORFLOW_DIR, "models")
TF_RESEARCH_DIR = os.path.join(TF_MODELS_DIR, "research")
TF_OBJECT_DETECTION_DIR = os.path.join(TF_RESEARCH_DIR, "object_detection")
TF_SLIM_DIR = os.path.join(TF_RESEARCH_DIR, "slim")

# Paths
CONFIG_JSON_PATH = os.path.join(ETA_DIR, "config.json")
ASCII_ART_PATH = os.path.join(RESOURCES_DIR, "eta-ascii.txt")
DEFAULT_FONT_PATH = os.path.join(RESOURCES_DIR, "Arial.ttf")
DEFAULT_LOGO_CONFIG_PATH = os.path.join(
    RESOURCES_DIR, "default-logo-config.json"
)


# Package metadata
NAME = _META["name"]
VERSION = _META["version"]
DESCRIPTION = _META["summary"]
AUTHOR = _META["author"]
AUTHOR_EMAIL = _META["author-email"]
URL = _META["home-page"]
LICENSE = _META["license"]
VERSION_LONG = "{} v{}, {}".format(NAME, VERSION, AUTHOR)
