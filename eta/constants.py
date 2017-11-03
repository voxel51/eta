'''
ETA package-wide constants.

IMPORTANT: this module should not import any ETA modules!

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import json
import os


# Directories
ETA_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIGS_DIR = os.path.join(ETA_DIR, "configs")
DEFAULT_CACHE_DIR = os.path.join(ETA_DIR, "cache")


# Paths
VERSION_JSON_PATH = os.path.join(ETA_DIR, "version.json")


# Version
with open(VERSION_JSON_PATH, "rt") as f:
    _VERSION = json.load(f)
NAME = _VERSION["name"]
VERSION = _VERSION["version"]
DESCRIPTION = _VERSION["description"]
AUTHOR = _VERSION["author"]
CONTACT = _VERSION["contact"]
URL = _VERSION["url"]
LICENSE = _VERSION["license"]
