'''
ETA package-wide constants.

IMPORTANT: this module should not import any ETA modules!

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
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import json
import os


# Directories
ETA_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(ETA_DIR)
EXAMPLES_DIR = os.path.join(BASE_DIR, 'examples')


# Paths
VERSION_JSON_PATH = os.path.join(ETA_DIR, "version.json")
ASCII_ART_PATH = os.path.join(ETA_DIR, "eta-ascii.txt")
CONFIG_JSON_PATH = os.path.join(BASE_DIR, "config.json")


# Version
with open(VERSION_JSON_PATH, "rt") as f:
    _VER = json.load(f)
NAME = _VER["name"]
VERSION = _VER["version"]
DESCRIPTION = _VER["description"]
AUTHOR = _VER["author"]
CONTACT = _VER["contact"]
URL = _VER["url"]
LICENSE = _VER["license"]


# ASCII art
with open(ASCII_ART_PATH, "rt") as f:
    ASCII_ART = f.read()
