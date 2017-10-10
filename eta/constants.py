'''
ETA package-wide constants.

IMPORTANT: this module should not import any ETA modules!

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
import json
import os


# Directories
ETA_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIGS_DIR = os.path.join(ETA_DIR, "configs")
DEFAULT_CACHE_DIR = os.path.join(ETA_DIR, "cache")


# Paths
VERSION_JSON_PATH = os.path.join(ETA_DIR, "version.json")


# Version
with open(VERSION_JSON_PATH) as f:
    _version = json.load(f)
NAME = str(_version["name"])
VERSION = str(_version["version"])
DESCRIPTION = str(_version["description"])
AUTHOR = str(_version["author"])
CONTACT = str(_version["contact"])
URL = str(_version["url"])
LICENSE = str(_version["license"])
