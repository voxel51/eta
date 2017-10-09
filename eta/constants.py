'''
ETA package-wide constants.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
import os

from core.config import Config


# Directories
ETA_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIGS_DIR = os.path.join(ETA_DIR, "configs")
DEFAULT_CACHE_DIR = os.path.join(ETA_DIR, "cache")


# Paths
VERSION_JSON_PATH = os.path.join(ETA_DIR, "version.json")


class _VersionConfig(Config):
    '''Version info.'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.version = self.parse_string(d, "version")
        self.description = self.parse_string(d, "description")
        self.author = self.parse_string(d, "author")
        self.contact = self.parse_string(d, "contact")
        self.url = self.parse_string(d, "url")
        self.license = self.parse_string(d, "license")


_version = _VersionConfig.from_json(VERSION_JSON_PATH)


# Version info
NAME = _version.name
VERSION = _version.version
DESCRIPTION = _version.description
AUTHOR = _version.author
CONTACT = _version.contact
URL = _version.url
LICENSE = _version.license
