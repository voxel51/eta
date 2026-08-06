#!/usr/bin/env python
"""
Installs ETA.

All static package metadata lives in pyproject.toml; this shim exists to
support the RELEASE_VERSION environment variable, which the publish
workflow uses to build release candidate versions.

Copyright 2017-2026, Voxel51, Inc.
voxel51.com
"""

import os

from setuptools import setup

VERSION = "0.17.0"


def get_version():
    if "RELEASE_VERSION" in os.environ:
        version = os.environ["RELEASE_VERSION"]
        if not version.startswith(VERSION):
            raise ValueError(
                "Release version does not match version: %s and %s"
                % (version, VERSION)
            )
        return version

    return VERSION


setup(version=get_version())
