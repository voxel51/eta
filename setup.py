#!/usr/bin/env python
'''
Installs ETA.

Copyright 2017-2020, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
'''
from setuptools import setup, find_packages


# @note the version info below should be kept in-sync with `eta/version.json`
setup(
    name="ETA",
    version="0.1.0",
    description="Extensible Toolkit for Analytics",
    author="Voxel51, Inc.",
    author_email="info@voxel51.com",
    url="https://github.com/voxel51/eta",
    license="BSD 4-clause",
    packages=find_packages(),
    classifiers=[
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
    scripts=["eta/eta"],
)
