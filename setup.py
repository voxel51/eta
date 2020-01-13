#!/usr/bin/env python
'''
Installs the ETA package.

Copyright 2017-2019, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
'''
from setuptools import setup, find_packages

setup(
    include_package_data=True,
    name="ETA",
    version="0.1.0",
    description="Extensible Toolkit for Analytics",
    author="Voxel51, Inc.",
    contact="support@voxel51.com",
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
