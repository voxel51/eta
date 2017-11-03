#!/usr/bin/env python
'''
Install ETA package.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
from setuptools import setup, find_packages

from eta import constants


setup(
    name=constants.NAME,
    version=constants.VERSION,
    description=constants.DESCRIPTION,
    author=constants.AUTHOR,
    author_email=constants.CONTACT,
    url=constants.URL,
    license=constants.LICENSE,
    packages=find_packages(),
    classifiers=[
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
)
