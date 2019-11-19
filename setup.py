#!/usr/bin/env python
'''
Installs the ETA package.

Copyright 2017-2019, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
'''
from setuptools import setup, find_packages

import eta.constants as etac


setup(
    name=etac.NAME,
    version=etac.VERSION,
    description=etac.DESCRIPTION,
    author=etac.AUTHOR,
    author_email=etac.CONTACT,
    url=etac.URL,
    license=etac.LICENSE,
    packages=find_packages(),
    classifiers=[
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
    scripts=["eta/eta"],
)
