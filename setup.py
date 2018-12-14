#!/usr/bin/env python
'''
Install ETA package.

Copyright 2017-2018, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
'''
from setuptools import setup, find_packages

import eta.constants as c


setup(
    name=c.NAME,
    version=c.VERSION,
    description=c.DESCRIPTION,
    author=c.AUTHOR,
    author_email=c.CONTACT,
    url=c.URL,
    license=c.LICENSE,
    packages=find_packages(),
    classifiers=[
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
    scripts=["eta/eta"],
)
