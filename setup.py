#!/usr/bin/env python
"""
Installs ETA.

Copyright 2017-2020, Voxel51, Inc.
voxel51.com
"""
from setuptools import setup, find_packages


setup(
    name="ETA",
    version="0.1.0",
    description="Extensible Toolkit for Analytics",
    author="Voxel51, Inc.",
    author_email="info@voxel51.com",
    url="https://github.com/voxel51/eta",
    license="BSD-4-Clause",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "argcomplete",
        "boto3",
        "dill",
        "future",
        "glob2",
        "google-api-python-client",
        "google-cloud-storage",
        "importlib-metadata; python_version<'3.8'",
        "numpy",
        "opencv-python-headless<5,>=4.1",
        "Pillow<7,>=6.2",
        "pysftp",
        "python-dateutil",
        "pytz",
        "requests",
        "retrying",
        "six",
        "sortedcontainers",
        "tabulate",
        "tzlocal",
    ],
    classifiers=[
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
    scripts=["eta/eta"],
    python_requires=">=2.7",
)
