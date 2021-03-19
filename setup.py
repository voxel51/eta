#!/usr/bin/env python
"""
Installs ETA.

Copyright 2017-2021, Voxel51, Inc.
voxel51.com
"""
import os
from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel


VERSION = "0.3.1"


class BdistWheelCustom(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        # Pure Python, so build a wheel for any Python version
        self.universal = True


with open("README.md", "r") as fh:
    long_description = fh.read()


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


setup(
    name="voxel51-eta",
    version=get_version(),
    description="Extensible Toolkit for Analytics",
    author="Voxel51, Inc.",
    author_email="info@voxel51.com",
    url="https://github.com/voxel51/eta",
    license="Apache",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "argcomplete",
        "dill",
        "future",
        "glob2",
        "importlib-metadata; python_version<'3.8'",
        "ndjson",
        "numpy",
        "opencv-python-headless<5,>=4.1",
        "packaging",
        "patool",
        "Pillow>=6.2",
        "python-dateutil",
        "pytz",
        "requests",
        "retrying",
        "six",
        "scikit-image",
        "sortedcontainers",
        "Sphinx",
        "tabulate",
        "tzlocal",
    ],
    extras_require={
        "storage": [
            "boto3",
            "google-api-python-client",
            "google-cloud-storage",
            "httplib2<=0.15",
            "pysftp",
        ],
    },
    classifiers=[
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    entry_points={"console_scripts": ["eta=eta.core.cli:main"]},
    python_requires=">=2.7",
    cmdclass={"bdist_wheel": BdistWheelCustom},
)
