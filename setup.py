#!/usr/bin/env python
"""
Installs ETA.

Copyright 2017-2021, Voxel51, Inc.
voxel51.com
"""
import os
from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel


VERSION = "0.5.3"


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
        "tabulate",
        "tzlocal",
    ],
    extras_require={
        "pipeline": ["blockdiag", "Sphinx", "sphinxcontrib-napoleon"],
        "storage": [
            "boto3",
            "google-api-python-client",
            "google-cloud-storage",
            "httplib2<=0.15",
            "pysftp",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={"console_scripts": ["eta=eta.core.cli:main"]},
    python_requires=">=2.7",
    cmdclass={"bdist_wheel": BdistWheelCustom},
)
