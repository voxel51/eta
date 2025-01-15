#!/usr/bin/env python
"""
Installs ETA.

Copyright 2017-2025, Voxel51, Inc.
voxel51.com
"""

from importlib import metadata
import os
import re
from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel


VERSION = "0.14.0"


class BdistWheelCustom(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        # Pure Python, so build a wheel for any Python version
        self.universal = True


INSTALL_REQUIRES = [
    "argcomplete",
    "dill",
    "future",
    "glob2",
    "jsonlines",
    "numpy",
    "packaging",
    "Pillow>=6.2",
    "py7zr",
    "python-dateutil",
    "pytz",
    "rarfile",
    "requests",
    "retrying",
    "six",
    "scikit-image",
    "sortedcontainers",
    "tabulate",
    "tzlocal",
    "urllib3",
]


CHOOSE_INSTALL_REQUIRES = [
    (
        (
            "opencv-python",
            "opencv-contrib-python",
            "opencv-contrib-python-headless",
        ),
        "opencv-python-headless<5,>=4.1",
    )
]


def choose_requirement(mains, secondary):
    chosen = secondary
    for main in mains:
        try:
            name = re.split(r"[!<>=]", main)[0]
            metadata.version(name)
            chosen = main
            break
        except metadata.PackageNotFoundError:
            pass

    return str(chosen)


def get_install_requirements(install_requires, choose_install_requires):
    for mains, secondary in choose_install_requires:
        install_requires.append(choose_requirement(mains, secondary))

    return install_requires


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
    install_requires=get_install_requirements(
        INSTALL_REQUIRES, CHOOSE_INSTALL_REQUIRES
    ),
    extras_require={
        "pipeline": ["blockdiag", "Sphinx", "sphinxcontrib-napoleon"],
        "storage": [
            "azure-identity",
            "azure-storage-blob>=12.4.0",
            "boto3>=1.15",
            "google-api-python-client",
            "google-cloud-storage>=1.36",
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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={"console_scripts": ["eta=eta.core.cli:main"]},
    python_requires=">=3.9",
    cmdclass={"bdist_wheel": BdistWheelCustom},
)
