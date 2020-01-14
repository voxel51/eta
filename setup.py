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
    install_requires=[
        "Cython==0.28.5",
        "Jinja2==2.10.1",
        "Pillow==6.2.0",
        "argcomplete==1.11.0",
        "blockdiag==1.5.3",
        "contextlib2==0.5.5",
        "dill==0.2.7.1",
        "future==0.16.0",
        "glob2==0.6",
        "h5py==2.10.0",
        "lxml==4.3.0",
        "numpy==1.16.3",
        "opencv-python-headless==4.1.0.25",
        "protobuf==3.6.1",
        "python-dateutil==2.7.0",
        "pytz==2019.3",
        "requests-toolbelt==0.8.0",
        "requests==2.21.0",
        "retrying==1.3.3",
        "scikit-learn==0.19.2",
        "scipy==0.19.1",
        "setuptools==36.5.0",
        "simplejson==3.8.1",
        "six==1.11.0",
        "tabulate==0.8.5",
        "tzlocal==2.0.0",
    ],
    extras_require={
        "storage": [
            "boto3==1.10.9",
            "google-api-python-client==1.6.5",
            "google-cloud-storage==1.7.0",
            "pysftp==0.2.9",
            "retrying==1.3.3",
        ],
        "dev": [
            "Sphinx==1.7.5",
            "pycodestyle==2.3.1",
            "sphinxcontrib-napoleon==0.6.1",
            'pylint==1.9.4;python_version<"3"',
            'pylint==2.3.1;python_version>="3"',
        ],
    },
    scripts=["eta/eta"],
)
