#!/usr/bin/env python
"""
Downloads example data from Google Drive.

Usage:
    python download_data.py

Copyright 2017-2022, Voxel51, Inc.
voxel51.com
"""
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging
import os

import eta.core.web as etaw
import eta.core.utils as etau


logger = logging.getLogger(__name__)


FILE_ID = "0B7phNvpRqNdpNEVpVjE2VXQxOWc"


logger.info("Downloading example data from Google Drive")
path = os.path.join(os.path.dirname(__file__), "data.zip")
etaw.download_google_drive_file(FILE_ID, path=path)
etau.extract_zip(path, delete_zip=True)
