#!/usr/bin/env python
'''
Download a file from Google Drive by ID.

Copyright 2017, Voxel51, LLC
voxel51.com
'''
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
import sys

import eta.core.module as mo
import eta.core.web as etaw


logger = logging.getLogger(__name__)


class GDriveDownloadConfig(mo.BaseModuleConfig):
    '''Clip configuration settings.'''

    def __init__(self, d):
        super(GDriveDownloadConfig, self).__init__(d)
        self.filename = self.parse_string(d, "filename")
        self.google_drive_id = self.parse_string(d, "google_drive_id")


def run(config_path, pipeline_config_path=None):
    '''Run the gdrive_download module.

    Args:
        config_path: path to a ClipConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    config = GDriveDownloadConfig.from_json(config_path)
    mo.setup(config, pipeline_config_path=pipeline_config_path)

    logger.info("Downloading %s from google drive", config.filename)
    try:
        etaw.download_google_drive_file(
                config.google_drive_id, path=config.filename)
    except etaw.WebSessionError:
        logger.error("Could not download google drive file id %s",
                     config.google_drive_id)
    except:
        logger.error("Unknown error occurred")


if __name__ == "__main__":
    run(*sys.argv[1:])
