#!/usr/bin/env python
'''
Downloads files from Google Drive.

Copyright 2017-2018, Voxel51, LLC
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
import os
import sys

from eta.core.config import Config
import eta.core.module as etam
import eta.core.web as etaw


logger = logging.getLogger(__name__)


class GoogleDriveDownloadConfig(etam.BaseModuleConfig):
    '''Google Drive download configuration settings.'''

    def __init__(self, d):
        super(GoogleDriveDownloadConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    '''Data configuration settings.'''

    def __init__(self, d):
        self.output_path = self.parse_string(d, "output_path")
        self.google_drive_id = self.parse_string(d, "google_drive_id")


def _download_files(download_config):
    for data in download_config.data:
        logger.info("Downloading %s from Google Drive", data.output_path)
        try:
            etaw.download_google_drive_file(
                data.google_drive_id, path=data.output_path)
        except etaw.WebSessionError:
            logger.error(
                "Could not download Google Drive file with ID %s",
                data.google_drive_id
            )
        except:
            logger.error("Unknown error occurred")


def run(config_path, pipeline_config_path=None):
    '''Run the google_drive_download module.

    Args:
        config_path: path to a GoogleDriveDownloadConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    download_config = GoogleDriveDownloadConfig.from_json(config_path)
    etam.setup(download_config, pipeline_config_path=pipeline_config_path)
    _download_files(download_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
