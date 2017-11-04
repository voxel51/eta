'''
Encapsulates an npz file that stores model weights locally in a cache and has
the ability to pull a version of the weights down from the net.

@todo: this should probably be:
    (1) extended to allow for methods to visualize, display, and debug weights
    (2) refactored out so the actual backing store code can be generally used

Copyright 2017, Voxel51, LLC
voxel51.com

Jason Corso, jjc@voxel51.com
Brian Moore, brian@voxel51.com
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

import os

import numpy as np

from eta.core.config import Config, Configurable
from eta import constants
from eta.core import utils
from eta.core import web


class WeightsConfig(Config):
    '''Weights configuration settings.'''

    def __init__(self, d):
        self.cache_dir = self.parse_string(
            d, "cache_dir", default=constants.DEFAULT_CACHE_DIR)
        self.filename = self.parse_string(d, "filename")
        self.url = self.parse_string(d, "url", default=None)
        self.google_drive_id = self.parse_string(
            d, "google_drive_id", default=None)

    @property
    def path(self):
        return os.path.join(self.cache_dir, self.filename)


class Weights(Configurable, dict):
    '''Weights class that encapsulates model weights and can load them from the
    net if needed (if paths are provided).

    Provides a dictionary interface to the loaded weights.
    '''

    def __init__(self, config):
        '''Initializes a Weights instance.

        Args:
            config: a WeightsConfig instance

        Raises:
            OSError: if the weights file was not found on disk and no web
                address was provided.
        '''
        self.validate(config)
        self.config = config

        if not os.path.isfile(self.config.path):
            utils.ensure_basedir(self.config.path)

            # Download the weights from the web.
            if self.config.google_drive_id:
                web.download_google_drive_file(
                    self.config.google_drive_id, path=self.config.path)
            elif self.config.url:
                web.download_file(
                    self.config.url, path=self.config.path)
            else:
                raise OSError(
                    "Weights file '%s' not found and no web address was "
                    "provided" % self.config.path
                )

        # Load weights from local file.
        self.update(np.load(self.config.path))
