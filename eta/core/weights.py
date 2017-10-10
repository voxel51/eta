'''
Encapsulates an npz file that stores model weights locally in a cache and has
the ability to pull a version of the  weights down from the net.

@todo: this should probably be:
    (1) extended to allow for methods to visualize, display, and debug weights
    (2) refactored out so the actual backing store code can be generally used

Copyright 2017, Voxel51, LLC
voxel51.com

Jason Corso, jjc@voxel51.com
'''
import os

import numpy as np

from config import Config, Configurable
from eta import constants
import utils
import web


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


class Weights(Configurable):
    '''Weights class that encapsulates model weights and can load them from the
    net if needed (if paths are provided).

    @todo: Would be great to make this class act like the actual dictionary it
    loads, by overloading/implementing the same methods.
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
        self.data = None

        if not os.path.isfile(self.config.path):
            utils.ensure_dir(self.config.path)

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
        self.data = np.load(self.config.path)
