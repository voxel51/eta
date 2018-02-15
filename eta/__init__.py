'''
ETA package initialization.

Copyright 2017-2018, Voxel51, LLC
voxel51.com

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

import logging
import os

import eta.constants as etac
from eta.core.config import Config
import eta.core.log as etal
import eta.core.utils as etau


logger = logging.getLogger(__name__)


def startup_message():
    '''Logs ETA startup message.'''
    logger.info("Starting...\n" + etac.ASCII_ART)
    logger.info("%s %s, %s", etac.NAME, etac.VERSION, etac.AUTHOR)
    logger.info("Revision %s\n", etau.get_eta_rev())


class EtaConfig(Config):
    '''Sytem-wide ETA configuration settings.'''

    def __init__(self, d):
        self.module_dirs = self.parse_array(d, "module_dirs", default=[])


# Default logging behavior
etal.basic_setup()


# Load config
# @todo allow configs to be overridden by environment variables
if os.path.isfile(etac.CONFIG_JSON_PATH):
    config = EtaConfig.from_json(etac.CONFIG_JSON_PATH)
else:
    logger.warning("Could not find config '%s'", etac.CONFIG_JSON_PATH)
    config = EtaConfig.default()
