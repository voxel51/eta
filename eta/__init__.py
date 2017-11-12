'''
ETA package initialization.

Copyright 2017, Voxel51, LLC
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

import eta.constants as c
from eta.core import log
from eta.core import utils


logger = logging.getLogger(__name__)


def startup_message():
    '''Logs ETA startup message.'''
    logger.info("Starting...\n" + c.ASCII_ART)
    logger.info("%s %s, %s", c.NAME, c.VERSION, c.AUTHOR)
    logger.info("Revision %s\n", utils.get_eta_rev())


# Default logging behavior
log.basic_setup()
