'''
Core logging infrastructure.

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
import os
import sys

import eta
from eta.core import utils
from eta.core.config import Config


root_logger = logging.getLogger()
logger = logging.getLogger(__name__)


# Basic logging defaults
DEFAULT_BASIC_LEVEL = logging.INFO
DEFAULT_BASIC_FORMAT = "%(message)s"

# Custom logging defaults
DEFAULT_STREAM_TO_STDOUT = True
DEFAULT_STDOUT_FORMAT = "%(message)s"
DEFAULT_STDOUT_LEVEL = "INFO"
DEFAULT_FILENAME = None
DEFAULT_FILE_FORMAT = \
    "%(asctime)s - %(name)-18s - %(levelname)-8s - %(message)s"
DEFAULT_FILE_LEVEL = "INFO"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_ENCODING = "utf8"


def flush():
    '''Flush logging handlers.

    It is only necessary to call this method when multiple processes are
    writing to a single log file (e.g., when running a pipeline).
    '''
    for h in root_logger.handlers:
        h.flush()


def reset():
    '''Reset logging.

    Performs the following tasks:
        - removes all existing handlers from the root logger
        - sets the root logger level to DEBUG (the effective logging level is
            determined on a per-handler basis)
        - routes all uncaught exceptions to the root logger
    '''
    root_logger.handlers = []
    root_logger.setLevel(logging.DEBUG)
    sys.excepthook = _exception_logger


def basic_setup(level=DEFAULT_BASIC_LEVEL, fmt=DEFAULT_BASIC_FORMAT):
    '''Setup basic logging to stdout.

    Args:
        level: the logging level. The default is DEFAULT_BASIC_LEVEL
        fmt: the logging format. The default is DEFAULT_BASIC_FORMAT
    '''
    reset()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt))
    handler.setLevel(level)
    root_logger.addHandler(handler)


def custom_setup(lc, overwrite=False):
    '''Setup custom logging.

    Args:
        lc: a LoggingConfig instance
        overwrite: whether to overwrite (True) any existing log, or append
            to it (False). The default is False
    '''
    # Messages to log after setup
    msgs = []

    # Reset logging
    reset()
    msgs.append("Resetting logging")
    msgs.append("Logging all uncaught exceptions")

    # Stdout logging
    if lc.stream_to_stdout:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter(fmt=lc.stdout_format, datefmt=lc.datefmt))
        stream_handler.setLevel(getattr(logging, lc.stdout_level))
        root_logger.addHandler(stream_handler)
        msgs.append("Logging to stdout at level %s" % lc.stdout_level)

    # File logging
    if lc.filename:
        if overwrite and os.path.isfile(lc.filename):
            msgs.append("Deleting existing log '%s'" % lc.filename)
            os.remove(lc.filename)

        file_handler = logging.FileHandler(
            lc.filename, mode="at", encoding=lc.encoding)
        file_handler.setFormatter(
            logging.Formatter(fmt=lc.file_format, datefmt=lc.datefmt))
        file_handler.setLevel(getattr(logging, lc.file_level))
        root_logger.addHandler(file_handler)
        msgs.append(
            "Logging to '%s' at level %s" % (lc.filename, lc.file_level))

    msgs.append("Logging initialized\n")

    # Initial logging output
    eta.startup_message()
    for msg in msgs:
        logger.info(msg)


def _exception_logger(*exc_info):
    root_logger.error("Uncaught exception", exc_info=exc_info)


class LoggingConfig(Config):
    '''Logging configuration settings.'''

    def __init__(self, d):
        self.stream_to_stdout = self.parse_bool(
            d, "stream_to_stdout", default=DEFAULT_STREAM_TO_STDOUT)
        self.stdout_format = self.parse_string(
            d, "stdout_format", default=DEFAULT_STDOUT_FORMAT)
        self.stdout_level = self.parse_string(
            d, "stdout_level", default=DEFAULT_STDOUT_LEVEL)
        self.filename = self.parse_string(
            d, "filename", default=DEFAULT_FILENAME)
        self.file_format = self.parse_string(
            d, "file_format", default=DEFAULT_FILE_FORMAT)
        self.file_level = self.parse_string(
            d, "file_level", default=DEFAULT_FILE_LEVEL)
        self.datefmt = self.parse_string(
            d, "datefmt", default=DEFAULT_DATEFMT)
        self.encoding = self.parse_string(
            d, "encoding", default=DEFAULT_ENCODING)
