'''
Job infrastructure for running modules in a pipeline.

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
import subprocess
import sys

from eta.core.config import Config
from eta.core import log
from eta.core import utils

logger = logging.getLogger(__name__)


def run(job_config, overwrite=True):
    '''Run the job specified by the JobConfig.

    If the job completes succesfully, the hash of the config file is written to
    disk.

    If the job raises an error, execution is terminated immediately.

    Args:
        job_config: a JobConfig instance
        overwrite: overwrite mode. When True, always run the job. When False,
            only run the job if the config file has changed since the last time
            the job was (succesfully) run

    Returns:
        True/False: if the job was run

    Raises:
        JobConfigError: if the JobConfig was invalid
    '''
    logger.info("Starting job '%s'", job_config.name)

    with utils.WorkingDir(job_config.working_dir):
        # Check config hash
        hasher = utils.MD5FileHasher(job_config.config_path)
        if hasher.has_changed:
            logger.info("Config '%s' changed" % job_config.config_path)
            should_run = True
        elif hasher.has_record:
            if overwrite:
                logger.info("Overwriting existing job output")
                should_run = True
            else:
                logger.info("Skipping job '%s'\n", job_config.name)
                should_run = False
        else:
            should_run = True

        if should_run:
            logger.info("Working directory: %s", os.getcwd())
            log.flush()

            # Run the job
            success = _run(job_config)
            if not success:
                logger.error("Job '%s' failed... exiting now", job_config.name)
                sys.exit()

            # Write config hash
            hasher.write()

            logger.info("Job '%s' complete\n", job_config.name)

        return should_run


def _run(job_config):
    # Construct command
    if job_config.binary:
        args = [
            job_config.binary,          # binary
            job_config.config_path,     # config file
        ]
    elif job_config.script:
        args = [
            job_config.interpreter,     # interpreter
            job_config.script,          # script
            job_config.config_path,     # config file
        ]
    elif job_config.custom:
        # Run custom command-line
        args = (
            job_config.custom +         # custom args
            [job_config.config_path]    # config file
        )
    else:
        raise JobConfigError("Invalid JobConfig")

    # Run command
    success = utils.call(args)
    return success


class JobConfigError(Exception):
    pass


class JobConfig(Config):
    '''Job configuration settings'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name", default="job")
        self.working_dir = self.parse_string(d, "working_dir", default=".")
        self.interpreter = self.parse_string(
            d, "interpreter", default="python")
        self.script = self.parse_string(d, "script", default=None)
        self.binary = self.parse_string(d, "binary", default=None)
        self.custom = self.parse_array(d, "custom", default=None)
        self.config_path = self.parse_string(d, "config_path")


if __name__ == "__main__":
    run(JobConfig.from_json(sys.argv[1]))
