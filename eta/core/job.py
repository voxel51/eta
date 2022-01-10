"""
Core job infrastructure for running modules in a pipeline.

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
import sys

from eta.core.config import Config
import eta.core.logging as etal
import eta.core.utils as etau


logger = logging.getLogger(__name__)


def run(job_config, pipeline_status, overwrite=True):
    """Run the job specified by the JobConfig.

    If the job completes succesfully, the hash of the config file is written to
    disk.

    Args:
        job_config: a JobConfig instance
        pipeline_status: a PipelineStatus instance
        overwrite: overwrite mode. When True, always run the job. When False,
            only run the job if the config file has changed since the last time
            the job was (succesfully) run

    Returns:
        (ran_job, success), where:
            ran_job: True/False if the job was actually run
            success: True/False if execution terminated succesfully

    Raises:
        JobConfigError: if the JobConfig was invalid
    """
    job_status = pipeline_status.add_job(job_config.name)

    with etau.WorkingDir(job_config.working_dir):
        # Check config hash
        hasher = etau.MD5FileHasher(job_config.config_path)
        if hasher.has_changed:
            logger.info("Config %s changed", job_config.config_path)
            should_run = True
        elif hasher.has_record:
            if overwrite:
                logger.info("Overwriting existing job output")
                should_run = True
            else:
                logger.info("Skipping job %s", job_config.name)
                should_run = False
        else:
            should_run = True

        if should_run:
            logger.info("Working directory: %s", os.getcwd())

            # Run job
            logger.info("Starting job %s", job_config.name)
            job_status.start()
            success = _run(job_config)
            if not success:
                # Job failed
                logger.error("Job %s failed... exiting now", job_config.name)
                job_status.fail()
                return should_run, False

            # Job complete!
            logger.info("Job %s complete", job_config.name)
            hasher.write()  # write config hash
            job_status.complete()
        else:
            # Skip job
            job_status.skip()

        return should_run, True


def _run(job_config):
    # Construct command
    if job_config.binary:
        args = [job_config.binary]  # binary
    elif job_config.script:
        args = [
            job_config.interpreter,  # interpreter
            job_config.script,  # script
        ]
    elif job_config.custom:
        # Run custom command-line
        args = job_config.custom  # custom args
    else:
        raise JobConfigError("Invalid JobConfig")

    # Add config files
    args.append(job_config.config_path)  # module config
    if job_config.pipeline_config_path:
        args.append(job_config.pipeline_config_path)  # pipeline config

    # Run command
    etal.flush()  # must flush because subprocess will append to same logfile
    success = etau.call(args)

    return success


class JobConfigError(Exception):
    """Exception raised when an invalid JobConfig is encountered."""

    pass


class JobConfig(Config):
    """Job configuration settings"""

    def __init__(self, d):
        self.name = self.parse_string(d, "name", default="job")
        self.working_dir = self.parse_string(d, "working_dir", default=None)
        self.interpreter = self.parse_string(
            d, "interpreter", default="python"
        )
        self.script = self.parse_string(d, "script", default=None)
        self.binary = self.parse_string(d, "binary", default=None)
        self.custom = self.parse_array(d, "custom", default=None)
        self.config_path = self.parse_string(d, "config_path")
        self.pipeline_config_path = self.parse_string(
            d, "pipeline_config_path", default=None
        )
