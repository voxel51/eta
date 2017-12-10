'''
Pipeline infrastructure for running series of jobs.

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
import sys

from eta.core.config import Config
from eta.core import job
from eta.core import log
from eta.core import utils


logger = logging.getLogger(__name__)


def run(pipeline_config_path):
    '''Run the pipeline specified by the PipelineConfig.

    Args:
        pipeline_config_path: path to a PipelineConfig file
    '''
    # Load config
    pipeline_config = PipelineConfig.from_json(pipeline_config_path)

    # Setup logging
    log.custom_setup(pipeline_config.logging_config, overwrite=True)

    # Run pipeline
    logger.info("Starting pipeline '%s'\n", pipeline_config.name)
    overwrite = pipeline_config.overwrite
    ran_job = False
    with utils.WorkingDir(pipeline_config.working_dir):
        for job_config in pipeline_config.jobs:
            if ran_job and not overwrite:
                logger.info(
                    "Config change detected, running all remaining jobs")
                overwrite = True

            job_config.pipeline_config_path = pipeline_config_path
            ran_job = job.run(job_config, overwrite=overwrite)

    logger.info("Pipeline '%s' complete", pipeline_config.name)


class PipelineConfig(Config):
    '''Pipeline configuration settings'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name", default="pipeline")
        self.working_dir = self.parse_string(d, "working_dir", default=".")
        self.overwrite = self.parse_bool(d, "overwrite", default=True)
        self.jobs = self.parse_object_array(
            d, "jobs", job.JobConfig, default=[])
        self.logging_config = self.parse_object(
            d, "logging_config", log.LoggingConfig,
            default=log.LoggingConfig.default())


if __name__ == "__main__":
    run(sys.argv[1])
