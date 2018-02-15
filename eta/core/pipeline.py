'''
Pipeline infrastructure for running series of jobs.

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
import sys

from eta.core.config import Config, Configurable
from eta.core.diagram import BlockDiagram
import eta.core.job as etaj
import eta.core.log as etal
import eta.core.module as etam
import eta.core.utils as etau


logger = logging.getLogger(__name__)


def run(pipeline_config_path):
    '''Run the pipeline specified by the PipelineConfig.

    Args:
        pipeline_config_path: path to a PipelineConfig file
    '''
    # Load config
    pipeline_config = PipelineConfig.from_json(pipeline_config_path)

    # Convert to absolute path so jobs can find the pipeline config later
    # regardless of their working directory
    pipeline_config_path = os.path.abspath(pipeline_config_path)

    # Setup logging
    etal.custom_setup(pipeline_config.logging_config, rotate=True)

    # Run pipeline
    logger.info("Starting pipeline '%s'\n", pipeline_config.name)
    overwrite = pipeline_config.overwrite
    ran_job = False
    with etau.WorkingDir(pipeline_config.working_dir):
        for job_config in pipeline_config.jobs:
            if ran_job and not overwrite:
                logger.info(
                    "Config change detected, running all remaining jobs")
                overwrite = True

            job_config.pipeline_config_path = pipeline_config_path
            ran_job = etaj.run(job_config, overwrite=overwrite)

    logger.info("Pipeline '%s' complete", pipeline_config.name)


class PipelineConfig(Config):
    '''Pipeline configuration settings'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name", default="pipeline")
        self.working_dir = self.parse_string(d, "working_dir", default=".")
        self.overwrite = self.parse_bool(d, "overwrite", default=True)
        self.jobs = self.parse_object_array(
            d, "jobs", etaj.JobConfig, default=[])
        self.logging_config = self.parse_object(
            d, "logging_config", etal.LoggingConfig,
            default=etal.LoggingConfig.default())


class PipelineMetadataConfig(Config):
    '''Pipeline metadata configuration class'''

    def __init__(self, d):
        self.info = self.parse_object(d, "info", InfoConfig)
        self.modules = self.parse_array(d, "modules")
        self.connections = self.parse_object_array(
            d, "connections", ConnectionConfig)


class InfoConfig(Config):
    '''Pipeline info configuration class'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.version = self.parse_string(d, "version")
        self.description = self.parse_string(d, "description")
        self.id = self.parse_string(d, "id")


class ConnectionConfig(Config):
    '''Connection edge configuration class.'''

    def __init__(self, d):
        self.source = self.parse_string(d, "source")
        self.sink = self.parse_string(d, "sink")


class PipelineMetadata(Configurable, BlockDiagram):
    '''Class the encapsulates the architecture of a pipeline.'''

    def __init__(self, config):
        '''Initializes a PipelineMetadata instance.

        Args:
            config: a PipelineMetadataConfig instance

        Raises:
            PipelineMetadataError: if there was an error parsing the pipeline
                definition
        '''
        self.validate(config)
        self.config = config
        self.modules = []
        self.parse_metadata()

    def parse_metadata(self):
        '''Parses the pipeline metadata config.'''
        # Load modules
        for name in self.config.modules:
            self.modules.append(etam.load_metadata(name))

        # Parse connections
        for c in self.config.connections:
            # @todo implement

    def _to_blockdiag(self, path):
        bp = BlockdiagPipeline(self.config.info.name)
        # @todo implement
        bp.write(path)


class PipelineMetadataError(Exception):
    pass


if __name__ == "__main__":
    run(sys.argv[1])
