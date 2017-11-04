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

import sys

from eta.core.config import Config
from eta.core import job
from eta.core import utils


def run(pipeline_config):
    '''Run the pipeline specified by the PipelineConfig.'''
    print("Starting pipeline '%s'" % pipeline_config.name)

    overwrite = pipeline_config.overwrite
    ran_job = False
    with utils.WorkingDir(pipeline_config.working_dir):
        for job_config in pipeline_config.jobs:
            if ran_job and not overwrite:
                print("Config change detected, running all remaining jobs")
                overwrite = True

            ran_job = job.run(job_config, overwrite=overwrite)

    print("Pipeline '%s' complete" % pipeline_config.name)


class PipelineConfig(Config):
    '''Pipeline configuration settings'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name", default="")
        self.working_dir = self.parse_string(d, "working_dir", default=".")
        self.overwrite = self.parse_bool(d, "overwrite", default=True)
        self.jobs = self.parse_object_array(d, "jobs", job.JobConfig)


if __name__ == "__main__":
    run(PipelineConfig.from_json(sys.argv[1]))
