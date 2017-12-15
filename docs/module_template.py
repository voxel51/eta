#!/usr/bin/env python
'''
{{Description Of The Module}}

Copyright 2017, Voxel51, LLC
voxel51.com
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

import eta.core.module as etam


logger = logging.getLogger(__name__)


class {{ModuleTemplate}}Config(etam.BaseModuleConfig):
    '''Module configuration settings.'''

    def __init__(self, d):
        super({{ModuleTemplate}}Config, self).__init__(d)


def main(config_path, pipeline_config_path=None):
    '''Run the module.

    Args:
        config_path: path to {{ModuleTemplate}}Config file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    config = {{ModuleTemplate}}Config.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
