'''
Core pipeline building system.

Copyright 2018, Voxel51, LLC
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
from future.utils import iteritems
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging
import os
import time

import eta
from eta.core.config import Config, Configurable
import eta.core.job as etaj
import eta.core.log as etal
import eta.core.module as etam
import eta.core.pipeline as etap
import eta.core.types as etat


logger = logging.getLogger(__name__)


PIPELINE_JSON_FILE = "pipeline.json"
PIPELINE_LOG_FILE = "pipeline.log"
MODULE_CONFIG_EXT = ".config"


def _get_timestamp():
    return time.strftime("%Y.%m.%d-%H:%M:%S")


class PipelineBuilder(object):
    '''Class that handles building a pipeline based on a PipelineRequest.'''

    def __init__(self, request):
        '''Creates a PipelineBuilder instance.

        Args:
            request: a PipelineRequest instance
        '''
        self.request = request

        self.timestamp = None
        self.config_dir = ""
        self.output_dir = ""
        self.pipeline_config = None
        self.pipeline_config_path = ""
        self.module_configs = {}
        self.module_config_paths = {}

    def build(self):
        '''Builds the pipeline and writes the associated config files.'''
        self.timestamp = _get_timestamp()
        self.config_dir = self._get_config_dir()
        self.output_dir = self._get_output_dir()

        self._build_pipeline_config()
        self._build_module_configs()

    def _build_pipeline_config(self):
        # Build job configs
        # @todo handle non-py executables
        jobs = []
        for module in self.request.metadata.execution_order:
            metadata = self.request.metadata.modules[module].metadata
            jobs.append(
                etaj.JobConfig.builder().set(
                    name=module,
                    working_dir=".",
                    script=etam.find_exe(metadata.info.exe),
                    config_path=self._get_module_config_path(module),
                ).build()
            )

        # Build logging config
        logging_config = etal.LoggingConfig.builder().set(
            filename=self._get_log_path()
        ).build()

        # Build pipeline config
        self.pipeline_config = etap.PipelineConfig.builder().set(
            name=self.request.name,
            working_dir=".",
            overwrite=False,
            jobs=jobs,
            logging_config=logging_config,
        ).build()

        # Write pipeline config
        self.pipeline_config_path = self._get_pipeline_config_path()
        logger.info("Writing pipeline config '%s'", self.pipeline_config_path)
        self.pipeline_config.write_json(self.pipeline_config_path)

    def _build_module_configs(self):
        for module in self.request.metadata.execution_order:
            metadata = self.request.metadata.modules[module].metadata

            # Build field configs
            inputs = {}
            outputs = {}
            parameters = {}
            # @todo implement this

            # Build module config
            module_config = GenericModuleConfig.builder().set(
                inputs=GenericFieldConfig(inputs),
                outputs=GenericFieldConfig(outputs),
                parameters=GenericFieldConfig(parameters),
            ).build()

            # Write module config
            module_config_path = self._get_module_config_path(module)
            logger.info("Writing module config '%s'", module_config_path)
            module_config.write_json(module_config_path)

            self.module_configs[module] = module_config
            self.module_config_paths[module] = module_config_path

    def _get_module_config_path(self, module):
        return os.path.join(self.config_dir, module + MODULE_CONFIG_EXT)

    def _get_pipeline_config_path(self):
        return os.path.join(self.config_dir, PIPELINE_JSON_FILE)

    def _get_log_path(self):
        return os.path.join(self.config_dir, PIPELINE_LOG_FILE)

    def _get_data_path(self, field, params):
        return field.type.gen_path(self.output_dir, params)

    def _get_config_dir(self):
        return os.path.join(
            eta.config.config_dir,
            self.request.name + os.sep + self.timestamp
        )

    def _get_output_dir(self):
        return os.path.join(
            eta.config.output_dir,
            self.request.name + os.sep + self.timestamp
        )


class GenericModuleConfig(etam.BaseModuleConfig):
    '''Generic module configuration class.'''

    def __init__(self, d):
        super(GenericModuleConfig, self).__init__(d)
        self.inputs = self.parse_object(
            d, "inputs", GenericFieldConfig, default=GenericFieldConfig())
        self.outputs = self.parse_dict(
            d, "outputs", GenericFieldConfig, default=GenericFieldConfig())
        self.parameters = self.parse_dict(
            d, "parameters", GenericFieldConfig, default=GenericFieldConfig())


class GenericFieldConfig(Config):
    '''Generic module field configuration class.'''

    def __init__(self, d):
        for k, v in iteritems(d):
            setattr(self, k, v)


class PipelineRequestConfig(Config):
    '''Pipeline request configuration class.'''

    def __init__(self, d):
        self.pipeline = self.parse_string(d, "pipeline")
        self.inputs = self.parse_dict(d, "inputs", default={})
        self.parameters = self.parse_dict(d, "parameters", default={})


class PipelineRequest(Configurable):
    '''Pipeline request class.

    Attributes:
        name: the name of the pipeline to be run
        metadata: the PipelineMetadata instance for the pipeline
        inputs: a dictionary mapping input names to input paths
        parameters: a dictionary mapping <module>.<parameter> names to
            parameter values
    '''

    def __init__(self, config):
        '''Creates a new PipelineRequest instance.

        Args:
            config: a PipelineRequestConfig instance.

        Raises:
            PipelineRequestError: if the pipeline request was invalid
        '''
        self.validate(config)

        self.name = config.pipeline
        self.metadata = etap.load_metadata(config.pipeline)
        self.inputs = config.inputs
        self.parameters = config.parameters

        self._validate_inputs()
        self._validate_parameters()

    def _validate_inputs(self):
        # Validate inputs
        for iname, ival in iteritems(self.inputs):
            if not self.metadata.has_input(iname):
                raise PipelineRequestError(
                    "Pipeline '%s' has no input '%s'" % (self.name, iname))
            if not self.metadata.is_valid_input(iname, ival):
                raise PipelineRequestError((
                    "'%s' is not a valid value for input '%s' of pipeline "
                    "'%s'") % (ival, iname, self.name)
                )

        # Ensure that mandatory inputs were supplied
        for miname, miobj in iteritems(self.metadata.inputs):
            if miobj.is_mandatory and miname not in self.inputs:
                raise PipelineRequestError((
                    "Mandatory input '%s' of pipeline '%s' was not "
                    "specified") % (miname, self.name)
                )

    def _validate_parameters(self):
        # Validate parameters
        for pname, pval in iteritems(self.parameters):
            if not self.metadata.has_parameter(pname):
                raise PipelineRequestError(
                    "Pipeline '%s' has no parameter '%s'" % (self.name, pname))
            if not self.metadata.is_valid_parameter(pname, pval):
                raise PipelineRequestError((
                    "'%s' is not a valid value for parameter '%s' of pipeline "
                    "'%s'") % (pval, pname, self.name)
                )

        # Ensure that mandatory parmeters were supplied
        for mpname, mpobj in iteritems(self.metadata.parameters):
            if mpobj.is_mandatory and mpname not in self.parameters:
                raise PipelineRequestError((
                    "Mandatory parameter '%s' of pipeline '%s' was not "
                    "specified") % (mpname, self.name)
                )


class PipelineRequestError(Exception):
    pass
