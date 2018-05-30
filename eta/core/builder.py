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

from collections import defaultdict
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
import eta.core.utils as etau


logger = logging.getLogger(__name__)


PIPELINE_CONFIG_FILE = "pipeline.json"
PIPELINE_LOGFILE_FILE = "pipeline.log"
PIPELINE_STATUS_FILE = "status.json"
MODULE_CONFIG_EXT = ".json"


class PipelineBuildRequestConfig(Config):
    '''Pipeline build request configuration class.'''

    def __init__(self, d):
        self.pipeline = self.parse_string(d, "pipeline")
        self.inputs = self.parse_dict(d, "inputs", default={})
        self.parameters = self.parse_dict(d, "parameters", default={})


class PipelineBuildRequest(Configurable):
    '''Pipeline build request class.

    A pipeline build request is valid if all of the following are true:
        - a pipeline of the specified name exists
        - all required pipeline inputs are provided and have valid values
        - all required pipeline parameters are provided and have valid values

    Note that input/parameter fields set to `None` are ignored (and thus must
    be optional in order for the build request to be valid).

    Attributes:
        pipeline: the (name of the) pipeline to run
        metadata: the PipelineMetadata instance for the pipeline
        inputs: a dictionary mapping input names to input paths
        parameters: a dictionary mapping <module>.<parameter> names to
            parameter values
    '''

    def __init__(self, config):
        '''Creates a new PipelineBuildRequest instance.

        Args:
            config: a PipelineBuildRequestConfig instance.

        Raises:
            PipelineBuildRequestError: if the pipeline request was invalid
        '''
        self.validate(config)

        self.pipeline = config.pipeline
        self.metadata = etap.load_metadata(config.pipeline)
        self.inputs = etau.remove_none_values(config.inputs)
        self.parameters = etau.remove_none_values(config.parameters)

        self._validate_inputs()
        self._validate_parameters()

    def _validate_inputs(self):
        # Validate inputs
        for iname, ipath in iteritems(self.inputs):
            if not self.metadata.has_input(iname):
                raise PipelineBuildRequestError(
                    "Pipeline '%s' has no input '%s'" % (self.pipeline, iname))
            if not self.metadata.is_valid_input(iname, ipath):
                raise PipelineBuildRequestError((
                    "'%s' is not a valid value for input '%s' of pipeline "
                    "'%s'") % (ipath, iname, self.pipeline)
                )

        # Ensure that required inputs were supplied
        for miname, miobj in iteritems(self.metadata.inputs):
            if miobj.is_required and miname not in self.inputs:
                raise PipelineBuildRequestError((
                    "Required input '%s' of pipeline '%s' was not "
                    "supplied") % (miname, self.pipeline)
                )

    def _validate_parameters(self):
        # Validate parameters
        for pname, pval in iteritems(self.parameters):
            if not self.metadata.has_tunable_parameter(pname):
                raise PipelineBuildRequestError(
                    "Pipeline '%s' has no tunable parameter '%s'" % (
                        self.pipeline, pname))
            if not self.metadata.is_valid_parameter(pname, pval):
                raise PipelineBuildRequestError((
                    "'%s' is not a valid value for parameter '%s' of pipeline "
                    "'%s'") % (pval, pname, self.pipeline)
                )

        # Ensure that required parmeters were supplied
        for mpname, mpobj in iteritems(self.metadata.parameters):
            if mpobj.is_required and mpname not in self.parameters:
                raise PipelineBuildRequestError((
                    "Required parameter '%s' of pipeline '%s' was not "
                    "specified") % (mpname, self.pipeline)
                )


class PipelineBuildRequestError(Exception):
    pass


class PipelineBuilder(object):
    '''Class for building a pipeline based on a PipelineBuildRequest.

    Attributes:
        request: the PipelineBuildRequest instance
        timestamp: the time when the pipeline was built
        config_dir: the directory where the pipeline and module configuration
            files were written
        output_dir: the base directory where pipeline outputs will be written
            when the pipeline is run
        pipeline_config_path: the path to the pipeline config file to run the
            pipeline
        pipeline_status_path: the path to the pipeline status JSON file that
            will be generated when the pipeline is run
        pipeline_logfile_path: the path to the pipeline logfile that will be
            generated when the pipeline is run
        outputs: a dictionary mapping pipeline outputs to the paths where they
            will be written when the pipeline is run
    '''

    def __init__(self, request):
        '''Creates a PipelineBuilder instance.

        Args:
            request: a PipelineBuildRequest instance
        '''
        self.request = request
        self.timestamp = None
        self.config_dir = None
        self.output_dir = None
        self.pipeline_config_path = None
        self.pipeline_logfile_path = None
        self.pipeline_status_path = None
        self.outputs = {}

        self._concrete_data_params = etat.ConcreteDataParams()

    def build(self):
        '''Builds the pipeline and writes the associated config files.'''
        #
        # Using local time could cause non-uniqueness around timezone changes
        # and daylight savings time, but we'll assume that users would rather
        # have human-readable directory names than using gmtime() to ensure
        # uniqueness.
        #
        # Note that uniqueness is not an issue in the typical container-based
        # deployment scenario where only a single pipeline is run in each
        # image.
        #
        self.timestamp = time.localtime()
        self.config_dir = self._get_config_dir()
        self.output_dir = self._get_output_dir()
        self.outputs = {}

        self._build_pipeline_config()
        self._build_module_configs()

    def _build_pipeline_config(self):
        # Build job configs
        # @todo handle non-py executables
        job_builders = []
        for module in self.request.metadata.execution_order:
            metadata = self.request.metadata.modules[module].metadata
            job_builders.append(
                etaj.JobConfig.builder()
                    .set(name=module)
                    .set(working_dir=".")
                    .set(script=etam.find_exe(metadata.info.exe))
                    .set(config_path=self._get_module_config_path(module))
                    .validate()
            )

        # Build logging config
        self.pipeline_logfile_path = self._get_pipeline_logfile_path()
        logging_config_builder = (etal.LoggingConfig.builder()
            .set(filename=self.pipeline_logfile_path)
            .validate())

        # Build pipeline config
        self.pipeline_status_path = self._get_pipeline_status_path()
        pipeline_config_builder = (etap.PipelineConfig.builder()
            .set(name=self.request.pipeline)
            .set(working_dir=".")
            .set(status_path=self.pipeline_status_path)
            .set(overwrite=False)
            .set(jobs=job_builders)
            .set(logging_config=logging_config_builder)
            .validate())

        # Write pipeline config
        self.pipeline_config_path = self._get_pipeline_config_path()
        logger.info("Writing pipeline config '%s'", self.pipeline_config_path)
        pipeline_config_builder.write_json(self.pipeline_config_path)

    def _build_module_configs(self):
        pmeta = self.request.metadata  # PipelineMetadata

        # Populate module I/O
        module_inputs = defaultdict(dict)
        module_outputs = defaultdict(dict)
        for module in pmeta.execution_order:
            mmeta = pmeta.modules[module].metadata  # ModuleMetadata

            # Populate inputs
            iconns = _get_incoming_connections(module, pmeta.connections)
            for iname, inode in iteritems(mmeta.inputs):
                if iname in iconns:
                    isrc = iconns[iname]
                    if isrc.is_pipeline_input:
                        # Get input from pipeline
                        ipath = self.request.inputs[isrc.node]
                        module_inputs[module][iname] = ipath
                    # Other inputs are populated by connected outputs...

            # Populate outputs
            oconns = _get_outgoing_connections(module, pmeta.connections)
            for oname, onode in iteritems(mmeta.outputs):
                if oname in oconns:
                    # Record output
                    opath = self._get_data_path(module, onode)
                    module_outputs[module][oname] = opath

                    osrc = oconns[oname]
                    if osrc.is_pipeline_output:
                        # Record pipeline output
                        self.outputs[osrc.node] = opath
                    else:
                        # Pass output to connected inputs
                        module_inputs[osrc.module][osrc.node] = opath

        # Populate module parameters
        module_params = defaultdict(dict)
        for param_str, param in iteritems(pmeta.parameters):
            val, found = _get_param_value(param_str, param, self.request)
            if found:
                module_params[param.module][param.name] = val

        # Generate module configs
        for module in pmeta.execution_order:
            # Build module config
            data = etau.join_dicts(
                module_inputs[module], module_outputs[module])
            module_config_builder = (etam.GenericModuleConfig.builder()
                .set(data=[data])
                .set(parameters=module_params[module])
                .validate())

            # Write module config
            module_config_path = self._get_module_config_path(module)
            logger.info("Writing module config '%s'", module_config_path)
            module_config_builder.write_json(module_config_path)

    def _get_timestamp_str(self):
        return time.strftime("%Y.%m.%d-%H.%M.%S", self.timestamp)

    def _get_config_dir(self):
        time_str = self._get_timestamp_str()
        return os.path.join(
            eta.config.config_dir, self.request.pipeline, time_str)

    def _get_output_dir(self):
        time_str = self._get_timestamp_str()
        return os.path.join(
            eta.config.output_dir, self.request.pipeline, time_str)

    def _get_pipeline_config_path(self):
        return os.path.join(self.config_dir, PIPELINE_CONFIG_FILE)

    def _get_pipeline_logfile_path(self):
        return os.path.join(self.config_dir, PIPELINE_LOGFILE_FILE)

    def _get_pipeline_status_path(self):
        return os.path.join(self.output_dir, PIPELINE_STATUS_FILE)

    def _get_module_config_path(self, module):
        return os.path.join(self.config_dir, module + MODULE_CONFIG_EXT)

    def _get_data_path(self, module, node):
        params = self._concrete_data_params.render_for(node.name)
        basedir = os.path.join(self.output_dir, module)
        return node.type.gen_path(basedir, params)


def _get_param_value(param_str, param, request):
    '''Gets the value for the parameter, resolving it if necessary.

    Args:
        param_str: the <module>.<parameter> string of the parameter
        param: a PipelineParameter instance describing the parameter
        request: the PipelineBuildRequest instance

    Returns:
        val: the parameter value, or None if a value could not be found
        found: True/False
    '''
    if param_str in request.parameters:
        # User-set parameter
        val = request.parameters[param_str]
        found = True
    elif param.has_set_value:
        # Pipeline-set parameter
        val = param.set_value
        found = True
    elif param.has_default_value:
         # Module-default value
         val = param.default_value
         found = True
    else:
        val = None
        found = False

    if found:
        # Resolve parameter value
        val = etat.resolve_value(val, param.param.type)

    return val, found


def _get_incoming_connections(module, connections):
    '''Gets the incoming connections for the given module.

    Args:
        module: the module name
        connections: a list of PipelineConnection instances describing the
            module I/O connections in the pipeline

    Returns:
        a dictionary mapping the names of the inputs of the given module to
            the PipelineNode instances describing the nodes that they are
            connected to
    '''
    return {
        c.sink.node: c.source
        for c in connections if c.sink.module == module
    }


def _get_outgoing_connections(module, connections):
    '''Gets the outgoing connections for the given module.

    Args:
        module: the module name
        connections: a list of PipelineConnection instances describing the
            module I/O connections in the pipeline

    Returns:
        a dictionary mapping the names of the outputs of the given module to
            the PipelineNode instances describing the nodes that they are
            connected to
    '''
    return {
        c.source.node: c.sink
        for c in connections if c.source.module == module
    }
