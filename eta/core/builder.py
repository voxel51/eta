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
from future.utils import iteritems, itervalues
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict
import copy
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
        self.outputs = self.parse_dict(d, "outputs", default={})
        self.parameters = self.parse_dict(d, "parameters", default={})
        self.logging_config = self.parse_object(
            d, "logging_config", etal.LoggingConfig, default=None)


class PipelineBuildRequest(Configurable):
    '''Pipeline build request class.

    A pipeline build request is valid if all of the following are true:
        - a pipeline of the specified name exists
        - all required pipeline inputs are provided and have valid values
        - any output paths specified must refer to valid pipeline output names
            and the specified paths must be valid paths for each output type
        - all required pipeline parameters are provided and have valid values

    Note that any fields set to `None` are ignored, so any inputs/parameters
    that are `None` must be optional in order for the request to be valid.

    Attributes:
        pipeline: the name of the pipeline to run
        metadata: the PipelineMetadata instance for the pipeline
        inputs: a dictionary mapping input names to input paths
        outputs: a dictionary mapping output names to output paths (if any)
        parameters: a dictionary mapping <module>.<parameter> names to
            parameter values
        logging_config: the LoggingConfig for the pipeline (if any)
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
        self.outputs = etau.remove_none_values(config.outputs)
        self.parameters = etau.remove_none_values(config.parameters)
        self.logging_config = config.logging_config

        self._validate_inputs()
        self._validate_outputs()
        self._validate_parameters()

    def _validate_inputs(self):
        # Validate inputs
        for iname, ipath in iteritems(self.inputs):
            if not self.metadata.has_input(iname):
                raise PipelineBuildRequestError(
                    "Pipeline '%s' has no input '%s'" % (self.pipeline, iname))
            if not self.metadata.is_valid_input(iname, ipath):
                raise PipelineBuildRequestError(
                    "'%s' is not a valid value for input '%s' of pipeline "
                    "'%s'" % (ipath, iname, self.pipeline))
            # Convert to absolute paths
            self.inputs[iname] = os.path.abspath(ipath)

        # Ensure that required inputs were supplied
        for miname, miobj in iteritems(self.metadata.inputs):
            if miobj.is_required and miname not in self.inputs:
                raise PipelineBuildRequestError(
                    "Required input '%s' of pipeline '%s' was not "
                    "supplied" % (miname, self.pipeline))

    def _validate_outputs(self):
        # Validate outputs
        for oname, opath in iteritems(self.outputs):
            if not self.metadata.has_output(oname):
                raise PipelineBuildRequestError(
                    "Pipeline '%s' has no output '%s'" % (
                        self.pipeline, oname))
            if not self.metadata.is_valid_output(oname, opath):
                raise PipelineBuildRequestError(
                    "'%s' is not a valid value for output '%s' of pipeline "
                    "'%s'" % (opath, oname, self.pipeline))
            # Convert to absolute paths
            self.outputs[oname] = os.path.abspath(opath)

    def _validate_parameters(self):
        # Validate parameters
        for pname, pval in iteritems(self.parameters):
            if not self.metadata.has_tunable_parameter(pname):
                raise PipelineBuildRequestError(
                    "Pipeline '%s' has no tunable parameter '%s'" % (
                        self.pipeline, pname))
            if not self.metadata.is_valid_parameter(pname, pval):
                raise PipelineBuildRequestError(
                    "'%s' is not a valid value for parameter '%s' of pipeline "
                    "'%s'" % (pval, pname, self.pipeline))
            # Convert any data parameters to absolute paths
            if self.metadata.parameters[pname].is_data:
                self.parameters[pname] = os.path.abspath(pval)

        # Ensure that required parmeters were supplied
        for mpname, mpobj in iteritems(self.metadata.parameters):
            if mpobj.is_required and mpname not in self.parameters:
                raise PipelineBuildRequestError(
                    "Required parameter '%s' of pipeline '%s' was not "
                    "specified" % (mpname, self.pipeline))


class PipelineBuildRequestError(Exception):
    '''Exception raised when an invalid PipelineBuildRequest is encountered.'''
    pass


class PipelineBuilder(object):
    '''Class for building a pipeline based on a PipelineBuildRequest.

    Attributes:
        request: the PipelineBuildRequest instance used to build the pipeline
        timestamp: the time when the pipeline was built
        config_dir: the directory where the pipeline and module configuration
            files were written and where the pipeline log will be written
        output_dir: the base directory where pipeline outputs and pipeline
            status will be written when the pipeline is run
        pipeline_config_path: the path to the pipeline config file to run the
            pipeline
        pipeline_status_path: the path to the pipeline status JSON file that
            will be generated when the pipeline is run
        pipeline_logfile_path: the path to the pipeline logfile that will be
            generated when the pipeline is run
        pipeline_outputs: a dictionary mapping pipeline outputs to the paths
            where they will be written when the pipeline is run. These
            paths are all within `output_dir`, and they are populated for all
            pipeline outputs regardless of whether the output was included in
            the `outputs` dictionary of the PipelineBuildRequest
        outputs: the outputs dictionary from the PipelineBuildRequest, which
            specifies where to publish certain pipeline outputs after the
            pipeline is run
    '''

    def __init__(self, request):
        '''Creates a PipelineBuilder instance.

        Args:
            request: a PipelineBuildRequest instance
        '''
        self.request = request
        self.outputs = self.request.outputs
        self._concrete_data_params = etat.ConcreteDataParams()
        self.reset()

    def reset(self):
        '''Resets the builder so that another pipeline can be built.'''
        self.timestamp = None
        self.config_dir = None
        self.output_dir = None
        self.pipeline_config_path = None
        self.pipeline_logfile_path = None
        self.pipeline_status_path = None
        self.pipeline_outputs = {}

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

        # Generate paths, if necessary
        if not self.config_dir:
            self.config_dir = self._get_config_dir()
        if not self.output_dir:
            self.output_dir = self._get_output_dir()
        if not self.pipeline_config_path:
            self.pipeline_config_path = self._make_pipeline_config_path()
        if not self.pipeline_logfile_path:
            self.pipeline_logfile_path = self._make_pipeline_logfile_path()
        if not self.pipeline_status_path:
            self.pipeline_status_path = self._make_pipeline_status_path()
        self.pipeline_outputs = {}

        self._build_pipeline_config()
        self._build_module_configs()

    def run(self):
        '''Runs the built pipeline and publishes any outputs to their
        specified locations.

        Returns:
            True/False whether the pipeline completed successfully

        Raises:
            PipelineBuilderError: if the pipeline hasn't been built
        '''
        if not self.pipeline_config_path:
            raise PipelineBuilderError(
                "You must build the pipeline before running it")

        # Run pipeline
        success = etap.run(self.pipeline_config_path)
        if not success:
            return False

        # Publish outputs
        for oname, opath in iteritems(self.outputs):
            ppath = self.pipeline_outputs[oname]
            if os.path.isfile(ppath):
                # Output is a file
                etau.copy_file(ppath, opath, check_ext=True)
            elif os.path.isdir(ppath):
                # Output is a directory
                etau.copy_dir(ppath, opath)
            else:
                # Assume the output is a sequence
                etau.copy_sequence(ppath, opath, check_ext=True)

        return True

    def cleanup(self):
        '''Cleans up the configs and output files generated when the pipeline
        was run, if necessary. Published outputs are NOT deleted.

        Raises:
            PipelineBuilderError: if the pipeline hasn't been built
        '''
        if not self.config_dir or not self.output_dir:
            raise PipelineBuilderError(
                "You must build and run the pipeline before you can clean "
                "it up")

        try:
            etau.delete_dir(self.config_dir)
            logger.info("Deleted config directory '%s'", self.config_dir)
        except OSError:
            pass

        try:
            etau.delete_dir(self.output_dir)
            logger.info("Deleted output directory '%s'", self.output_dir)
        except OSError:
            pass

    def _build_pipeline_config(self):
        # Build job configs
        # @todo handle non-py executables
        jobs = []
        for module in self.request.metadata.execution_order:
            metadata = self.request.metadata.modules[module].metadata
            jobs.append(
                etaj.JobConfig.builder()
                    .set(name=module)
                    .set(script=etam.find_exe(metadata))
                    .set(config_path=self._get_module_config_path(module))
                    .validate())

        # Handle logging
        if self.request.logging_config is not None:
            logging_config = copy.deepcopy(self.request.logging_config)
            if logging_config.filename:
                # Accept the provided logfile location
                self.pipeline_logfile_path = logging_config.filename
            else:
                # Generate a pipeline log in our automated location
                logging_config.filename = self.pipeline_logfile_path
        else:
            logging_config = (etal.LoggingConfig.builder()
                .set(filename=self.pipeline_logfile_path)
                .validate())

        # Build pipeline config
        pipeline_config_builder = (etap.PipelineConfig.builder()
            .set(name=self.request.pipeline)
            .set(status_path=self.pipeline_status_path)
            .set(overwrite=False)
            .set(jobs=jobs)
            .set(logging_config=logging_config)
            .validate())

        # Write pipeline config
        logger.info("Writing pipeline config '%s'", self.pipeline_config_path)
        pipeline_config_builder.write_json(self.pipeline_config_path)

    def _build_module_configs(self):
        pmeta = self.request.metadata  # PipelineMetadata
        module_inputs = defaultdict(dict)
        module_outputs = defaultdict(dict)

        # Distribute pipeline inputs
        for iname, ipath in iteritems(self.request.inputs):
            for sink in pmeta.get_input_sinks(iname):
                module_inputs[sink.module][sink.node] = ipath

        # Get path hints from published outputs
        path_hints = {}
        for poname, popath in iteritems(self.request.outputs):
            source = pmeta.get_output_source(poname)
            path_hints[str(source)] = popath

        # Propagate module connections
        for module in pmeta.execution_order:
            mmeta = pmeta.modules[module].metadata  # ModuleMetadata
            oconns = pmeta.get_outgoing_connections(module)
            for oname, osinks in iteritems(oconns):
                if oname in module_outputs[module]:
                    # This output has multiple outgoing connections; we
                    # already generated its path
                    opath = module_outputs[module][oname]
                else:
                    # Set output path
                    onode = mmeta.outputs[oname]
                    opath = self._get_data_path(module, onode, path_hints)
                    module_outputs[module][oname] = opath

                for osink in osinks:
                    if osink.is_pipeline_output:
                        # Set pipeline output
                        self.pipeline_outputs[osink.node] = opath
                    else:
                        # Pass output to connected module input
                        module_inputs[osink.module][osink.node] = opath

        # Populate module parameters
        module_params = defaultdict(dict)
        for param in itervalues(pmeta.parameters):
            val = _get_param_value(param, self.request)
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

    def _make_pipeline_config_path(self):
        return os.path.join(self.config_dir, PIPELINE_CONFIG_FILE)

    def _make_pipeline_logfile_path(self):
        return os.path.join(self.config_dir, PIPELINE_LOGFILE_FILE)

    def _make_pipeline_status_path(self):
        return os.path.join(self.output_dir, PIPELINE_STATUS_FILE)

    def _get_module_config_path(self, module):
        return os.path.join(self.config_dir, module + MODULE_CONFIG_EXT)

    def _get_data_path(self, module, node, path_hints):
        basedir = os.path.join(self.output_dir, module)
        hint = _get_path_hint(module, node.name, path_hints)
        params = self._concrete_data_params.render_for(node.name, hint=hint)
        return node.type.gen_path(basedir, params)


class PipelineBuilderError(Exception):
    '''Exception raised when an invalid action is taken with a
    PipelineBuilder.
    '''
    pass


def _get_path_hint(module, output, path_hints):
    '''Gets a path hint for the given module output, if possible.

    Args:
        module: the module name
        output: the module output name
        path_hints: a dict mapping PipelineNode strings to paths
    '''
    node_str = etap.PipelineNode.get_node_str(module, output)
    return path_hints.get(node_str, None)


def _get_param_value(param, request):
    '''Gets the value for the parameter.

    Args:
        param: a PipelineParameter instance describing the parameter
        request: the PipelineBuildRequest instance

    Returns:
        val: the parameter value
    '''
    if param.param_str in request.parameters:
        # User-set parameter
        val = request.parameters[param.param_str]
    elif param.has_set_value:
        # Pipeline-set parameter
        val = param.set_value
    else:
        # Module-default value
        val = param.default_value

    return val
