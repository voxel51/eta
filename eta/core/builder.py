"""
Core pipeline building system.

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
import eta.core.logging as etal
import eta.core.module as etam
import eta.core.pipeline as etap
import eta.core.status as etas
import eta.core.types as etat
import eta.core.utils as etau


logger = logging.getLogger(__name__)


PIPELINE_CONFIG_FILE = "pipeline.json"
PIPELINE_LOGFILE_FILE = "pipeline.log"
PIPELINE_STATUS_FILE = "status.json"
MODULE_CONFIG_EXT = ".json"


def find_last_built_pipeline():
    """Finds the pipeline config file for the last built pipeline.

    The last built pipeline is the pipeline whose generated configuration
    directory has the most recent timestamp.

    Returns:
        the path to the pipline config file for the last built pipeline, or
            None if no config directories were found
    """
    builds = _list_all_built_pipelines()
    if not builds:
        return None

    config_dir = max(builds, key=os.path.basename)
    return os.path.join(config_dir, PIPELINE_CONFIG_FILE)


def find_all_built_pipelines():
    """Finds the pipeline config files for all built pipelines.

    Returns:
        a list of paths to pipline config files
    """
    builds = _list_all_built_pipelines()
    return [os.path.join(b, PIPELINE_CONFIG_FILE) for b in builds]


def _list_all_built_pipelines():
    config_patt = os.path.join(eta.config.config_dir, "*", "*")
    return etau.get_glob_matches(config_patt)


def cleanup_pipeline(pipeline_config_path):
    """Cleans up the given built pipeline.

    This function simply deletes the generated config and output directories
    for the pipeline, if necessary. Published outputs are not deleted.

    Args:
        pipeline_config_path: the path to the pipeline config file for a
            built pipeline

    Raises:
        OSError: if the pipeline was not a generated pipeline in the ETA config
            directory
    """
    eta_config_dir = os.path.realpath(eta.config.config_dir)
    eta_output_dir = os.path.realpath(eta.config.output_dir)

    config_dir = os.path.dirname(os.path.realpath(pipeline_config_path))
    output_dir = os.path.join(
        eta_output_dir, config_dir[(len(eta_config_dir) + 1) :]
    )

    if not config_dir.startswith(eta_config_dir):
        raise OSError(
            "Expected pipeline '%s' to be in the ETA config directory '%s'"
            % (pipeline_config_path, eta_config_dir)
        )

    logger.info("Cleaning up pipeline '%s'", pipeline_config_path)

    try:
        etau.delete_dir(config_dir)
        logger.info("*** Deleted config directory '%s'", config_dir)
    except OSError:
        pass

    try:
        etau.delete_dir(output_dir)
        logger.info("*** Deleted output directory '%s'", output_dir)
    except OSError:
        pass


def cleanup_last_built_pipeline():
    """Cleans up the last built pipeline, if any."""
    pipeline_config_path = find_last_built_pipeline()
    if pipeline_config_path:
        cleanup_pipeline(pipeline_config_path)


def cleanup_all_built_pipelines():
    """Cleans up all built pipelines."""
    for pipeline_config_path in find_all_built_pipelines():
        cleanup_pipeline(pipeline_config_path)


class ModuleConfig(etam.BaseModuleConfig):
    """Module configuration class.

    This generic class is used by `PipelineBuilder` to build module
    configuration files of any type.
    """

    def __init__(self, d):
        super(ModuleConfig, self).__init__(d)
        self.data = self.parse_array(d, "data", default=[])
        self.parameters = self.parse_dict(d, "parameters", default={})


class PipelineBuildRequestConfig(Config):
    """Pipeline build request configuration class."""

    def __init__(self, d):
        self.name = self.parse_string(d, "name", default=None)
        self.pipeline = self.parse_string(d, "pipeline")
        self.inputs = self.parse_dict(d, "inputs", default={})
        self.outputs = self.parse_dict(d, "outputs", default={})
        self.parameters = self.parse_dict(d, "parameters", default={})
        self.eta_config = self.parse_dict(d, "eta_config", default={})
        self.logging_config = self.parse_object(
            d, "logging_config", etal.LoggingConfig, default=None
        )


class PipelineBuildRequest(Configurable):
    """Pipeline build request class.

    A pipeline build request is valid if all of the following are true:
        - a pipeline of the specified name exists
        - all required pipeline inputs are provided and have valid values
        - any output paths specified must refer to valid pipeline output names
            and the specified paths must be either (a) valid paths for the
            associated types, or (b) None, in which case their paths will be
            automatically populated when the pipeline is built
        - all required pipeline parameters are provided and have valid values

    Note that any fields set to `None` are ignored, so any inputs/parameters
    that are `None` must be optional in order for the request to be valid.

    Attributes:
        name: a name for this pipeline run
        pipeline: the name of the pipeline to run
        metadata: the PipelineMetadata instance for the pipeline
        inputs: a dictionary mapping input names to input paths
        outputs: a dictionary mapping output names to output paths (if any)
        parameters: a dictionary mapping <module>.<parameter> names to
            parameter values
        eta_config: a dictionary of custom ETA config settings for the pipeline
            (if any)
        logging_config: the LoggingConfig for the pipeline (if any)
    """

    def __init__(self, config):
        """Creates a new PipelineBuildRequest instance.

        Args:
            config: a PipelineBuildRequestConfig instance.

        Raises:
            PipelineBuildRequestError: if the pipeline request was invalid
        """
        self.validate(config)

        self.name = config.name
        self.pipeline = config.pipeline
        self.metadata = etap.load_metadata(config.pipeline)
        self.inputs = etau.remove_none_values(config.inputs)
        self.outputs = config.outputs
        self.parameters = etau.remove_none_values(config.parameters)
        self.eta_config = config.eta_config
        self.logging_config = config.logging_config

        self._validate_inputs()
        self._validate_outputs()
        self._validate_parameters()

    def _validate_inputs(self):
        # Validate inputs
        for iname, ipath in iteritems(self.inputs):
            if not self.metadata.has_input(iname):
                raise PipelineBuildRequestError(
                    "Pipeline '%s' has no input '%s'" % (self.pipeline, iname)
                )
            if not self.metadata.is_valid_input(iname, ipath):
                raise PipelineBuildRequestError(
                    "'%s' is not a valid value for input '%s' of pipeline "
                    "'%s'" % (ipath, iname, self.pipeline)
                )
            # Convert to absolute paths
            self.inputs[iname] = os.path.abspath(ipath)

        # Ensure that required inputs were supplied
        for miname, miobj in iteritems(self.metadata.inputs):
            if miobj.is_required and miname not in self.inputs:
                raise PipelineBuildRequestError(
                    "Required input '%s' of pipeline '%s' was not "
                    "supplied" % (miname, self.pipeline)
                )

    def _validate_outputs(self):
        # Validate outputs
        for oname, opath in iteritems(self.outputs):
            if not self.metadata.has_output(oname):
                raise PipelineBuildRequestError(
                    "Pipeline '%s' has no output '%s'" % (self.pipeline, oname)
                )
            if not opath:
                continue
            if not self.metadata.is_valid_output(oname, opath):
                raise PipelineBuildRequestError(
                    "'%s' is not a valid value for output '%s' of pipeline "
                    "'%s'" % (opath, oname, self.pipeline)
                )
            # Convert to absolute paths
            self.outputs[oname] = os.path.abspath(opath)

    def _validate_parameters(self):
        # Validate parameters
        for pname, pval in iteritems(self.parameters):
            if not self.metadata.has_tunable_parameter(pname):
                raise PipelineBuildRequestError(
                    "Pipeline '%s' has no tunable parameter '%s'"
                    % (self.pipeline, pname)
                )
            if not self.metadata.is_valid_parameter(pname, pval):
                raise PipelineBuildRequestError(
                    "'%s' is not a valid value for parameter '%s' of pipeline "
                    "'%s'" % (pval, pname, self.pipeline)
                )
            # Convert any data parameters to absolute paths
            if self.metadata.parameters[pname].is_data:
                self.parameters[pname] = os.path.abspath(pval)

        # Ensure that required parmeters were supplied
        for mpname, mpobj in iteritems(self.metadata.parameters):
            if mpobj.is_required and mpname not in self.parameters:
                raise PipelineBuildRequestError(
                    "Required parameter '%s' of pipeline '%s' was not "
                    "specified" % (mpname, self.pipeline)
                )


class PipelineBuildRequestError(Exception):
    """Exception raised when an invalid PipelineBuildRequest is encountered."""

    pass


class PipelineBuilder(object):
    """Class for building a pipeline based on a PipelineBuildRequest.

    Attributes:
        request: the PipelineBuildRequest instance used to build the pipeline
        optimized: whether the pipeline was optimized during building
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
        execution_order: a list of modules defining the order the pipeline
            modules will be executed
        module_inputs: a dictionary mapping module names to dictionaries
            of module input names and their associated paths
        module_outputs: a dictionary mapping module names to dictionaries
            of module output names and their associated paths in `output_dir`
        module_parameters: a dictionary mapping modules names to dictionaries
            containing the module parameter names and their associated values
        pipeline_outputs: a dictionary mapping pipeline outputs to their
            associated paths. If the pipeline is optimized, this dictionary
            will contain the same outputs from the PipelineBuildRequest. If
            the pipeline is not optimized, it will also contain paths in
            `output_dir` for any pipeline outputs that were not included in the
            outputs dictionary
    """

    def __init__(self, request):
        """Creates a PipelineBuilder instance.

        Args:
            request: a PipelineBuildRequest instance
        """
        self.request = request

        self.optimized = None
        self.timestamp = None
        self.config_dir = None
        self.output_dir = None
        self.pipeline_config_path = None
        self.pipeline_status_path = None
        self.pipeline_logfile_path = None
        self.execution_order = None
        self.module_inputs = None
        self.module_outputs = None
        self.module_parameters = None
        self.pipeline_outputs = None

        self._concrete_data_params = etat.ConcreteDataParams()

        self.reset()

    def reset(self):
        """Resets the builder so that another pipeline can be built."""
        self.optimized = None
        self.timestamp = None
        self.config_dir = None
        self.output_dir = None
        self.pipeline_config_path = None
        self.pipeline_status_path = None
        self.pipeline_logfile_path = None
        self.execution_order = []
        self.module_inputs = defaultdict(dict)
        self.module_outputs = defaultdict(dict)
        self.module_parameters = defaultdict(dict)
        self.pipeline_outputs = {}

    def build(self, optimized=True):
        """Builds the pipeline and writes the associated config files.

        Args:
            optimized: whether to optimize the pipeline by omitting any modules
                that are not necessary to generate the requested outputs. By
                default, this is True
        """
        self.reset()
        self.optimized = optimized

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

        self._populate_pipeline_connections()
        if self.optimized:
            self._optimize_pipeline()
        self._build_pipeline_config()
        self._build_module_configs()

    def run(self):
        """Runs the built pipeline.

        Returns:
            True/False whether the pipeline completed successfully

        Raises:
            PipelineBuilderError: if the pipeline hasn't been built
        """
        if not self.pipeline_config_path:
            raise PipelineBuilderError(
                "No pipeline config found; you must build the pipeline before "
                "running it"
            )

        return etap.run(self.pipeline_config_path)

    def get_status(self):
        """Gets the PipelineStatus for the last run pipeline.

        Returns:
            a PipelineStatus instance

        Raises:
            PipelineBuilderError: if the pipeline hasn't been built and run
        """
        if not os.path.exists(self.pipeline_status_path):
            raise PipelineBuilderError(
                "No pipeline status found; you must build and run the "
                "pipeline before getting its status"
            )

        return etas.PipelineStatus.from_json(self.pipeline_status_path)

    def cleanup(self):
        """Cleans up the configs and output files generated when the pipeline
        was run, if necessary. Published outputs are NOT deleted.

        Raises:
            PipelineBuilderError: if the pipeline hasn't been built
        """
        if not self.config_dir or not self.output_dir:
            raise PipelineBuilderError(
                "You must build and run the pipeline before you can clean "
                "it up"
            )

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

    def _populate_pipeline_connections(self):
        # Get PipelineMetadata
        pmeta = self.request.metadata

        # Distribute pipeline inputs
        for piname, pipath in iteritems(self.request.inputs):
            for sink in pmeta.get_input_sinks(piname):
                self.module_inputs[sink.module][sink.node] = pipath

        # Distribute published outputs
        for poname, popath in iteritems(self.request.outputs):
            if popath:
                source = pmeta.get_output_source(poname)
                self.module_outputs[source.module][source.node] = popath

        # Propagate module connections
        for module in pmeta.execution_order:
            mmeta = pmeta.modules[module].metadata  # ModuleMetadata
            oconns = pmeta.get_outgoing_connections(module)
            for oname, osinks in iteritems(oconns):
                if oname in self.module_outputs[module]:
                    # This is either a published output or a node with multiple
                    # outgoing connections; in either case, we already have
                    # its path
                    opath = self.module_outputs[module][oname]
                else:
                    # Set output path
                    onode = mmeta.outputs[oname]
                    opath = self._get_data_path(module, onode)
                    self.module_outputs[module][oname] = opath

                for osink in osinks:
                    if osink.is_pipeline_output:
                        # Set pipeline output
                        self.pipeline_outputs[osink.node] = opath
                    else:
                        # Pass output to connected module input
                        self.module_inputs[osink.module][osink.node] = opath

        # Populate module parameters
        for param in itervalues(pmeta.parameters):
            val = _get_param_value(param, self.request)
            # We specifically omit any parameters that are set to `None` so
            # that the module's default parameter will be used
            if val is not None:
                self.module_parameters[param.module][param.name] = val

        # Set execution order
        self.execution_order = pmeta.execution_order

    def _optimize_pipeline(self):
        logger.info("Optimizing pipeline")

        # Get PipelineMetadata
        pmeta = self.request.metadata

        # Compute active nodes
        active_modules = set()
        active_inputs = defaultdict(set)
        active_outputs = defaultdict(set)
        queue = [
            pmeta.get_output_source(oname) for oname in self.request.outputs
        ]
        while queue:
            node = queue.pop(0)
            active_outputs[node.module].add(node.node)

            if node.is_module_output and node.module not in active_modules:
                active_modules.add(node.module)
                for conn in pmeta.get_incoming_connections(node.module):
                    queue.append(conn.source)
                    active_inputs[conn.sink.module].add(conn.sink.node)

        # Prune inactive modules
        for module in self.execution_order:
            if module not in active_modules:
                logger.debug("*** Pruning inactive module '%s'", module)
                self.module_inputs.pop(module, None)
                self.module_outputs.pop(module, None)
                self.module_parameters.pop(module, None)

        # Delete inactive outputs
        for module in list(self.module_outputs.keys()):
            mmeta = pmeta.modules[module].metadata
            for oname in list(self.module_outputs[module].keys()):
                if oname not in active_outputs[module]:
                    if not mmeta.get_output(oname).is_required:
                        logger.debug(
                            "*** Pruning unnecessary output '%s' from module "
                            "'%s'",
                            oname,
                            module,
                        )
                        del self.module_outputs[module][oname]

        # Update execution order
        self.execution_order = [
            module
            for module in self.execution_order
            if module in active_modules
        ]

    def _build_pipeline_config(self):
        # Build job configs
        # @todo handle non-py executables
        jobs = []
        for module in self.execution_order:
            metadata = self.request.metadata.modules[module].metadata
            jobs.append(
                etaj.JobConfig.builder()
                .set(name=module)
                .set(script=etam.find_exe(module_metadata=metadata))
                .set(config_path=self._get_module_config_path(module))
                .validate()
            )
        if not jobs:
            logger.warning("Pipeline contains no jobs...")

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
            logging_config = (
                etal.LoggingConfig.builder()
                .set(filename=self.pipeline_logfile_path)
                .validate()
            )

        # Build pipeline config
        name = self.request.name or self.request.pipeline
        pipeline_config = (
            etap.PipelineConfig.builder()
            .set(name=name)
            .set(status_path=self.pipeline_status_path)
            .set(overwrite=False)
            .set(jobs=jobs)
            .set(eta_config=self.request.eta_config)
            .set(logging_config=logging_config)
            .validate()
        )

        # Write pipeline config
        logger.info("Writing pipeline config '%s'", self.pipeline_config_path)
        pipeline_config.write_json(self.pipeline_config_path)

    def _build_module_configs(self):
        for module in self.execution_order:
            # Build module config
            data = etau.join_dicts(
                self.module_inputs[module], self.module_outputs[module]
            )
            module_config = (
                ModuleConfig.builder()
                .set(data=[data])
                .set(parameters=self.module_parameters[module])
                .validate()
            )

            # Write module config
            module_config_path = self._get_module_config_path(module)
            logger.info("Writing module config '%s'", module_config_path)
            module_config.write_json(module_config_path)

    def _get_timestamp_str(self):
        return time.strftime("%Y.%m.%d-%H.%M.%S", self.timestamp)

    def _get_config_dir(self):
        time_str = self._get_timestamp_str()
        return os.path.join(
            eta.config.config_dir, self.request.pipeline, time_str
        )

    def _get_output_dir(self):
        time_str = self._get_timestamp_str()
        return os.path.join(
            eta.config.output_dir, self.request.pipeline, time_str
        )

    def _make_pipeline_config_path(self):
        return os.path.join(self.config_dir, PIPELINE_CONFIG_FILE)

    def _make_pipeline_logfile_path(self):
        return os.path.join(self.config_dir, PIPELINE_LOGFILE_FILE)

    def _make_pipeline_status_path(self):
        return os.path.join(self.output_dir, PIPELINE_STATUS_FILE)

    def _get_module_config_path(self, module):
        return os.path.join(self.config_dir, module + MODULE_CONFIG_EXT)

    def _get_data_path(self, module, node):
        basedir = os.path.join(self.output_dir, module)
        params = self._concrete_data_params.render_for(node.name)
        return node.type.gen_path(basedir, params)


class PipelineBuilderError(Exception):
    """Exception raised when an invalid action is taken with a PipelineBuilder.
    """

    pass


def _get_param_value(param, request):
    """Gets the value for the parameter.

    Args:
        param: a PipelineParameter instance describing the parameter
        request: the PipelineBuildRequest instance

    Returns:
        val: the parameter value
    """
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
