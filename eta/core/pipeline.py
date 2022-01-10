"""
Core pipeline infrastructure.

See `docs/pipelines_dev_guide.md` for detailed information about the design of
the ETA pipeline system.

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
from future.utils import iteritems

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict
from glob import glob
import logging
import os
import sys

import eta
from eta.core.config import Config, Configurable
from eta.core.diagram import HasBlockDiagram, BlockdiagPipeline
import eta.core.graph as etag
import eta.core.job as etaj
import eta.core.logging as etal
import eta.core.module as etam
import eta.core.status as etas
import eta.core.types as etat
import eta.core.utils as etau
import eta.core.video as etav


logger = logging.getLogger(__name__)


PIPELINE_INPUT_NAME = "INPUT"
PIPELINE_OUTPUT_NAME = "OUTPUT"


def run(
    pipeline_config_or_path,
    pipeline_status=None,
    mark_as_complete=True,
    handle_failures=True,
    rotate_logs=True,
    force_overwrite=False,
):
    """Runs the pipeline specified by the given PipelineConfig.

    Args:
        pipeline_config_or_path: a PipelineConfig, a dict representation of
            one, or a path to one on disk. If a config is provided in-memory,
            it is writeen to a temporary directory on disk while the pipeline
            executes
        pipeline_status: an optional PipelineStatus instance. By default, a
            PipelineStatus object is created that logs its status to the path
            specified in the provided PipelineConfig
        mark_as_complete: whether to mark the PipelineStatus as complete when
            the pipeline finishes. By default, this is True
        handle_failures: whether to gracefully handle job failures and return
            a success flag (True) or raise a PipelineExecutionError if a job
            fails (False). The default is True
        rotate_logs: whether to rotate any existing pipeline log(s) before
            running. By default, this is True
        force_overwrite: whether to force the pipeline to overwrite any
            existing outputs by setting the `overwrite` flag of the
            PipelineConfig. By default, the PipelineConfig's `overwrite` flag
            is used

    Returns:
        True/False whether the pipeline completed successfully

    Raises:
        PipelineExecutionError: if the pipeline failed and `handle_failures` is
            False
    """
    # Parse PipelineConfig
    if etau.is_str(pipeline_config_or_path):
        # Found a path to a pipeline config on disk
        temp_dir = None
        pipeline_config_path = pipeline_config_or_path
        pipeline_config = PipelineConfig.from_json(pipeline_config_path)
    else:
        # Found an in-memory pipeline config
        temp_dir = etau.TempDir()
        pipeline_config_path = os.path.join(
            temp_dir.__enter__(), "pipeline.json"
        )
        if isinstance(pipeline_config_or_path, dict):
            pipeline_config = PipelineConfig.from_dict(pipeline_config_or_path)
        else:
            pipeline_config = pipeline_config_or_path

        pipeline_config.write_json(pipeline_config_path)

    # Force overwrite, if requested
    if force_overwrite:
        pipeline_config.overwrite = True

    # Setup logging
    etal.custom_setup(pipeline_config.logging_config, rotate=rotate_logs)

    # Apply config settings
    eta.set_config_settings(**pipeline_config.eta_config)

    # Create a PipelineStatus instance, if necessary
    if not pipeline_status:
        pipeline_status = _make_pipeline_status(pipeline_config)

    # Convert to absolute path so jobs can find the pipeline config later
    # regardless of their working directory
    pipeline_config_path = os.path.abspath(pipeline_config_path)

    # Run pipeline
    status = _run(
        pipeline_config,
        pipeline_config_path,
        pipeline_status,
        mark_as_complete,
        handle_failures,
    )

    # Cleanup, if necessary
    if temp_dir is not None:
        temp_dir.__exit__()

    return status


def _make_pipeline_status(pipeline_config):
    pipeline_status = etas.PipelineStatus(name=pipeline_config.name)

    if pipeline_config.status_path:
        pcb = _make_publish_status_callback(pipeline_config.status_path)
        pipeline_status.set_publish_callback(pcb)

    return pipeline_status


def _make_publish_status_callback(status_path):
    def _publish_status(pipeline_status):
        pipeline_status.write_json(status_path)
        logger.info("Pipeline status written to '%s'", status_path)

    return _publish_status


def _run(
    pipeline_config,
    pipeline_config_path,
    pipeline_status,
    mark_as_complete,
    handle_failures,
):
    # Starting pipeline
    logger.info("Pipeline %s started", pipeline_config.name)
    pipeline_status.start()
    pipeline_status.publish()

    # Run jobs
    overwrite = pipeline_config.overwrite
    ran_last_job = False
    with etau.WorkingDir(pipeline_config.working_dir):
        for job_config in pipeline_config.jobs:
            if ran_last_job and not overwrite:
                overwrite = True

            # Run job
            job_config.pipeline_config_path = pipeline_config_path
            ran_last_job, success = etaj.run(
                job_config, pipeline_status, overwrite=overwrite
            )

            if not success:
                if not handle_failures:
                    raise PipelineExecutionError(
                        "Pipeline %s failed; aborting" % pipeline_config.name
                    )

                # Pipeline failed
                logger.info("Pipeline %s failed", pipeline_config.name)
                pipeline_status.fail()
                pipeline_status.publish()
                return False

    if mark_as_complete:
        # Pipeline complete
        logger.info("Pipeline %s complete", pipeline_config.name)
        pipeline_status.complete()

    pipeline_status.publish()
    return True


class PipelineExecutionError(Exception):
    """Exception raised when a pipeline fails."""

    pass


def load_all_metadata():
    """Loads all pipeline metadata files.

    Assumes any JSON files in the `eta.config.pipeline_dirs` directories are
    pipeline metadata files.

    Returns:
        a dictionary mapping pipeline names to PipelineMetadata instances

    Raises:
        PipelineMetadataError: if any of the pipeline metadata files are
            invalid
    """
    return {k: _load_metadata(v) for k, v in iteritems(find_all_metadata())}


def load_metadata(pipeline_name):
    """Loads the pipeline metadata file for the pipeline with the given name.

    Pipeline metadata files must be JSON files in one of the directories in
    `eta.config.pipeline_dirs`.

    Args:
        pipeline_name: the name of the pipeline

    Returns:
        the PipelineMetadata instance for the given pipeline

    Raises:
        PipelineMetadataError: if the pipeline metadata file could not be found
            or was invalid
    """
    return _load_metadata(find_metadata(pipeline_name))


def _load_metadata(config):
    metadata = PipelineMetadata.from_json(config)
    name = os.path.splitext(os.path.basename(config))[0]
    if metadata.info.name != name:
        raise PipelineMetadataError(
            "Name '%s' from PipelineMetadata must match pipeline name '%s'"
            % (metadata.info.name, name)
        )

    return metadata


def find_all_metadata():
    """Finds all pipeline metadata files.

    Assumes any JSON files in the `eta.config.pipeline_dirs` directories are
    pipeline metadata files. To load these files, use `load_all_metadata()`.

    Returns:
        a dictionary mapping pipeline names to absolute paths to
            PipelineMetadata files
    """
    d = {}
    pdirs = etau.make_search_path(eta.config.pipeline_dirs)
    for pdir in pdirs:
        for path in glob(os.path.join(pdir, "*.json")):
            name = os.path.splitext(os.path.basename(path))[0]
            if name not in d:
                d[name] = path
            else:
                logger.debug(
                    "Pipeline '%s' already exists; ignoring %s", name, path
                )

    return d


def find_metadata(pipeline_name):
    """Finds the pipeline metadata file for the pipeline with the given name.

    Pipeline metadata files must be JSON files in one of the directories in
    `eta.config.pipeline_dirs`.

    Args:
        pipeline_name: the name of the pipeline

    Returns:
        the absolute path to the PipelineMetadata file

    Raises:
        PipelineMetadataError: if the pipeline could not be found
    """
    try:
        return find_all_metadata()[pipeline_name]
    except KeyError:
        raise PipelineMetadataError(
            "Could not find pipeline '%s'" % pipeline_name
        )


def is_video_pipeline(metadata):
    """Determines whether this pipeline is a video pipeline, i.e., if it has
    exactly one input and that input is a video type.

    Args:
        metadata: a PipelineMetadata instance

    Returns:
        True/False
    """
    if metadata.num_inputs == 1:
        input_name = list(metadata.inputs.keys())[0]
        input_type = metadata.get_input_type(input_name)
        if issubclass(input_type, etat.Video):
            return True

    return False


def get_metadata_for_video_input(request):
    """Returns VideoMetadata for the input of the given pipeline request, which
    must be a video pipeline.

    Args:
        request: a PipelineBuildRequest

    Returns:
        a VideoMetadata instance, or None
    """
    if is_video_pipeline(request.metadata):
        input_name = list(request.metadata.inputs.keys())[0]
        video_path = request.inputs[input_name]
        try:
            return etav.VideoMetadata.build_for(video_path)
        except:
            pass

    return None


class PipelineConfig(Config):
    """Pipeline configuration class."""

    def __init__(self, d):
        self.name = self.parse_string(d, "name", default="pipeline")
        self.working_dir = self.parse_string(d, "working_dir", default=None)
        self.status_path = self.parse_string(d, "status_path", default=None)
        self.overwrite = self.parse_bool(d, "overwrite", default=True)
        self.jobs = self.parse_object_array(
            d, "jobs", etaj.JobConfig, default=[]
        )
        self.eta_config = self.parse_dict(d, "eta_config", default={})
        self.logging_config = self.parse_object(
            d,
            "logging_config",
            etal.LoggingConfig,
            default=etal.LoggingConfig.default(),
        )


class PipelineMetadataConfig(Config):
    """Pipeline metadata configuration class."""

    def __init__(self, d):
        self.info = self.parse_object(d, "info", PipelineInfoConfig)
        self.inputs = self.parse_array(d, "inputs")
        self.outputs = self.parse_array(d, "outputs")
        self.modules = self.parse_object_dict(
            d, "modules", PipelineModuleConfig
        )
        self.connections = self.parse_object_array(
            d, "connections", PipelineConnectionConfig
        )


class PipelineInfoConfig(Config):
    """Pipeline info configuration class."""

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.version = self.parse_string(d, "version")
        self.description = self.parse_string(d, "description")


class PipelineModuleConfig(Config):
    """Pipeline module configuration class."""

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.tunable_parameters = self.parse_array(d, "tunable_parameters")
        self.set_parameters = self.parse_dict(d, "set_parameters")


class PipelineConnectionConfig(Config):
    """Pipeline connection configuration class."""

    def __init__(self, d):
        self.source = self.parse_string(d, "source")
        self.sink = self.parse_string(d, "sink")


class PipelineNodeType(object):
    """Class enumerating the types of pipeline nodes."""

    PIPELINE_INPUT = 1
    PIPELINE_OUTPUT = 2
    MODULE_INPUT = 3
    MODULE_OUTPUT = 4


class PipelineInfo(Configurable):
    """Pipeline info descriptor.

    Attributes:
        name: the name of the pipeline
        type: the eta.core.types.Type of the pipeline
        version: the pipeline version
        description: a free text description of the pipeline
    """

    def __init__(self, config):
        self.validate(config)

        self.name = config.name
        self.type = self._parse_type(config.type)
        self.version = config.version
        self.description = config.description

    @staticmethod
    def _parse_type(type_str):
        type_ = etat.parse_type(type_str)
        if not etat.is_pipeline(type_):
            raise PipelineMetadataError(
                "'%s' is not a valid pipeline type" % type_
            )
        return type_


class PipelineInput(object):
    """Class describing a pipeline input.

    Attributes:
        name: the input name
        inputs: a list ModuleInput instance(s) of the module input(s) that the
            pipeline input is connected to
    """

    def __init__(self, name, inputs):
        self.name = name
        self.inputs = inputs

    @property
    def is_required(self):
        """Returns True/False if this input is required."""
        return any(inp.is_required for inp in self.inputs)

    def is_valid_path(self, path):
        """Returns True/False if `path` is a valid path for this input."""
        return all(inp.is_valid_path(path) for inp in self.inputs)


class PipelineOutput(object):
    """Class describing a pipeline output.

    Attributes:
        name: the output name
        output: the ModuleOutput instance of the module output that the
            pipeline output is connected to
    """

    def __init__(self, name, output):
        self.name = name
        self.output = output

    def is_valid_path(self, path):
        """Returns True/False if `path` is a valid path for this output."""
        return self.output.is_valid_path(path)


class PipelineParameter(object):
    """Class describing a pipeline parameter.

    A pipeline parameter is *active* if it is tunable and/or its value is set
    by the pipeline.

    A pipeline parameter is required (must be set by the end-user) if it is
    required by the module but has no value set by the pipeline.

    Attributes:
        module: the module associated with this parameter
        name: the name of this parameter
        param: the ModuleParameter instance of the module parameter
        is_tunable: whether the parameter is tunable
        set_value: the value set by the pipeline, or None if it is not set
    """

    def __init__(self, module, name, param, is_tunable, set_value=None):
        """Creates a PipelineParameter instance.

        Args:
            module: the module associated with this parameter
            name: the name of this parameter
            param: the ModuleParameter instance of the module parameter
            is_tunable: whether the parameter is tunable
            set_value: an optional pipeline-set value for the parameter

        Raise:
            PipelineMetadataError: is the pipeline parameter is invalid.
        """
        self.module = module
        self.name = name
        self.param = param
        self.is_tunable = is_tunable
        self.set_value = set_value
        self._validate()

    @property
    def is_builtin(self):
        """Returns True/False if this parameter is a Builtin."""
        return self.param.is_builtin

    @property
    def is_data(self):
        """Returns True/False if this parameter is Data."""
        return self.param.is_data

    @property
    def is_required(self):
        """Returns True/False if this parameter must be set by the user."""
        return self.param.is_required and not self.has_set_value

    @property
    def has_set_value(self):
        """Returns True/False if this parameter has a value set by the
        pipeline.
        """
        return self.set_value is not None

    @property
    def default_value(self):
        """Gets the default value for this parameter."""
        if self.is_required:
            raise PipelineMetadataError(
                "Pipeline parameter '%s' is required, so it has no default "
                "value" % self.param_str
            )

        if self.has_set_value:
            return self.set_value

        return self.param.default_value

    @property
    def param_str(self):
        """Returns the pipeline node string describing this parameter."""
        return PipelineNode.get_node_str(self.module, self.name)

    def is_valid_value(self, val):
        """Returns True/False if `val` is a valid value for this parameter."""
        return self.param.is_valid_value(val)

    def _validate(self):
        if not self.is_tunable and not self.has_set_value:
            raise PipelineMetadataError(
                "Pipeline parameter '%s' must be tunable or its value must "
                "be set by this pipeline" % self.param_str
            )


class PipelineModule(Configurable):
    """Class describing a module in the context of a pipeline.

    A pipeline module definition is valid if every *required* module parameter
    is "active", i.e., satisfies at least one of the following conditions:
        - the parameter is exposed to the end-user as tunable
        - the parameter is set by the pipeline

    Attributes:
        name: the name of the module in the pipeline
        metadata: the ModuleMetadata instance for the module
        parameters: a dictionary mapping <module>.<parameter> strings to
            PipelineParameter instances describing the active parameters
    """

    def __init__(self, name, config):
        """Creates a PipelineModule instance.

        Args:
            name: a name for the module
            config: a PipelineModuleConfig instance

        Raises:
            PipelineMetadataError: if the pipeline module configuration was
                invalid
        """
        self.validate(config)

        self.name = name
        self.metadata = etam.load_metadata(config.name)
        self.parameters = {}

        self._parse_parameters(
            config.tunable_parameters, config.set_parameters
        )

    def _parse_parameters(self, tunable_parameters, set_parameters):
        # Verify parameter settings
        self._verify_has_parameters(tunable_parameters)
        self._verify_has_parameters(set_parameters.keys())
        self._verify_parameter_values(set_parameters)

        for pname, param in iteritems(self.metadata.parameters):
            # Verify that required parameters are active
            is_tunable = pname in tunable_parameters
            is_set = pname in set_parameters
            is_active = is_tunable or is_set
            if param.is_required and not is_active:
                raise PipelineMetadataError(
                    "Required parameter '%s' of module '%s' must be set or "
                    "exposed as tunable" % (pname, self.metadata.info.name)
                )

            # Record active parameter
            if is_active:
                set_value = set_parameters[pname] if is_set else None
                pparam = PipelineParameter(
                    self.name, pname, param, is_tunable, set_value=set_value
                )
                self.parameters[pparam.param_str] = pparam

    def _verify_has_parameters(self, param_names):
        for name in param_names:
            if not self.metadata.has_parameter(name):
                raise PipelineMetadataError(
                    "Module '%s' has no parameter '%s'"
                    % (self.metadata.info.name, name)
                )

    def _verify_parameter_values(self, param_dict):
        for name, val in iteritems(param_dict):
            if not self.metadata.is_valid_parameter(name, val):
                raise PipelineMetadataError(
                    "'%s' is an invalid value for parameter '%s' of module "
                    "'%s'" % (val, name, self.metadata.info.name)
                )


class PipelineNode(object):
    """Class representing a node in a pipeline.

    Attributes:
        module: the module name, or INPUT or OUTPUT for pipeline endpoints
        node: the node name
    """

    def __init__(self, module, node, _type):
        self.module = module
        self.node = node
        self._type = _type

    def __str__(self):
        return "%s.%s" % (self.module, self.node)

    def is_same_node(self, node):
        """Returns True/False if the given node is equal to this node, i.e.,
        if they refer to the same node of the same module.
        """
        return str(self) == str(node)

    def is_same_node_str(self, node_str):
        """Returns True/False if the given node string refers to this node."""
        return str(self) == node_str

    @property
    def is_pipeline_node(self):
        """Returns True/False if this node is a pipeline input or output."""
        return self.is_pipeline_input or self.is_pipeline_output

    @property
    def is_pipeline_input(self):
        """Returns True/False if this node is a pipeline input."""
        return self._type == PipelineNodeType.PIPELINE_INPUT

    @property
    def is_pipeline_output(self):
        """Returns True/False if this node is a pipeline output."""
        return self._type == PipelineNodeType.PIPELINE_OUTPUT

    @property
    def is_module_node(self):
        """Returns True/False if this node is a module input or output."""
        return self.is_module_input or self.is_module_output

    @property
    def is_module_input(self):
        """Returns True/False if this node is a module input."""
        return self._type == PipelineNodeType.MODULE_INPUT

    @property
    def is_module_output(self):
        """Returns True/False if this node is a module output."""
        return self._type == PipelineNodeType.MODULE_OUTPUT

    @staticmethod
    def get_input_str(name):
        """Gets the node string for a pipeline input with the given name."""
        return "%s.%s" % (PIPELINE_INPUT_NAME, name)

    @staticmethod
    def get_output_str(name):
        """Gets the node string for a pipeline output with the given name."""
        return "%s.%s" % (PIPELINE_OUTPUT_NAME, name)

    @staticmethod
    def get_node_str(module, node):
        """Gets the node string for a pipeline node with the given module and
        node names.
        """
        return "%s.%s" % (module, node)


class PipelineConnection(object):
    """Class representing a connection between two nodes in a pipeline.

    Attributes:
        source: the source PipelineNode
        sink: the sink PipelineNode
    """

    def __init__(self, source, sink):
        self.source = source
        self.sink = sink

    def __str__(self):
        return "%s -> %s" % (self.source, self.sink)


class PipelineMetadata(Configurable, HasBlockDiagram):
    """Class the encapsulates the architecture of a pipeline.

    A pipeline definition is valid if all of the following conditions are met:
        - every pipeline input is connected to at least one module input
        - every pipeline output is connected to exactly one module output
        - every module input either has exactly one incoming connection or is
            not required
        - every module output either has at least one outgoing connection or
            is not required
        - every *required* module parameter is either exposed to the end-user
            as tunable or is set by the pipeline
        - the module graph defined by the pipeline is acyclic

    Attributes:
        info: a PipelineInfo instance describing the pipeline
        inputs: a dictionary mapping input names to PipelineInput instances
        outputs: a dictionary mapping output names to PipelineOutput instances
        parameters: a dictionary mapping <module>.<parameter> strings to
            PipelineParameter instances describing the *active* module
            parameters of the pipeline
        modules: a dictionary mapping module names to PipelineModule instances
        execution_order: a list of module names defining the order in which the
            modules should be executed
        nodes: a list of PipelineNode instances describing the sources and
            sinks for all pipeline connections
        connections: a list of PipelineConnection instances describing the
            connections between pipeline inputs, modules, and pipeline outputs
    """

    def __init__(self, config):
        """Initializes a PipelineMetadata instance.

        Args:
            config: a PipelineMetadataConfig instance

        Raises:
            PipelineMetadataError: if the pipeline definition was invalid
        """
        self.validate(config)
        self.config = config

        self.info = None
        self.inputs = {}
        self.outputs = {}
        self.parameters = {}
        self.modules = {}
        self.execution_order = []
        self.nodes = []
        self.connections = []

        self._concrete_data_params = etat.ConcreteDataParams()
        self._parse_metadata(config)

    @property
    def num_inputs(self):
        """Returns the number of inputs for the pipeline."""
        return len(self.inputs)

    @property
    def num_outputs(self):
        """Returns the number of outputs for the pipeline."""
        return len(self.outputs)

    @property
    def num_parameters(self):
        """Returns the number of active parameters for the pipeline."""
        return len(self.parameters)

    @property
    def num_modules(self):
        """Returns the number of modules in the pipeline."""
        return len(self.modules)

    def has_input(self, name):
        """Returns True/False if the pipeline has an input `name`."""
        return name in self.inputs

    def has_output(self, name):
        """Returns True/False if the pipeline has an output `name`."""
        return name in self.outputs

    def has_module(self, name):
        """Returns True/False if the pipeline has a module `name`."""
        return name in self.modules

    def has_tunable_parameter(self, param_str):
        """Returns True/False if this pipeline has tunable parameter
        `param_str`.
        """
        return (
            param_str in self.parameters
            and self.parameters[param_str].is_tunable
        )

    def get_input_type(self, name):
        """Returns the `eta.core.types.Type` of input `name`."""
        #
        # @todo propertly support pipeline input types, rather than simply
        # returning the type of the first connected module input
        #
        return self.inputs[name].inputs[0].type

    def get_output_type(self, name):
        """Returns the `eta.core.types.Type` of output `name`."""
        return self.outputs[name].output.type

    def recommend_output_filename(self, name):
        """Returns a recommended filename for output `name`."""
        output_type = self.get_output_type(name)
        params = self._concrete_data_params.render_for(name)
        return os.path.basename(output_type.gen_path("", params))

    def is_valid_input(self, name, path):
        """Returns True/False if `path` is a valid path for input `name`."""
        return self.inputs[name].is_valid_path(path)

    def is_valid_output(self, name, path):
        """Returns True/False if `path` is a valid path for output `name`."""
        return self.outputs[name].is_valid_path(path)

    def is_valid_parameter(self, param_str, val):
        """Returns True/False if `val` is a valid value for tunable parameter
        `param_str`.
        """
        return self.parameters[param_str].is_valid_value(val)

    def get_input_sinks(self, name):
        """Gets the sinks for the given input.

        Args:
            name: the pipeline input name

        Returns:
            a list of PipelineNode instances that the input is connected to
        """
        node_str = PipelineNode.get_input_str(name)
        return _get_sinks_with_source(node_str, self.connections)

    def get_output_source(self, name):
        """Gets the source for the given output.

        Args:
            name: the pipeline output name

        Returns:
            the PipelineNode instance that the output is connected to
        """
        node_str = PipelineNode.get_output_str(name)
        return _get_sources_with_sink(node_str, self.connections)[0]

    def get_incoming_connections(self, module):
        """Gets the incoming connections for the given module.

        Args:
            module: the module name

        Returns:
            a list of PipelineConnections describing the incoming connections
                for the given module
        """
        iconns = []
        for c in self.connections:
            if c.sink.module == module:
                iconns.append(c)
        return iconns

    def get_outgoing_connections(self, module):
        """Gets the outgoing connections for the given module.

        Args:
            module: the module name

        Returns:
            a dictionary mapping the names of the outputs of the given module
                to lists of PipelineNode instances describing the nodes that
                they are connected to
        """
        oconns = defaultdict(list)
        for c in self.connections:
            if c.source.module == module:
                oconns[c.source.node].append(c.sink)
        return dict(oconns)

    def to_blockdiag(self):
        """Returns a BlockdiagPipeline representation of this pipeline."""
        bp = BlockdiagPipeline(self.info.name)
        for name in self.execution_order:
            bp.add_module(name, self.modules[name].metadata.to_blockdiag())
        for n in self.nodes:
            if n.is_pipeline_input:
                bp.add_input(n.node)
            if n.is_pipeline_output:
                bp.add_output(n.node)
        for c in self.connections:
            bp.add_connection(c.source, c.sink)
        return bp

    def _register_node(self, node):
        for _node in self.nodes:
            if _node.is_same_node(node):
                return _node
        self.nodes.append(node)
        return node

    def _parse_metadata(self, config):
        # Parse info
        self.info = PipelineInfo(config.info)

        # Parse modules
        for name, module_config in iteritems(config.modules):
            module = PipelineModule(name, module_config)
            self.modules[name] = module
            self.parameters.update(module.parameters)

        # Parse connections
        for c in config.connections:
            # Parse nodes
            source = _parse_pipeline_node_str(
                c.source, config.inputs, config.outputs, self.modules
            )
            sink = _parse_pipeline_node_str(
                c.sink, config.inputs, config.outputs, self.modules
            )

            # Make sure we don't duplicate nodes
            source = self._register_node(source)
            sink = self._register_node(sink)

            # Record connection
            connection = _create_node_connection(source, sink, self.modules)
            self.connections.append(connection)

        # Parse inputs
        for name in config.inputs:
            self.inputs[name] = _parse_input(
                name, self.connections, self.modules
            )

        # Parse outputs
        for name in config.outputs:
            self.outputs[name] = _parse_output(
                name, self.connections, self.modules
            )

        # Validate connections
        _validate_module_connections(self.modules, self.connections)

        # Compute execution order
        self.execution_order = _compute_execution_order(self.connections)


class PipelineMetadataError(Exception):
    """Exception raised when an invalid pipeline metadata file is
    encountered.
    """

    pass


def _parse_input(name, connections, modules):
    """Parses the pipeline input with the given name.

    A pipeline input is properly configured if it is connected to at least one
    module input.

    Args:
        name: the pipeline input name
        connections: a list of PipelineConnection instances
        modules: a dictionary mapping module names to ModuleMetadata instances

    Returns:
        a PipelineInput instance describing the pipeline input

    Raises:
        PipelineMetadataError: if the input was not properly connected
    """
    node_str = PipelineNode.get_input_str(name)
    sinks = _get_sinks_with_source(node_str, connections)
    if not sinks:
        raise PipelineMetadataError(
            "Pipeline input '%s' is not connected to any modules" % name
        )

    nodes = [
        modules[sink.module].metadata.get_input(sink.node) for sink in sinks
    ]
    return PipelineInput(name, nodes)


def _parse_output(name, connections, modules):
    """Parses the pipeline output with the given name.

    A pipeline output is properly configured if it is connected to exactly one
    module output.

    Args:
        name: the pipeline output name
        connections: a list of PipelineConnection instances
        modules: a dictionary mapping module names to ModuleMetadata instances

    Returns:
        a PipelineOutput instance describing the pipeline output

    Raises:
        PipelineMetadataError: if the output was not properly connected
    """
    node_str = PipelineNode.get_output_str(name)
    sources = _get_sources_with_sink(node_str, connections)
    if len(sources) != 1:
        raise PipelineMetadataError(
            "Pipeline output '%s' must be connected to exactly one module "
            "output, but was connected to %d" % (name, len(sources))
        )

    source = sources[0]
    node = modules[source.module].metadata.get_output(source.node)
    return PipelineOutput(name, node)


def _validate_module_connections(modules, connections):
    """Ensures that the modules connections are valid.

    The module connections are valid if:
        - every module input either has exactly one incoming connection or is
            not required
        - every module output either has at least one outgoing connection or
            is not required

    Args:
        modules: a dictionary mapping module names to ModuleMetadata instances
        connections: a list of PipelineConnection instances

    Raises:
        PipelineMetadataError: if the modules were not properly connected
    """
    for mname, module in iteritems(modules):
        # Validate inputs
        for iname, node in iteritems(module.metadata.inputs):
            node_str = PipelineNode.get_node_str(mname, iname)
            num_sources = len(_get_sources_with_sink(node_str, connections))
            if num_sources == 0 and node.is_required:
                raise PipelineMetadataError(
                    "Module '%s' input '%s' is required but has no incoming "
                    "connection" % (mname, iname)
                )
            if num_sources > 1:
                raise PipelineMetadataError(
                    "Module '%s' input '%s' must have one incoming connection "
                    "but instead has %d incoming connections"
                    % (mname, iname, num_sources)
                )

        # Validate outputs
        for oname, node in iteritems(module.metadata.outputs):
            node_str = PipelineNode.get_node_str(mname, oname)
            sinks = _get_sinks_with_source(node_str, connections)
            if not sinks and node.is_required:
                raise PipelineMetadataError(
                    "Module '%s' output '%s' is required but has no outgoing "
                    "connections" % (mname, oname)
                )


def _compute_execution_order(connections):
    """Computes a valid execution order for the pipeline defined by the given
    module connections.

    Args:
        connections: a list of PipelineConnection instances

    Returns:
        a list defining a valid execution order for the modules

    Raises:
        PipelineMetadataError: if the module connections form a cyclic graph
    """
    module_graph = etag.DirectedGraph()
    for c in connections:
        module_graph.add_edge(c.source.module, c.sink.module)

    try:
        execution_order = module_graph.sort()
        execution_order.remove(PIPELINE_INPUT_NAME)
        execution_order.remove(PIPELINE_OUTPUT_NAME)
    except etag.CyclicGraphError:
        raise PipelineMetadataError(
            "Unable to compute a valid execution order because the module "
            "connections form a cyclic graph."
        )

    return execution_order


def _get_sinks_with_source(node_str, connections):
    """Returns the sink nodes that are connected to the source node with the
    given string representation.

    Args:
        node_str: a string representation of a PipelineNode
        connections: a list of PipelineConnection instances

    Returns:
        a list of PipelineNodes that are connected to the given source node.
    """
    return [c.sink for c in connections if c.source.is_same_node_str(node_str)]


def _get_sources_with_sink(node_str, connections):
    """Returns the source nodes that are connected to the sink node with the
    given string representation.

    Args:
        node_str: a string representation of a PipelineNode
        connections: a list of PipelineConnection instances

    Returns:
        a list of PipelineNodes that are connected to the given sink node.
    """
    return [c.source for c in connections if c.sink.is_same_node_str(node_str)]


def _parse_module_node_str(node_str):
    """Parses a module node string.

    Args:
        node_str: a string of the form <module>.<node>

    Returns:
        the module and node components of the module node string

    Raises:
        PipelineMetadataError: if the module node string was invalid
    """
    try:
        module, node = node_str.split(".")
    except ValueError:
        raise PipelineMetadataError(
            "Expected '%s' to have form <module>.<node>" % node_str
        )

    return module, node


def _parse_pipeline_node_str(node_str, inputs, outputs, modules):
    """Parses a pipeline node string.

    Args:
        node_str: a string of the form <module>.<node>
        inputs: a list of pipeline inputs
        outputs: a list of pipeline outputs
        modules: a dictionary mapping module names to PipelineModule instances

    Returns:
        a PipelineNode instance describing the node

    Raises:
        PipelineMetadataError: if the pipeline node string was invalid
    """
    module, node = _parse_module_node_str(node_str)

    if module == PIPELINE_INPUT_NAME:
        if node not in inputs:
            raise PipelineMetadataError("Pipeline has no input '%s'" % node)

        _type = PipelineNodeType.PIPELINE_INPUT
    elif module == PIPELINE_OUTPUT_NAME:
        if node not in outputs:
            raise PipelineMetadataError("Pipeline has no output '%s'" % node)

        _type = PipelineNodeType.PIPELINE_OUTPUT
    else:
        try:
            meta = modules[module].metadata
        except KeyError:
            raise PipelineMetadataError(
                "Module '%s' not found in pipeline" % module
            )

        if meta.has_input(node):
            _type = PipelineNodeType.MODULE_INPUT
        elif meta.has_output(node):
            _type = PipelineNodeType.MODULE_OUTPUT
        else:
            raise PipelineMetadataError(
                "Module '%s' has no input or output named '%s'"
                % (module, node)
            )

    return PipelineNode(module, node, _type)


def _create_node_connection(source, sink, modules):
    """Creates a pipeline connection between two nodes.

    Args:
        source: the source PipelineNode
        sink: the sink PipelineNode
        modules: a dictionary mapping module names to PipelineModule instances

    Returns:
        a PipelineConnection instance describing the connection

    Raises:
        PipelineMetadataError: if the connection was invalid
    """
    if source.is_pipeline_input and not sink.is_module_input:
        raise PipelineMetadataError(
            "'%s' must be connected to a module input" % source
        )
    if source.is_pipeline_output or source.is_module_input:
        raise PipelineMetadataError(
            "'%s' cannot be a connection source" % source
        )
    if sink.is_pipeline_input or sink.is_module_output:
        raise PipelineMetadataError("'%s' cannot be a connection sink" % sink)
    if source.is_module_output and sink.is_module_input:
        src = modules[source.module].metadata.get_output(source.node)
        snk = modules[sink.module].metadata.get_input(sink.node)
        if not issubclass(src.type, snk.type):
            raise PipelineMetadataError(
                "Module output '%s' ('%s') is not a valid input to module "
                "'%s' ('%s')" % (source, src.type, sink, snk.type)
            )

    return PipelineConnection(source, sink)


if __name__ == "__main__":
    run(sys.argv[1])
