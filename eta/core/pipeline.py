'''
Core pipeline infrastructure.

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
from future.utils import iteritems
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import OrderedDict
from glob import glob
import logging
import os
import sys

import eta
from eta.core.config import Config, Configurable
from eta.core.diagram import HasBlockDiagram, BlockdiagPipeline
import eta.core.job as etaj
import eta.core.log as etal
import eta.core.module as etam
import eta.core.types as etat
import eta.core.utils as etau


logger = logging.getLogger(__name__)


PIPELINE_INPUT_NAME = "INPUT"
PIPELINE_OUTPUT_NAME = "OUTPUT"


def load_all_metadata():
    '''Loads all pipeline metadata files.

    Assumes any JSON files in the `eta.config.pipeline_dirs` directories are
    pipeline metadata files.

    Returns:
        a dictionary mapping pipeline names to PipelineMetadata instances

    Raises:
        PipelineMetadataError: if any of the pipeline metadata files are
            invalid
    '''
    return {k: _load_metadata(v) for k, v in iteritems(find_all_metadata())}


def load_metadata(pipeline_name):
    '''Loads the pipeline metadata file for the pipeline with the given name.

    Pipeline metadata files must JSON files in one of the directories in
    `eta.config.pipeline_dirs`.

    Args:
        pipeline_name: the name of the pipeline

    Returns:
        the PipelineMetadata instance for the given pipeline

    Raises:
        PipelineMetadataError: if the pipeline metadata file could not be found
            or was invalid
    '''
    return _load_metadata(find_metadata(pipeline_name))


def _load_metadata(config):
    metadata = PipelineMetadata.from_json(config)
    name = os.path.splitext(os.path.basename(config))[0]
    if metadata.info.name != name:
        raise PipelineMetadataError(
            "Name '%s' from PipelineMetadata must match pipeline name '%s'" % (
                metadata.info.name, name))

    return metadata


def find_all_metadata():
    '''Finds all pipeline metadata files.

    Assumes any JSON files in the `eta.config.pipeline_dirs` directories are
    pipeline metadata files. To load these files, use `load_all_metadata()`.

    Returns:
        a dictionary mapping pipeline names to pipeline metadata filenames

    Raises:
        PipelineMetadataError: if the pipeline names are not unique
    '''
    d = {}
    for pdir in eta.config.pipeline_dirs:
        for path in glob(os.path.join(pdir, "*.json")):
            name = os.path.splitext(os.path.basename(path))[0]
            if name in d:
                raise PipelineMetadataError(
                    "Found two '%s' pipelines. Names must be unique." % name)
            d[name] = path

    return d


def find_metadata(pipeline_name):
    '''Finds the pipeline metadata file for the pipeline with the given name.

    Pipeline metadata files must be JSON files in one of the directories in
    `eta.config.pipeline_dirs`.

    Returns:
        the path to the pipeline metadata file

    Raises:
        PipelineMetadataError: if the pipeline could not be found
    '''
    try:
        return find_all_metadata()[pipeline_name]
    except KeyError:
        raise PipelineMetadataError(
            "Could not find pipeline '%s'" % pipeline_name)


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
    '''Pipeline configuration class.'''

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
    '''Pipeline metadata configuration class.'''

    def __init__(self, d):
        self.info = self.parse_object(d, "info", PipelineInfoConfig)
        self.inputs = self.parse_array(d, "inputs")
        self.outputs = self.parse_array(d, "outputs")
        self.modules = self.parse_object_array(
            d, "modules", PipelineModuleConfig)
        self.connections = self.parse_object_array(
            d, "connections", PipelineConnectionConfig)


class PipelineInfoConfig(Config):
    '''Pipeline info configuration class.'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.version = self.parse_string(d, "version")
        self.description = self.parse_string(d, "description")


class PipelineModuleConfig(Config):
    '''Pipeline module configuration class.'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.tunable_parameters = self.parse_array(d, "tunable_parameters")
        self.set_parameters = self.parse_dict(d, "set_parameters")


class PipelineConnectionConfig(Config):
    '''Pipeline connection configuration class.'''

    def __init__(self, d):
        self.source = self.parse_string(d, "source")
        self.sink = self.parse_string(d, "sink")


class PipelineNodeType(object):
    '''Class enumerating the types of pipeline nodes.'''

    PIPELINE_INPUT = 1
    PIPELINE_OUTPUT = 2
    MODULE_INPUT = 3
    MODULE_OUTPUT = 4


class PipelineInfo(Configurable):
    '''Pipeline info descriptor.

    Attributes:
        name: the name of the pipeline
        type: the eta.core.types.Type of the pipeline
        version: the pipeline version
        description: a free text description of the pipeline
    '''

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
                    "'%s' is not a valid pipeline type" % type_)
        return type_


class PipelineInput(object):
    '''Class describing a pipeline input.

    Attributes:
        name: the input name
        fields: the list ModuleField instance(s) of the module input field(s)
            that the pipeline input is connected to
    '''

    def __init__(self, name, fields):
        self.name = name
        self.fields = fields

    @property
    def is_mandatory(self):
        '''Returns True/False if this input is mandatory.'''
        return any(field.is_mandatory for field in self.fields)

    def is_valid_value(self, val):
        '''Returns True/False if `val` is a valid value for this input.'''
        return all(field.is_valid_value(val) for field in self.fields)


class PipelineOutput(object):
    '''Class describing a pipeline output.

    Attributes:
        name: the output name
        field: the ModuleField instance of the module output field that the
            pipeline output is connected to
    '''

    def __init__(self, name, field):
        self.name = name
        self.field = field

    def is_valid_value(self, val):
        '''Returns True/False if `val` is a valid value for this output.'''
        return self.field.is_valid_value(val)


# This exists so that None can be a valid value for parameters
class no_set_val(object):
    pass


class PipelineParameter(object):
    '''Class describing a tunable pipeline parameter.

    Attributes:
        param_str: the <module>.<parameter> string for the parameter
        field: the ModuleField instance of the module parameter that the
            pipeline parameter corresponds to
        is_tunable: whether the parameter is tunable
    '''

    def __init__(self, param_str, field, is_tunable, set_val=no_set_val):
        self.param_str = param_str
        self.field = field
        self.is_tunable = is_tunable
        self.set_val = set_val

    @property
    def has_set_val(self):
        '''Returns True/False if this parameter has a value set by the
        pipeline.
        '''
        return self.set_val is not no_set_val

    @property
    def is_mandatory(self):
        '''Returns True/False if this parameter must be set by the
        end-user.
        '''
        return self.field.is_mandatory and not self.has_set_val

    def is_valid_value(self, val):
        '''Returns True/False if `val` is a valid value for this parameter.'''
        return self.field.is_valid_value(val)


class PipelineModule(Configurable):
    '''Pipeline module class.

    A pipeline module definition is valid if every module parameter is either
    tunable, is set by the pipeline, or is exposed to the end-user as tunable.

    A module parameter is active if it is exposed as tunable and/or is set by
    the pipeline.

    Attributes:
        name: the name of the module
        metadata: the ModuleMetadata instance for the module
        active_parameters: a dictionary mapping <module>.<parameter> strings to
            PipelineParameter instances describing the active parameters of the
            module
    '''

    def __init__(self, config):
        '''Creates a PipelineModule instance.

        Args:
            config: a PipelineModuleConfig instance

        Raises:
            PipelineMetadataError: if the module configuration was invalid
        '''
        self.validate(config)

        self.name = config.name
        self.metadata = etam.load_metadata(config.name)
        self.active_parameters = {}

        self._parse_parameters(
            config.tunable_parameters, config.set_parameters)

    def _parse_parameters(self, tunable_parameters, set_parameters):
        # Verify parameter settings
        self._verify_has_parameters(tunable_parameters)
        self._verify_has_parameters(set_parameters.keys())
        self._verify_parameter_values(set_parameters)

        # Verify that each parameter is accounted for
        for pname, param in iteritems(self.metadata.parameters):
            is_tunable = pname in tunable_parameters
            is_set = pname in set_parameters
            is_active = is_tunable or is_set
            if param.is_mandatory and not is_active:
                raise PipelineMetadataError((
                    "Module '%s' parameter '%s' must be set, tunable, or "
                    "have a default value") % (self.name, pname))

            # Record active parameter
            if is_active:
                param_str = PipelineNode.get_node_str(self.name, pname)
                set_val = set_parameters[pname] if is_set else no_set_val
                self.active_parameters[param_str] = PipelineParameter(
                    param_str, param, is_tunable, set_val=set_val)

    def _verify_has_parameters(self, param_names):
        for name in param_names:
            if not self.metadata.has_parameter(name):
                raise PipelineMetadataError(
                    "Module '%s' has no parameter '%s'" % (self.name, name))

    def _verify_parameter_values(self, param_dict):
        for name, val in param_dict:
            if not self.metadata.is_valid_parameter(name, val):
                raise PipelineMetadataError((
                    "'%s' is an invalid value for parameter '%s' of module "
                    "'%s'") % (val, name, self.name))


class PipelineNode(object):
    '''Class representing a node in a pipeline.

    Attributes:
        module: the module name
        field: the field name
    '''

    def __init__(self, module, field, _type):
        self.module = module
        self.field = field
        self._type = _type

    def __str__(self):
        return "%s.%s" % (self.module, self.field)

    def is_same_node(self, node):
        '''Returns True/False if the given node is equal to this node, i.e.,
        if they refer to the same field of the same module.
        '''
        return str(self) == str(node)

    def is_same_node_str(self, node_str):
        '''Returns True/False if the given node string refers to this node.'''
        return str(self) == node_str

    @property
    def is_pipeline_node(self):
        '''Returns True/False if this node is a pipeline input or output.'''
        return self.is_pipeline_input or self.is_pipeline_output

    @property
    def is_pipeline_input(self):
        '''Returns True/False if this node is a pipeline input.'''
        return self._type == PipelineNodeType.PIPELINE_INPUT

    @property
    def is_pipeline_output(self):
        '''Returns True/False if this node is a pipeline output.'''
        return self._type == PipelineNodeType.PIPELINE_OUTPUT

    @property
    def is_module_node(self):
        '''Returns True/False if this node is a module input or output.'''
        return self.is_module_input or self.is_module_output

    @property
    def is_module_input(self):
        '''Returns True/False if this node is a module input.'''
        return self._type == PipelineNodeType.MODULE_INPUT

    @property
    def is_module_output(self):
        '''Returns True/False if this node is a module output.'''
        return self._type == PipelineNodeType.MODULE_OUTPUT

    @staticmethod
    def get_input_str(name):
        '''Gets the node string for a pipeline input with the given name.'''
        return "%s.%s" % (PIPELINE_INPUT_NAME, name)

    @staticmethod
    def get_output_str(name):
        '''Gets the node string for a pipeline output with the given name.'''
        return "%s.%s" % (PIPELINE_OUTPUT_NAME, name)

    @staticmethod
    def get_node_str(module, field):
        '''Gets the node string for a pipeline node with the given module and
        field names.
        '''
        return "%s.%s" % (module, field)


class PipelineConnection(object):
    '''Class representing a connection between two nodes in a pipeline.

    Attributes:
        source: the source PipelineNode
        sink: the sink PipelineNode
    '''

    def __init__(self, source, sink):
        self.source = source
        self.sink = sink

    def __str__(self):
        return "%s -> %s" % (self.source, self.sink)

    @property
    def is_module_connection(self):
        '''Returns True/False if this connection is between module nodes.'''
        return self.source.is_module_node and self.sink.is_module_node


class PipelineMetadata(Configurable, HasBlockDiagram):
    '''Class the encapsulates the architecture of a pipeline.

    A pipeline definition is valid if the following conditions are met:
        - every pipeline input is connected to at least one module input
        - every pipeline output is connected to exactly one module output
        - every module input either has a default value or has exactly one
            incoming connection
        - every module output either has a default value or has at least one
            outgoing connection
        - every module parameter is either tunable, is set by the pipeline, or
            is exposed to the end-user as tunable

    Attributes:
        info: a PipelineInfo instance describing the pipeline
        inputs: a dictionary mapping input names to PipelineInput instances
        outputs: a dictionary mapping output names to PipelineOutput instances
        parameters: a dictionary mapping <module>.<parameter> strings to
            PipelineParameter instances
        modules: a dictionary mapping module names to PipelineModule instances
        nodes: a list of PipelineNode instances describing the sources and
            sinks for all pipeline connections
        connections: a list of PipelineConnection instances describing the
            connections between pipeline inputs, modules, and pipeline outputs
    '''

    def __init__(self, config):
        '''Initializes a PipelineMetadata instance.

        Args:
            config: a PipelineMetadataConfig instance

        Raises:
            PipelineMetadataError: if the pipeline definition was invalid
        '''
        self.validate(config)

        self.info = None
        self.inputs = {}
        self.outputs = {}
        self.parameters = {}
        self.modules = OrderedDict()  # module order encodes execution order
        self.nodes = []
        self.connections = []

        self._parse_metadata(config)

    def has_input(self, name):
        '''Returns True/False if the pipeline has an input `name`.'''
        return name in self.inputs

    def has_output(self, name):
        '''Returns True/False if the pipeline has an output `name`.'''
        return name in self.outputs

    def has_module(self, name):
        '''Returns True/False if the pipeline has a module `name`.'''
        return name in self.modules

    def has_parameter(self, param_str):
        '''Returns True/False if this pipeline has parameter `param_str`.'''
        return param_str in self.parameters

    def is_valid_input(self, name, val):
        '''Returns True/False if `val` is a valid value for input `name`.'''
        return self.inputs[name].is_valid_value(val)

    def is_valid_output(self, name, val):
        '''Returns True/False if `val` is a valid vale for output `name`.'''
        return self.outputs[name].is_valid_value(val)

    def is_valid_parameter(self, param_str, val):
        '''Returns True/False if `val` is a valid value for parameter
        `param_str`.
        '''
        return self.parameters[param_str].is_valid_value(val)

    def to_blockdiag(self):
        '''Returns a BlockdiagPipeline representation of this pipeline.'''
        bp = BlockdiagPipeline(self.info.name)
        for name, module in iteritems(self.modules):
            bp.add_module(name, module.metadata.to_blockdiag())
        for n in self.nodes:
            if n.is_pipeline_input:
                bp.add_input(n.field)
            if n.is_pipeline_output:
                bp.add_output(n.field)
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
        for module_config in config.modules:
            module = PipelineModule(module_config)
            self.modules[module.name] = module
            self.parameters.update(module.active_parameters)

        # Parse connections
        for c in config.connections:
            # Parse nodes
            source = _parse_node_str(
                c.source, config.inputs, config.outputs, self.modules)
            sink = _parse_node_str(
                c.sink, config.inputs, config.outputs, self.modules)

            # Make sure we don't duplicate nodes
            source = self._register_node(source)
            sink = self._register_node(sink)

            # Record connection
            connection = _create_node_connection(source, sink, self.modules)
            self.connections.append(connection)

        # Parse inputs
        for name in config.inputs:
            self.inputs[name] = _parse_input(
                name, self.connections, self.modules)

        # Parse outputs
        for name in config.outputs:
            self.outputs[name] = _parse_output(
                name, self.connections, self.modules)

        # Validate connections
        _validate_module_connections(self.modules, self.connections)


class PipelineMetadataError(Exception):
    pass


def _parse_input(name, connections, modules):
    '''Parses the pipeline input with the given name.

    Args:
        name: the pipeline input name
        connections: a list of PipelineConnection instances
        modules: a dictionary mapping module names to ModuleMetadata instances

    Returns:
        a PipelineInput instance describing the pipeline input

    Raises:
        PipelineMetadataError: if the input was not properly connected
    '''
    node_str = PipelineNode.get_input_str(name)
    sinks = _get_sinks_with_source(node_str, connections)
    if len(sinks) == 0:
        raise PipelineMetadataError(
            "Pipeline input '%s' is not connected to any modules" % name)

    fields = [
        modules[node.module].metadata.get_input_field(node.field)
        for node in sinks
    ]
    return PipelineInput(name, fields)


def _parse_output(name, connections, modules):
    '''Parses the pipeline output with the given name.

    Args:
        name: the pipeline output name
        connections: a list of PipelineConnection instances
        modules: a dictionary mapping module names to ModuleMetadata instances

    Returns:
        a PipelineOutput instance describing the pipeline output

    Raises:
        PipelineMetadataError: if the output was not properly connected
    '''
    node_str = PipelineNode.get_output_str(name)
    sources = _get_sources_with_sink(node_str, connections)
    if len(sources) != 1:
        raise PipelineMetadataError((
            "Pipeline output '%s' must be connected to one module "
            "output, but was connected to %d") % (name, len(sources))
        )

    node = sources[0]
    field = modules[node.module].metadata.get_output_field(node.field)
    return PipelineOutput(name, field)


def _validate_module_connections(modules, connections):
    '''Ensures that the modules connections are valid, i.e., that:
        - every module input either has a default value or one incoming
            connection
        - every module output either has a default value or at least one
            outgoing connection

    Args:
        modules: a dictionary mapping module names to ModuleMetadata instances
        connections: a list of PipelineConnection instances

    Raises:
        PipelineMetadataError: if the modules were not properly connected
    '''
    for mname, module in iteritems(modules):
        # Validate inputs
        for iname, field in iteritems(module.metadata.inputs):
            node_str = PipelineNode.get_node_str(mname, iname)
            num_sources = len(_get_sources_with_sink(node_str, connections))
            if num_sources == 0 and field.is_mandatory:
                raise PipelineMetadataError((
                    "Module '%s' input '%s' has no default value and no "
                    "incoming connection") % (mname, iname)
                )
            if num_sources > 1:
                raise PipelineMetadataError((
                    "Module '%s' input '%s' must have one incoming connection "
                    "but instead has %d connections") % (
                        mname, iname, num_sources)
                )

        # Validate outputs
        for oname, field in iteritems(module.metadata.outputs):
            node_str = PipelineNode.get_node_str(mname, oname)
            num_sinks = len(_get_sinks_with_source(node_str, connections))
            if num_sinks == 0 and field.is_mandatory:
                raise PipelineMetadataError((
                    "Module '%s' output '%s' has no default value and no "
                    "outgoing connections") % (mname, oname)
                )


def _get_sinks_with_source(node_str, connections):
    '''Returns the sink nodes that are connected to the source node with the
    given string representation.

    Args:
        node_str: a string representation of a PipelineNode
        connections: a list of PipelineConnection instances

    Returns:
        a list of PipelineNodes that are connected to the given source node.
    '''
    return [c.sink for c in connections if c.source.is_same_node_str(node_str)]


def _get_sources_with_sink(node_str, connections):
    '''Returns the source nodes that are connected to the sink node with the
    given string representation.

    Args:
        node_str: a string representation of a PipelineNode
        connections: a list of PipelineConnection instances

    Returns:
        a list of PipelineNodes that are connected to the given sink node.
    '''
    return [c.source for c in connections if c.sink.is_same_node_str(node_str)]


def _parse_module_field_str(field_str):
    '''Parses a module field string.

    Args:
        field_str: a string of the form <module>.<field>

    Returns:
        the module and field components of the module field string

    Raises:
        PipelineMetadataError: if the module field string was invalid
    '''
    try:
        module, field = field_str.split(".")
    except ValueError:
        raise PipelineMetadataError(
            "Expected '%s' to have form <module>.<field>" % field_str)

    return module, field


def _parse_node_str(node_str, inputs, outputs, modules):
    '''Parses a pipeline node string.

    Args:
        node_str: a string of the form <module>.<field>
        inputs: a list of pipeline inputs
        outputs: a list of pipeline outputs
        modules: a dictionary mapping module names to PipelineModule
            instances

    Returns:
        a PipelineNode instance describing the node

    Raises:
        PipelineMetadataError: if the pipeline node string was invalid
    '''
    module, field = _parse_module_field_str(node_str)

    if module == PIPELINE_INPUT_NAME:
        if field not in inputs:
            raise PipelineMetadataError(
                "Pipeline has no input '%s'" % field)

        _type = PipelineNodeType.PIPELINE_INPUT
    elif module == PIPELINE_OUTPUT_NAME:
        if field not in outputs:
            raise PipelineMetadataError(
                "Pipeline has no output '%s'" % field)

        _type = PipelineNodeType.PIPELINE_OUTPUT
    else:
        try:
            meta = modules[module].metadata
        except KeyError:
            raise PipelineMetadataError(
                "Module '%s' not found in pipeline" % module)

        if meta.has_input(field):
            _type = PipelineNodeType.MODULE_INPUT
        elif meta.has_output(field):
            _type = PipelineNodeType.MODULE_OUTPUT
        else:
            raise PipelineMetadataError(
                "Module '%s' has no input or output named '%s'" % (
                    module, field))

    return PipelineNode(module, field, _type)


def _create_node_connection(source, sink, modules):
    '''Creates a pipeline connection between two nodes.

    Args:
        source: the source PipelineNode
        sink: the sink PipelineNode
        modules: a dictionary mapping module names to PipelineModule
            instances

    Returns:
        a PipelineConnection instance describing the connection

    Raises:
        PipelineMetadataError: if the connection was invalid
    '''
    if source.is_pipeline_input and not sink.is_module_input:
        raise PipelineMetadataError(
            "'%s' must be connected to a module input" % source)
    if source.is_pipeline_output or source.is_module_input:
        raise PipelineMetadataError(
            "'%s' cannot be a connection source" % source)
    if sink.is_pipeline_input or  sink.is_module_output:
        raise PipelineMetadataError(
        "'%s' cannot be a connection sink" % sink)
    if source.is_module_output and sink.is_module_input:
        src = modules[source.module].metadata.get_output_field(source.field)
        snk = modules[sink.module].metadata.get_input_field(sink.field)
        if not issubclass(src.type, snk.type):
            raise PipelineMetadataError(
                (
                    "Module output '%s' ('%s') is not a valid input "
                    "to module '%s' ('%s')"
                ) % (source, src.type, sink, snk.type)
            )

    return PipelineConnection(source, sink)


if __name__ == "__main__":
    run(sys.argv[1])
