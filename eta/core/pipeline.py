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


class PipelineInfo(Configurable):
    '''Pipeline info descriptor.'''

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


class PipelineModuleConfig(Config):
    '''Pipeline module configuration class.'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.set_parameters = self.parse_raw(d, "set_parameters")
        self.tunable_parameters = self.parse_array(d, "tunable_parameters")


class PipelineModule(Configurable):
    '''Pipeline module class.'''

    def __init__(self, config):
        self.validate(config)
        self.name = config.name
        self.metadata = etam.load_metadata(config.name)
        self.set_parameters = config.set_parameters
        self.tunable_parameters = config.tunable_parameters


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


class PipelineNode(object):
    '''Class representing a node in a pipeline.'''

    def __init__(self, module, field, _type):
        '''Creates a new PipelineNode instance.

        Args:
            module: the module name string
            field: the field name string
            _type: the PipelineNodeType of the node
        '''
        self.module = module
        self.field = field
        self._type = _type

    def __str__(self):
        return "%s.%s" % (self.module, self.field)

    def is_same_node(self, node):
        '''Returns True/False if the given node string refers to this node.'''
        return str(self) == str(node)

    @property
    def is_pipeline_input(self):
        '''Returns True/False if this node is a pipeline input.'''
        return self._type == PipelineNodeType.PIPELINE_INPUT

    @property
    def is_pipeline_output(self):
        '''Returns True/False if this node is a pipeline output.'''
        return self._type == PipelineNodeType.PIPELINE_OUTPUT

    @property
    def is_module_input(self):
        '''Returns True/False if this node is a module input.'''
        return self._type == PipelineNodeType.MODULE_INPUT

    @property
    def is_module_output(self):
        '''Returns True/False if this node is a module output.'''
        return self._type == PipelineNodeType.MODULE_OUTPUT


class PipelineConnection(object):
    '''Class representing a connection between two nodes in a pipeline.'''

    def __init__(self, source, sink):
        '''Creates a new PipelineConnection instance.

        Args:
            source: the source PipelineNode
            sink: the sink PipelineNode
        '''
        self.source = source
        self.sink = sink

    def __str__(self):
        return "%s -> %s" % (self.source, self.sink)


class PipelineMetadata(Configurable, HasBlockDiagram):
    '''Class the encapsulates the architecture of a pipeline.

    Attributes:
        info: a PipelineInfo instance describing the pipeline
        modules: a dictionary mapping module names to PipelineModule instances
        nodes: a list of PipelineNode instances describing the connection
            pipeline-level sources and sinks for all pipeline-level connections
        connections: a list of PipelineConnection instances describing the
            pipeline-level connections between pipeline inputs, modules, and
            pipeline outputs
    '''

    def __init__(self, config):
        '''Initializes a PipelineMetadata instance.

        Args:
            config: a PipelineMetadataConfig instance

        Raises:
            PipelineMetadataError: if there was an error parsing the pipeline
                definition
        '''
        self.validate(config)
        self.info = None
        self.inputs = None
        self.outputs = None
        self.modules = OrderedDict()
        self.nodes = []
        self.connections = []
        self._parse_metadata(config)

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
        self.info = PipelineInfo(config.info)
        self.inputs = config.inputs
        self.outputs = config.outputs

        # Parse modules
        for module_config in config.modules:
            module = PipelineModule(module_config)
            self.modules[module.name] = module

        # Parse connections
        for c in config.connections:
            # Parse nodes
            source = _parse_node_str(
                c.source, self.inputs, self.outputs, self.modules)
            sink = _parse_node_str(
                c.sink, self.inputs, self.outputs, self.modules)

            # Make sure we don't duplicate nodes
            source = self._register_node(source)
            sink = self._register_node(sink)

            # Record connection
            connection = _create_node_connection(source, sink, self.modules)
            self.connections.append(connection)


class PipelineMetadataError(Exception):
    pass


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
    try:
        module, field = node_str.split(".")
    except ValueError:
        raise PipelineMetadataError(
            "Expected '%s' to have form <module>.<field>" % node_str)

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
        PipelineMetadataError: if the pipeline connection was invalid
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
        src = modules[source.module].metadata.get_output(source.field)
        snk = modules[sink.module].metadata.get_input(sink.field)
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
