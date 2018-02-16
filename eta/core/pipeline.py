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
import logging
import os
import sys

import eta
from eta.core.config import Config, Configurable
from eta.core.diagram import BlockDiagram, BlockdiagPipeline
import eta.core.job as etaj
import eta.core.log as etal
import eta.core.module as etam
import eta.core.types as etat
import eta.core.utils as etau


logger = logging.getLogger(__name__)


def load_metadata(pipeline_name):
    '''Loads the pipeline metadata file for the pipeline with the given name.

    Args:
        pipeline_name: the name of the pipeline

    Returns:
        the PipelineMetadata instance for the given pipeline

    Raises:
        PipelineMetadataError: if the pipeline metadata file could not be found
    '''
    return PipelineMetadata.from_json(find_metadata(pipeline_name))


def find_metadata(pipeline_name):
    '''Finds the pipeline metadata file for the pipeline with the given name.

    Modules must be located in one of the directories in the
    `eta.config.pipeline_dirs` list

    Args:
        pipeline_name: the name of the pipeline

    Returns:
        the absolute path to the pipeline metadata file

    Raises:
        PipelineMetadata: if the pipeline metadata file could not be found
    '''
    for d in eta.config.pipeline_dirs:
        abspath = os.path.join(d, pipeline_name + ".json")
        if os.path.isfile(abspath):
            return abspath

    raise PipelineMetadata("Could not find pipeline '%s'" % pipeline_name)


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
        self.modules = self.parse_array(d, "modules")
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


class PipelineConnectionConfig(Config):
    '''Pipeline connection configuration class.'''

    def __init__(self, d):
        self.source = self.parse_string(d, "source")
        self.sink = self.parse_string(d, "sink")


class PipelineNode(object):
    '''Class representing a node in a pipeline.'''

    PIPELINE_INPUT_NAME = "INPUT"
    PIPELINE_OUTPUT_NAME = "OUTPUT"
    PIPELINE_INPUT = 1
    PIPELINE_OUTPUT = 2
    MODULE_INPUT = 3
    MODULE_OUTPUT = 4

    def __init__(self, node_str, modules):
        '''Creates a new PipelineNode instance.

        Args:
            node_str: a string of the form <module>.<field>
            modules: a dictionary mapping module names to ModuleMetadata
                instances

        Raises:
            PipelineMetadataError: if the pipeline node string was invalid
        '''
        m, f, _t = self._parse_node_str(node_str, modules)
        self.module = m
        self.field = f
        self._type = _t

    def __str__(self):
        return "%s.%s" % (self.module, self.field)

    def is_same_node(self, node):
        '''Returns True/False if the given node string refers to this node.'''
        return str(self) == str(node)

    @property
    def is_pipeline_input(self):
        '''Returns True/False if this node is a pipeline input.'''
        return self._type == PipelineNode.PIPELINE_INPUT

    @property
    def is_pipeline_output(self):
        '''Returns True/False if this node is a pipeline output.'''
        return self._type == PipelineNode.PIPELINE_OUTPUT

    @property
    def is_module_input(self):
        '''Returns True/False if this node is a module input.'''
        return self._type == PipelineNode.MODULE_INPUT

    @property
    def is_module_output(self):
        '''Returns True/False if this node is a module output.'''
        return self._type == PipelineNode.MODULE_OUTPUT

    @staticmethod
    def _parse_node_str(node_str, modules):
        try:
            module, field = node_str.split(".")
        except ValueError:
            raise PipelineMetadataError(
                "Expected '%s' to have form <module>.<field>" % node_str)

        if module == PipelineNode.PIPELINE_INPUT_NAME:
            _type = PipelineNode.PIPELINE_INPUT
        elif module == PipelineNode.PIPELINE_OUTPUT_NAME:
            _type = PipelineNode.PIPELINE_OUTPUT
        else:
            try:
                mod = modules[module]
            except KeyError:
                raise PipelineMetadataError(
                    "Module '%s' not found in pipeline" % module)

            if mod.has_input(field):
                _type = PipelineNode.MODULE_INPUT
            elif mod.has_output(field):
                _type = PipelineNode.MODULE_OUTPUT
            else:
                raise PipelineMetadataError(
                    "Module '%s' has no input or output named '%s'" % (
                        module, field))

        return module, field, _type


class PipelineConnection(object):
    '''Class representing a connection between two nodes in a pipeline.'''

    def __init__(self, source, sink, modules):
        '''Creates a new PipelineConnection instance.

        Args:
            source: the source PipelineNode
            sink: the sink PipelineNode
            modules: a dictionary mapping module names to ModuleMetadata
                instances

        Raises:
            PipelineMetadataError: if the pipeline connection was invalid
        '''
        self._validate_connection(source, sink, modules)
        self.source = source
        self.sink = sink

    def __str__(self):
        return "%s -> %s" % (self.source, self.sink)

    @staticmethod
    def _validate_connection(source, sink, modules):
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
            src = modules[source.module].get_output(source.field)
            snk = modules[sink.module].get_input(sink.field)
            if not issubclass(src.type, snk.type):
                raise PipelineMetadataError(
                    (
                        "Module output '%s' ('%s') is not a valid input "
                        "to module '%s' ('%s')"
                    ) % (source, src.type, sink, snk.type)
                )


class PipelineMetadata(Configurable, BlockDiagram):
    '''Class the encapsulates the architecture of a pipeline.

    Attributes:
        info: a PipelineInfo instance describing the pipeline
        modules: a dictionary mapping module names to ModuleMetadata instances
        nodes: a list of PipelineNode instances describing the connection
            pipeline-level sources and sinks for all pipeline-level connections
        connections: a list of PipelineConnection instances describing the
            pipeline-level connections between pipeline inputs, modules and
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
        self.modules = OrderedDict()
        self.nodes = []
        self.connections = []
        self._parse_metadata(config)

    def to_blockdiag(self):
        '''Returns a BlockdiagPipeline representation of this pipeline.'''
        bp = BlockdiagPipeline(self.info.name)
        for name, mm in iteritems(self.modules):
            bp.add_module(name, mm.to_blockdiag())
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

        # Parse modules
        for name in config.modules:
            self.modules[name] = etam.load_metadata(name)

        # Parse connections
        for c in config.connections:
            # Get (unique) nodes
            source = self._register_node(PipelineNode(c.source, self.modules))
            sink = self._register_node(PipelineNode(c.sink, self.modules))

            # Record connection
            connection = PipelineConnection(source, sink, self.modules)
            self.connections.append(connection)


class PipelineMetadataError(Exception):
    pass


if __name__ == "__main__":
    run(sys.argv[1])
