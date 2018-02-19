'''
Core module infrastructure.

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
import os

import eta
from eta.core.config import Config, Configurable
from eta.core.diagram import HasBlockDiagram, BlockdiagModule
import eta.core.log as etal
import eta.core.types as etat
import eta.core.utils as etau


def load_all_metadata():
    '''Loads all module metadata files.

    Assumes any JSON files in the `eta.config.module_dirs` directories are
    module metadata files.

    Returns:
        a dictionary mapping module names to ModuleMetadata instances

    Raises:
        ModuleMetadataError: if any of the module metadata files are invalid
    '''
    return {k: _load_metadata(v) for k, v in iteritems(find_all_metadata())}


def load_metadata(module_name):
    '''Loads the module metadata file for the module with the given name.

    Module metadata files must JSON files in one of the directories in
    `eta.config.module_dirs`.

    Args:
        module_name: the name of the module

    Returns:
        the ModuleMetadata instance for the given module

    Raises:
        ModuleMetadataError: if the module metadata file could not be found
            or was invalid
    '''
    return _load_metadata(find_metadata(module_name))


def _load_metadata(config):
    metadata = ModuleMetadata.from_json(config)
    name = os.path.splitext(os.path.basename(config))[0]
    if metadata.info.name != name:
        raise ModuleMetadataError(
            "Name '%s' from ModuleMetadata must match module name '%s'" % (
                metadata.info.name, name))

    return metadata


def find_all_metadata():
    '''Finds all module metadata files.

    Assumes any JSON files in the `eta.config.module_dirs` directories are
    module metadata files. To load these files, use `load_all_metadata()`.

    Returns:
        a dictionary mapping module names to module metadata filenames

    Raises:
        ModuleMetadataError: if the module names are not unique
    '''
    d = {}
    for pdir in eta.config.module_dirs:
        for path in glob(os.path.join(pdir, "*.json")):
            name = os.path.splitext(os.path.basename(path))[0]
            if name in d:
                raise ModuleMetadataError(
                    "Found two '%s' modules. Names must be unique." % name)
            d[name] = path

    return d


def find_metadata(module_name):
    '''Finds the module metadata file for the module with the given name.

    Module metadata files must be JSON files in one of the directories in
    `eta.config.module_dirs`.

    Returns:
        the path to the module metadata file

    Raises:
        ModuleMetadataError: if the module could not be found
    '''
    try:
        return find_all_metadata()[module_name]
    except KeyError:
        raise ModuleMetadataError(
            "Could not find module '%s'" % module_name)


# @todo should pass a PipelineConfig instance here, not just the path. The need
# to use PipelineConfig here is causing a circular import with eta.core.module.
# This suggests bad design...
def setup(module_config, pipeline_config_path=None):
    '''Perform module setup.

    If a pipeline config is provided, it overrides any applicable values in
    the module config.

    Args:
        module_config: a Config instance derived from BaseModuleConfig
        pipeline_config_path: an optional path to a PipelineConfig
    '''
    # Set/override module config settings
    if pipeline_config_path:
        from eta.core.pipeline import PipelineConfig
        pipeline_config = PipelineConfig.from_json(pipeline_config_path)
        module_config.logging_config = pipeline_config.logging_config

    # Setup logging
    etal.custom_setup(module_config.logging_config)


class BaseModuleConfig(Config):
    '''Base module configuration class.'''

    def __init__(self, d):
        self.logging_config = self.parse_object(
            d, "logging_config", etal.LoggingConfig,
            default=etal.LoggingConfig.default())


class ModuleMetadataConfig(Config):
    '''Module metadata configuration class.'''

    def __init__(self, d):
        self.info = self.parse_object(d, "info", ModuleInfoConfig)
        self.inputs = self.parse_object_array(d, "inputs", ModuleFieldConfig)
        self.outputs = self.parse_object_array(d, "outputs", ModuleFieldConfig)
        self.parameters = self.parse_object_array(
            d, "parameters", ModuleFieldConfig)


class ModuleInfoConfig(Config):
    '''Module info configuration class.'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.version = self.parse_string(d, "version")
        self.description = self.parse_string(d, "description")
        self.exe = self.parse_string(d, "exe")


# This exists so that None can be a default value for Config fields
class mandatory(object):
    pass


class ModuleFieldConfig(Config):
    '''Module field descriptor configuration.'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.description = self.parse_string(d, "description")
        self.default = self.parse_raw(d, "default", default=mandatory)


class ModuleInfo(Configurable):
    '''Module info descriptor.

    Attributes:
        name: the name of the module
        type: the eta.core.types.Type of the module
        version: the version of the module
        description: a free text description of the module
        exe: the executable for the module
    '''

    def __init__(self, config):
        self.validate(config)

        self.name = config.name
        self.type = self._parse_type(config.type)
        self.version = config.version
        self.description = config.description
        self.exe = config.exe

    @staticmethod
    def _parse_type(type_str):
        type_ = etat.parse_type(type_str)
        if not etat.is_module(type_):
            raise ModuleMetadataError(
                "'%s' is not a valid module type" % type_)
        return type_


class ModuleField(Configurable):
    '''Module field descriptor.


    Attributes:
        name: the name of the field
        type: the eta.core.types.Type of the field
        description: a free text description of the field
        default: the default value (if any) of the field
    '''

    def __init__(self, config):
        self.validate(config)

        self.name = config.name
        self.type = self._parse_type(config.type)
        self.description = config.description
        self.default = config.default

    @property
    def is_mandatory(self):
        '''Returns True/False indicating whether this field is mandatory.'''
        return self.default is mandatory

    @staticmethod
    def _parse_type(type_str):
        type_ = etat.parse_type(type_str)
        if not etat.is_data(type_):
            raise ModuleMetadataError(
                "'%s' is not a valid data type" % type_)
        return type_


class ModuleMetadata(Configurable, HasBlockDiagram):
    '''Class the encapsulates the architecture of a module.

    Attributes:
        info: a ModuleInfo instance describing the module
        inputs: a dictionary mapping input names to ModuleField instances
            describing the inputs
        outputs: a dictionary mapping output names to ModuleField instances
            describing the outputs
        parameters: a dictionary mapping parameter names to ModuleField
            instances describing the parameters
    '''

    def __init__(self, config):
        '''Initializes a ModuleMetadata instance.

        Args:
            config: a ModuleMetadataConfig instance

        Raises:
            ModuleMetadataError: if there was an error parsing the module
                definition
        '''
        self.validate(config)
        self.info = None
        self.inputs = OrderedDict()
        self.outputs = OrderedDict()
        self.parameters = OrderedDict()
        self._parse_metadata(config)

    def has_input(self, name):
        '''Returns True/False if the module has an input with the given
        name.
        '''
        return name in self.inputs

    def has_output(self, name):
        '''Returns True/False if the module has an output with the given
        name.
        '''
        return name in self.outputs

    def has_parameter(self, name):
        '''Returns True/False if the module has a parameter with the given
        name.
        '''
        return name in self.parameters

    def get_input(self, name):
        '''Returns the input ModuleField with the given name.'''
        return self.inputs[name]

    def get_output(self, name):
        '''Returns the output ModuleField with the given name.'''
        return self.outputs[name]

    def get_parameter(self, name):
        '''Returns the parameter ModuleField with the given name.'''
        return self.parameters[name]

    def to_blockdiag(self):
        '''Returns a BlockdiagModule representation of this module.'''
        bm = BlockdiagModule(self.info.name)
        for name in self.inputs:
            bm.add_input(name)
        for name in self.outputs:
            bm.add_output(name)
        for name in self.parameters:
            bm.add_parameter(name)
        return bm

    def _parse_metadata(self, config):
        self.info = ModuleInfo(config.info)
        for i in config.inputs:
            self.inputs[i.name] = ModuleField(i)
        for o in config.outputs:
            self.outputs[o.name] = ModuleField(o)
        for p in config.parameters:
            self.parameters[p.name] = ModuleField(p)


class ModuleMetadataError(Exception):
    pass
