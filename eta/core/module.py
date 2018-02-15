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
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import os

import eta
import eta.constants as etac
from eta.core.config import Config, Configurable
from eta.core.diagram import BlockDiagram, BlockdiagModule
import eta.core.log as etal
import eta.core.utils as etau


def load_metadata(module_name):
    '''Loads the module metadata file for the module with the given name.

    Args:
        module_name: the name of the module

    Returns:
        the ModuleMetadata instance for the given module

    Raises:
        ModuleMetadataError: if the module metadata file could not be found
    '''
    return ModuleMetadata.from_json(find_metadata(module_name))


def find_metadata(module_name):
    '''Find the module metadata file for the module with the given name.

    Modules must be located in one of the directories in the
    `eta.config.module_dirs` list

    Args:
        module_name: the name of the module

    Returns:
        the absolute path to the module metadata file

    Raises:
        ModuleMetadataError: if the module metadata file could not be found
    '''
    for d in eta.config.module_dirs:
        abspath = os.path.join(d, module_name + ".json")
        if os.path.isfile(abspath):
            return abspath

    raise ModuleMetadataError("Could not find module '%s'" % module_name)


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
    '''Base module configuration settings.'''

    def __init__(self, d):
        self.logging_config = self.parse_object(
            d, "logging_config", etal.LoggingConfig,
            default=etal.LoggingConfig.default())


class ModuleMetadataConfig(Config):
    '''Module metadata configuration class.'''

    def __init__(self, d):
        self.info = self.parse_object(d, "info", InfoConfig)
        self.inputs = self.parse_object_array(d, "inputs", FieldConfig)
        self.outputs = self.parse_object_array(d, "outputs", FieldConfig)
        self.parameters = self.parse_object_array(d, "parameters", FieldConfig)


class InfoConfig(Config):
    '''Module info.'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.version = self.parse_string(d, "version")
        self.description = self.parse_string(d, "description")
        self.id = self.parse_string(d, "id")
        self.exe = self.parse_string(d, "exe")


# This exists so that None can be a default value for Config fields
class mandatory(object):
    pass


class FieldConfig(Config):
    '''Module field descriptor.'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.description = self.parse_string(d, "description")
        self.default = self.parse_raw(d, "default", default=mandatory)

    @property
    def is_mandatory(self):
        '''Returns True/False indicating whether this field is mandatory.'''
        return self.default is mandatory


class ModuleMetadata(Configurable, BlockDiagram):
    '''Class the encapsulates the architecture of a module.'''

    def __init__(self, config):
        '''Initializes a ModuleMetadata instance.

        Args:
            config: a ModuleMetadataConfig instance

        Raises:
            ModuleMetadataError: if there was an error parsing the module
                definition
        '''
        self.validate(config)
        self.config = config
        self.parse_metadata()

    def parse_metadata(self):
        '''Parses the module metadata config.'''
        # Verify types
        for i in self.config.inputs:
            self.verify_field_type(i)
        for o in self.config.outputs:
            self.verify_field_type(o)
        for p in self.config.parameters:
            self.verify_field_type(p)

    @staticmethod
    def verify_field_type(field):
        '''Verifies that the field has a valid type.

        Args:
            field: a FieldConfig instance

        Raises:
            ModuleMetadataError if the type is invalid.
        '''
        try:
            etau.get_class(field.type)
        except ImportError:
            raise ModuleMetadataError(
                "Field '%s' has unknown type '%s'" % (field.name, field.type))

    def _to_blockdiag(self, path):
        bm = BlockdiagModule(self.config.info.name)
        for i in self.config.inputs:
            bm.add_input(i.name)
        for o in self.config.outputs:
            bm.add_output(o.name)
        for p in self.config.parameters:
            bm.add_parameter(p.name)
        bm.write(path)


class ModuleMetadataError(Exception):
    pass
