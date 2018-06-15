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
        ModuleMetadataError: if the module metadata file could not be found
    '''
    try:
        return find_all_metadata()[module_name]
    except KeyError:
        raise ModuleMetadataError(
            "Could not find module '%s'" % module_name)


def find_exe(module_exe):
    '''Finds the given module executable.

    Module executables must be in one of the directories in
    `eta.config.module_dirs`.

    Returns:
        the path to the module executable

    Raises:
        ModuleMetadataError: if the module executable could not be found
    '''
    for pdir in eta.config.module_dirs:
        exe_path = os.path.join(pdir, module_exe)
        if os.path.isfile(exe_path):
            return exe_path

    raise ModuleMetadataError(
        "Could not find module executable '%s'" % module_exe)


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
        module_config.base.logging_config = pipeline_config.logging_config

    # Setup logging
    etal.custom_setup(module_config.base.logging_config)


class BaseModuleConfig(Config):
    '''Base module configuration class that defines common configuration
    fields that all modules must support.

    All fields defined here should provide default values.

    Attributes:
        base: an `eta.core.module.BaseModuleConfigSettings` instance defining
            module configuration parameters
    '''

    def __init__(self, d):
        self.base = self.parse_object(
            d, "base", BaseModuleConfigSettings,
            default=BaseModuleConfigSettings.default(),
        )


class BaseModuleConfigSettings(Config):
    '''Base module configuration settings that all modules must support.

    All fields defined here should provide default values.

    Attributes:
        logging_config: an `eta.core.log.LoggingConfig` instance defining
            the logging configuration settings for the module
    '''

    def __init__(self, d):
        self.logging_config = self.parse_object(
            d, "logging_config", etal.LoggingConfig,
            default=etal.LoggingConfig.default())


class GenericModuleConfig(Config):
    '''Generic module configuration class.

    This class is used by `eta.core.builder.PipelineBuilder` to build
    module configuration files.
    '''

    def __init__(self, d):
        self.data = self.parse_array(d, "data", default=[])
        self.parameters = self.parse_dict(d, "parameters", default={})


class ModuleMetadataConfig(Config):
    '''Module metadata configuration class.'''

    def __init__(self, d):
        self.info = self.parse_object(d, "info", ModuleInfoConfig)
        self.inputs = self.parse_object_array(d, "inputs", ModuleInputConfig)
        self.outputs = self.parse_object_array(
            d, "outputs", ModuleOutputConfig)
        self.parameters = self.parse_object_array(
            d, "parameters", ModuleParameterConfig)

    def attributes(self):
        return ["info", "inputs", "outputs", "parameters"]


class ModuleInfoConfig(Config):
    '''Module info configuration class.'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.version = self.parse_string(d, "version")
        self.description = self.parse_string(d, "description")
        self.exe = self.parse_string(d, "exe")

    def attributes(self):
        return ["name", "type", "version", "description", "exe"]


class ModuleInputConfig(Config):
    '''Module input descriptor configuration.'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.description = self.parse_string(d, "description")
        self.required = self.parse_bool(d, "required", default=True)

    def attributes(self):
        return ["name", "type", "description", "required"]


class ModuleOutputConfig(Config):
    '''Module output descriptor configuration.'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.description = self.parse_string(d, "description")
        self.required = self.parse_bool(d, "required", default=True)

    def attributes(self):
        return ["name", "type", "description", "required"]


class ModuleParameterConfig(Config):
    '''Module parameter descriptor configuration.'''

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.description = self.parse_string(d, "description")
        self.required = self.parse_bool(d, "required", default=True)
        self.default = self.parse_raw(d, "default", default=None)

    def attributes(self):
        return ["name", "type", "description", "required", "default"]


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


class ModuleInput(Configurable):
    '''Module input descriptor.

    Module inputs must be subclasses of eta.core.types.Data.

    Attributes:
        name: the name of the input
        type: the eta.core.types.Type of the input
        description: a free text description of the input
        required: whether the input is required
    '''

    def __init__(self, config):
        '''Creates a new ModuleInput instance.

        Args:
            config: a ModuleInputConfig instance

        Raises:
        '''
        self.validate(config)

        self.name = config.name
        self.type = self._parse_type(config.type)
        self.description = config.description
        self.required = config.required

    def is_valid_path(self, path):
        '''Returns True/False indicating whether the given path is a valid
        setting for this input.'''
        return self.type.is_valid_path(path)

    @property
    def is_required(self):
        '''Returns True/False if this input is required.'''
        return self.required

    def _parse_type(self, type_str):
        type_ = etat.parse_type(type_str)
        if not etat.is_data(type_):
            raise ModuleMetadataError((
                "Module input '%s' has type '%s' but must be a subclass "
                "of Data") % (self.name, type_))
        return type_


class ModuleOutput(Configurable):
    '''Module output descriptor.

    Module outputs must be subclasses of eta.core.types.ConcreteData.

    Attributes:
        name: the name of the output
        type: the eta.core.types.Type of the output
        description: a free text description of the output
        required: whether the output is required
    '''

    def __init__(self, config):
        '''Creates a new ModuleOutput instance.

        Args:
            config: a ModuleOutputConfig instance

        Raises:
        '''
        self.validate(config)

        self.name = config.name
        self.type = self._parse_type(config.type)
        self.description = config.description
        self.required = config.required

    def is_valid_path(self, path):
        '''Returns True/False indicating whether the given path is a valid
        setting for this output.'''
        return self.type.is_valid_path(path)

    @property
    def is_required(self):
        '''Returns True/False if this output is required.'''
        return self.required

    def _parse_type(self, type_str):
        type_ = etat.parse_type(type_str)
        if not etat.is_concrete_data(type_):
            raise ModuleMetadataError((
                "Module output '%s' has type '%s' but must be a subclass "
                "of ConcreteData") % (self.name, type_))
        return type_


class ModuleParameter(Configurable):
    '''Module parameter descriptor.

    Module parameters must be subclasses of eta.core.types.Builtin or
    eta.core.types.ConcreteData.

    Attributes:
        name: the name of the parameter
        type: the eta.core.types.Type of the parameter
        description: a free text description of the parameter
        required: whether the parameter is required
        default: an optional default value for the parameter
    '''

    def __init__(self, config):
        self.validate(config)

        self.name = config.name
        self.type = self._parse_type(config.name, config.type)
        self.description = config.description
        self.required = config.required
        self.default = config.default

        if self.has_default_value:
            self._validate_default()

    def is_valid_value(self, val):
        '''Returns True/False indicating whether the given value is a valid
        setting for this parameter.'''
        if self.is_builtin:
            return self.type.is_valid_value(val)
        return self.type.is_valid_path(val)

    @property
    def is_required(self):
        '''Returns True/False if this parameter is required.'''
        return self.required

    @property
    def has_default_value(self):
        '''Returns True/false if this parameter has a default value.'''
        return self.default is not None

    @property
    def is_builtin(self):
        '''Returns True/False if this parameter is a Builtin.'''
        return etat.is_builtin(self.type)

    @property
    def is_data(self):
        '''Returns True/False if this parameter is Data.'''
        return etat.is_data(self.type)

    @staticmethod
    def _parse_type(name, type_str):
        type_ = etat.parse_type(type_str)
        if not etat.is_builtin(type_) and not etat.is_concrete_data(type_):
            raise ModuleMetadataError((
                "Module parameter '%s' has type '%s' but must be a subclass "
                "of Builtin or ConcreteData") % (name, type_))
        return type_

    def _validate_default(self):
        if self.is_builtin:
            is_valid = self.type.is_valid_value(self.default)
        else:
            is_valid = self.type.is_valid_path(self.default)
        if not is_valid:
            raise ModuleMetadataError((
                "Default value '%s' is invalid for module parameter '%s' of "
                "'%s'") % (self.default, self.name, self.type))


class ModuleMetadata(Configurable, HasBlockDiagram):
    '''Class that encapsulates the architecture of a module.

    A module definition is valid if all of the following are true:
        - the module has at least one input and output
        - all input, output, and parameter names are mutually unique
        - all inputs have types that are subclasses of eta.core.types.Data
        - all outputs have types that are subclasses of
            eta.core.types.ConcreteData
        - all parameters have types that are subclasses of
            eta.core.types.Builtin or eta.core.types.ConcreteData
        - any default parameters are valid values for their associated types

    Attributes:
        info: a ModuleInfo instance describing the module
        inputs: a dictionary mapping input names to ModuleInput instances
            describing the inputs
        outputs: a dictionary mapping output names to ModuleOutput instances
            describing the outputs
        parameters: a dictionary mapping parameter names to ModuleParameter
            instances describing the parameters
    '''

    def __init__(self, config):
        '''Initializes a ModuleMetadata instance.

        Args:
            config: a ModuleMetadataConfig instance

        Raises:
            ModuleMetadataError: if the module definition was invalid
        '''
        self.validate(config)

        self.info = None
        self.inputs = OrderedDict()
        self.outputs = OrderedDict()
        self.parameters = OrderedDict()

        self._parse_metadata(config)

    def has_input(self, name):
        '''Returns True/False if the module has an input `name`.'''
        return name in self.inputs

    def has_output(self, name):
        '''Returns True/False if the module has an output `name`.'''
        return name in self.outputs

    def has_parameter(self, name):
        '''Returns True/False if the module has a parameter `name`.'''
        return name in self.parameters

    def is_valid_input(self, name, path):
        '''Returns True/False if `path` is a valid path for input `name`.'''
        return self.get_input(name).is_valid_path(path)

    def is_valid_output(self, name, path):
        '''Returns True/False if `path` is a valid path for output `name`.'''
        return self.get_output(name).is_valid_path(path)

    def is_valid_parameter(self, name, val):
        '''Returns True/False if `val` is a valid value for parameter
        `name`.
        '''
        return self.get_parameter(name).is_valid_value(val)

    def get_input(self, name):
        '''Returns the ModuleInput instance for input `name`.'''
        return self.inputs[name]

    def get_output(self, name):
        '''Returns the ModuleOutput instance for output `name`.'''
        return self.outputs[name]

    def get_parameter(self, name):
        '''Returns the ModuleParameter instance for parameter `name`.'''
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

        if not config.inputs:
            raise ModuleMetadataError(
                "Module '%s' must have at least one input" % self.info.name)
        for i in config.inputs:
            self._verify_uniqueness(i.name)
            self.inputs[i.name] = ModuleInput(i)

        if not config.outputs:
            raise ModuleMetadataError(
                "Module '%s' must have at least one output" % self.info.name)
        for o in config.outputs:
            self._verify_uniqueness(o.name)
            self.outputs[o.name] = ModuleOutput(o)

        for p in config.parameters:
            self._verify_uniqueness(p.name)
            self.parameters[p.name] = ModuleParameter(p)

    def _verify_uniqueness(self, name):
        is_duplicate = (
            name in self.inputs or
            name in self.outputs or
            name in self.parameters
        )
        if is_duplicate:
            raise ModuleMetadataError(
                "Duplicate field '%s' found for module '%s'" % (
                    name, self.info.name))


class ModuleMetadataError(Exception):
    pass
