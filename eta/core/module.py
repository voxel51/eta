"""
Core module infrastructure.

See `docs/modules_dev_guide.md` for detailed information about the design and
usage of ETA modules.

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

from collections import OrderedDict
from glob import glob
import logging
import os

import eta
from eta.core.config import Config, ConfigError, Configurable
from eta.core.diagram import HasBlockDiagram, BlockdiagModule
import eta.core.logging as etal
import eta.core.serial as etas
import eta.core.types as etat
import eta.core.utils as etau


logger = logging.getLogger(__name__)


def run(module_name, module_config_or_path):
    """Runs the specified module with the given config.

    This is a convenience function for running modules programmatically. This
    function is not used directly by pipelines when running modules, and, as
    such, it does not support providing a PipelineConfig instance to use.

    Args:
        module_name: the name of the module
        module_config_or_path: a ModuleConfig, a dict representation of one,
            or path to one on disk. If a config is provided in-memory, it is
            written to a temporary directory on disk while the module executes

    Returns:
        True/False whether the module completed successfully
    """
    if etau.is_str(module_config_or_path):
        # Found a path to a module config on disk
        return _run(module_name, module_config_or_path)

    # Found an in-memory module config
    with etau.TempDir() as d:
        module_config_path = os.path.join(d, "config.json")
        etas.write_json(module_config_or_path, module_config_path)
        return _run(module_name, module_config_path)


def _run(module_name, module_config_path):
    module_exe = find_exe(module_name)
    args = ["python", module_exe, module_config_path]
    return etau.call(args)


def load_all_metadata():
    """Loads all module metadata files.

    Assumes any JSON files in the `eta.config.module_dirs` directories are
    module metadata files.

    Returns:
        a dictionary mapping module names to ModuleMetadata instances

    Raises:
        ModuleMetadataError: if any of the module metadata files are invalid
    """
    return {k: _load_metadata(v) for k, v in iteritems(find_all_metadata())}


def load_metadata(module_name):
    """Loads the module metadata file for the module with the given name.

    Module metadata files must JSON files in one of the directories in
    `eta.config.module_dirs`.

    Args:
        module_name: the name of the module

    Returns:
        the ModuleMetadata instance for the given module

    Raises:
        ModuleMetadataError: if the module metadata file could not be found
            or was invalid
    """
    return _load_metadata(find_metadata(module_name))


def _load_metadata(config):
    metadata = ModuleMetadata.from_json(config)
    name = os.path.splitext(os.path.basename(config))[0]
    if metadata.info.name != name:
        raise ModuleMetadataError(
            "Name '%s' from ModuleMetadata must match module name '%s'"
            % (metadata.info.name, name)
        )

    return metadata


def find_all_metadata():
    """Finds all module metadata files.

    Assumes any JSON files in the `eta.config.module_dirs` directories are
    module metadata files. To load these files, use `load_all_metadata()`.

    Returns:
        a dictionary mapping module names to (absolute paths to) module
            metadata filenames
    """
    d = {}
    mdirs = etau.make_search_path(eta.config.module_dirs)
    for mdir in mdirs:
        for path in glob(os.path.join(mdir, "*.json")):
            name = os.path.splitext(os.path.basename(path))[0]
            if name not in d:
                d[name] = path
            else:
                logger.debug(
                    "Module '%s' already exists; ignoring %s", name, path
                )

    return d


def find_metadata(module_name):
    """Finds the module metadata file for the module with the given name.

    Module metadata files must be JSON files in one of the directories in
    `eta.config.module_dirs`.

    Args:
        module_name: the name of the module

    Returns:
        the (absolute) path to the module metadata file

    Raises:
        ModuleMetadataError: if the module metadata file could not be found
    """
    try:
        return find_all_metadata()[module_name]
    except KeyError:
        raise ModuleMetadataError("Could not find module '%s'" % module_name)


def find_exe(module_name=None, module_metadata=None):
    """Finds the executable for the given module.

    Exactly one keyword argument must be supplied.

    Args:
        module_name: the name of the module
        module_metadata: the ModuleMetadata instance for the module

    Returns:
        the (absolute) path to the module executable

    Raises:
        ModuleMetadataError: if the module executable could not be found
    """
    if module_metadata is None:
        module_metadata = load_metadata(module_name)
    meta_path = find_metadata(module_metadata.info.name)
    exe_path = os.path.join(
        os.path.dirname(meta_path), module_metadata.info.exe
    )
    if not os.path.isfile(exe_path):
        raise ModuleMetadataError(
            "Could not find module executable '%s'" % exe_path
        )

    return exe_path


#
# @todo should pass a PipelineConfig instance here, not just the path. The need
# to use PipelineConfig here is causing a circular import with
# eta.core.module, which suggests this is a bad design...
#
def setup(module_config, pipeline_config_path=None):
    """Perform module setup.

    If a pipeline config is provided, it overrides any applicable values in
    the module config.

    Args:
        module_config: a Config instance derived from BaseModuleConfig
        pipeline_config_path: an optional path to a PipelineConfig
    """
    if pipeline_config_path:
        # Load pipeline config
        from eta.core.pipeline import PipelineConfig

        pipeline_config = PipelineConfig.from_json(pipeline_config_path)

        # Inherit settings from pipeline
        module_config.base.eta_config.update(pipeline_config.eta_config)
        module_config.base.logging_config = pipeline_config.logging_config

    # Setup logging
    etal.custom_setup(module_config.base.logging_config)

    # Apply config settings
    eta.set_config_settings(**module_config.base.eta_config)


class BaseModuleConfig(Config):
    """Base module configuration class that defines common configuration
    fields that all modules must support.

    All fields defined here should provide default values.

    Attributes:
        base: an `eta.core.module.BaseModuleConfigSettings` instance defining
            module configuration parameters
    """

    def __init__(self, d):
        self.base = self.parse_object(
            d, "base", BaseModuleConfigSettings, default=None
        )
        if self.base is None:
            self.base = BaseModuleConfigSettings.default()


class BaseModuleConfigSettings(Config):
    """Base module configuration settings that all modules must support.

    All fields defined here should provide default values.

    Attributes:
        eta_config: a dictionary defining custom ETA config settings to apply
            before running the module
        logging_config: an `eta.core.logging.LoggingConfig` instance defining
            the logging configuration settings for the module
    """

    def __init__(self, d):
        self.eta_config = self.parse_dict(d, "eta_config", default={})
        self.logging_config = self.parse_object(
            d,
            "logging_config",
            etal.LoggingConfig,
            default=etal.LoggingConfig.default(),
        )


class ModuleMetadataConfig(Config):
    """Module metadata configuration class."""

    def __init__(self, d):
        self.info = self.parse_object(d, "info", ModuleInfoConfig)
        self.inputs = self.parse_object_array(d, "inputs", ModuleInputConfig)
        self.outputs = self.parse_object_array(
            d, "outputs", ModuleOutputConfig
        )
        self.parameters = self.parse_object_array(
            d, "parameters", ModuleParameterConfig
        )

    def attributes(self):
        return ["info", "inputs", "outputs", "parameters"]


class ModuleInfoConfig(Config):
    """Module info configuration class."""

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.version = self.parse_string(d, "version")
        self.description = self.parse_string(d, "description")
        self.exe = self.parse_string(d, "exe")

    def attributes(self):
        return ["name", "type", "version", "description", "exe"]


class ModuleInputConfig(Config):
    """Module input descriptor configuration."""

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.description = self.parse_string(d, "description")
        self.required = self.parse_bool(d, "required", default=True)

    def attributes(self):
        return ["name", "type", "description", "required"]


class ModuleOutputConfig(Config):
    """Module output descriptor configuration."""

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.description = self.parse_string(d, "description")
        self.required = self.parse_bool(d, "required", default=True)

    def attributes(self):
        return ["name", "type", "description", "required"]


class ModuleParameterConfig(Config):
    """Module parameter descriptor configuration."""

    def __init__(self, d):
        self.name = self.parse_string(d, "name")
        self.type = self.parse_string(d, "type")
        self.description = self.parse_string(d, "description")
        self.required = self.parse_bool(d, "required", default=True)
        if not self.required:
            self.default = self.parse_raw(d, "default")
        elif "default" in d:
            raise ConfigError(
                "Module parameter '%s' is required, so it should not have a "
                "default value" % self.name
            )

    def attributes(self):
        attrs = ["name", "type", "description", "required"]
        if not self.required:
            attrs.append("default")

        return attrs


class ModuleInfo(Configurable):
    """Module info descriptor.

    Attributes:
        name: the name of the module
        type: the eta.core.types.Type of the module
        version: the version of the module
        description: a free text description of the module
        exe: the executable for the module
    """

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
                "'%s' is not a valid module type" % type_
            )
        return type_


class ModuleInput(Configurable):
    """Module input descriptor.

    Module inputs must be subclasses of eta.core.types.Data.

    Attributes:
        name: the name of the input
        type: the eta.core.types.Type of the input
        description: a free text description of the input
        required: whether the input is required
    """

    def __init__(self, config):
        """Creates a new ModuleInput instance.

        Args:
            config: a ModuleInputConfig instance

        Raises:
        """
        self.validate(config)

        self.name = config.name
        self.type = self._parse_type(config.type)
        self.description = config.description
        self.required = config.required

    def is_valid_path(self, path):
        """Returns True/False indicating whether the given path is a valid
        setting for this input."""
        return self.type.is_valid_path(path)

    @property
    def is_required(self):
        """Returns True/False if this input is required."""
        return self.required

    def _parse_type(self, type_str):
        type_ = etat.parse_type(type_str)
        if not etat.is_data(type_):
            raise ModuleMetadataError(
                (
                    "Module input '%s' has type '%s' but must be a subclass "
                    "of Data"
                )
                % (self.name, type_)
            )
        return type_


class ModuleOutput(Configurable):
    """Module output descriptor.

    Module outputs must be subclasses of eta.core.types.ConcreteData.

    Attributes:
        name: the name of the output
        type: the eta.core.types.Type of the output
        description: a free text description of the output
        required: whether the output is required
    """

    def __init__(self, config):
        """Creates a new ModuleOutput instance.

        Args:
            config: a ModuleOutputConfig instance

        Raises:
        """
        self.validate(config)

        self.name = config.name
        self.type = self._parse_type(config.type)
        self.description = config.description
        self.required = config.required

    def is_valid_path(self, path):
        """Returns True/False indicating whether the given path is a valid
        setting for this output."""
        return self.type.is_valid_path(path)

    @property
    def is_required(self):
        """Returns True/False if this output is required."""
        return self.required

    def _parse_type(self, type_str):
        type_ = etat.parse_type(type_str)
        if not etat.is_concrete_data(type_):
            raise ModuleMetadataError(
                (
                    "Module output '%s' has type '%s' but must be a subclass "
                    "of ConcreteData"
                )
                % (self.name, type_)
            )
        return type_


class ModuleParameter(Configurable):
    """Module parameter descriptor.

    Module parameters must be subclasses of eta.core.types.Builtin or
    eta.core.types.ConcreteData.

    Attributes:
        name: the name of the parameter
        type: the eta.core.types.Type of the parameter
        description: a free text description of the parameter
        required: whether the parameter is required
    """

    def __init__(self, config):
        self.validate(config)

        self.name = config.name
        self.type = self._parse_type(config.name, config.type)
        self.description = config.description
        self.required = config.required
        if not self.required:
            self._default = config.default
            self._validate_default()

    def is_valid_value(self, val):
        """Returns True/False indicating whether the given value is a valid
        setting for this parameter."""
        if self.is_builtin:
            return self.type.is_valid_value(val)
        return self.type.is_valid_path(val)

    @property
    def is_required(self):
        """Returns True/False if this parameter is required."""
        return self.required

    @property
    def is_builtin(self):
        """Returns True/False if this parameter is a Builtin."""
        return etat.is_builtin(self.type)

    @property
    def is_data(self):
        """Returns True/False if this parameter is Data."""
        return etat.is_data(self.type)

    @property
    def default_value(self):
        """Gets the default value for this parameter."""
        if self.is_required:
            raise ModuleMetadataError(
                "Module parameter '%s' is required, so it has no default "
                "value" % self.name
            )
        return self._default

    @staticmethod
    def _parse_type(name, type_str):
        type_ = etat.parse_type(type_str)
        if not etat.is_builtin(type_) and not etat.is_concrete_data(type_):
            raise ModuleMetadataError(
                "Module parameter '%s' has type '%s' but must be a subclass "
                "of Builtin or ConcreteData" % (name, type_)
            )
        return type_

    def _validate_default(self):
        if self._default is None:
            # We always allow None, which implies that the module can function
            # without this parameter being set to a valid typed value
            is_valid = True
        elif self.is_builtin:
            is_valid = self.type.is_valid_value(self._default)
        else:
            is_valid = self.type.is_valid_path(self._default)
        if not is_valid:
            raise ModuleMetadataError(
                "Default value '%s' is invalid for module parameter '%s' of "
                "'%s'" % (self._default, self.name, self.type)
            )


class ModuleMetadata(Configurable, HasBlockDiagram):
    """Class that encapsulates the architecture of a module.

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
    """

    def __init__(self, config):
        """Initializes a ModuleMetadata instance.

        Args:
            config: a ModuleMetadataConfig instance

        Raises:
            ModuleMetadataError: if the module definition was invalid
        """
        self.validate(config)
        self.config = config

        self.info = None
        self.inputs = OrderedDict()
        self.outputs = OrderedDict()
        self.parameters = OrderedDict()

        self._parse_metadata(config)

    def has_input(self, name):
        """Returns True/False if the module has an input `name`."""
        return name in self.inputs

    def has_output(self, name):
        """Returns True/False if the module has an output `name`."""
        return name in self.outputs

    def has_parameter(self, name):
        """Returns True/False if the module has a parameter `name`."""
        return name in self.parameters

    def is_valid_input(self, name, path):
        """Returns True/False if `path` is a valid path for input `name`."""
        return self.get_input(name).is_valid_path(path)

    def is_valid_output(self, name, path):
        """Returns True/False if `path` is a valid path for output `name`."""
        return self.get_output(name).is_valid_path(path)

    def is_valid_parameter(self, name, val):
        """Returns True/False if `val` is a valid value for parameter
        `name`.
        """
        return self.get_parameter(name).is_valid_value(val)

    def get_input(self, name):
        """Returns the ModuleInput instance for input `name`."""
        return self.inputs[name]

    def get_output(self, name):
        """Returns the ModuleOutput instance for output `name`."""
        return self.outputs[name]

    def get_parameter(self, name):
        """Returns the ModuleParameter instance for parameter `name`."""
        return self.parameters[name]

    def to_blockdiag(self):
        """Returns a BlockdiagModule representation of this module."""
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
                "Module '%s' must have at least one input" % self.info.name
            )
        for i in config.inputs:
            self._verify_uniqueness(i.name)
            self.inputs[i.name] = ModuleInput(i)

        if not config.outputs:
            raise ModuleMetadataError(
                "Module '%s' must have at least one output" % self.info.name
            )
        for o in config.outputs:
            self._verify_uniqueness(o.name)
            self.outputs[o.name] = ModuleOutput(o)

        for p in config.parameters:
            self._verify_uniqueness(p.name)
            self.parameters[p.name] = ModuleParameter(p)

    def _verify_uniqueness(self, name):
        if name == self.info.name:
            raise ModuleMetadataError(
                "Module '%s' cannot have a field with the same name" % name
            )

        is_duplicate = (
            name in self.inputs
            or name in self.outputs
            or name in self.parameters
        )
        if is_duplicate:
            raise ModuleMetadataError(
                "Module '%s' cannot have duplicate field '%s'"
                % (self.info.name, name)
            )


class ModuleMetadataError(Exception):
    """Exception raised when an invalid module metadata file is encountered."""

    pass
