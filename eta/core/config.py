'''
Core tools for defining, reading and writing configuration files.

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
import six
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import numbers
import os
import sys

import eta.core.serial as etas
from eta.core.serial import Serializable
import eta.core.utils as etau


class NoDefault(object):
    '''A placeholder class that is typically used to distinguish between an
    argument that has _no default value_ and an argument that has the default
    value `None`.
    '''

    def __bool__(self):
        '''NoDefault instances always evaluate to False.'''
        return False


# A singleton NoDefault value that should be used whenever one needs to allow
# `None` to be a default value
no_default = NoDefault()


class Configurable(object):
    '''Base class for classes that can be initialized with a Config instance.

    Enforces the convention that configurable class `Foo` has configuration
    class `FooConfig`.
    '''

    @classmethod
    def from_json(cls, json_path):
        '''Encapsulates the common behavior of instantiating a Configurable
        class from a Config JSON file.

        Args:
            json_path: path to a JSON config of type <cls>Config

        Returns:
            an instance of cls instantiated from the config
        '''
        config_cls = Configurable.parse(
            cls.__name__, module_name=cls.__module__)[1]
        return cls(config_cls.from_json(json_path))

    @classmethod
    def from_dict(cls, d):
        '''Encapsulates the common behavior of instantiating a Configurable
        class from a Config JSON dictionary.

        Args:
            d: a JSON dictionary of type <cls>Config

        Returns:
            an instance of cls instantiated from the config
        '''
        config_cls = Configurable.parse(
            cls.__name__, module_name=cls.__module__)[1]
        return cls(config_cls.from_dict(d))

    @classmethod
    def validate(cls, config):
        '''Validates that the config instance is of the correct type.

        Raises:
            ConfigurableError: if config is not an instance of <cls>Config
        '''
        actual = config.__class__.__name__
        expected = cls.__name__ + "Config"
        if expected != actual:
            raise ConfigurableError(
                "Found Config '%s'; expected '%s'" % (actual, expected))

    @staticmethod
    def parse(class_name, module_name=None):
        '''Parses a Configurable subclass name string.

        Assumes both the Configurable class and the Config class are defined
        in the same module. The module containing the classes will be loaded
        if necessary.

        Args:
            class_name: a string containing the name of the Configurable class,
                e.g. "ClassName", or a fully-qualified class name, e.g.
                "eta.core.config.ClassName"
            module_name: a string containing the fully-qualified module name,
                e.g. "eta.core.config", or None if class_name includes the
                module name. Set module_name = __name__ to load a class from
                the calling module

        Returns:
            cls: the Configurable class
            config_cls: the Config class associated with cls
        '''
        if module_name is None:
            module_name, class_name = class_name.rsplit(".", 1)

        cls = etau.get_class(class_name, module_name=module_name)
        config_cls = etau.get_class(
            class_name + "Config", module_name=module_name)
        return cls, config_cls


class ConfigurableError(Exception):
    '''Exception raised when an invalid Configurable is encountered.'''
    pass


class ConfigBuilder(Serializable):
    '''A class for building Config instances programmatically.'''

    def __init__(self, cls):
        '''Creates a ConfigBuilder instance for the given class.

        Args:
            cls: the Config subclass to build.
        '''
        self._cls = cls
        self._attributes = []
        self._is_validated = False

    def set(self, **kwargs):
        '''Sets the given attributes.

        Args:
            **kwargs: a dictionary of attributes and values to set

        Returns:
            the ConfigBuilder instance
        '''
        for k, v in iteritems(kwargs):
            setattr(self, k, v)
            self._attributes.append(k)
        self._is_validated = False
        return self

    def validate(self):
        '''Validates that the ConfigBuilder instance is ready to be built or
        serialized.

        Returns:
            the ConfigBuilder instance

        Raises:
            ConfigError: if the required attributes were not provided to build
                the specified Config subclass
        '''
        self.build()
        self._is_validated = True
        return self

    def build(self):
        '''Builds the Config subclass instance from this builder.

        Returns:
            the Config subclass instance

        Raises:
            ConfigError: if the required attributes were not provided to build
                the specified Config subclass
        '''
        return self._cls.from_dict(self._serialize())

    def serialize(self):
        '''Serializes the ConfigBuilder into a dictionary.

        Raises:
            ConfigBuilderError: if the builder has not been validated
        '''
        if not self._is_validated:
            raise ConfigBuilderError(
                "Must call validate() before serializing a ConfigBuilder")

        return self._serialize()

    def _serialize(self):
        return super(ConfigBuilder, self).serialize(
            attributes=self._attributes)

    @classmethod
    def from_json(cls, *args, **kwargs):
        raise NotImplementedError("ConfigBuilders cannot be read from JSON")


class ConfigBuilderError(Exception):
    '''Exception raised when an invalid ConfigBuilder action is taken.'''
    pass


class Config(Serializable):
    '''Base class for reading JSON configuration files.

    Config subclasses should implement constructors that take a JSON dictionary
    as input and parse the desired fields using the static methods defined by
    this class.
    '''

    @classmethod
    def load_default(cls):
        '''Loads the default config instance from file.

        Subclasses must implement this method if they intend to support
        default instances.
        '''
        raise NotImplementedError("subclass must implement load_default()")

    @classmethod
    def default(cls):
        '''Returns the default config instance.

        By default, this method instantiates the class from an empty
        dictionary, which will only succeed if all attributes are optional.
        Otherwise, subclasses should override this method to provide the
        desired default configuration.
        '''
        return cls({})

    @classmethod
    def from_dict(cls, d):
        '''Constructs a Config object from a JSON dictionary.

        Config subclass constructors accept JSON dictionaries, so this method
        simply passes the dictionary to cls().

        Args:
            d: a JSON dictionary containing the fields expected by cls

        Returns:
            an instance of cls
        '''
        return cls(d)

    @classmethod
    def builder(cls):
        '''Returns a ConfigBuilder instance for this class.'''
        return ConfigBuilder(cls)

    @classmethod
    def from_kwargs(cls, **kwargs):
        '''Constructs a Config object from keyword arguments.

        Args:
            **kwargs: keyword arguments that define the fields expected by cls

        Returns:
            an instance of cls
        '''
        return cls(kwargs)

    @staticmethod
    def parse_object(d, key, cls, default=no_default):
        '''Parses an object attribute.

        Args:
            d: a JSON dictionary
            key: the key to parse
            cls: the class of the d[key]
            default: a default value to return if key is not present

        Returns:
            an instance of cls

        Raises:
            ConfigError: if the field value was the wrong type or no default
                value was provided and the key was not found in the dictionary
        '''
        val, found = _parse_key(d, key, dict, default)
        return cls(val) if found else val

    @staticmethod
    def parse_object_array(d, key, cls, default=no_default):
        '''Parses an array of objects.

        Args:
            d: a JSON dictionary
            key: the key to parse
            cls: the class of the elements of list d[key]
            default: a default value to return if key is not present

        Returns:
            a list of cls instances

        Raises:
            ConfigError: if the field value was the wrong type or no default
                value was provided and the key was not found in the dictionary
        '''
        val, found = _parse_key(d, key, list, default)
        return [cls(obj) for obj in val] if found else val

    @staticmethod
    def parse_array(d, key, default=no_default):
        '''Parses a raw array attribute.

        Args:
            d: a JSON dictionary
            key: the key to parse
            default: a default value to return if key is not present

        Returns:
            a list (e.g., of strings from the raw JSON dictionary value)

        Raises:
            ConfigError: if the field value was the wrong type or no default
                value was provided and the key was not found in the dictionary
        '''
        return _parse_key(d, key, list, default)[0]

    @staticmethod
    def parse_dict(d, key, default=no_default):
        '''Parses an dictionary attribute.

        Args:
            d: a JSON dictionary
            key: the key to parse
            default: a default value to return if key is not present

        Returns:
            a dictionary

        Raises:
            ConfigError: if the field value was the wrong type or no default
                value was provided and the key was not found in the dictionary
        '''
        return _parse_key(d, key, dict, default)[0]

    @staticmethod
    def parse_object_dict(d, key, cls, default=no_default):
        '''Parses a dictionary whose values are objects.

        Args:
            d: a JSON dictionary
            key: the key to parse
            cls: the class of the values of dictionary d[key]
            default: a default value to return if key is not present

        Returns:
            a dictionary whose values are cls instances

        Raises:
            ConfigError: if the field value was the wrong type or no default
                value was provided and the key was not found in the dictionary
        '''
        val, found = _parse_key(d, key, dict, default)
        return {k: cls(v) for k, v in iteritems(val)} if found else val

    @staticmethod
    def parse_string(d, key, default=no_default):
        '''Parses a string attribute.

        Args:
            d: a JSON dictionary
            key: the key to parse
            default: a default value to return if key is not present

        Returns:
            a string

        Raises:
            ConfigError: if the field value was the wrong type or no default
                value was provided and the key was not found in the dictionary
        '''
        val = _parse_key(d, key, six.string_types, default)[0]
        return str(val) if val is not None else val

    @staticmethod
    def parse_number(d, key, default=no_default):
        '''Parses a number attribute.

        Args:
            d: a JSON dictionary
            key: the key to parse
            default: a default value to return if key is not present

        Returns:
            a number (e.g. int, float)

        Raises:
            ConfigError: if the field value was the wrong type or no default
                value was provided and the key was not found in the dictionary
        '''
        return _parse_key(d, key, numbers.Number, default)[0]

    @staticmethod
    def parse_bool(d, key, default=no_default):
        '''Parses a boolean value.

        Args:
            d: a JSON dictionary
            key: the key to parse
            default: a default value to return if key is not present

        Returns:
            True/False

        Raises:
            ConfigError: if the field value was the wrong type or no default
                value was provided and the key was not found in the dictionary
        '''
        return _parse_key(d, key, bool, default)[0]

    @staticmethod
    def parse_raw(d, key, default=no_default):
        '''Parses a raw (arbitrary) JSON field.

        Args:
            d: a JSON dictionary
            key: the key to parse
            default: a default value to return if key is not present

        Returns:
            the raw (untouched) value of the given field

        Raises:
            ConfigError: if no default value was provided and the key was not
                found in the dictionary
        '''
        return _parse_key(d, key, None, default)[0]


class ConfigContainer(etas.Container):
    '''Abstract base class for containers that store lists of `Config` class
    instances.

    This class cannot be instantiated directly. Instead a subclass should
    be created for each type of data to be stored. Subclasses MUST set the
    following members:
        -  `_ELE_CLS`: the class of the element stored in the container

    In addition, sublasses MAY override the following members:
        - `_ELE_CLS_FIELD`: the name of the private attribute that will store
            the class of the elements in the container
        - `_ELE_ATTR`: the name of the attribute that will store the elements
            in the container

    ConfigContainer subclasses embed their class names and underlying data
    instance class names in their JSON representations, so they can be read
    reflectively from disk.

    Attributes:
        <configs>: a list of Config instances. The field name <configs> is
            specified by the `_ELE_ATTR` member of the ConfigContainer
            subclass, and the class of the Config instances is specified by the
            `_ELE_CLS` member
    '''

    #
    # The class of the configs stored in the container
    #
    # Subclasses MUST set this field
    #
    _ELE_CLS = None

    #
    # The name of the private attribute that will store the class of the
    # configs in the container
    #
    # Subclasses MAY override this field
    #
    _ELE_CLS_FIELD = "_CONFIG_CLS"

    #
    # The name of the attribute that will store the configs in the container
    #
    # Subclasses MAY override this field
    #
    _ELE_ATTR = "configs"

    @property
    def _config(self):
        '''The list of Config instances stored in this container, independent
        of the container-specific name of the attribute in which they are
        stored.
        '''
        return self.__elements__

    @classmethod
    def get_config_class(cls):
        '''Gets the class of Config stored in this container.'''
        return cls._ELE_CLS

    @classmethod
    def get_config_class_name(cls):
        '''Returns the fully-qualified class name string of the Config
        instances in this container.
        '''
        return etau.get_class_name(cls._ELE_CLS)

    @classmethod
    def _validate(cls):
        '''Adds a validation to all subclasses that enforces only Config's can
        be stored in this container.
        '''
        super(ConfigContainer, cls)._validate()

        if not issubclass(cls._ELE_CLS, Config):
            raise etas.ContainerError(
                "_ELE_CLS for a ConfigContainer does not inherit from Config")


class ConfigError(Exception):
    '''Exception raised when an invalid Config instance is encountered.'''
    pass


class EnvConfig(Serializable):
    '''Base class for reading JSON configuration files whose values can be
    specified or overridden via environment variables.

    EnvConfig subclasses should implement constructors that take a possibly
    empty JSON dictionary as input and parse the desired fields using the
    static methods defined by this class.
    '''

    @staticmethod
    def parse_string_array(d, key, env_var=None, default=no_default):
        '''Parses a string array attribute.

        Args:
            d: a JSON dictionary
            key: the key to parse
            env_var: an optional environment variable to load the attribute
                from rather than using the JSON dictionary
            default: an optional default value to return if key is not present

        Returns:
            a list of strings

        Raises:
            EnvConfigError: if the environment variable, the dictionary key, or
                a default value was not provided.
        '''
        return _parse_env_var_or_key(d, key, list, env_var, str, True, default)

    @staticmethod
    def parse_string(d, key, env_var=None, default=no_default):
        '''Parses a string attribute.

        Args:
            d: a JSON dictionary
            key: the key to parse
            env_var: an optional environment variable to load the attribute
                from rather than using the JSON dictionary
            default: an optional default value to return if key is not present

        Returns:
            a string

        Raises:
            EnvConfigError: if the environment variable, the dictionary key, or
                a default value was not provided.
        '''
        val = _parse_env_var_or_key(
            d, key, six.string_types, env_var, str, False, default)
        return str(val) if val is not None else val

    @staticmethod
    def parse_number(d, key, env_var=None, default=no_default):
        '''Parses a number attribute.

        Args:
            d: a JSON dictionary
            key: the key to parse
            env_var: an optional environment variable to load the attribute
                from rather than using the JSON dictionary
            default: an optional default value to return if key is not present

        Returns:
            a number (e.g. int, float)

        Raises:
            EnvConfigError: if the environment variable, the dictionary key, or
                a default value was not provided.
        '''
        return _parse_env_var_or_key(
            d, key, numbers.Number, env_var, float, False, default)

    @staticmethod
    def parse_bool(d, key, env_var=None, default=no_default):
        '''Parses a boolean value.

        Args:
            d: a JSON dictionary
            key: the key to parse
            env_var: an optional environment variable to load the attribute
                from rather than using the JSON dictionary
            default: a default value to return if key is not present

        Returns:
            True/False

        Raises:
            EnvConfigError: if the environment variable, the dictionary key, or
                a default value was not provided.
        '''
        env_t = lambda v: str(v).lower() in ("yes", "true", "1")
        return _parse_env_var_or_key(
            d, key, bool, env_var, env_t, False, default)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an EnvConfig object from a JSON dictionary.

        EnvConfig subclass constructors accept JSON dictionaries, so this
        method simply passes the dictionary to cls().

        Args:
            d: a JSON dictionary containing the fields expected by cls

        Returns:
            an instance of cls
        '''
        return cls(d)

    @classmethod
    def from_json(cls, path):
        '''Constructs an EnvConfig object from a JSON file.

        EnvConfig instances allow their values to be overriden by environment
        variables, so, if the JSON file does not exist, this method silently
        loads an empty dictionary in its place.
        '''
        d = etas.read_json(path) if os.path.isfile(path) else {}
        return cls.from_dict(d)


class EnvConfigError(Exception):
    '''Exception raised when an invalid EnvConfig instance is encountered.'''
    pass


def _parse_key(d, key, t, default):
    if key in d:
        val = d[key]
        if t is None or isinstance(val, t):
            # Return provided value
            return val, True

        if val is not None:
            raise ConfigError(
                "Expected key '%s' of %s; found %s" % (key, t, type(val)))

    if default is not no_default:
        # Return default value
        return default, False

    raise ConfigError("Expected key '%s' of %s" % (key, t))


def _parse_env_var_or_key(d, key, t, env_var, env_t, sep, default):
    val = os.environ.get(env_var)
    if val:
        # Return value(s) from environment variable
        try:
            return [env_t(vi) for vi in val.split(":")] if sep else env_t(val)
        except ValueError:
            raise EnvConfigError(
                "Failed to parse environment variable '%s' using %s",
                env_var, env_t)

    if key in d:
        val = d[key]
        if t is None or isinstance(val, t):
            # Return provided value
            return val

        if val is not None:
            raise EnvConfigError(
                "Expected key '%s' of %s; found %s" % (key, t, type(val)))

    if default is not no_default:
        # Return default value
        return default

    raise EnvConfigError(
        "Expected environment variable '%s' or key '%s'" % (env_var, key))
