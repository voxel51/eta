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
import six
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import numbers
import sys

from eta.core.serial import Serializable
import eta.core.utils as ut


class Configurable(object):
    '''Base class for classes that can be initialized with a Config instance.

    Enforces the convention that configurable class `Foo` has configuration
    class `FooConfig`.
    '''

    @classmethod
    def from_json(cls, json_path):
        '''Encapsulates the common behavior of loading a configuration from a
        JSON file and then instantiating the class.

        Args:
            json_path: path to a JSON config of type <cls>Config

        Returns:
            an instance of cls instantiated from the config
        '''
        cls_, config_cls = Configurable.parse(
            cls.__name__, module_name=cls.__module__)
        assert cls == cls_, "Expected %s, found %s" % (cls, cls_)
        config = config_cls.from_json(json_path)
        return cls(config)

    @classmethod
    def validate(cls, config):
        '''Validates that the config instance is of the correct type.

        Raises:
            ConfigurableError: if config is not an instance of <cls>Config
        '''
        actual = config.__class__.__name__
        expected = cls.__name__ + "Config"
        if expected != actual:
            raise ConfigurableError(actual, expected)

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

        cls = ut.get_class(class_name, module_name=module_name)
        config_cls = ut.get_class(
            class_name + "Config", module_name=module_name)
        return cls, config_cls


class ConfigurableError(Exception):
    def __init__(self, actual, expected):
        message = "Found config '%s', expected '%s'" % (actual, expected)
        super(ConfigurableError, self).__init__(message)


# This exists so that None can be a default value for Config fields
class no_default(object):
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
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        val, found = Config._parse_key(d, key, dict, default=default)
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
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        val, found = Config._parse_key(d, key, list, default=default)
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
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        return Config._parse_key(d, key, list, default=default)[0]

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
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        val = Config._parse_key(d, key, six.string_types, default=default)[0]
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
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        return Config._parse_key(d, key, numbers.Number, default=default)[0]

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
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        return Config._parse_key(d, key, bool, default=default)[0]

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
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        return Config._parse_key(d, key, None, default=default)[0]

    @staticmethod
    def _parse_key(d, key, t, default=no_default):
        if key in d:
            if t is None or isinstance(d[key], t):
                return d[key], True
            raise ConfigError(
                "Expected key '%s' of %s; found %s" % (key, t, type(d[key])))
        elif default is not no_default:
            return default, False
        raise ConfigError("Expected key '%s' of %s" % (key, t))


class ConfigError(Exception):
    pass
