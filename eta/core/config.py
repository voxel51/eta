'''
Core tools for defining, reading and writing configuration files.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import numbers
import sys

from eta.core.serial import Serializable


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
        cls_, config_cls = Configurable.parse(cls.__module__, cls.__name__)
        assert cls == cls_, "Expected %s, found %s" % (cls, cls_)
        config = config_cls.from_json(json_path)
        return cls(config)

    @classmethod
    def validate(cls, config):
        '''Validate that the config instance is of the correct type.

        Raises:
            ConfigurableError: if config is not an instance of <cls>Config
        '''
        actual = config.__class__.__name__
        expected = cls.__name__ + "Config"
        if expected != actual:
            raise ConfigurableError(actual, expected)

    @staticmethod
    def parse(module_name, class_name):
        '''Parse a Configurable subclass name string.

        Args:
            module_name: a string. The name of the module (usually, __name__)
                containing the Configurable subclass and its associated Config
                subclass
            class_name: a string. The name of the Configurable subclass

        Returns:
            cls: the Configurable subclass
            config_cls: the Config subclass associated with cls
        '''
        cls = getattr(sys.modules[module_name], class_name)
        config_cls = getattr(sys.modules[module_name], class_name + "Config")
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
    def from_dict(cls, d):
        '''Constructs a Config object from a JSON dictionary.

        Config subclass constructors accept JSON dictionaries, so this method
        simply passes the dictionary to cls()

        Args:
            d: a JSON dictionary containing the fields expected by cls

        Returns:
            an instance of cls
        '''
        return cls(d)

    @staticmethod
    def parse_object(d, key, cls, default=no_default):
        '''Parses an object attribute.

        Normally, if the key is not present, the default value should be a JSON
        dictionary, which is passed to cls(). However, in the special case when
        default=None, None is returned directly.

        Args:
            d: a JSON dictionary
            key: the key to parse
            cls: the class of the d[key]
            default: default value if key is not present

        Returns:
            an instance of cls

        Raises:
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        val = Config._parse_key(d, key, dict, default=default)
        return cls(val) if val is not None else None

    @staticmethod
    def parse_object_array(d, key, cls, default=no_default):
        '''Parses an array of objects.

        Args:
            d: a JSON dictionary
            key: the key to parse
            cls: the class of the elements of list d[key]
            default: default value if key is not present

        Returns:
            a list of cls instances

        Raises:
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        objects = Config._parse_key(d, key, list, default=default)
        return [cls(obj) for obj in objects]

    @staticmethod
    def parse_array(d, key, default=no_default):
        '''Parses a raw array attribute.

        Args:
            d: a JSON dictionary
            key: the key to parse
            default: default value if key is not present

        Returns:
            a list (e.g., of strings from the raw JSON dictionary value)

        Raises:
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        return Config._parse_key(d, key, list, default=default)

    @staticmethod
    def parse_string(d, key, default=no_default):
        '''Parses a string attribute.

        Args:
            d: a JSON dictionary
            key: the key to parse
            default: default value if key is not present

        Returns:
            a string

        Raises:
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        return Config._parse_key(d, key, str, default=default)

    @staticmethod
    def parse_number(d, key, default=no_default):
        '''Parses a number attribute.

        Args:
            d: a JSON dictionary
            key: the key to parse
            default: default value if key is not present

        Returns:
            a number (e.g. int, float)

        Raises:
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        return Config._parse_key(d, key, numbers.Number, default=default)

    @staticmethod
    def parse_bool(d, key, default=no_default):
        '''Parses a boolean value.

        Args:
            d: a JSON dictionary
            key: the key to parse
            default: default value if key is not present

        Returns:
            True/False

        Raises:
            ConfigError: if no default value was provided and the key was
                not present in the dictionary.
        '''
        return Config._parse_key(d, key, bool, default=default)

    @staticmethod
    def _parse_key(d, key, t, default=no_default):
        if key in d and isinstance(d[key], t):
            return d[key]
        elif default is not no_default:
            return default
        else:
            raise ConfigError("Expected key '%s' of %s" % (key, str(t)))


class ConfigError(Exception):
    pass
