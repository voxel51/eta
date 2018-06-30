'''
Core data structures for working with data that can be read/written to disk.

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

from collections import OrderedDict
import datetime as dt
import dill as pickle
import json
import os

import numpy as np

import eta.core.utils as etau


def load_json(path_or_str):
    '''Loads JSON from argument.

    Args:
        path_or_str: can either be the path to a JSON file or a JSON string

    Returns:
        the JSON dictionary
    '''
    if os.path.isfile(path_or_str):
        return read_json(path_or_str)
    return json.loads(path_or_str)


def read_json(path):
    '''Reads JSON from file.'''
    with open(path, "rt") as f:
        return json.load(f)


def write_json(obj, path, pretty_print=True):
    '''Writes JSON object to file, creating the output directory if necessary.

    Args:
        obj: is either an object that can be directly dumped to a JSON file or
            an instance of a subclass of Serializable
        path: the output path
        pretty_print: when True (default), the resulting JSON will be outputted
            to be human readable; when False, it will be compact with no
            extra spaces or newline characters
    '''
    if is_serializable(obj):
        obj = obj.serialize()
    s = json_to_str(obj, pretty_print=pretty_print)

    etau.ensure_basedir(path)
    with open(path, "wt") as f:
        f.write(s)


def json_to_str(obj, pretty_print=True):
    '''Converts the JSON object to a string.

    Args:
        obj: a JSON dictionary or an instance of a Serializable subclass
        pretty_print: when True (default), the string will be formatted to be
            human readable; when False, it will be compact with no extra spaces
            or newline characters
    '''
    if is_serializable(obj):
        obj = obj.serialize()
    kwargs = {"indent": 4} if pretty_print else {}
    s = json.dumps(
        obj, separators=(",", ": "), cls=EtaJSONEncoder, ensure_ascii=False,
        **kwargs
    )
    return str(s)


class Picklable(object):
    '''Mixin class for objects that can be pickled.'''

    def pickle(self, path):
        '''Saves the instance to disk in a pickle. '''
        etau.ensure_basedir(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        '''Loads the pickle from disk and returns the instance.'''
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def is_pickle_path(path):
        '''Checks the path to see if it has a pickle extension.'''
        return path.endswith(".pkl")


class Serializable(object):
    '''Base class for objects that can be serialized.

    Subclasses must implement from_dict(), which defines how to construct a
    serializable object from a JSON dictionary.
    '''

    def __str__(self):
        '''Returns the string representation of this object as it would be
        written to JSON.'''
        return self.to_str()

    def serialize(self, attributes=None):
        '''Serializes the object into a dictionary.

        Args:
            attributes: an optional list of attributes to serialize. By default
                this list is obtained by calling self.attributes()

        Subclasses can override this method, but, by default, Serializable
        objects are serialized by:
            a) calling self.attributes()
            b) invoking serialize() on serializable values and leaving other
               values untouched
            c) applying b) element-wise on list values
        '''
        if attributes is None:
            attributes = self.attributes()
        return OrderedDict((a, _recurse(getattr(self, a))) for a in attributes)

    def attributes(self):
        '''Returns a list of class attributes to be serialized.

        Subclasses can override this method, but, by default, all attributes
        in vars(self) are returned, minus private attributes (those starting
        with "_").

        In particular, subclasses may choose to override this method if they
        want their JSON files to be organized in a particular way.
        '''
        return [a for a in vars(self) if not a.startswith("_")]

    def to_str(self, pretty_print=True):
        '''Returns the string representation of this object as it would be
        written to JSON.

        Args:
            pretty_print: if True (default), the string will be formatted to be
                human readable; when False, it will be compact with no extra
                spaces or newline characters
        '''
        return json_to_str(self, pretty_print=pretty_print)

    def write_json(self, path, pretty_print=True):
        '''Serializes the object and writes it to disk.

        Args:
            path: the output path
            pretty_print: when True (default), the resulting JSON will be
                outputted to be human readable; when False, it will be compact
                with no extra spaces or newline characters
        '''
        write_json(self.serialize(), path, pretty_print=pretty_print)

    @classmethod
    def from_dict(cls, d):
        '''Constructs a Serializable object from a JSON dictionary.

        Subclasses must implement this method if they intend to support being
        read from disk.
        '''
        raise NotImplementedError("subclass must implement from_dict()")

    @classmethod
    def from_json(cls, path):
        '''Constructs a Serializable object from a JSON file.

        Subclasses may override this method, but, by default, this method
        simply reads the JSON and calls from_dict(), which subclasses must
        implement.
        '''
        return cls.from_dict(read_json(path))


class Container(Serializable):
    '''Abstract base class for flexible containers that store lists of
    `Serializable` elements.

    Container subclasses embed their class names and underlying element class
    names in their JSON representations, so they can be read reflectively from
    disk.

    This class cannot be instantiated directly.

    This class currently has only two direct subclasses, which bifurcate the
    container implementation into two distinct categories:
        - `eta.core.data.DataContainer`: base class for containers that store
            lists of `Serializable` data instances
        - `eta.core.config.ConfigContainer`: base class for containers that
            store lists of `Config` instances

    See `DataContainer` and `ConfigContainer` for concrete usage examples.
    '''

    #
    # The name of the private attribute that will store the class
    #
    # Subclasses MAY override this field
    #
    _CLS_FIELD = "_CLS"

    #
    # The class of the element stored in the container
    #
    # Subclasses MUST set this field
    #
    _ELE_CLS = None

    #
    # The name of the private attribute that will store the class of the
    # elements in the container
    #
    # Subclasses MUST set this field
    #
    _ELE_CLS_FIELD = None

    #
    # The name of the attribute that will store the elements in the container
    #
    # Subclasses MUST set this field
    #
    _ELE_ATTR = None

    def __init__(self, **kwargs):
        '''Constructs a Container subclass.

        Args:
            <element>: an optional list of elements to store in the Container.
            The appropriate name of this keyword argument is determined by the
            `_ELE_ATTR` member of the Container subclass.

        Raises:
            ContainerError: if there was a problem parsing the input
        '''
        self._validate()

        if kwargs and self._ELE_ATTR not in kwargs:
            raise ContainerError(
                "Expected elements to be provided in keyword argument '%s'; "
                "found keys %s" % (self._ELE_ATTR, list(kwargs.keys())))
        elements = kwargs.get(self._ELE_ATTR, [])

        for e in elements:
            if not isinstance(e, self._ELE_CLS):
                raise ContainerError(
                    "Container %s expects elements of type %s but found "
                    "%s" % (self.__class__, self._ELE_CLS, e.__class__))

        setattr(self, self._ELE_ATTR, elements)

    def __getitem__(self, index):
        return self.__elements__.__getitem__(index)

    def __iter__(self):
        return iter(self.__elements__)

    @property
    def __elements__(self):
        return getattr(self, self._ELE_ATTR)

    @classmethod
    def get_class_name(cls):
        '''Returns the fully-qualified class name string of this container.'''
        return etau.get_class_name(cls)

    def add(self, instance):
        '''Adds an element to the container.

        Args:
            instance: an instance of `_ELE_CLS`
        '''
        self.__elements__.append(instance)

    @property
    def size(self):
        '''Returns the number of elements in the container.'''
        return len(self.__elements__)

    def count_matches(self, filters, match=any):
        '''Counts the number of elements in the Container that match the
        given filters.

        Args:
            filters: a list of functions that accept instances of class
                `_ELE_CLS`and return True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
        '''
        return self.get_matches(filters, match=match).size

    def get_matches(self, filters, match=any):
        '''Returns a container containing only instances that match the
        filters.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
        '''
        return self.__class__(**{
            self._ELE_ATTR:
            list(filter(
                lambda o: match(f(o) for f in filters), self.__elements__))
        })

    def attributes(self):
        '''Returns the list of class attributes that will be serialized by this
        Container.
        '''
        return [self._CLS_FIELD, self._ELE_CLS_FIELD, self._ELE_ATTR]

    def serialize(self):
        '''Serializes the object into a dictionary.

        Containers have a custom serialization implementation that embeds the
        class name and element class name into the JSON, which enables
        reflective parsing when reading from disk.
        '''
        d = OrderedDict()
        d[self._CLS_FIELD] = self.get_class_name()
        d[self._ELE_CLS_FIELD] = etau.get_class_name(self._ELE_CLS)
        d[self._ELE_ATTR] = [o.serialize() for o in self.__elements__]
        return d

    @classmethod
    def from_dict(cls, d):
        '''Constructs a Container from a JSON dictionary.

        If the dictionary has the `cls._CLS_FIELD` and `cls._ELE_CLS_FIELD`
        keys, they are used to infer the Container class and underlying element
        classes, respectively, and this method can be invoked on any
        `Container` subclass that has the same `_ELE_CLS_FIELD` setting.

        Otherwise, this method must be called on the same concrete `Container`
        subclass from which the JSON was generated.
        '''
        if cls._CLS_FIELD in d and cls._ELE_CLS_FIELD in d:
            # Parse reflectively
            cls = etau.get_class(d[cls._CLS_FIELD])
            ele_cls = etau.get_class(d[cls._ELE_CLS_FIELD])
        else:
            # Parse using provided class
            ele_cls = cls._ELE_CLS

        return cls(**{
            cls._ELE_ATTR: [ele_cls.from_dict(dd) for dd in d[cls._ELE_ATTR]]
        })

    @classmethod
    def _validate(cls):
        '''Validates that a concrete Container subclass definition is valid.'''
        if cls._ELE_CLS is None:
            raise ContainerError(
                "Cannot instantiate a Container for which _ELE_CLS is None")
        if cls._ELE_ATTR is None:
            raise ContainerError(
                "Cannot instantiate a Container for which _ELE_ATTR is None")
        if not issubclass(cls._ELE_CLS, Serializable):
            raise ContainerError(
                "%s is not Serializable" % cls._ELE_CLS)


class ContainerError(Exception):
    '''Exception raised when an invalid Container is encountered.'''
    pass


def is_serializable(obj):
    '''Returns True if the object is serializable (i.e., implements the
    Serializable interface) and False otherwise.
    '''
    return isinstance(obj, Serializable)


def _recurse(v):
    if isinstance(v, list):
        return [_recurse(vi) for vi in v]
    elif is_serializable(v):
        return v.serialize()
    return v


class EtaJSONEncoder(json.JSONEncoder):
    '''Extends basic JSONEncoder to handle numpy scalars/arrays and datatime
    data-types.
    '''

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        return super(EtaJSONEncoder, self).default(obj)
