'''
Core data structures for working with data that can be read/written to disk.

Copyright 2017-2019, Voxel51, Inc.
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
import copy
import datetime as dt
import dill as pickle
import json
import os
import pickle as _pickle
import pprint
import uuid

import numpy as np

import eta.core.utils as etau


def load_json(path_or_str):
    '''Loads JSON from the input argument.

    The input argument can be any of the following:
        (a) the path to a JSON file on disk
        (b) a string that can be directly parsed via `json.loads`
        (c) a string containing a comma-seperated list of key=val values
            defining a JSON dictionary, where each value must be parsable via
            `json.loads(val)`

    Args:
        path_or_str: the JSON path or string any of the above supported formats

    Returns:
        the loaded JSON

    Raises:
        ValueError: if no JSON could be decoded
    '''
    try:
        # Parse from JSON string
        return _load_json(path_or_str)
    except ValueError:
        if os.path.isfile(path_or_str):
            # Read from disk
            return read_json(path_or_str)

        try:
            # Try to parse comma-seperated list of key=value pairs
            d = {}
            for chunk in path_or_str.split(","):
                key, value = chunk.split("=")
                d[key] = _load_json(value)
            return d
        except ValueError:
            raise ValueError("Unable to load JSON from '%s'" % path_or_str)


def _load_json(str_or_bytes):
    try:
        return json.loads(str_or_bytes)
    except TypeError:
        # Must be a Python version for which json.loads() cannot handle bytes
        return json.loads(str_or_bytes.decode("utf-8"))


def read_json(path):
    '''Reads JSON from file.

    Args:
        path: the path to the JSON file

    Returns:
        a dict or list containing the loaded JSON

    Raises:
        ValueError: if the JSON file was invalid
    '''
    try:
        with open(path, "rt") as f:
            return json.load(f)
    except ValueError:
        raise ValueError("Unable to parse JSON file '%s'" % path)


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
    if isinstance(obj, Serializable):
        obj = obj.serialize()
    kwargs = {"indent": 4} if pretty_print else {}
    s = json.dumps(
        obj, separators=(",", ": "), cls=ETAJSONEncoder, ensure_ascii=False,
        **kwargs
    )
    return str(s)


def pretty_str(obj):
    '''Wrapper for the pprint.pformat function that generates a formatted
    string representation of the input object.
    '''
    return pprint.pformat(obj, indent=4, width=79)


def read_pickle(path):
    '''Loads the object from the given .pkl file.

    This function attempts to gracefully load a python 2 pickle in python 3
    (if a unicode error is encountered) by assuming "latin1" encoding.

    Args:
        path: the path to the .pkl file

    Returns:
        the loaded instance
    '''
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(path, "rb") as f:
            # https://stackoverflow.com/q/28218466
            return _pickle.load(f, encoding="latin1")


def write_pickle(obj, path):
    '''Writes the object to disk at the given path as a .pkl file.

    Args:
        obj: the pickable object
        path: the path to write the .pkl file
    '''
    etau.ensure_basedir(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


class Serializable(object):
    '''Base class for objects that can be serialized in JSON format.

    Subclasses must implement `from_dict()`, which defines how to construct a
    serializable object from a JSON dictionary.

    Serializable objects can be easily converted read and written from any of
    the following formats:
        - JSON files on disk
        - JSON dictionaries
        - JSON strings

    For example, you can do the following with any class `SerializableClass`
    that is a subclass of `Serializable`:

    ```
    json_path = "/path/to/data.json"

    obj = SerializableClass(...)

    s = obj.to_str()
    obj1 = SerializableClass.from_str(s)

    d = obj.serialize()
    obj2 = SerializableClass.from_dict(d)

    obj.write_json(json_path)
    obj3 = SerializableClass.from_json(json_path)
    ```

    Serializable objects can optionally be serialized in "reflective" mode,
    in which case their class names are embedded in their JSON representations.
    This allows for reading Serializable JSON instances of arbitrary types
    polymorphically via the Serializable interface. For example:

    ```
    json_path = "/path/to/data.json"

    obj = SerializableClass(...)

    s = obj.to_str(reflective=True)
    obj1 = Serializable.from_str(s)  # returns a SerializableClass

    d = obj.serialize(reflective=True)
    obj2 = Serializable.from_dict(d)  # returns a SerializableClass

    obj.write_json(json_path, reflective=True)
    obj3 = Serializable.from_json(json_path)  # returns a SerializableClass
    ```
    '''

    def __str__(self):
        return self.to_str()

    @classmethod
    def get_class_name(cls):
        '''Returns the fully-qualified class name string of this object.'''
        return etau.get_class_name(cls)

    def attributes(self):
        '''Returns a list of class attributes to be serialized.

        This method is called internally by `serialize()` to determine the
        class attributes to serialize.

        Subclasses can override this method, but, by default, all attributes in
        vars(self) are returned, minus private attributes, i.e., those starting
        with "_". The order of the attributes in this list is preserved when
        serializing objects, so a common pattern is for subclasses to override
        this method if they want their JSON files to be organized in a
        particular way.

        Returns:
            a list of class attributes to be serialized
        '''
        return [a for a in vars(self) if not a.startswith("_")]

    def custom_attributes(self, dynamic=False, private=False):
        '''Returns a customizable list of class attributes.

        By default, all attributes in vars(self) are returned, minus private
        attributes (those starting with "_").

        Args:
            dynamic: whether to include dynamic properties, e.g., those defined
                by getter/setter methods or the `@property` decorator. By
                default, this is False
            private: whether to include private properties, i.e., those
                starting with "_". By default, this is False

        Returns:
            a list of class attributes
        '''
        if dynamic:
            attrs = [a for a in dir(self) if not callable(getattr(self, a))]
        else:
            attrs = vars(self)
        if not private:
            attrs = [a for a in attrs if not a.startswith("_")]
        return attrs

    def serialize(self, reflective=False):
        '''Serializes the object into a dictionary.

        Serialization is applied recursively to all attributes in the object,
        including element-wise serialization of lists and dictionary values.

        Args:
            reflective: whether to include reflective attributes when
                serializing the object. By default, this is False

        Returns:
            a JSON dictionary representation of the object
        '''
        d = self._prepare_serial_dict(reflective)
        for a in self.attributes():
            d[a] = _recurse(getattr(self, a), reflective)
        return d

    def _prepare_serial_dict(self, reflective):
        d = OrderedDict()
        if reflective:
            d["_CLS"] = self.get_class_name()
        return d

    def to_str(self, pretty_print=True, **kwargs):
        '''Returns a string representation of this object.

        Args:
            pretty_print: if True (default), the string will be formatted to be
                human readable; when False, it will be compact with no extra
                spaces or newline characters
            **kwargs: optional keyword arguments for `self.serialize()`

        Returns:
            a string representation of the object
        '''
        obj = self.serialize(**kwargs)
        return json_to_str(obj, pretty_print=pretty_print)

    def write_json(self, path, pretty_print=True, **kwargs):
        '''Serializes the object and writes it to disk.

        Args:
            path: the output path
            pretty_print: when True (default), the resulting JSON will be
                outputted to be human readable; when False, it will be compact
                with no extra spaces or newline characters
            **kwargs: optional keyword arguments for `self.serialize()`
        '''
        obj = self.serialize(**kwargs)
        write_json(obj, path, pretty_print=pretty_print)

    @classmethod
    def from_dict(cls, d, *args, **kwargs):
        '''Constructs a Serializable object from a JSON dictionary.

        Subclasses must implement this method if they intend to support being
        read from disk.

        Args:
            d: a JSON dictionary representation of a Serializable object
            *args: optional class-specific positional arguments
            **kwargs: optional class-specific keyword arguments

        Returns:
            an instance of the Serializable class
        '''
        if "_CLS" in d:
            #
            # Parse reflectively
            #
            # Note that "_CLS" is popped from the dictionary here. This is
            # crucial because if the subclass does not implement `from_dict`,
            # this method will be called again and we need to raise a
            # NotImplementedError next time around!
            #
            cls = etau.get_class(d.pop("_CLS"))
            return cls.from_dict(d, *args, **kwargs)

        raise NotImplementedError("subclass must implement from_dict()")

    @classmethod
    def from_str(cls, s, *args, **kwargs):
        '''Constructs a Serializable object from a JSON string.

        Subclasses may override this method, but, by default, this method
        simply parses the string and calls from_dict(), which subclasses must
        implement.

        Args:
            s: a JSON string representation of a Serializable object
            *args: optional positional arguments for `self.from_dict()`
            **kwargs: optional keyword arguments for `self.from_dict()`

        Returns:
            an instance of the Serializable class
        '''
        return cls.from_dict(_load_json(s), *args, **kwargs)

    @classmethod
    def from_json(cls, path, *args, **kwargs):
        '''Constructs a Serializable object from a JSON file.

        Subclasses may override this method, but, by default, this method
        simply reads the JSON and calls from_dict(), which subclasses must
        implement.

        Args:
            path: the path to the JSON file on disk
            *args: optional positional arguments for `self.from_dict()`
            **kwargs: optional keyword arguments for `self.from_dict()`

        Returns:
            an instance of the Serializable class
        '''
        return cls.from_dict(read_json(path), *args, **kwargs)


def _recurse(v, reflective):
    if isinstance(v, Serializable):
        return v.serialize(reflective=reflective)
    elif isinstance(v, set):
        v = list(v)  # convert sets to lists
    if isinstance(v, list):
        return [_recurse(vi, reflective) for vi in v]
    elif isinstance(v, dict):
        return OrderedDict(
            (ki, _recurse(vi, reflective)) for ki, vi in iteritems(v))
    return v


class DirectoryContainer(Serializable):
    '''Container like object that represents a list serializable objects in a
    directory on disk.

    USE WITH CARE.
    '''

    _ELE_CLS = None

    _ELE_CLS_FIELD = None

    _ELE_DIR_ATTR = None

    _ELE_MAP_ATTR = None

    def __init__(self, **kwargs):
        if self._ELE_DIR_ATTR not in kwargs:
            raise ContainerError("bad")
        setattr(self, self._ELE_DIR_ATTR, kwargs[self._ELE_DIR_ATTR])

        setattr(
            self,
            self._ELE_MAP_ATTR,
            kwargs.get(
                self._ELE_MAP_ATTR,
                list(etau.multiglob(".json", root=self._ele_dir+"/**/*"))
            )
        )

    def __iter__(self):
        return (self._load_ele(path) for path in self._paths)

    def __getitem__(self, idx):
        return self._load_ele(self._ele_path(idx))

    def __setitem__(self, idx, ele):
        ele.write_json(self._ele_path(idx))

    def __delitem__(self, idx):
        etau.delete_file(self._ele_path(idx))
        del self._ele_map[idx]

    def __len__(self):
        return len(self._ele_map)

    def __bool__(self):
        return bool(self._ele_map)

    def add(self, ele):
        self.append(ele)

    def add_container(self, container):
        for ele in container:
            self.add(ele)

    def append(self, ele):
        self[len(self)] = ele

    def attributes(self):
        return [self._ELE_DIR_ATTR, self._ELE_MAP_ATTR]

    def filter_elements(self, filters, match=any):
        self._filter_elements(filters, match)

    def delete_inds(self, inds):
        for idx in sorted(inds, reverse=True):
            del self[idx]

    def keep_inds(self, inds):
        inds = set(inds)
        for idx, path in self._files():
            if idx not in inds:
                del self[idx]

    def extract_inds(self, new_ele_dir, inds):
        inds = set(inds)
        container = copy.deepcopy(self)
        setattr(container, self._ELE_DIR_ATTR, new_ele_dir)
        for idx, ele in self:
            if idx in inds:
                container.add(ele)
        return container

    def clear(self):
        etau.delete_dir(self._ele_dir)
        setattr(self, self._ELE_MAP_ATTR, [])

    @property
    def size(self):
        return len(self)

    def count_matches(self, filters, match=any):
        return len(self._filter_elements(filters, match=match))

    def get_matches(self, new_ele_dir, filters, match=any):
        inds = self._filter_elements(filters, match)
        container = copy.deepcopy(self)
        setattr(container, self._ELE_DIR_ATTR, new_ele_dir)
        for idx in inds:
            container.add(self[idx])
        return container

    def sort_by(self, attr, reverse=False):
        def field_none_last(uuid_str:
            ele = self[idx]
            val = getattr(ele, attr)
            return ((val is None) ^ reverse, val)  # always puts None last

        setattr(
            self, self._ELE_MAP_ATTR, sorted(
                self._ele_map, reverse=reverse, key=field_none_last))

    def serialize(self, reflective=False):
        d = OrderedDict()
        if reflective:
            d["_CLS"] = self.get_class_name()
            d[self._ELE_CLS_FIELD] = etau.get_class_name(self._ELE_CLS)
        d.update(super(DirectoryContainer, self).serialize(reflective=False))
        return d

    @property
    def _paths(self):
        return (self._ele_path(idx) for idx in range(0, len(self._ele_map)))

    @property
    def _ele_dir(self):
        return getattr(self, self._ELE_DIR_ATTR)

    def _ele_subdir_pattern(self, idx):
        return os.path.join(*list(self._ele_filename(idx)[:5]))

    @property
    def _ele_map(self):
        return getattr(self, self._ELE_MAP_ATTR)

    def _ele_filename(self, idx):
        if idx > len(self):
            raise ContainerError("out of bounds")
        elif idx == len(self):
            self._ele_map.append(str(uuid.uuid4()))
        return "%s.json" % self._ele_map[idx]

    def _ele_path(self, idx):
        return os.path.join(
            self._ele_dir,
            self._ele_subdir_pattern(idx),
            self._ele_filename(idx)
        )

    def _load_ele(self, path):
        return self._ELE_CLS.from_json(path)

    def _filter_elements(self, filters, match):
        return list(filter(
            lambda o: match(f(self[o]) for f in filters), self._ele_map))

    def _validate(self):
        '''Validates that a Container instance is valid.'''
        if self._ELE_CLS is None:
            raise ContainerError(
                "Cannot instantiate a Container for which _ELE_CLS is None")
        if self._ELE_DIR_ATTR is None:
            raise ContainerError(
                "Cannot instantiate a Container for which _ELE_ATTR is None")
        if self._ELE_MAP_ATTR is None:
            raise ContainerError(
                "Cannot instantiate a Container for which _ELE_ATTR is None")
        if not issubclass(self._ELE_CLS, Serializable):
            raise ContainerError(
                "%s is not Serializable" % self._ELE_CLS)

    @classmethod
    def from_dict(cls, d):
        '''Constructs a Container from a JSON dictionary.

        If the dictionary has the `"_CLS"` and `cls._ELE_CLS_FIELD`
        keys, they are used to infer the Container class and underlying element
        classes, respectively, and this method can be invoked on any
        `Container` subclass that has the same `_ELE_CLS_FIELD` setting.

        Otherwise, this method must be called on the same concrete `Container`
        subclass from which the JSON was generated.
        '''
        if cls._ELE_CLS_FIELD is None:
            raise ContainerError(
                "%s is an abstract container and cannot be used to load a "
                "JSON dictionary. Please use a Container subclass that "
                "defines its `_ELE_CLS_FIELD` member" % cls)

        if "_CLS" in d:
            if cls._ELE_CLS_FIELD not in d:
                raise ContainerError(
                    "Cannot use %s to reflectively load this container "
                    "because the expected field '%s' was not found in the "
                    "JSON dictionary" % (cls, cls._ELE_CLS_FIELD))

            # Parse reflectively
            cls = etau.get_class(d["_CLS"])
            ele_cls = etau.get_class(d[cls._ELE_CLS_FIELD])
        else:
            # Validates the cls settings
            cls()
            # Parse using provided class
            ele_cls = cls._ELE_CLS

        return cls(**d)

class Container(Serializable):
    '''Abstract base class for flexible containers that store homogeneous lists
    of elements of a `Serializable` class.

    Container subclasses embed their class names and underlying element class
    names in their JSON representations, so they can be read reflectively from
    disk.

    This class cannot be instantiated directly.

    This class currently has only two direct subclasses, which bifurcate the
    container implementation into two distinct categories:
        - `eta.core.config.ConfigContainer`: base class for containers that
            store lists of `Config` instances
        - `eta.core.data.DataContainer`: base class for containers that store
            lists of arbitrary `Serializable` data instances

    See `ConfigContainer` and `DataContainer` for concrete usage examples.
    '''

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
        '''Creates a Container instance.

        Args:
            <elements>: an optional list of elements to store in the Container.
                The appropriate name of this keyword argument is determined by
                the `_ELE_ATTR` member of the Container subclass

        Raises:
            ContainerError: if there was an error while creating the container
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
        return self.__elements__[index]

    def __setitem__(self, index, value):
        self.__elements__[index] = value

    def __delitem__(self, index):
        del self.__elements__[index]

    def __iter__(self):
        return iter(self.__elements__)

    def __len__(self):
        return len(self.__elements__)

    def __bool__(self):
        return self.size > 0

    @property
    def __elements__(self):
        return getattr(self, self._ELE_ATTR)

    def add(self, instance):
        '''Adds an element to the container.

        Args:
            instance: an instance of `_ELE_CLS`
        '''
        self.__elements__.append(instance)

    def add_container(self, container):
        '''Adds the elements in the given container to this container.

        Args:
            container: a Container instance
        '''
        self.__elements__.extend(container.__elements__)

    def filter_elements(self, filters, match=any):
        '''Removes elements that don't match the given filters from the
        container.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
        '''
        elements = self._filter_elements(filters, match)
        setattr(self, self._ELE_ATTR, elements)

    def delete_inds(self, inds):
        '''Deletes the elements from the container with the given indices.

        Args:
            inds: a list of indices of the elements to delete
        '''
        for idx in sorted(inds, reverse=True):
            del self.__elements__[idx]

    def keep_inds(self, inds):
        '''Keeps only the elements in the container with the given indices.

        Args:
            inds: an iterable of indices of the elements to keep
        '''
        inds = set(inds)
        elements = [e for i, e in enumerate(self.__elements__) if i in inds]
        setattr(self, self._ELE_ATTR, elements)

    def extract_inds(self, inds):
        '''Creates a new container having only the elements with the given
        indices.

        Args:
            inds: a list of indices of the elements to keep

        Returns:
            a Container
        '''
        container = copy.deepcopy(self)
        container.keep_inds(inds)
        return container

    def clear(self):
        '''Deletes all elements from the container.'''
        setattr(self, self._ELE_ATTR, [])

    @property
    def size(self):
        '''Returns the number of elements in the container.'''
        return len(self.__elements__)

    def count_matches(self, filters, match=any):
        '''Counts the number of elements in the container that match the
        given filters.

        Args:
            filters: a list of functions that accept instances of class
                `_ELE_CLS`and return True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`

        Returns:
            the number of elements in the container that match the filters
        '''
        return self.get_matches(filters, match=match).size

    def get_matches(self, filters, match=any):
        '''Gets elements matching the given filters.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`

        Returns:
            a copy of the container containing only the elements that match
                the filters
        '''
        elements = self._filter_elements(filters, match)
        return self.__class__(**{self._ELE_ATTR: elements})

    def sort_by(self, attr, reverse=False):
        '''Sorts the elements in the container by the given attribute.

        Elements whose attribute is None are always put at the end of the list.

        Args:
            attr: the element attribute to sort by
            reverse: whether to sort in descending order. The default is False
        '''
        def field_none_last(ele):
            val = getattr(ele, attr)
            return ((val is None) ^ reverse, val)  # always puts None last

        setattr(
            self, self._ELE_ATTR, sorted(
                self.__elements__, reverse=reverse, key=field_none_last))

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.'''
        return [self._ELE_ATTR]

    def serialize(self, reflective=False):
        '''Serializes the container into a dictionary.

        Args:
            reflective: whether to include reflective attributes when
                serializing the object. By default, this is False

        Returns:
            a JSON dictionary representation of the container
        '''
        d = OrderedDict()
        if reflective:
            d["_CLS"] = self.get_class_name()
            d[self._ELE_CLS_FIELD] = etau.get_class_name(self._ELE_CLS)
        d.update(super(Container, self).serialize(reflective=False))
        return d

    @classmethod
    def from_dict(cls, d):
        '''Constructs a Container from a JSON dictionary.

        If the dictionary has the `"_CLS"` and `cls._ELE_CLS_FIELD`
        keys, they are used to infer the Container class and underlying element
        classes, respectively, and this method can be invoked on any
        `Container` subclass that has the same `_ELE_CLS_FIELD` setting.

        Otherwise, this method must be called on the same concrete `Container`
        subclass from which the JSON was generated.
        '''
        if cls._ELE_CLS_FIELD is None:
            raise ContainerError(
                "%s is an abstract container and cannot be used to load a "
                "JSON dictionary. Please use a Container subclass that "
                "defines its `_ELE_CLS_FIELD` member" % cls)

        if "_CLS" in d:
            if cls._ELE_CLS_FIELD not in d:
                raise ContainerError(
                    "Cannot use %s to reflectively load this container "
                    "because the expected field '%s' was not found in the "
                    "JSON dictionary" % (cls, cls._ELE_CLS_FIELD))

            # Parse reflectively
            cls = etau.get_class(d["_CLS"])
            ele_cls = etau.get_class(d[cls._ELE_CLS_FIELD])
        else:
            # Validates the cls settings
            cls()
            # Parse using provided class
            ele_cls = cls._ELE_CLS

        return cls(**{
            cls._ELE_ATTR: [ele_cls.from_dict(dd) for dd in d[cls._ELE_ATTR]]
        })

    def _filter_elements(self, filters, match):
        return list(
            filter(lambda o: match(f(o) for f in filters), self.__elements__))

    def _validate(self):
        '''Validates that a Container instance is valid.'''
        if self._ELE_CLS is None:
            raise ContainerError(
                "Cannot instantiate a Container for which _ELE_CLS is None")
        if self._ELE_ATTR is None:
            raise ContainerError(
                "Cannot instantiate a Container for which _ELE_ATTR is None")
        if not issubclass(self._ELE_CLS, Serializable):
            raise ContainerError(
                "%s is not Serializable" % self._ELE_CLS)


class ContainerError(Exception):
    '''Exception raised when an invalid Container is encountered.'''
    pass


class Picklable(object):
    '''Mixin class for objects that can be pickled.'''

    def pickle(self, path):
        '''Saves the instance to disk as a .pkl file.

        Args:
            path: the path to write the .pkl file
        '''
        write_pickle(self, path)

    @classmethod
    def from_pickle(cls, path):
        '''Loads the object from the given .pkl file.

        Args:
            path: the path to the .pkl file

        Returns:
            the loaded instance
        '''
        return read_pickle(path)

    @staticmethod
    def is_pickle_path(path):
        '''Checks the path to see if it has a pickle (.pkl) extension.'''
        return path.endswith(".pkl")


class NpzWriteable(object):
    '''Base class for dictionary-like objects that contain numpy.array values
    that can be written to disk as .npz files.
    '''

    @staticmethod
    def is_npz_path(path):
        '''Returns True/False if the provided path is an .npz file.'''
        return path.endswith(".npz")

    def attributes(self):
        '''Returns a list of class attributes that will be included when
        writing an .npz file.
        '''
        return [a for a in vars(self) if not a.startswith("_")]

    def write_npz(self, path):
        '''Writes the instance to disk in .npz format.

        Args:
            path: the path to write the .npz file
        '''
        etau.ensure_basedir(path)
        d = {a: getattr(self, a) for a in self.attributes()}
        np.savez_compressed(path, **d)

    @classmethod
    def from_npz(cls, path):
        '''Loads the .npz file from disk and returns an NpzWriteable instance.

        Args:
            path: the path to an .npz file

        Returns:
            an NpzWriteable instance
        '''
        return cls(**np.load(path))


class ETAJSONEncoder(json.JSONEncoder):
    '''Extends basic JSONEncoder to handle ETA conventions and types.'''

    def default(self, obj):
        if isinstance(obj, Serializable):
            return obj.serialize()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        return super(ETAJSONEncoder, self).default(obj)
