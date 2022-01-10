"""
Core data structures for working with data that can be read/written to disk.

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
from future.utils import iteritems, itervalues

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from base64 import b64encode, b64decode
from collections import OrderedDict
import copy
import datetime as dt
import dill as pickle
import glob
import io
import json
import logging
import numbers
import os
import pickle as _pickle
import pprint
from uuid import uuid4
import zlib

import ndjson
import numpy as np

import eta.core.utils as etau


logger = logging.getLogger(__name__)


def load_json(path_or_str):
    """Loads JSON from the input argument.

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
    """
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
                k, v = etau.remove_escape_chars(chunk, ",").split("=")
                d[k] = _load_json(v)

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
    """Reads JSON from file.

    Args:
        path: the path to the JSON file

    Returns:
        a dict or list containing the loaded JSON

    Raises:
        ValueError: if the JSON file was invalid
    """
    try:
        with open(path, "rt") as f:
            return json.load(f)
    except ValueError:
        raise ValueError("Unable to parse JSON file '%s'" % path)


def write_json(obj, path, pretty_print=False):
    """Writes JSON object to file.

    Args:
        obj: is either an object that can be directly dumped to a JSON file or
            an instance of a subclass of Serializable
        path: the output path
        pretty_print: whether to render the JSON in human readable format with
            newlines and indentations. By default, this is False
    """
    s = json_to_str(obj, pretty_print=pretty_print)
    etau.ensure_basedir(path)
    with open(path, "wt") as f:
        f.write(s)


def read_ndjson(path):
    """Reads the NDJSON from file.

    Args:
        path: the path to the NDJSON file

    Returns:
        a list of JSON dicts
    """
    with open(path, "rt") as f:
        return ndjson.load(f)


def load_ndjson(path_or_str):
    """Loads NDJSON from the input argument.

    The input argument can be any of the following:
        (a) the path to a NDJSON file on disk
        (b) a string that can be directly parsed via `ndjson.loads`

    Args:
        path_or_str: the NDJSON path or string

    Returns:
        the loaded NDJSON

    Raises:
        ValueError: if no NDJSON could be decoded
    """
    if os.path.isfile(path_or_str):
        return read_ndjson(path_or_str)

    return _load_ndjson(path_or_str)


def _load_ndjson(str_or_bytes):
    try:
        return ndjson.loads(str_or_bytes)
    except TypeError:
        # Must be a Python version for which ndjson.loads() cannot handle bytes
        return ndjson.loads(str_or_bytes.decode("utf-8"))


def write_ndjson(obj, path, append=False):
    """Writes the list of JSON dicts to disk in NDJSON format.

    Args:
        obj: a list of JSON dicts
        path: the output path
        append: whether to append to an existing file, if necessary
    """
    etau.ensure_basedir(path)

    if append and os.path.exists(path) and os.path.getsize(path) > 0:
        prefix = os.linesep
    else:
        prefix = ""

    mode = "at" if append else "wt"
    with open(path, mode) as f:
        f.write(prefix + ndjson.dumps(obj))


def json_to_str(obj, pretty_print=True):
    """Converts the JSON object to a string.

    Args:
        obj: a JSON dictionary or an instance of a Serializable subclass
        pretty_print: whether to render the JSON in human readable format with
            newlines and indentations. By default, this is True
    """
    if isinstance(obj, Serializable):
        obj = obj.serialize()

    kwargs = {"indent": 4} if pretty_print else {}
    s = json.dumps(
        obj,
        separators=(",", ": "),
        cls=ETAJSONEncoder,
        ensure_ascii=False,
        **kwargs
    )
    return str(s)


def pretty_str(obj):
    """Wrapper for the pprint.pformat function that generates a formatted
    string representation of the input object.
    """
    return pprint.pformat(obj, indent=4, width=79)


def read_pickle(path):
    """Loads the object from the given .pkl file.

    This function attempts to gracefully load a python 2 pickle in python 3
    (if a unicode error is encountered) by assuming "latin1" encoding.

    Args:
        path: the path to the .pkl file

    Returns:
        the loaded instance
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(path, "rb") as f:
            # https://stackoverflow.com/q/28218466
            return _pickle.load(f, encoding="latin1")


def write_pickle(obj, path):
    """Writes the object to disk at the given path as a .pkl file.

    Args:
        obj: the pickable object
        path: the path to write the .pkl file
    """
    etau.ensure_basedir(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def serialize_numpy_array(array):
    """Serializes a numpy array.

    Args:
        array: a numpy array

    Returns:
        the serialized string
    """
    #
    # We currently serialize in numpy format. Other alternatives considered
    # were `pickle.dumps(array)` and HDF5
    #
    with io.BytesIO() as f:
        allow_pickle = array.dtype == object
        np.save(f, array, allow_pickle=allow_pickle)
        bytes_str = zlib.compress(f.getvalue())

    return b64encode(bytes_str).decode("ascii")


def deserialize_numpy_array(numpy_str, allow_pickle=False):
    """Loads a serialized numpy array from string.

    Args:
        numpy_str: serialized numpy array string
        allow_pickle: whether to allow loading pickled objects, which is
            necessary if the array has ``dtype=object``

    Returns:
        the numpy array
    """
    #
    # We currently serialize in numpy format. Other alternatives considered
    # were `pickle.loads(numpy_str)` and HDF5
    #
    bytes_str = zlib.decompress(b64decode(numpy_str.encode("ascii")))
    with io.BytesIO(bytes_str) as f:
        return np.load(f, allow_pickle=allow_pickle)


class Serializable(object):
    """Base class for objects that can be serialized in JSON format.

    Subclasses must implement `from_dict()`, which defines how to construct a
    serializable object from a JSON dictionary.

    Serializable objects can be easily converted read and written from any of
    the following formats:
        - JSON files on disk
        - JSON dictionaries
        - JSON strings

    For example, you can do the following with any class `SerializableClass`
    that is a subclass of `Serializable`::

        json_path = "/path/to/data.json"

        obj = SerializableClass(...)

        s = obj.to_str()
        obj1 = SerializableClass.from_str(s)

        d = obj.serialize()
        obj2 = SerializableClass.from_dict(d)

        obj.write_json(json_path)
        obj3 = SerializableClass.from_json(json_path)

    Serializable objects can optionally be serialized in "reflective" mode,
    in which case their class names are embedded in their JSON representations.
    This allows for reading Serializable JSON instances of arbitrary types
    polymorphically via the Serializable interface. For example::

        json_path = "/path/to/data.json"

        obj = SerializableClass(...)

        s = obj.to_str(reflective=True)
        obj1 = Serializable.from_str(s)  # returns a SerializableClass

        d = obj.serialize(reflective=True)
        obj2 = Serializable.from_dict(d)  # returns a SerializableClass

        obj.write_json(json_path, reflective=True)
        obj3 = Serializable.from_json(json_path)  # returns a SerializableClass
    """

    def __str__(self):
        return self.to_str()

    def copy(self):
        """Returns a deep copy of the object.

        Returns:
            a Serializable instance
        """
        return copy.deepcopy(self)

    @classmethod
    def get_class_name(cls):
        """Returns the fully-qualified class name string of this object."""
        return etau.get_class_name(cls)

    def attributes(self):
        """Returns a list of class attributes to be serialized.

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
        """
        return [a for a in vars(self) if not a.startswith("_")]

    def custom_attributes(self, dynamic=False, private=False):
        """Returns a customizable list of class attributes.

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
        """
        if dynamic:
            attrs = [a for a in dir(self) if not callable(getattr(self, a))]
        else:
            attrs = vars(self)
        if not private:
            attrs = [a for a in attrs if not a.startswith("_")]
        return attrs

    def serialize(self, reflective=False):
        """Serializes the object into a dictionary.

        Serialization is applied recursively to all attributes in the object,
        including element-wise serialization of lists and dictionary values.

        Args:
            reflective: whether to include reflective attributes when
                serializing the object. By default, this is False

        Returns:
            a JSON dictionary representation of the object
        """
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
        """Returns a string representation of this object.

        Args:
            pretty_print: whether to render the JSON in human readable format
                with newlines and indentations. By default, this is True
            **kwargs: optional keyword arguments for `self.serialize()`

        Returns:
            a string representation of the object
        """
        obj = self.serialize(**kwargs)
        return json_to_str(obj, pretty_print=pretty_print)

    def write_json(self, path, pretty_print=False, **kwargs):
        """Serializes the object and writes it to disk.

        Args:
            path: the output path
            pretty_print: whether to render the JSON in human readable format
                with newlines and indentations. By default, this is False
            **kwargs: optional keyword arguments for `self.serialize()`
        """
        obj = self.serialize(**kwargs)
        write_json(obj, path, pretty_print=pretty_print)

    @classmethod
    def from_dict(cls, d, *args, **kwargs):
        """Constructs a Serializable object from a JSON dictionary.

        Subclasses must implement this method if they intend to support being
        read from disk.

        Args:
            d: a JSON dictionary representation of a Serializable object
            *args: optional class-specific positional arguments
            **kwargs: optional class-specific keyword arguments

        Returns:
            an instance of the Serializable class
        """
        if "_CLS" in d:
            #
            # Parse reflectively
            #
            # Note that "_CLS" is popped from the dictionary here. This is
            # crucial because if the subclass does not implement `from_dict`,
            # this method will be called again and we need to raise a
            # NotImplementedError next time around!
            #
            serializable_cls = etau.get_class(d.pop("_CLS"))
            return serializable_cls.from_dict(d, *args, **kwargs)

        raise NotImplementedError("subclass must implement from_dict()")

    @classmethod
    def from_str(cls, s, *args, **kwargs):
        """Constructs a Serializable object from a JSON string.

        Subclasses may override this method, but, by default, this method
        simply parses the string and calls from_dict(), which subclasses must
        implement.

        Args:
            s: a JSON string representation of a Serializable object
            *args: optional positional arguments for `self.from_dict()`
            **kwargs: optional keyword arguments for `self.from_dict()`

        Returns:
            an instance of the Serializable class
        """
        return cls.from_dict(_load_json(s), *args, **kwargs)

    @classmethod
    def from_json(cls, path, *args, **kwargs):
        """Constructs a Serializable object from a JSON file.

        Subclasses may override this method, but, by default, this method
        simply reads the JSON and calls from_dict(), which subclasses must
        implement.

        Args:
            path: the path to the JSON file on disk
            *args: optional positional arguments for `self.from_dict()`
            **kwargs: optional keyword arguments for `self.from_dict()`

        Returns:
            an instance of the Serializable class
        """
        return cls.from_dict(read_json(path), *args, **kwargs)


class ExcludeNoneAttributes(Serializable):
    """Mixin for Serializable classes that exclude None-valued attributes when
    serializing.

    This class must appear BEFORE Serializable in the inheritence list of the
    class::

        class Foo(ExcludeNone, Serializable):
            ...
    """

    def attributes(self):
        """Returns a list of class attributes to be serialized.

        Any attributes whose value is None are omitted.

        Returns:
            the list of attributes
        """
        attrs = super(ExcludeNoneAttributes, self).attributes()
        return [a for a in attrs if getattr(self, a) is not None]


def _recurse(v, reflective):
    if isinstance(v, Serializable):
        return v.serialize(reflective=reflective)

    if isinstance(v, set):
        v = list(v)  # convert sets to lists

    if isinstance(v, list):
        return [_recurse(vi, reflective) for vi in v]

    if isinstance(v, dict):
        return OrderedDict(
            (str(ki), _recurse(vi, reflective)) for ki, vi in iteritems(v)
        )

    if isinstance(v, np.ndarray):
        return v.tolist()

    if hasattr(v, "serialize") and callable(v.serialize):
        return v.serialize()

    if hasattr(v, "to_dict") and callable(v.to_dict):
        return v.to_dict()

    return v


class Set(Serializable):
    """Abstract base class for flexible sets that store homogeneous elements
    of a `Serializable` class and provides O(1) lookup of elements by a
    subclass-configurable element attribute.

    This class cannot be instantiated directly. Instead a subclass should
    be created for each type of element to be stored. Subclasses MUST set the
    following members:
        -  `_ELE_CLS`: the class of the element stored in the set
        -  `_ELE_KEY_ATTR`: the name of the element attribute to use to perform
            element lookups

    In addition, sublasses MAY override the following members:
        - `_ELE_CLS_FIELD`: the name of the private attribute that will store
            the class of the elements in the set
        - `_ELE_ATTR`: the name of the attribute that will store the elements
            in the set

    Set subclasses can optionally embed their class names and underlying
    element class names in their JSON representations, so they can be read
    reflectively from disk.

    Examples::

        from eta.core.serial import Set
        from eta.core.geometry import LabeledPointSet

        points = LabeledPointSet()
        points.write_json("points.json", reflective=True)

        points2 = Set.from_json("points.json")
        print(points2.__class__)  # LabeledPointSet, not Set!
    """

    #
    # The class of the element stored in the set
    #
    # Subclasses MUST set this field
    #
    _ELE_CLS = None

    #
    # The name of the element attribute that will be used when looking up
    # elements by key in the set
    #
    # Subclasses MUST set this field
    #
    _ELE_KEY_ATTR = None

    #
    # The name of the private attribute that will store the class of the
    # elements in the set
    #
    # Subclasses MAY set this field
    #
    _ELE_CLS_FIELD = "_ELEMENT_CLS"

    #
    # The name of the attribute that will store the elements in the set
    #
    # Subclasses MAY set this field
    #
    _ELE_ATTR = "elements"

    def __init__(self, **kwargs):
        """Creates a Set instance.

        Args:
            <elements>: an optional iterable of elements to store in the Set.
                The appropriate name of this keyword argument is determined by
                the `_ELE_ATTR` member of the Set subclass

        Raises:
            SetError: if there was an error while creating the set
        """
        self._validate()

        if kwargs and self._ELE_ATTR not in kwargs:
            raise SetError(
                "Expected elements to be provided in keyword argument '%s'; "
                "found keys %s" % (self._ELE_ATTR, list(kwargs.keys()))
            )
        elements = kwargs.get(self._ELE_ATTR, None) or []

        for e in elements:
            if not isinstance(e, self._ELE_CLS):
                raise SetError(
                    "Set %s expects elements of type %s but found "
                    "%s" % (self.__class__, self._ELE_CLS, e.__class__)
                )

        self.clear()
        self.add_iterable(elements)

    def __getitem__(self, key):
        return self.__elements__[key]

    def __setitem__(self, key, element):
        self.__elements__[key] = element

    def __delitem__(self, key):
        del self.__elements__[key]

    def __contains__(self, key):
        return key in self.__elements__

    def __iter__(self):
        return iter(itervalues(self.__elements__))

    def __len__(self):
        return len(self.__elements__)

    def __bool__(self):
        return bool(self.__elements__)

    @property
    def __elements__(self):
        return getattr(self, self._ELE_ATTR)

    def keys(self):
        """Returns an iterator over the keys of the elements in the set."""
        return iter(self.__elements__)

    @classmethod
    def get_element_class(cls):
        """Gets the class of elements stored in this set."""
        return cls._ELE_CLS

    @classmethod
    def get_element_class_name(cls):
        """Returns the fully-qualified class name string of the element
        instances in this set.
        """
        return etau.get_class_name(cls._ELE_CLS)

    @classmethod
    def get_key(cls, element):
        """Gets the key for the given element, i.e., its `_ELE_KEY_ATTR` field.

        Args:
            element: an instance of `_ELE_CLS`

        Returns:
            the key for the element
        """
        return getattr(element, cls._ELE_KEY_ATTR)

    @property
    def is_empty(self):
        """Whether this set has no elements."""
        return not bool(self)

    def clear(self):
        """Deletes all elements from the set."""
        setattr(self, self._ELE_ATTR, OrderedDict())

    def empty(self):
        """Returns an empty copy of the set.

        Subclasses may override this method, but, by default, this method
        constructs an empty set via `self.__class__()`

        Returns:
            an empty Set
        """
        return self.__class__()

    def add(self, element):
        """Adds an element to the set.

        Args:
            element: an instance of `_ELE_CLS`
        """
        key = self.get_key(element)
        if key is None:
            key = str(uuid4())

        self[key] = element

    def add_set(self, set_):
        """Adds the elements in the given set to this set.

        Args:
            set_: a Set of `_ELE_CLS` objects
        """
        for key in set_.keys():
            self[key] = set_[key]

    def add_iterable(self, elements):
        """Adds the elements in the given iterable to the set.

        Args:
            elements: an iterable of `_ELE_CLS` objects
        """
        for element in elements:
            self.add(element)

    def filter_elements(self, filters, match=any):
        """Removes elements that don't match the given filters from the set.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
        """
        elements = self._filter_elements(filters, match)
        setattr(self, self._ELE_ATTR, elements)

    def pop_elements(self, filters, match=any):
        """Pops elements that match the given filters from the set.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`

        Returns:
            a Set of elements matching the given filters
        """
        _set = self.empty()
        _set.add_iterable(self._pop_elements(filters, match))
        return _set

    def delete_keys(self, keys):
        """Deletes the elements from the set with the given keys.

        Args:
            keys: an iterable of keys of the elements to delete
        """
        for key in keys:
            del self[key]

    def keep_keys(self, keys):
        """Keeps only the elements in the set with the given keys.

        Args:
            keys: an iterable of keys of the elements to keep
        """
        elements = self._get_elements_with_keys(keys)
        setattr(self, self._ELE_ATTR, elements)

    def extract_keys(self, keys):
        """Returns a new set having only the elements with the given keys.

        The elements are passed by reference, not copied.

        Args:
            keys: an iterable of keys of the elements to keep

        Returns:
            a Set with the requested elements
        """
        _set = self.empty()
        for key in keys:
            _set.add(self[key])

        return _set

    def pop_keys(self, keys):
        """Pops elements with the given keys from the set.

        Args:
            keys: an iterable of keys of the elements to keep

        Returns:
            a Set with the popped elements
        """
        _set = self.empty()
        _set.add_iterable(self._pop_elements_with_keys(keys))
        return _set

    def count_matches(self, filters, match=any):
        """Counts the number of elements in the set that match the given
        filters.

        Args:
            filters: a list of functions that accept instances of class
                `_ELE_CLS`and return True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`

        Returns:
            the number of elements in the set that match the filters
        """
        elements = self._filter_elements(filters, match)
        return len(elements)

    def get_matches(self, filters, match=any):
        """Returns a set of elements matching the given filters.

        The elements are passed by reference, not copied.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`

        Returns:
            a Set with elements matching the filters
        """
        _set = self.empty()
        _set.add_iterable(itervalues(self._filter_elements(filters, match)))
        return _set

    def sort_by(self, attr, reverse=False):
        """Sorts the elements in the set by the given attribute.

        Elements whose attribute is None are always put at the end of the set.

        Args:
            attr: the element attribute to sort by
            reverse: whether to sort in descending order. The default is False
        """

        def field_none_last(key_ele_pair):
            val = getattr(key_ele_pair[1], attr)
            return ((val is None) ^ reverse, val)  # always puts None last

        elements = OrderedDict(
            sorted(
                iteritems(self.__elements__),
                reverse=reverse,
                key=field_none_last,
            )
        )
        setattr(self, self._ELE_ATTR, elements)

    def attributes(self):
        """Returns the list of class attributes that will be serialized."""
        # `_ELE_ATTR` is omitted here because it is serialized manually
        return []

    def serialize(self, reflective=False):
        """Serializes the set into a dictionary.

        Args:
            reflective: whether to include reflective attributes when
                serializing the object. By default, this is False

        Returns:
            a JSON dictionary representation of the set
        """
        d = OrderedDict()
        if reflective:
            d["_CLS"] = self.get_class_name()
            d[self._ELE_CLS_FIELD] = etau.get_class_name(self._ELE_CLS)

        d.update(super(Set, self).serialize(reflective=False))

        #
        # Note that we serialize the elements as a list; the keys are
        # re-generated during de-serialization
        #
        elements_list = list(itervalues(self.__elements__))
        d[self._ELE_ATTR] = _recurse(elements_list, False)

        return d

    @classmethod
    def from_iterable(cls, elements):
        """Constructs a Set from an iterable of elements.

        Args:
            elements: an iterable of elements

        Returns:
            a Set
        """
        _set = cls()
        _set.add_iterable(elements)
        return _set

    @classmethod
    def from_numeric_patt(cls, pattern):
        """Constructs a Set from a numeric pattern of elements on disk.

        Args:
             pattern: a string with one or more numeric patterns, like
                "/path/to/labels/%05d.json"

        Returns:
            a Set
        """
        parse_method = etau.get_pattern_matches
        return cls._from_element_patt(pattern, parse_method)

    @classmethod
    def from_glob_patt(cls, pattern):
        """Constructs a Set from a glob pattern of elements on disk.

        Args:
            pattern: a glob pattern like "/path/to/labels/*.json"

        Returns:
            a Set
        """
        parse_method = glob.glob
        return cls._from_element_patt(pattern, parse_method)

    @classmethod
    def from_dict(cls, d, **kwargs):
        """Constructs a Set from a JSON dictionary.

        If the dictionary has the `"_CLS"` and `cls._ELE_CLS_FIELD`
        keys, they are used to infer the Set class and underlying element
        classes, respectively, and this method can be invoked on any
        `Set` subclass that has the same `_ELE_CLS_FIELD` setting.

        Otherwise, this method must be called on the same concrete `Set`
        subclass from which the JSON was generated.

        Args:
            d: a JSON dictionary representation of a Set object
            **kwargs: optional keyword arguments that have already been parsed
                by a subclass

        Returns:
            a Set
        """
        set_cls = cls._validate_dict(d)
        elements = [
            set_cls._ELE_CLS.from_dict(dd) for dd in d[set_cls._ELE_ATTR]
        ]
        return set_cls(
            **etau.join_dicts({set_cls._ELE_ATTR: elements}, kwargs)
        )

    @classmethod
    def _from_element_patt(cls, pattern, parse_method):
        _set = cls()
        for element_path in parse_method(pattern):
            _set.add(cls.get_element_class().from_json(element_path))

        return _set

    def _get_elements_with_keys(self, keys):
        if etau.is_str(keys):
            logger.debug("Wrapping single key as a list")
            keys = [keys]

        return OrderedDict(
            (k, e) for k, e in iteritems(self.__elements__) if k in set(keys)
        )

    def _pop_elements_with_keys(self, keys):
        pop = []
        for key in keys:
            e = self.__elements__.pop(key, None)
            if e is not None:
                pop.append(e)

        return pop

    def _filter_elements(self, filters, match):
        match_fcn = lambda e: match(f(e) for f in filters)

        return OrderedDict(
            (k, e) for k, e in iteritems(self.__elements__) if match_fcn(e)
        )

    def _pop_elements(self, filters, match):
        match_fcn = lambda e: match(f(e) for f in filters)

        pop = []
        del_keys = []
        for k, e in iteritems(self.__elements__):
            if match_fcn(e):
                pop.append(e)
                del_keys.append(k)

        for k in del_keys:
            del self[k]

        return pop

    def _validate(self):
        if self._ELE_CLS is None:
            raise SetError(
                "Cannot instantiate a Set for which _ELE_CLS is None"
            )

        if self._ELE_ATTR is None:
            raise SetError(
                "Cannot instantiate a Set for which _ELE_ATTR is None"
            )

        if not issubclass(self._ELE_CLS, Serializable):
            raise SetError("%s is not Serializable" % self._ELE_CLS)

        if self._ELE_KEY_ATTR is None:
            raise SetError(
                "Cannot instantiate a Set for which _ELE_KEY_ATTR is None"
            )

    @classmethod
    def _validate_dict(cls, d):
        if cls._ELE_CLS_FIELD is None:
            raise SetError(
                "%s is an abstract set and cannot be used to load a "
                "JSON dictionary. Please use a Set subclass that "
                "defines its `_ELE_CLS_FIELD` member" % cls
            )

        if "_CLS" not in d:
            return cls

        if cls._ELE_CLS_FIELD not in d:
            raise SetError(
                "Cannot use %s to reflectively load this set because the "
                "expected field '%s' was not found in the JSON "
                "dictionary" % (cls, cls._ELE_CLS_FIELD)
            )

        return etau.get_class(d["_CLS"])


class BigMixin(object):
    """Mixin class for "big" Serializable classes, which store their elements
    on disk rather than in-memory.

    Subclasses must call BigMixin's constructor to properly initialize the
    Big object, and they must implement the `add_by_path()` and
    `add_iterable()` methods.

    `backing_dir` is an optional keyword argument for any methods that require
    initializing a new instance of a Big iterable. When `backing_dir` is
    `None`, a temporary directory is used and is deleted when the Big iterable
    is garbage collected.

    `move()` can be used to move the Big iterable between a set (persistent)
    backing directory and a temporary backing directory.
    """

    def __init__(self, backing_dir):
        """Initializes the base BigMixin.

        Args:
            backing_dir: the backing directory to use, or None if using
                temporary storage
        """
        self._backing_dir = None
        self._uses_temporary_storage = False
        self._set_backing_dir(backing_dir)

    def __del__(self):
        if self.uses_temporary_storage and os.path.exists(self.backing_dir):
            etau.delete_dir(self.backing_dir)

    @property
    def backing_dir(self):
        """The backing directory for this Big iterable."""
        return self._backing_dir

    @property
    def uses_temporary_storage(self):
        """Whether this Big iterable is backed by temporary storage."""
        return self._uses_temporary_storage

    def add_by_path(self, path):
        """Adds an element to the Big iterable via its path on disk.

        Args:
            path: the path to the element JSON file on disk
        """
        raise NotImplementedError("subclasses must implement add_by_path()")

    def add_iterable(self, elements):
        """Adds the elements in the given iterable to the Big iterable.

         Args:
            elements: an iterable of elements
        """
        raise NotImplementedError("subclasses must implement add_iterable()")

    def clear(self):
        """Deletes all elements from the Big iterable."""
        super(BigMixin, self).clear()
        etau.delete_dir(self.backing_dir)
        etau.ensure_dir(self.backing_dir)

    def copy(self, backing_dir=None):
        """Creates a deep copy of this Big iterable backed by the given
        directory.

        Args:
            backing_dir: optional backing directory to use for the new big
                iterable. If provided, it must be empty or non-existent

        Returns:
            a Big iterable
        """
        new_big = self.empty(backing_dir=backing_dir)
        new_big.add_iterable(self)
        return new_big

    def empty(self, backing_dir=None):
        """Returns an empty copy of the Big iterable backed by the given
        directory.

        Subclasses may override this method, but, by default, this method
        constructs an empty Big iterable via
        `self.__class__(backing_dir=backing_dir)`

        Args:
            backing_dir: optional backing directory to use for the new Big
                iterable. If provided, it must be empty or non-existent

        Returns:
            an empty Big iterable
        """
        return self.__class__(backing_dir=backing_dir)

    def move(self, backing_dir=None):
        """Moves the backing directory of the Big iterable to the given
        location.

        When `backing_dir` is not provided, it is moved to a temporary
        directory. Therefore, this method can be used to move a Big iterable
        out of the current `backing_dir` and into a temporary storage state.

        Args:
            backing_dir: optional backing directory to use for the new Big
                iterable. If provided, it must be empty or non-existent
        """
        if backing_dir is not None:
            etau.ensure_empty_dir(backing_dir)
        orig_backing_dir = self.backing_dir
        self._set_backing_dir(backing_dir)
        etau.move_dir(orig_backing_dir, self.backing_dir)

    def to_archive(self, archive_path, delete_backing_dir=False):
        """Writes the Big iterable to a self-contained archive file.

        The archive contains both a JSON index and the raw element JSON files
        organized in the directory structure shown below. The filename (without
        extension) defines the root directory inside the archive::

            <root>/
                index.json
                <elements>/
                    <uuid>.json

        Note that deleting the backing directory is a more efficient way to
        create the archive because it avoids data duplication, but it also
        invalidates the current Big iterable.

        Args:
            archive_path: the path + extension to write the output archive
            delete_backing_dir: whether to delete the original backing
                directory when creating the archive. By default, this is False
        """
        with etau.TempDir() as tmp_dir:
            name = os.path.splitext(os.path.basename(archive_path))[0]
            rootdir = os.path.join(tmp_dir, name)
            index_path = os.path.join(rootdir, "index.json")
            ele_dir = os.path.join(rootdir, self._ELE_ATTR)

            if delete_backing_dir:
                self.move(backing_dir=ele_dir)
                big = self
            else:
                big = self.copy(backing_dir=ele_dir)

            #
            # The backing directory embedded in the JSON is not actually used
            # (neither here nor in `from_archive`), but we set it relative to
            # the root of the archive, for completeness.
            #
            full_backing_dir = big.backing_dir
            big._backing_dir = "./" + self._ELE_ATTR
            big.write_json(index_path)
            big._backing_dir = full_backing_dir

            etau.make_archive(rootdir, archive_path)

    @classmethod
    def from_archive(
        cls, archive_path, backing_dir=None, delete_archive=False
    ):
        """Loads a Big iterable from an archive created by `to_archive()`.

        Args:
            archive_path: the path to the archive to load
            backing_dir: optional backing directory to use for the new Big
                iterable. If provided, it must be empty or non-existent
            delete_archive: whether to delete the archive after unpacking it.
                By default, this is False

        Returns:
            a Big iterable
        """
        with etau.TempDir() as tmp_dir:
            name = os.path.splitext(os.path.basename(archive_path))[0]
            rootdir = os.path.join(tmp_dir, name)
            index_path = os.path.join(rootdir, "index.json")
            tmp_backing_dir = os.path.join(rootdir, cls._ELE_ATTR)

            etau.extract_archive(
                archive_path, outdir=tmp_dir, delete_archive=delete_archive
            )
            big = cls.from_json(index_path)
            big._backing_dir = tmp_backing_dir
            big.move(backing_dir=backing_dir)

        return big

    @classmethod
    def from_paths(cls, paths, backing_dir=None):
        """Creates a Big iterable from a list of element JSON files.

        Args:
            backing_dir: optional backing directory to use for the Big
                iterable. If provided, it must be empty or non-existent
            paths: an iterable of paths to element JSON files

        Returns:
            a Big iterable
        """
        big = cls(backing_dir=backing_dir)
        for path in paths:
            big.add_by_path(path)
        return big

    @classmethod
    def from_dir(cls, source_dir, backing_dir=None):
        """Creates a Big iterable from an unstructured directory of element
        JSON files on disk.

        The source directory is traversed recursively.

        Args:
            backing_dir: optional backing directory to use for the new Big
                iterable. If provided, it must be empty or non-existent
            source_dir: the source directory from which to ingest elements

        Returns:
            a Big iterable
        """
        paths = etau.multiglob(".json", root=source_dir + "/**/*")
        return cls.from_paths(paths, backing_dir=backing_dir)

    def _set_backing_dir(self, backing_dir):
        if backing_dir is not None:
            self._uses_temporary_storage = False
            self._backing_dir = os.path.abspath(backing_dir)
        else:
            self._uses_temporary_storage = True
            self._backing_dir = etau.make_temp_dir()


class BigSet(BigMixin, Set):
    """Set that stores a (potentially huge) list of `Serializable`
    objects. The elements are stored on disk in a backing directory; accessing
    any element in the list causes an immediate READ from disk, and
    adding/setting an element causes an immediate WRITE to disk.

    BigSets store a `backing_dir` attribute that specifies the path on
    disk to the serialized elements. If a backing directory is explicitly
    provided by a user, the directory will be maintained after the BigSet
    object is deleted; if no backing directory is specified, a temporary
    backing directory is used and is deleted when the BigSet instance is
    garbage collected.

    BigSets maintain an OrderedDict of keys determined by `_ELE_KEY_ATTR` and
    values of uuids which are used to locate elements on disk. This OrderedDict
    is included in lieu of the actual elements when serializing BigSet
    instances.

    To read/write archives of BigSet that also include their elements,
    use the `to_archive()` and `from_archive()` methods, respectively.

    This class cannot be instantiated directly. Instead a subclass should
    be created for each type of element to be stored. Subclasses MUST set the
    following members:
        -  `_ELE_CLS`: the class of the element stored in the set
        -  `_ELE_KEY_ATTR`: the name of the element attribute to use to perform
            element lookups

    In addition, sublasses MAY override the following members:
        - `_ELE_CLS_FIELD`: the name of the private attribute that will store
            the class of the elements in the set
        - `_ELE_ATTR`: the name of the attribute that will store the elements
            in the set
    """

    def __init__(self, backing_dir=None, **kwargs):
        """Creates a BigSet instance.

        Args:
            backing_dir: an optional backing directory in which the elements
                are/will be stored. If omitted, a temporary backing directory
                is used
            <elements>: an optional dictionary or list of (key, uuid) tuples
                for elements in the set. The appropriate name of this keyword
                argument is determined by the `_ELE_ATTR` member of the BigSet
                subclass

        Raises:
            SetError: if there was an error while creating the set
        """
        self._validate()
        super(BigSet, self).__init__(backing_dir)

        elements = kwargs.get(self._ELE_ATTR, None) or []
        if elements:
            etau.ensure_dir(self.backing_dir)
        else:
            etau.ensure_empty_dir(self.backing_dir)

        setattr(self, self._ELE_ATTR, OrderedDict(elements))

    def __getitem__(self, key):
        return self._load_ele(self._ele_path(key))

    def __setitem__(self, key, element):
        if key not in self:
            self.__elements__[key] = self._make_uuid()

        element.write_json(self._ele_path(key))

    def __delitem__(self, key):
        etau.delete_file(self._ele_path(key))
        super(BigSet, self).__delitem__(key)

    def __iter__(self):
        return iter(self._load_ele(path) for path in self._ele_paths)

    @property
    def set_cls(self):
        """Returns the Set class associated with this BigSet."""
        module, dot, big_cls = etau.get_class_name(self).rpartition(".")
        cls_name = module + dot + big_cls[len("Big") :]
        return etau.get_class(cls_name)

    def empty_set(self):
        """Returns an empty in-memory Set version of this BigSet.

        Subclasses may override this method, but, by default, this method makes
        the empty Set via `self.set_cls()`

        Returns:
            an empty Set
        """
        return self.set_cls()

    def add(self, element):
        """Adds an element to the set.

        Args:
            element: an instance of `_ELE_CLS`
        """
        key = self.get_key(element)
        if key is None:
            key = str(uuid4())
        self[key] = element

    def add_by_path(self, path, key=None):
        """Adds an element to the set via its path on disk.

        Args:
            path: the path to the element JSON file on disk
            key: optional key value to the element. If not provided, the
                element is loaded and the key is read
        """
        if key is None:
            # Must load element to get key
            self.add(self._load_ele(path))
            return

        if key not in self:
            # Must add key to set
            self.__elements__[key] = self._make_uuid()

        etau.copy_file(path, self._ele_path(key))

    def add_set(self, set_):
        """Adds the given set's elements to the set.

        Args:
            set_: a Set of `_ELE_CLS` objects
        """
        if isinstance(set_, BigSet):
            # Copy BigSet elements via disk to avoid loading into memory
            for key in set_.keys():
                path = set_._ele_path(key)
                self.add_by_path(path, key=key)
        else:
            self.add_iterable(set_)

    def add_iterable(self, elements):
        """Adds the elements in the given iterable to the set.

        Args:
            elements: an iterable of `_ELE_CLS` objects
        """
        for element in elements:
            self.add(element)

    def filter_elements(self, filters, match=any):
        """Removes elements that don't match the given filters from the set.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
        """
        elements = self._filter_elements(filters, match)
        self.keep_keys(elements.keys())

    def pop_elements(self, filters, match=any, big=True, backing_dir=None):
        """Pops elements that match the given filters from the set.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
            big: whether to create a BigSet (True) or Set (False). By default,
                this is True
            backing_dir: an optional backing directory to use for the new
                BigSet. If provided, the directory must be empty or
                non-existent. Only relevant if `big == True`

        Returns:
            a BigSet or Set of elements matching the given filters
        """
        elements = self._filter_elements(filters, match)
        return self.pop_keys(elements.keys(), big=big, backing_dir=backing_dir)

    def keep_keys(self, keys):
        """Keeps only the elements in the set with the given keys.

        Args:
            keys: an iterable of keys of the elements to keep
        """
        self.delete_keys(set(self.__elements__.keys()) - set(keys))

    def extract_keys(self, keys, big=True, backing_dir=None):
        """Returns a set having only the elements with the given keys.

        Args:
            keys: an iterable of keys of the elements to keep
            big: whether to create a BigSet (True) or Set (False). By default,
                this is True
            backing_dir: an optional backing directory to use for the new
                BigSet. If provided, the directory must be empty or
                non-existent. Only relevant if `big == True`

        Returns:
            a BigSet or Set with the requested elements
        """
        if not big:
            # Return results in a Set
            _set = self.empty_set()
            for key in keys:
                _set.add(self[key])

            return _set

        # Return results in a BigSet
        _set = self.empty(backing_dir=backing_dir)
        for key in keys:
            path = self._ele_path(key)
            _set.add_by_path(path, key=key)

        return _set

    def pop_keys(self, keys, big=True, backing_dir=None):
        """Pops elements from the set with the given keys.

        Args:
            keys: an iterable of keys of the elements to pop
            big: whether to create a BigSet (True) or Set (False). By default,
                this is True
            backing_dir: an optional backing directory to use for the new
                BigSet. If provided, the directory must be empty or
                non-existent. Only relevant if `big == True`

        Returns:
            a BigSet or Set with the popped elements
        """
        _set = self.extract_keys(keys, big=big, backing_dir=backing_dir)
        self.delete_keys(keys)
        return _set

    def get_matches(self, filters, match=any, big=True, backing_dir=None):
        """Gets elements matching the given filters.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
            big: whether to create a BigSet (True) or Set (False). By default,
                this is True
            backing_dir: an optional backing directory to use for the new
                BigSet. If provided, the directory must be empty or
                non-existent. Only relevant if `big == True`

        Returns:
            a Set or BigSet with elements matching the filters
        """
        elements = self._filter_elements(filters, match)
        return self.extract_keys(
            elements.keys(), big=big, backing_dir=backing_dir
        )

    def sort_by(self, attr, reverse=False):
        """Sorts the elements in the set by the given attribute.

        Elements whose attribute is None are always put at the end of the set.

        Args:
            attr: the element attribute to sort by
            reverse: whether to sort in descending order. The default is False
        """

        def field_none_last(key_ele_pair):
            val = getattr(self[key_ele_pair[0]], attr)
            return ((val is None) ^ reverse, val)  # always puts None last

        elements = OrderedDict(
            sorted(
                iteritems(self.__elements__),
                reverse=reverse,
                key=field_none_last,
            )
        )
        setattr(self, self._ELE_ATTR, elements)

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Returns:
            the list of attributes
        """
        return ["backing_dir"] + super(BigSet, self).attributes()

    def serialize(self, reflective=False):
        """Serializes the set into a dictionary.

        Args:
            reflective: whether to include reflective attributes when
                serializing the object. By default, this is False

        Returns:
            a JSON dictionary representation of the set
        """
        d = OrderedDict()
        if reflective:
            d["_CLS"] = self.get_class_name()
            d[self._ELE_CLS_FIELD] = etau.get_class_name(self._ELE_CLS)

        d.update(super(BigSet, self).serialize(reflective=False))

        #
        # Note that we serialize the dictionary into a list of (key, value)
        # tuples, to preserve order
        #
        d[self._ELE_ATTR] = _recurse(list(self.__elements__.items()), False)

        return d

    def to_set(self):
        """Loads a BigSet into an in-memory Set of the associated class.

        Returns:
            a Set
        """
        set_ = self.empty_set()
        set_.add_set(self)
        return set_

    @classmethod
    def from_set(cls, set_, backing_dir=None):
        """Constructs a BigSet with the given Set's elements.

        Args:
            set_: a Set
            backing_dir: an optional backing directory to use for the new
                BigSet. If provided, must be empty or non-existent

        Returns:
            a BigSet
        """
        big_set = cls(backing_dir=backing_dir)
        big_set.add_set(set_)
        return big_set

    @classmethod
    def from_iterable(cls, elements, backing_dir=None):
        """Constructs a BigSet from an iterable of elements.

        Args:
            elements: an iterable of elements
            backing_dir: an optional backing directory to use for the new
                BigSet. If provided, must be empty or non-existent

        Returns:
            a BigSet
        """
        big_set = cls(backing_dir=backing_dir)
        big_set.add_iterable(elements)
        return big_set

    @classmethod
    def from_numeric_patt(cls, pattern, backing_dir=None):
        """Constructs a BigSet from a numeric pattern of elements on disk.

        Args:
            pattern: a string with one or more numeric patterns, like
                "/path/to/labels/%05d.json"
            backing_dir: an optional backing directory to use for the new
                BigSet. If provided, must be empty or non-existent

        Returns:
            a BigSet
        """
        parse_method = etau.get_pattern_matches
        return cls._from_element_patt(pattern, parse_method, backing_dir)

    @classmethod
    def from_glob_patt(cls, pattern, backing_dir=None):
        """Constructs a BigSet from a glob pattern of elements on disk.

        Args:
            pattern: a glob pattern like "/path/to/labels/*.json"
            backing_dir: an optional backing directory to use for the new
                BigSet. If provided, must be empty or non-existent

        Returns:
            a BigSet
        """
        parse_method = glob.glob
        return cls._from_element_patt(pattern, parse_method, backing_dir)

    @classmethod
    def from_dict(cls, d, **kwargs):
        """Constructs a BigSet from a JSON dictionary.

        Args:
            d: a JSON dictionary representation of a BigSet object
            **kwargs: optional keyword arguments that have already been parsed
                by a subclass

        Returns:
            a BigSet
        """
        set_cls = cls._validate_dict(d)
        backing_dir = d.get("backing_dir", None)
        kwargs[set_cls._ELE_ATTR] = d[set_cls._ELE_ATTR]
        return set_cls(backing_dir=backing_dir, **kwargs)

    @classmethod
    def _from_element_patt(cls, pattern, parse_method, backing_dir):
        big_set = cls(backing_dir=backing_dir)
        for element_path in parse_method(pattern):
            big_set.add_by_path(element_path)

        return big_set

    def _filter_elements(self, filters, match):
        def run_filters(key):
            ele = self[key]
            return match(f(ele) for f in filters)

        return OrderedDict(
            (k, e) for k, e in iteritems(self.__elements__) if run_filters(k)
        )

    @property
    def _ele_paths(self):
        return iter(self._ele_path(key) for key in self.__elements__)

    def _ele_filename(self, key):
        if key not in self.__elements__:
            raise KeyError("Set key %d does not exist" % key)

        return "%s.json" % self.__elements__[key]

    def _ele_path(self, key):
        return os.path.join(self.backing_dir, self._ele_filename(key))

    def _ele_path_by_uuid(self, uuid):
        return os.path.join(self.backing_dir, "%s.json" % uuid)

    def _load_ele(self, path):
        return self._ELE_CLS.from_json(path)

    def _load_ele_by_uuid(self, uuid):
        return self._load_ele(self._ele_path_by_uuid(uuid))

    @staticmethod
    def _make_uuid():
        return str(uuid4())


class SetError(Exception):
    """Exception raised when an invalid Set is encountered."""

    pass


class Container(Serializable):
    """Abstract base class for flexible containers that store homogeneous lists
    of elements of a `Serializable` class.

    Containers provide native support for all common array operations like
    getting, setting, deleting, length, and slicing. So, for example,
    `container[:5]` will return a `Container` that contains the first 5
    elements of `container`.

    This class cannot be instantiated directly. Instead a subclass should
    be created for each type of element to be stored. Subclasses MUST set the
    following members:
        -  `_ELE_CLS`: the class of the element stored in the container

    In addition, sublasses MAY override the following members:
        - `_ELE_CLS_FIELD`: the name of the private attribute that will store
            the class of the elements in the container
        - `_ELE_ATTR`: the name of the attribute that will store the elements
            in the container

    Container subclasses can optionally embed their class names and underlying
    element class names in their JSON representations, so they can be read
    reflectively from disk.

    Examples::

        from eta.core.serial import Container
        from eta.core.geometry import LabeledPointContainer

        points = LabeledPointContainer()
        points.write_json("points.json", reflective=True)

        points2 = Container.from_json("points.json")
        print(points2.__class__)  # LabeledPointContainer, not Container!
    """

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
    # Subclasses MAY set this field
    #
    _ELE_CLS_FIELD = "_ELEMENT_CLS"

    #
    # The name of the attribute that will store the elements in the container
    #
    # Subclasses MAY set this field
    #
    _ELE_ATTR = "elements"

    def __init__(self, **kwargs):
        """Creates a Container instance.

        Args:
            <elements>: an optional iterable of elements to store in the
                Container. The appropriate name of this keyword argument is
                determined by the `_ELE_ATTR` member of the Container subclass

        Raises:
            ContainerError: if there was an error while creating the container
        """
        self._validate()

        if kwargs and self._ELE_ATTR not in kwargs:
            raise ContainerError(
                "Expected elements to be provided in keyword argument '%s'; "
                "found keys %s" % (self._ELE_ATTR, list(kwargs.keys()))
            )
        elements = kwargs.get(self._ELE_ATTR, None) or []

        for e in elements:
            if not isinstance(e, self._ELE_CLS):
                raise ContainerError(
                    "Container %s expects elements of type %s but found "
                    "%s" % (self.__class__, self._ELE_CLS, e.__class__)
                )

        self.clear()
        self.add_iterable(elements)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            inds = self._slice_to_inds(idx)
            return self.extract_inds(inds)

        return self.__elements__[idx]

    def __setitem__(self, idx, element):
        if isinstance(idx, slice):
            inds = self._slice_to_inds(idx)
            for ind, ele in zip(inds, element):
                self[ind] = ele

            return

        self.__elements__[idx] = element

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            inds = self._slice_to_inds(idx)
            self.delete_inds(inds)
            return

        del self.__elements__[idx]

    def __iter__(self):
        return iter(self.__elements__)

    def __len__(self):
        return len(self.__elements__)

    def __bool__(self):
        return bool(self.__elements__)

    @property
    def __elements__(self):
        return getattr(self, self._ELE_ATTR)

    @classmethod
    def get_element_class(cls):
        """Gets the class of elements stored in this container."""
        return cls._ELE_CLS

    @classmethod
    def get_element_class_name(cls):
        """Returns the fully-qualified class name string of the element
        instances in this container.
        """
        return etau.get_class_name(cls._ELE_CLS)

    @property
    def is_empty(self):
        """Whether this container has no elements."""
        return not bool(self)

    def clear(self):
        """Deletes all elements from the container."""
        setattr(self, self._ELE_ATTR, [])

    def empty(self):
        """Returns an empty copy of the container.

        Subclasses may override this method, but, by default, this method
        constructs an empty container via `self.__class__()`

        Returns:
            an empty Container
        """
        return self.__class__()

    def add(self, element):
        """Appends an element to the container.

        Args:
            element: an instance of `_ELE_CLS`
        """
        self.__elements__.append(element)

    def add_container(self, container):
        """Appends the elements in the given container to this container.

        Args:
            container: a Container of `_ELE_CLS` objects
        """
        self.__elements__.extend(container.__elements__)

    def add_iterable(self, elements):
        """Appends the elements in the given iterable to the container.

        Args:
            elements: an iterable of `_ELE_CLS` objects
        """
        self.__elements__.extend(list(elements))

    def prepend(self, element):
        """Prepends an element to the container.

        Args:
            element: an instance of `_ELE_CLS`
        """
        self.__elements__.insert(0, element)

    def prepend_container(self, container):
        """Prepends the elements in the given container to this container.

        Args:
            container: a Container of `_ELE_CLS` objects
        """
        self.__elements__[0:0] = container.__elements__

    def prepend_iterable(self, elements):
        """Prepends the elements in the given iterable to the container.

        Args:
            elements: an iterable of `_ELE_CLS` objects
        """
        self.__elements__[0:0] = list(elements)

    def filter_elements(self, filters, match=any):
        """Removes elements that don't match the given filters from the
        container.

        The order of the remaining elements in the container are preserved.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
        """
        elements = self._filter_elements(filters, match)
        setattr(self, self._ELE_ATTR, elements)

    def pop_elements(self, filters, match=any):
        """Pops elements that match the given filters from the container.

        The order of the elements in both containers are preserved.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`

        Returns:
            a Container of elements matching the given filters
        """
        container = self.empty()
        container.add_iterable(self._pop_elements(filters, match))
        return container

    def delete_inds(self, inds):
        """Deletes the elements from the container with the given indices.

        The order of the remaining elements in the container are preserved.

        Args:
            inds: an iterable of indices of the elements to delete
        """
        for idx in sorted(inds, reverse=True):
            del self[idx]

    def keep_inds(self, inds):
        """Keeps only the elements in the container with the given indices.

        The order of the remaining elements in the container are preserved.

        Args:
            inds: an iterable of indices of the elements to keep
        """
        elements = self._get_elements_with_inds(inds)
        setattr(self, self._ELE_ATTR, elements)

    def extract_inds(self, inds):
        """Creates a new container having only the elements with the given
        indices.

        The elements are passed by reference, not copied. The order of the
        elements in the returned container matches the input indices.

        Args:
            inds: an iterable of indices of the elements to keep

        Returns:
            a Container with the requested elements
        """
        container = self.empty()
        for idx in inds:
            container.add(self[idx])

        return container

    def pop_inds(self, inds):
        """Pops elements from the container with the given indices.

        The order of the elements in both containers are preserved.

        Args:
            inds: an iterable of indices of the elements to pop

        Returns:
            a Container with the popped elements
        """
        container = self.empty()
        container.add_iterable(self._pop_elements_with_inds(inds))
        return container

    def count_matches(self, filters, match=any):
        """Counts the number of elements in the container that match the
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
        """
        elements = self._filter_elements(filters, match)
        return len(elements)

    def get_matches(self, filters, match=any):
        """Gets elements matching the given filters.

        The elements are passed by reference, not copied. The order of the
        elements in the input container is preserved.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`

        Returns:
            a Container with elements that matched the filters
        """
        container = self.empty()
        container.add_iterable(self._filter_elements(filters, match))
        return container

    def get_matching_inds(self, filters, match=any):
        """Gets the indices of the elements matching the given filters.

        The indices are returned in ascending order.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`

        Returns:
            a list of indices
        """
        return self._get_matching_inds(filters, match)

    def sort_by(self, attr, reverse=False):
        """Sorts the elements in the container by the given attribute.

        Elements whose attribute is None are always put at the end of the list.

        Args:
            attr: the element attribute to sort by
            reverse: whether to sort in descending order. The default is False
        """

        def field_none_last(ele):
            val = getattr(ele, attr)
            return ((val is None) ^ reverse, val)  # always puts None last

        elements = sorted(
            self.__elements__, reverse=reverse, key=field_none_last
        )
        setattr(self, self._ELE_ATTR, elements)

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Returns:
            the list of attributes
        """
        return [self._ELE_ATTR]

    def serialize(self, reflective=False):
        """Serializes the container into a dictionary.

        Args:
            reflective: whether to include reflective attributes when
                serializing the object. By default, this is False

        Returns:
            a JSON dictionary representation of the container
        """
        d = OrderedDict()
        if reflective:
            d["_CLS"] = self.get_class_name()
            d[self._ELE_CLS_FIELD] = etau.get_class_name(self._ELE_CLS)

        d.update(super(Container, self).serialize(reflective=False))
        return d

    @classmethod
    def from_iterable(cls, elements):
        """Constructs a Container from an iterable of elements.

        Args:
            elements: an iterable of elements

        Returns:
            a Container
        """
        container = cls()
        container.add_iterable(elements)
        return container

    @classmethod
    def from_numeric_patt(cls, pattern):
        """Constructs a Container from a numeric pattern of elements on disk.

        Args:
            pattern: a string with one or more numeric patterns, like
                "/path/to/labels/%05d.json"

        Returns:
            a Container
        """
        parse_method = etau.get_pattern_matches
        return cls._from_element_patt(pattern, parse_method)

    @classmethod
    def from_glob_patt(cls, pattern):
        """Constructs a Container from a glob pattern of elements on disk.

        Args:
             pattern: a glob pattern like "/path/to/labels/*.json"

        Returns:
            a Container
        """
        parse_method = glob.glob
        return cls._from_element_patt(pattern, parse_method)

    @classmethod
    def from_dict(cls, d, **kwargs):
        """Constructs a Container from a JSON dictionary.

        If the dictionary has the `"_CLS"` and `cls._ELE_CLS_FIELD`
        keys, they are used to infer the Container class and underlying element
        classes, respectively, and this method can be invoked on any
        `Container` subclass that has the same `_ELE_CLS_FIELD` setting.

        Otherwise, this method must be called on the same concrete `Container`
        subclass from which the JSON was generated.

        Args:
            d: a JSON dictionary representation of a Container object
            **kwargs: optional keyword arguments that have already been parsed
                by a subclass

        Returns:
            a Container
        """
        container_cls = cls._validate_dict(d)
        elements = [
            container_cls._ELE_CLS.from_dict(dd)
            for dd in d[container_cls._ELE_ATTR]
        ]
        return container_cls(
            **etau.join_dicts({container_cls._ELE_ATTR: elements}, kwargs)
        )

    @classmethod
    def _from_element_patt(cls, pattern, parse_method):
        container = cls()
        for element_path in parse_method(pattern):
            container.add(cls.get_element_class().from_json(element_path))

        return container

    def _get_elements_with_inds(self, inds):
        if isinstance(inds, numbers.Integral):
            logger.debug("Wrapping single index as a list")
            inds = [inds]

        return [e for i, e in enumerate(self.__elements__) if i in set(inds)]

    def _pop_elements_with_inds(self, inds):
        inds = set(inds)

        keep = []
        pop = []
        for i, e in enumerate(self.__elements__):
            if i in inds:
                pop.append(e)
            else:
                keep.append(e)

        setattr(self, self._ELE_ATTR, keep)
        return pop

    def _filter_elements(self, filters, match):
        return list(
            filter(lambda e: match(f(e) for f in filters), self.__elements__)
        )

    def _pop_elements(self, filters, match):
        match_fcn = lambda e: match(f(e) for f in filters)

        keep = []
        pop = []
        for e in self.__elements__:
            if match_fcn(e):
                pop.append(e)
            else:
                keep.append(e)

        setattr(self, self._ELE_ATTR, keep)
        return pop

    def _get_matching_inds(self, filters, match):
        return [
            ind
            for ind, e in enumerate(self.__elements__)
            if match(f(e) for f in filters)
        ]

    def _slice_to_inds(self, sli):
        return range(len(self))[sli]

    def _validate(self):
        """Validates that a Container instance is valid."""
        if self._ELE_CLS is None:
            raise ContainerError(
                "Cannot instantiate a Container for which _ELE_CLS is None"
            )
        if self._ELE_ATTR is None:
            raise ContainerError(
                "Cannot instantiate a Container for which _ELE_ATTR is None"
            )
        if not issubclass(self._ELE_CLS, Serializable):
            raise ContainerError("%s is not Serializable" % self._ELE_CLS)

    @classmethod
    def _validate_dict(cls, d):
        if cls._ELE_CLS_FIELD is None:
            raise ContainerError(
                "%s is an abstract container and cannot be used to load a "
                "JSON dictionary. Please use a Container subclass that "
                "defines its `_ELE_CLS_FIELD` member" % cls
            )

        if "_CLS" not in d:
            return cls

        if cls._ELE_CLS_FIELD not in d:
            raise ContainerError(
                "Cannot use %s to reflectively load this container "
                "because the expected field '%s' was not found in the "
                "JSON dictionary" % (cls, cls._ELE_CLS_FIELD)
            )

        return etau.get_class(d["_CLS"])


class BigContainer(BigMixin, Container):
    """Container that stores a (potentially huge) list of `Serializable`
    objects. The elements are stored on disk in a backing directory; accessing
    any element in the list causes an immediate READ from disk, and
    adding/setting an element causes an immediate WRITE to disk.

    BigContainers provide native support for all common array operations like
    getting, setting, deleting, length, and slicing. In the case of slicing,
    a BigContainer slice will be returned as an in-memory instance of the
    corresponding `Container` version of the `BigContainer` class. So, for
    example, `big_container[:5]` will return a `Container` that contains the
    first 5 elements of `big_container`.

    BigContainers store a `backing_dir` attribute that specifies the path on
    disk to the serialized elements. If a backing directory is explicitly
    provided by a user, the directory will be maintained after the BigContainer
    object is deleted; if no backing directory is specified, a temporary
    backing directory is used and is deleted when the BigContainer instance
    is garbage collected.

    BigContainers maintain a list of uuids in their `_ELE_ATTR` field to locate
    elements on disk. This list is included in lieu of the actual elements when
    serializing BigContainer instances.

    To read/write archives of BigContainer that also include their elements,
    use the `to_archive()` and `from_archive()` methods, respectively.

    This class cannot be instantiated directly. Instead a subclass should
    be created for each type of element to be stored. Subclasses MUST set the
    following members:
        -  `_ELE_CLS`: the class of the element stored in the container

    In addition, sublasses MAY override the following members:
        - `_ELE_CLS_FIELD`: the name of the private attribute that will store
            the class of the elements in the container
        - `_ELE_ATTR`: the name of the attribute that will store the elements
            in the container

    Examples::

        import eta.core.geometry as etag

        point = etag.LabeledPoint("origin", etag.RelativePoint(0, 0))
        points = etag.BigLabeledPointContainer("/tmp/BigLabeledPointContainer")

        # Immediately writes the LabeledPoint to disk
        points.add(point)

        # Reads the LabeledPoint from disk
        print(points[0])

        # Only the index of the BigContainer is serialized
        print(points)
    """

    def __init__(self, backing_dir=None, **kwargs):
        """Creates a BigContainer instance.

        Args:
            backing_dir: an optional backing directory in which the elements
                are/will be stored. If omitted, a temporary backing directory
                is used
            <elements>: an optional list of uuids for elements in the
                container. The appropriate name of this keyword argument is
                determined by the `_ELE_ATTR` member of the BigContainer
                subclass

        Raises:
            ContainerError: if there was an error while creating the container
        """
        self._validate()
        super(BigContainer, self).__init__(backing_dir)

        elements = kwargs.get(self._ELE_ATTR, None) or []
        if elements:
            etau.ensure_dir(self.backing_dir)
        else:
            etau.ensure_empty_dir(self.backing_dir)

        setattr(self, self._ELE_ATTR, elements)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            inds = self._slice_to_inds(idx)
            return self.extract_inds(inds, big=False)

        if idx < 0:
            idx += len(self)

        return self._load_ele(self._ele_path(idx))

    def __setitem__(self, idx, element):
        if isinstance(idx, slice):
            inds = self._slice_to_inds(idx)
            for ind, ele in zip(inds, element):
                self[ind] = ele

            return

        if idx < 0:
            idx += len(self)

        element.write_json(self._ele_path(idx))

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            inds = self._slice_to_inds(idx)
            self.delete_inds(inds)
            return

        if idx < 0:
            idx += len(self)

        etau.delete_file(self._ele_path(idx))
        super(BigContainer, self).__delitem__(idx)

    def __iter__(self):
        return iter(self._load_ele(path) for path in self._ele_paths)

    @property
    def container_cls(self):
        """Returns the Container class associated with this BigContainer."""
        module, dot, big_cls = etau.get_class_name(self).rpartition(".")
        cls_name = module + dot + big_cls[len("Big") :]
        return etau.get_class(cls_name)

    def add_by_path(self, path):
        """Appends an element to the container via its path on disk.

        No validation is done. Subclasses may choose to override this method
        for such purposes.

        Args:
            path: the path to the element JSON file on disk
        """
        uuid = self._make_uuid()
        self.__elements__.append(uuid)
        etau.copy_file(path, self._ele_path_by_uuid(uuid))

    def empty_container(self):
        """Returns an empty in-memory Container version of this BigContainer.

        Subclasses may override this method, but, by default, this method makes
        the empty Container via `self.container_cls()`

        Returns:
            an empty Container
        """
        return self.container_cls()

    def add(self, element):
        """Appends an element to the container.

        Args:
            element: an instance of `_ELE_CLS`
        """
        self.__elements__.append(self._make_uuid())
        self[-1] = element

    def add_container(self, container):
        """Appends the given container's elements to the container.

        Args:
            container: a Container of `_ELE_CLS` objects
        """
        if isinstance(container, BigContainer):
            # Copy BigContainer elements via disk to avoid loading into memory
            for path in container._ele_paths:
                self.add_by_path(path)
        else:
            self.add_iterable(container)

    def add_iterable(self, elements):
        """Appends the elements in the given iterable to the container.

        Args:
            elements: an iterable of `_ELE_CLS` objects
        """
        for element in elements:
            self.add(element)

    def filter_elements(self, filters, match=any):
        """Removes elements that don't match the given filters from the
        container.

        The order of the remaining elements in the container is preserved.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
        """
        self.keep_inds(self._filter_elements(filters, match))

    def pop_elements(self, filters, match=any, big=True, backing_dir=None):
        """Pops elements that match the given filters from the container.

        The order of the elements in both containers are preserved.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
            big: whether to create a BigContainer (True) or Container (False).
                By default, this is True
            backing_dir: an optional backing directory to use for the new
                BigContainer. If provided, the directory must be empty or
                non-existent. Only relevant if `big == True`

        Returns:
            a BigContainer or Container of elements matching the given filters
        """
        inds = self._filter_elements(filters, match)
        return self.pop_inds(inds, big=big, backing_dir=backing_dir)

    def keep_inds(self, inds):
        """Keeps only the elements in the container with the given indices.

        The order of the remaining elements in the container is preserved.

        Args:
            inds: an iterable of indices of the elements to keep
        """
        self.delete_inds(set(range(len(self))) - set(inds))

    def extract_inds(self, inds, big=True, backing_dir=None):
        """Returns a container having only the elements with the given indices.

        The order of the elements in the returned container matches the input
        indices.

        Args:
            inds: a list of indices of the elements to keep
            big: whether to create a BigContainer (True) or Container (False).
                By default, this is True
            backing_dir: an optional backing directory to use for the new
                BigContainer. If provided, the directory must be empty or
                non-existent. Only relevant if `big == True`

        Returns:
            a BigContainer or Container with the requested elements
        """
        if not big:
            # Return results in a Container
            container = self.empty_container()
            for idx in inds:
                container.add(self[idx])

            return container

        # Return results in a BigContainer
        container = self.empty(backing_dir=backing_dir)
        for idx in inds:
            container.add_by_path(self._ele_path(idx))

        return container

    def pop_inds(self, inds, big=True, backing_dir=None):
        """Pops elements from the container with the given indices.

        The order of the elements in both containers are preserved.

        Args:
            inds: a list of indices of the elements to pop
            big: whether to create a BigContainer (True) or Container (False).
                By default, this is True
            backing_dir: an optional backing directory to use for the new
                BigContainer. If provided, the directory must be empty or
                non-existent. Only relevant if `big == True`

        Returns:
            a BigContainer or Container with the popped elements
        """
        container = self.extract_inds(inds, big=big, backing_dir=backing_dir)
        self.delete_inds(inds)
        return container

    def get_matches(self, filters, match=any, big=True, backing_dir=None):
        """Returns a container with elements matching the given filters.

        The order of the elements in the container is preserved.

        Args:
            filters: a list of functions that accept elements and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
            big: whether to create a BigContainer (True) or Container (False).
                By default, this is True
            backing_dir: an optional backing directory to use to create a
                BigContainer. If provided, the directory must be empty or
                non-existent. Only relevant if `big == True`

        Returns:
            a BigContainer or Container with elements that match the filters
        """
        inds = self._filter_elements(filters, match)
        return self.extract_inds(inds, big=big, backing_dir=backing_dir)

    def sort_by(self, attr, reverse=False):
        """Sorts the elements in the container by the given attribute.

        Elements whose attribute is None are always put at the end of the list.

        Args:
            attr: the element attribute to sort by
            reverse: whether to sort in descending order. The default is False
        """

        def field_none_last(uuid):
            ele = self._load_ele_by_uuid(uuid)
            val = getattr(ele, attr)
            return ((val is None) ^ reverse, val)  # always puts None last

        elements = sorted(
            self.__elements__, reverse=reverse, key=field_none_last
        )
        setattr(self, self._ELE_ATTR, elements)

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Returns:
            the list of attributes
        """
        return ["backing_dir"] + super(BigContainer, self).attributes()

    def to_container(self):
        """Loads a BigContainer into an in-memory Container of the associated
        class.

        Returns:
            a Container
        """
        container = self.empty_container()
        container.add_container(self)
        return container

    @classmethod
    def from_container(cls, container, backing_dir=None):
        """Constructs a BigContainer with the given Container's elements.

        Args:
            container: a Container
            backing_dir: an optional backing directory to use for the new
                BigContainer. If provided, must be empty or non-existent

        Returns:
            a BigContainer
        """
        big_container = cls(backing_dir=backing_dir)
        big_container.add_container(container)
        return big_container

    @classmethod
    def from_iterable(cls, elements, backing_dir=None):
        """Constructs a BigContainer from an iterable of elements.

        Args:
            elements: an iterable of elements
            backing_dir: an optional backing directory to use for the new
                BigContainer. If provided, must be empty or non-existent

        Returns:
            a BigContainer
        """
        big_container = cls(backing_dir=backing_dir)
        big_container.add_iterable(elements)
        return big_container

    @classmethod
    def from_numeric_patt(cls, pattern, backing_dir=None):
        """Constructs a BigContainer from a numeric pattern of elements on
        disk.

        Args:
            pattern: a string with one or more numeric patterns, like
                "/path/to/labels/%05d.json"
            backing_dir: an optional backing directory to use for the new
                BigContainer. If provided, must be empty or non-existent

        Returns:
            a BigContainer
        """
        parse_method = etau.get_pattern_matches
        return cls._from_element_patt(pattern, parse_method, backing_dir)

    @classmethod
    def from_glob_patt(cls, pattern, backing_dir=None):
        """Constructs a BigContainer from a glob pattern of elements on disk.

        Args:
            pattern: a glob pattern like "/path/to/labels/*.json"
            backing_dir: an optional backing directory to use for the new
                BigContainer. If provided, must be empty or non-existent

        Returns:
            a BigContainer
        """
        parse_method = glob.glob
        return cls._from_element_patt(pattern, parse_method, backing_dir)

    @classmethod
    def from_dict(cls, d, **kwargs):
        """Constructs a BigContainer from a JSON dictionary.

        Args:
            d: a JSON dictionary representation of a BigContainer object
            **kwargs: optional keyword arguments that have already been parsed
                by a subclass

        Returns:
            a BigContainer
        """
        container_cls = cls._validate_dict(d)
        return container_cls(**etau.join_dicts(d, kwargs))

    @classmethod
    def _from_element_patt(cls, pattern, parse_method, backing_dir):
        big_container = cls(backing_dir=backing_dir)
        for element_path in parse_method(pattern):
            big_container.add_by_path(element_path)

        return big_container

    def _filter_elements(self, filters, match):
        def run_filters(idx):
            ele = self[idx]
            return match(f(ele) for f in filters)

        return list(filter(run_filters, range(len(self))))

    def _get_matching_inds(self, filters, match):
        # For BigContainers, `_filter_elements` already does the job here
        return self._filter_elements(filters, match)

    @property
    def _ele_paths(self):
        return iter(self._ele_path(idx) for idx in range(len(self)))

    def _ele_filename(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Container index %d out of bounds" % idx)
        return "%s.json" % self.__elements__[idx]

    def _ele_path(self, idx):
        return os.path.join(self.backing_dir, self._ele_filename(idx))

    def _ele_path_by_uuid(self, uuid):
        return os.path.join(self.backing_dir, "%s.json" % uuid)

    def _load_ele(self, path):
        return self._ELE_CLS.from_json(path)

    def _load_ele_by_uuid(self, uuid):
        return self._load_ele(self._ele_path_by_uuid(uuid))

    @staticmethod
    def _make_uuid():
        return str(uuid4())


class ContainerError(Exception):
    """Exception raised when an invalid Container is encountered."""

    pass


class Picklable(object):
    """Mixin class for objects that can be pickled."""

    def pickle(self, path):
        """Saves the instance to disk as a .pkl file.

        Args:
            path: the path to write the .pkl file
        """
        write_pickle(self, path)

    @classmethod
    def from_pickle(cls, path):
        """Loads the object from the given .pkl file.

        Args:
            path: the path to the .pkl file

        Returns:
            the loaded instance
        """
        return read_pickle(path)

    @staticmethod
    def is_pickle_path(path):
        """Checks the path to see if it has a pickle (.pkl) extension."""
        return path.endswith(".pkl")


class NpzWriteable(object):
    """Base class for dictionary-like objects that contain numpy.array values
    that can be written to disk as .npz files.
    """

    @staticmethod
    def is_npz_path(path):
        """Returns True/False if the provided path is an .npz file."""
        return path.endswith(".npz")

    def attributes(self):
        """Returns a list of class attributes that will be included when
        writing an .npz file.
        """
        return [a for a in vars(self) if not a.startswith("_")]

    def write_npz(self, path):
        """Writes the instance to disk in .npz format.

        Args:
            path: the path to write the .npz file
        """
        etau.ensure_basedir(path)
        d = {a: getattr(self, a) for a in self.attributes()}
        np.savez_compressed(path, **d)

    @classmethod
    def from_npz(cls, path):
        """Loads the .npz file from disk and returns an NpzWriteable instance.

        Args:
            path: the path to an .npz file

        Returns:
            an NpzWriteable instance
        """
        return cls(**np.load(path))


class ETAJSONEncoder(json.JSONEncoder):
    """Extends basic JSONEncoder to handle ETA conventions and types."""

    # pylint: disable=method-hidden
    def default(self, obj):
        if isinstance(obj, Serializable):
            return obj.serialize()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return serialize_numpy_array(obj)
        if isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode("ascii")
        return super(ETAJSONEncoder, self).default(obj)
