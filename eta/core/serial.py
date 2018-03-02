'''
Core data structures for working with data that can be read/written to disk.

Copyright 2017, Voxel51, LLC
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
    else:
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
    if pretty_print:
        s = json.dumps(obj, indent=4, cls=EtaJSONEncoder, ensure_ascii=False)
    else:
        s = json.dumps(obj, cls=EtaJSONEncoder, ensure_ascii=False)
    return str(s)


class Picklable(object):
    '''Mixin class for objects that can be pickled.

    Subclasses need not implement anything.
    '''
    def pickle(self, path):
        '''Saves the instance to disk in a pickle. '''
        etau.ensure_basedir(path)
        with open(path, 'wb') as mf:
            pickle.dump(self, mf)

    @classmethod
    def from_pickle(cls, path):
        '''Loads the pickle from disk and returns the instance. '''
        with open(path, 'rb') as mf:
            M = pickle.load(mf)
        return M

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
        return json_to_str(self.serialize(), pretty_print=True)

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
