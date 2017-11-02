'''
Core data structures for working with data that can be read/written to disk.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
import collections
import json
import types

import numpy as np

import eta.core.utils as ut


def read_json(path):
    '''Reads JSON from file.'''
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    '''Writes JSON object to file, creating the output directory if necessary.

    Args:
        obj: is either an object that can be directly dumped to a JSON file or
            an instance of a subclass of serial.Serializable
        path: the output path
    '''
    if se.is_serializable(obj):
        obj = obj.serialize()
    ut.ensure_basedir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, cls=JSONNumpyEncoder)


class Serializable(object):
    '''Base class for objects that can be serialized.

    Subclasses must implement from_dict(), which defines how to construct a
    serializable object from a JSON dictionary.
    '''

    def serialize(self):
        '''Serializes the object into a dictionary.

        Subclasses can override this method, but, by default, Serializable
        objects are serialized by:
            a) calling self.attributes()
            b) invoking serialize() on serializable values and leaving other
               values untouched
            c) applying b) element-wise on list values
        '''
        return collections.OrderedDict(
            (a, _recurse(getattr(self, a))) for a in self.attributes()
        )

    def attributes(self):
        '''Returns a list of class attributes to be serialized.

        Subclasses can override this method, but, by default, all attributes
        in vars(self) are returned, minus private attributes (those starting
        with "_").

        In particular, subclasses may choose to override this method if they
        want their JSON files to be organized in a particular way.
        '''
        return [a for a in vars(self) if not a.startswith("_")]

    @classmethod
    def from_dict(cls, d):
        '''Constructs a Serializable object from a JSON dictionary.

        Subclasses must implement this method.
        '''
        raise NotImplementedError("subclass must implement from_dict()")

    @classmethod
    def from_json(cls, json_path):
        '''Constructs a Serializable object from a JSON file.

        Subclasses may override this method, but, by default, this method
        simply reads the JSON and calls from_dict(), which subclasses must
        implement.
        '''
        return cls.from_dict(read_json(json_path))


def is_serializable(obj):
    '''Returns True if the object is serializable (i.e., implements the
    Serializable interface) and False otherwise.
    '''
    return isinstance(obj, Serializable)


def _recurse(v):
    if isinstance(v, types.ListType):
        return [_recurse(vi) for vi in v]
    elif is_serializable(v):
        return v.serialize()
    return v


class JSONNumpyEncoder(json.JSONEncoder):
    '''Extends basic JSONEncoder to handle numpy scalars/arrays.'''

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(JSONNumpyEncoder, self).default(obj)
