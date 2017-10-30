'''
Core data structures for working with data that can be read/written to disk.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
import collections
import types

import eta.core.utils as ut


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
        return cls.from_dict(ut.read_json(json_path))


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
