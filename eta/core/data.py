'''
Core data structures for representing data and containers of data.

Copyright 2017-2018, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
Jason Corso, jason@voxel51.com
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

from eta.core.serial import Serializable
import eta.core.utils as etau


class DataContainer(Serializable):
    '''Base class for containers that store lists of data instances.

    This class should not be instantiated directly. Instead a subclass should
    be created for each type of data to be stored.

    By default, DataContainer subclasses embed their class names and
    underlying data instance class names in their JSON representations, so
    data containers can be read reflectively from disk.

    Examples:
        ```
        tags= LocalizedTags(...)
        tags.write_json("tags.json")
        tags2 = DataContainer.from_json("tags.json")
        print(tags2.__class__)  # LocalizedTags, not DataContainer
        ```

    Attributes:
        <data>: a list of data instances. The field name <data> is specified by
            the `_DATA_ATTR` member, and the class of the data instances is
            specified by the `_DATA_CLS` member
    '''

    # The class of the data stored in the container
    _DATA_CLS = None

    # The name of the attribute that will store the data in the container
    _DATA_ATTR = None

    def __init__(self, **kwargs):
        '''Constructs a DataContainer.

        Args:
            <data>: optional list of data instances to store; the actual name
            of the data is arbitrary and up to the subclass.  For example, if
            we store objects in an ObjectContainer, then we expect this to be
            `objects=...` here.  This field_name must match the `_DATA_ATTR` of
            the subclass.  The default for this is "data" in the event the
            subclass does not want to customize it.
        '''
        self._validate()

        if kwargs and self._DATA_ATTR not in kwargs:
            raise DataContainerError(
                "Expected data to be provided in keyword argument '%s'; "
                "found keys %s" % (self._DATA_ATTR, list(kwargs.keys())))
        data = kwargs.get(self._DATA_ATTR, [])

        for d in data:
            if not isinstance(d, self._DATA_CLS):
                raise DataContainerError(
                    "Data container %s expects data of type %s but found "
                    "%s" % (self.__class__, self._DATA_CLS, d.__class__))

        setattr(self, self._DATA_ATTR, data)

    def __getitem__(self, index):
        return self._data.__getitem__(index)

    def __iter__(self):
        return iter(self._data)

    @property
    def _CLS(self):
        return etau.get_class_name(self)

    @property
    def _data(self):
        '''Convenience accessor to get the list of data instances stored
        independent of what the name of that attribute is.
        '''
        return getattr(self, self._DATA_ATTR)

    @classmethod
    def _validate(cls):
        if cls._DATA_CLS is None:
            raise DataContainerError(
                "_DATA_CLS is None; note that you cannot instantiate "
                "DataContainer directly.")
        if cls._DATA_ATTR is None:
            raise DataContainerError(
                "_DATA_ATTR is None; note that you cannot instantiate "
                "DataContainer directly.")

    def add(self, instance):
        '''Adds a data instance to the container.

        Args:
            instance: an instance for this container
        '''
        self._data.append(instance)

    def attributes(self):
        return ["_CLS", "_DATA_CLS", "_DATA_ATTR", self._DATA_ATTR]

    def count_matches(self, filters, match=any):
        '''Counts number of data instances that match the filters.

        Args:
            filters: a list of functions that accept data instances and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
        '''
        return self.get_matches(filters, match=match).num

    @classmethod
    def from_dict(cls, d):
        '''Constructs an DataContainer from a JSON dictionary.

        If the JSON contains the reflective `_CLS` and `_DATA_CLS` fields, they
        are used to infer the underlying data classes, and this method can
        be invoked as `DataContainer.from_dict`. Otherwise, this method must
        be called on a concrete subclass of `DataContainer`.
        '''
        if "_CLS" in d and "_DATA_CLS" in d:
            # Parse reflectively
            cls = etau.get_class(d["_CLS"])
            data_cls = etau.get_class(d["_DATA_CLS"])
        else:
            # Parse using provided class
            cls._validate()
            data_cls = cls._DATA_CLS
        return cls(**{
            cls._DATA_ATTR:
            [data_cls.from_dict(dd) for dd in d[cls._DATA_ATTR]]
        })

    @classmethod
    def get_data_class(cls):
        '''Gets the class of data instances stored in this container.'''
        return cls._DATA_CLS

    def get_matches(self, filters, match=any):
        '''Returns a data container containing only instances that match the
        filters.

        Args:
            filters: a list of functions that accept data instances and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
        '''
        return self.__class__(**{
            self._DATA_ATTR:
            list(filter(lambda o: match(f(o) for f in filters), self._data))
        })

    @property
    def num(self):
        '''The number of data instances in the container.'''
        return len(self._data)

    def serialize(self):
        '''Custom serialization implementation for DataContainers that embeds
        the class name, and the data class name in the JSON to enable
        reflective parsing when reading from disk.
        '''
        d = OrderedDict()
        d["_CLS"] = etau.get_class_name(self)
        d["_DATA_CLS"] = etau.get_class_name(self._DATA_CLS)
        d[self._DATA_ATTR] = [o.serialize() for o in self._data]
        return d


class DataContainerError(Exception):
    '''Exception raised when an invalid DataContainer is encountered.'''
    pass
