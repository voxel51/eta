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

from eta.core.config import no_default
import eta.core.serial as etas
import eta.core.utils as etau


class DataContainer(etas.Container):
    '''Abstract base class for containers that store lists of `Serializable`
    data class instances.

    This class cannot be instantiated directly. Instead a subclass should
    be created for each type of data to be stored. Subclasses MUST set the
    following members:
        -  `_ELE_CLS`: the class of the element stored in the container

    In addition, sublasses MAY override the following members:
        - `_ELE_CLS_FIELD`: the name of the private attribute that will store
            the class of the elements in the container
        - `_ELE_ATTR`: the name of the attribute that will store the elements
            in the container

    DataContainer subclasses embed their class names and underlying data
    instance class names in their JSON representations, so they can be read
    reflectively from disk.

    Examples:
        ```
        from eta.core.data import DataContainer
        from eta.core.geometry import LabeledPointContainer

        tags = LabeledPointContainer(...)
        tags.write_json("tags.json")

        tags2 = DataContainer.from_json("tags.json")
        print(tags2.__class__)  # LabeledPointContainer, not DataContainer
        ```

    Attributes:
        <data>: a list of data instances. The field name <data> is specified by
            the `_ELE_ATTR` member of the DataContainer subclass, and the class
            of the data instances is specified by the `_ELE_CLS` member
    '''

    #
    # The class of the data stored in the container
    #
    # Subclasses MUST set this field
    #
    _ELE_CLS = None

    #
    # The name of the private attribute that will store the class of the
    # data in the container
    #
    # Subclasses MAY override this field
    #
    _ELE_CLS_FIELD = "_DATA_CLS"

    #
    # The name of the attribute that will store the data in the container
    #
    # Subclasses MAY override this field
    #
    _ELE_ATTR = "data"

    @classmethod
    def get_data_class(cls):
        '''Gets the class of data stored in this container.'''
        return cls._ELE_CLS

    @classmethod
    def get_data_class_name(cls):
        '''Returns the fully-qualified class name string of the data instances
        in this container.
        '''
        return etau.get_class_name(cls._ELE_CLS)


class DataRecords(DataContainer):
    '''Container for data records.

    DataRecords is a generic container of records each having a value for
    a certain set of fields.  A DataRecords is like a NOSQL database.
    '''

    _ELE_ATTR = "records"
    _ELE_CLS_FIELD = "_RECORDS_CLS"
    _ELE_CLS = None

    def __init__(self, record_cls, **kwargs):
        '''Instantiate a `DataRecords` instance using the element cls given.

        This functionality adds more flexibility than standard eta
        `Containers`, which require the element class to be set statically.
        '''
        self._ELE_CLS = record_cls
        super(DataRecords, self).__init__(**kwargs)

    def add_dict(self, d, record_cls=None):
        '''Adds the contents in d to this container.

        Returns the new size of the container.
        '''
        rc = record_cls
        if rc is None:
            rc = self._ELE_CLS

        assert rc is not None, "need record_cls to add DataRecords objects"

        dr = DataRecords(self._ELE_CLS,
            records=[rc.from_dict(dc) for dc in d["records"]])
        self.records += dr.records

        return len(self.records)

    def add_json(self, json_path, record_cls=None):
        ''' Adds the contents from the records json_path into this
        container.'''
        return self.add_dict(etas.read_json(json_path), record_cls)

    def build_keyset(self, field):
        ''' Builds a list of unique values present across records in `field`.
        '''
        keys = set()
        for r in self.records:
            keys.add(getattr(self.records, field))
        return list(keys)

    def build_lookup(self, field):
        ''' Builds a lookup dictionary, indexed by field, that has entries as
        lists of indices within this records container based on field.
        '''
        lud = {}

        for i, r in enumerate(self.records):
            attr = getattr(r, field)
            if attr in lud:
                lud[attr].append(i)
            else:
                lud[attr] = [i]

        return lud

    def build_subsets(self, field):
        ''' Builds a dictionary, indexed by `field`, that has entries as lists
        of records.  Caution: this creates new dictionary entries based on the
        individual values of field.
        '''
        sss = {}

        for r in self.records:
            attr = getattr(r, field)
            if attr in sss:
                sss[attr].append(r)
            else:
                sss[attr] = [r]

        return sss

    def cull(self, field, values, keep_values = False):
        ''' Cull records from our store based on the value in `field`.  If
        `keep_values` is True then the list `values` specifies which records to
        keep; otherwise, it specifies which records to remove (the default).

        Returns the number of records after the operation.
        '''

        sss = self.build_subsets(field)

        if keep_values:
            v_to_use = values
        else:
            v_to_use = list(set(sss.keys()) - set(values))

        records = []
        for v in v_to_use:
            records += sss[v]
        self.records = records

        return len(self.records)

    def slice(self, field):
        ''' For `field`, build a list of the entries in the DataRecords
        container.
        '''
        sss = []
        for r in self.records:
            sss.append(getattr(r, field))
        return sss

    @classmethod
    def from_dict(cls, d, record_cls=None):
        '''Constructs the containers from a dictionary.
        The record_cls is needed to know what types of records we are talking
        about.  However, it can be set also by changing the static class
        variable via DataRecords.set_record_cls(record_cls).  This one passed
        to from_dict will override the static class variable.  But, one needs
        to be set.
        '''
        rc = record_cls
        if rc is None:
            rc = cls._ELE_CLS

        assert rc is not None, "need record_cls to load a DataRecords object"

        return DataRecords(records=[rc.from_dict(dc) for dc in d["records"]])

    @classmethod
    def from_json(cls, json_path, record_cls=None):
        '''Constructs a DataRecords object from a JSON file.'''
        return cls.from_dict(etas.read_json(json_path), record_cls)

    @classmethod
    def get_record_cls(cls):
        '''Returns the current set class that will be used to instantiate
        records.'''
        return cls._ELE_CLS

    @classmethod
    def set_record_cls(cls, record_cls):
        '''Sets the class that will be used to instantiate records.'''
        cls._ELE_CLS = record_cls

    def subset_from_indices(self, indices):
        ''' Create a new DataRecords instance with the same settings as this
        one, and populate it with only those entries from arg indices.
        '''
        newdr = DataRecords()
        newdr.set_record_cls(self.get_record_cls())
        newdr.records = \
            [r for (i, r) in enumerate(self.records) if i in indices]
        return newdr


DEFAULT_DATA_RECORDS_FILENAME = "records.json"


class BaseDataRecord(etas.Serializable):
    '''Base class for all data records.

    Data records are flexible containers that function as dictionary-like
    classes that define the required, optional, and excluded keys that they
    support.

    @todo excluded is redundant. We should only serialize required and optional
    attributes; all others should be excluded by default.
    '''

    def __init__(self):
        '''Base constructor for all data records.'''
        self.clean_optional()

    def __getitem__(self, key):
        '''Provides dictionary-style `[key]` access to the attributes of the
        data record.
        '''
        return getattr(self, key)

    def attributes(self):
        '''Returns the list of attributes of the data record that are to be
        serialized, i.e., all attributes that are not in `excluded()`

        Returns:
            the list of attributes to be serialized
        '''
        return [a for a in vars(self) if a not in self.excluded()]

    def clean_optional(self):
        '''Deletes any optional attributes from the data record that are not
        set, i.e., those that are `no_default`.s

        Note that `None` is a valid value for an attribute.
        '''
        for o in self.optional():
            if hasattr(self, o) and getattr(self, o) is no_default:
                delattr(self, o)

    @classmethod
    def from_dict(cls, d):
        '''Constructs a data record from a JSON dictionary. All required
        attributes must be present in the dictionary, and any optional
        attributes that are present will also be stored.

        Args:
            a JSON dictonary containing (at minimum) all of the required
                attributes for the data record

        Returns:
            an instance of the data record

        Raises:
            KeyError: if a required attribute was not found in the input
                dictionary
        '''
        kwargs = {k: d[k] for k in cls.required()}                  # required
        kwargs.update({k: d[k] for k in cls.optional() if k in d})  # optional
        return cls(**kwargs)

    @classmethod
    def required(cls):
        '''Returns a list of attributes that are required by all instances of
        the data record. By default, an empty list is returned.
        '''
        return []

    @classmethod
    def optional(cls):
        '''Returns a list of attributes that are optionally included in the
        data record if they are present in the data dictionary. By default,
        an empty list is returned.
        '''
        return []

    @classmethod
    def excluded(cls):
        '''Return a list of attributes that should always be excluded when the
        data record is serialized. By default, an empty list is returned.
        '''
        return []


class LabeledVideoRecord(BaseDataRecord):
    '''A simple, reusable DataRecord for a labeled video.

    Args:
        video_path: the path to the video
        label: the label of the video
        group: an optional group attribute that provides additional information
            about the video. For example, if multiple video clips were sampled
            from a single video, this attribute can be used to specify the
            parent video
    '''
    def __init__(self, video_path, label, group=no_default):
        '''Creates a new LabeledVideoRecord instance.'''
        self.video_path = video_path
        self.label = label
        self.group = group

        self.clean_optional()

    @classmethod
    def optional(cls):
        return ["group"]

    @classmethod
    def required(cls):
        return ["video_path", "label"]
