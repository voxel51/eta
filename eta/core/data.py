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

import os

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


class DataFileSequence(etas.Serializable):
    '''Class representing a sequence of data files on the disk.

    Implements various `eta.core.types.*Sequence*`.

    This class has the ability to have immutable bounds, which is the
    default, where bounds mean the lowest and highest indices usable.
    Immutable bounds means that the class will provide error checking on the
    sequence indices based on their bounds at the time the instance was
    initialized.  This files themselves during this time can be changed (this
    class does not enforce immutability on them).  If the instance is created
    with `immutable_bounds=False` then the user of the class will be able to
    add files to the end of the sequence in order, but not at random.

    Examples of representable file sequences:
        /path/to/video/%05d.png
        /path/to/objects/%05d.json

    Attributes:
        sequence (str): The string representing the file sequence.
        extension (str): The file extension of the sequence.
        lower_bound (int): Index of smallest file in sequence.
        upper_bound (int): Index of largest file in sequence.
    '''

    def __init__(self, sequence_string, immutable_bounds=True):
        '''Initialize the DataFileSequence instance using the sequence
        string.

        Args:
            sequence_string: The printf-style string to be used for locating
                the sequence files on disk.  E.g., `/tmp/foo/file%05d.json`
            immutable_bounds (bool): [True] enforce the lower and upper bound
                on the indices for the sequence as they exit at initialization
        '''
        self.sequence = sequence_string
        self._extension = os.path.splitext(self.sequence)[1]
        self._lower_bound, self._upper_bound = \
            etau.parse_bounds_from_dir_pattern(self.sequence)
        self._immutable_bounds = immutable_bounds
        self._iter_index = None

    def __getitem__(self, index):
        '''Implements the get item interface to allow arbitrary access to the
        paths in the sequence.
        '''
        return self.gen_path(index)

    def __iter__(self):
        '''Implements the iterable interface on the sequence of path names.'''
        self._iter_index = self._lower_bound - 1
        return self

    def __next__(self):
        '''Implements the rest of the iterable interface.'''
        if self._iter_index is None:
            raise DataFileSequenceError("next called from outside iterable.")

        self._iter_index += 1
        if not self.check_bounds(self._iter_index):
            self._iter_index = None
            raise StopIteration

        return self.gen_path(self._iter_index)

    next = __next__  # Python 2 compatibility for iteration

    @property
    def extension(self):
        return self._extension

    @property
    def lower_bound(self):
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, value):
        if self._immutable_bounds:
            raise DataFileSequenceError(
                'Cannot set bounds for a sequence with `immutable_bounds`.')
        self._lower_bound = value

    @property
    def upper_bound(self):
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, value):
        if self._immutable_bounds:
            raise DataFileSequenceError(
                'Cannot set bounds for a sequence with `immutable_bounds`.')
        self._upper_bound = value

    @property
    def starts_at_one(self):
        return self._lower_bound == 1

    @property
    def starts_at_zero(self):
        return self._lower_bound == 0

    def check_bounds(self, index):
        '''Checks if the index is within the bounds for this sequence.

        Returns:
            True is index is valid.
        '''
        if index < self.lower_bound or index > self.upper_bound:
            return False
        return True

    def gen_path(self, index):
        '''Generate and return the path for the file at the index.

        Does error-checking for sequence bounds.
        '''
        if self._immutable_bounds:
            if not self.check_bounds(index):
                raise DataFileSequenceError(
                    "Index out of bounds for sequence.")
        else:
            # In the mutable bounds case, enforce the behavior that the user
            # can add a file to the beginning or the end of the sequence.
            if index < 0:
                raise DataFileSequenceError(
                    "Indices cannot be less than zero.")
            elif index == self.lower_bound - 1:
                self._lower_bound = index
            elif index == self.upper_bound + 1:
                self._upper_bound = index
            else:
                if not self.check_bounds(index):
                    raise DataFileSequenceError(
                        "Given index out of bounds for sequence with mutable "
                        "bounds: permitted to add to the beginning or end of "
                        "the sequence but not to add arbitrarily to it.")

        return self.sequence % index

    @classmethod
    def from_dict(cls, d):
        return cls(d["sequence"])


class DataFileSequenceError(Exception):
    '''Error raised for out of bounds requests when working with
    `DataFileSequence`s.
    '''
    pass


class DataRecords(DataContainer):
    '''Container for data records.

    DataRecords is a generic container of records each having a value for
    a certain set of fields.  A DataRecords is like a NOSQL database.
    '''

    _ELE_ATTR = "records"
    _ELE_CLS_FIELD = "_RECORDS_CLS"
    _ELE_CLS = None

    def __init__(self, record_cls=None, **kwargs):
        '''Instantiate a `DataRecords` instance using the element cls given.

        This functionality adds more flexibility than standard eta
        `Containers`, which require the element class to be set statically.
        '''
        if record_cls:
            self._ELE_CLS = record_cls
        super(DataRecords, self).__init__(**kwargs)

    def add_dict(self, d, record_cls=None):
        '''Adds the contents in d to this container.

        Returns the new size of the container.
        '''
        rc = record_cls
        if rc is None:
            rc = self._ELE_CLS

        if rc is None:
            raise DataRecordsError(
                "Need record_cls to add a DataRecords object")

        dr = DataRecords(
            self._ELE_CLS,
            records=[rc.from_dict(dc) for dc in d["records"]]
        )
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

    def cull(self, field, values, keep_values=False):
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

        if rc is None:
            raise DataRecordsError(
                "Need record_cls to load a DataRecords object")

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


class DataRecordsError(Exception):
    '''Exception raised for invalid DataRecords invocations.'''
    pass


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
        serialized.

        All private attributes (those starting with "_") and attributes in
        `excluded()` are omitted from this list.

        Returns:
            the list of attributes to be serialized
        '''
        attr = super(BaseDataRecord, self).attributes()
        return [a for a in attr if a not in self.excluded()]

    def clean_optional(self):
        '''Deletes any optional attributes from the data record that are not
        set, i.e., those that are `no_default`.

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
