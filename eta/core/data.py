'''
Core data structures for representing data and containers of data.

Copyright 2017-2020, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
Jason Corso, jason@voxel51.com
Tyler Ganter, tyler@voxel51.com
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

from collections import defaultdict
import os

import numpy as np

from eta.core.config import no_default
import eta.core.numutils as etan
from eta.core.serial import Container, NpzWriteable, Serializable
import eta.core.utils as etau


def majority_vote_categorical_attrs(attrs, confidence_weighted=False):
    '''Performs majority votes over the given attributes, which are assumed to
    be `CategoricalAttribute`s.

    A separate vote is performed for attributes of each name.

    If a list of AttributeContainers is provided, all attributes are combined
    into a single vote.

    Args:
        attrs: an AttributeContainer or list of AttributeContainers
        confidence_weighted: whether to weight the vote by confidence. By
            default, this is False

    Returns:
        an AttributeContainer containing the voted attributes
    '''
    if not isinstance(attrs, list):
        attrs = [attrs]

    accums = defaultdict(lambda: etan.Accumulator())
    for _attrs in attrs:
        for attr in _attrs:
            accums[attr.name].add(attr.value, weight=attr.confidence or 0.0)

    voted_attrs = AttributeContainer()
    for name, accum in iteritems(accums):
        value = accum.argmax(weighted=confidence_weighted)
        confidence = accum.get_average_weight(value) or None
        attr = CategoricalAttribute(name, value, confidence=confidence)
        voted_attrs.add(attr)

    return voted_attrs


class Attribute(Serializable):
    '''Base class for attributes.

    This class assumes the convention that attribute class `<AttributeClass>`
    defines an associated schema class `<AttributeClass>Schema` in the same
    module.
    '''

    def __init__(self, name, value, confidence=None):
        '''Constructs an Attribute instance.

        Args:
            name: the attribute name
            value: the attribute value
            confidence: an optional confidence of the value, in [0, 1]. By
                default, no confidence is stored
        '''
        self.type = etau.get_class_name(self)
        self.name = name
        self.value = self.parse_value(value)
        self.confidence = confidence

    @classmethod
    def get_schema_cls(cls):
        '''Gets the schema class for this attribute.'''
        class_name = etau.get_class_name(cls)
        return etau.get_class(class_name + "Schema")

    @classmethod
    def parse_value(cls, value):
        '''Parses the attribute value.'''
        raise NotImplementedError("subclass must implement parse_value()")

    def attributes(self):
        '''Returns the list of attributes to serialize.

        Optional attributes that were not provided (i.e., are None) are omitted
        from this list.
        '''
        _attrs = ["type", "name", "value"]
        if self.confidence is not None:
            _attrs.append("confidence")
        return _attrs

    @classmethod
    def _from_dict(cls, d):
        '''Internal implementation of `from_dict()`.

        Subclasses MUST implement this method, NOT `from_dict()`, if they
        contain custom fields. Moreover, such implementations must internally
        call this super method to ensure that the base `Attribute` is properly
        initialized.
        '''
        confidence = d.get("confidence", None)
        return cls(d["name"], d["value"], confidence=confidence)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an Attribute from a JSON dictionary.'''
        attr_cls = etau.get_class(d["type"])
        return attr_cls._from_dict(d)


class CategoricalAttribute(Attribute):
    '''Class encapsulating categorical attributes.'''

    def __init__(self, name, value, confidence=None, top_k_probs=None):
        '''Constructs a CategoricalAttribute instance.

        Args:
            name: the attribute name
            value: the attribute value
            confidence: an optional confidence of the value, in [0, 1]. By
                default, no confidence is stored
            top_k_probs: an optional dictionary mapping values to
                probabilities. By default, no probabilities are stored
        '''
        super(CategoricalAttribute, self).__init__(
            name, value, confidence=confidence)
        self.top_k_probs = top_k_probs

    @classmethod
    def parse_value(cls, value):
        '''Parses the attribute value.'''
        return value

    def attributes(self):
        '''Returns the list of attributes to serialize.

        Optional attributes that were not provided (i.e., are None) are omitted
        from this list.
        '''
        _attrs = super(CategoricalAttribute, self).attributes()
        if self.top_k_probs is not None:
            _attrs.append("top_k_probs")
        return _attrs

    @classmethod
    def _from_dict(cls, d):
        attr = super(CategoricalAttribute, cls)._from_dict(d)
        attr.top_k_probs = d.get("top_k_probs", None)
        return attr


class NumericAttribute(Attribute):
    '''Class encapsulating numeric attributes.'''

    @classmethod
    def parse_value(cls, value):
        '''Parses the attribute value.'''
        return float(value)


class BooleanAttribute(Attribute):
    '''Class encapsulating boolean attributes.'''

    @classmethod
    def parse_value(cls, value):
        '''Parses the attribute value.'''
        return bool(value)


class AttributeSchema(Serializable):
    '''Base class for attribute schemas.

    This class assumes the convention that attribute class `<AttributeClass>`
    defines an associated schema class `<AttributeClass>Schema` in the same
    module.
    '''

    def __init__(self, name):
        '''Initializes the AttributeSchema. All subclasses should call this
        constructor.

        Args:
            name: the name of the attribute
        '''
        self.name = name
        self.type = etau.get_class_name(self)[:-6]  # removes "Schema"
        self._attr_cls = etau.get_class(self.type)

    def get_attribute_class(self):
        '''Gets the Attribute class associated with this schema.'''
        return self._attr_cls

    def validate_type(self, attr):
        '''Validates that the attribute is of the correct class.

        Args:
            attr: an Attribute

        Raises:
            AttributeSchemaError: if the attribute is not of the class expected
                by the schema
        '''
        if not isinstance(attr, self._attr_cls):
            raise AttributeSchemaError(
                "Expected attribute '%s' to have type '%s'; found '%s'" %
                (attr.name, self.type, etau.get_class_name(attr)))

    def validate_attribute(self, attr):
        '''Validates that the attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Raises:
            AttributeSchemaError: if the attribute is not compliant with the
                schema
        '''
        if attr.name != self.name:
            raise AttributeSchemaError(
                "Expected name '%s'; found '%s'" % (self.name, attr.name))

        self.validate_type(attr)

        if not self.is_valid_value(attr.value):
            raise AttributeSchemaError(
                "Value '%s' of attribute '%s' is not allowed by the "
                "schema " % (attr.value, attr.name))

    def is_valid_attribute(self, attr):
        '''Returns True/False if the attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Returns:
            True/False
        '''
        try:
            self.validate_attribute(attr)
            return True
        except AttributeSchemaError:
            return False

    def is_valid_value(self, value):
        '''Returns True/False if value is valid for the attribute.'''
        raise NotImplementedError("subclass must implement is_valid_value()")

    def add_attribute(self, attr):
        '''Incorporates the given Attribute into the schema.'''
        raise NotImplementedError("subclass must implement add_attribute()")

    def merge_schema(self, schema):
        '''Incorporates the given AttributeSchema into the schema.'''
        raise NotImplementedError("subclass must implement merge_schema()")

    @staticmethod
    def get_kwargs(d):
        '''Extracts the relevant keyword arguments for this schema from the
        JSON dictionary.
        '''
        raise NotImplementedError("subclass must implement get_kwargs()")

    @classmethod
    def from_dict(cls, d):
        '''Constructs an AttributeSchema from a JSON dictionary.

        Note that this function reflectively parses the schema type from the
        dictionary, so subclasses do not need to implement this method.
        '''
        attr_cls = etau.get_class(d["type"])
        schema_cls = attr_cls.get_schema_cls()
        return schema_cls(d["name"], **schema_cls.get_kwargs(d))


class AttributeSchemaError(Exception):
    '''Error raised when an AttributeSchema is violated.'''
    pass


class CategoricalAttributeSchema(AttributeSchema):
    '''Class that encapsulates the schema of categorical attributes.'''

    def __init__(self, name, categories=None):
        '''Creates a CategoricalAttributeSchema instance.

        Args:
            name: the name of the attribute
            categories: a set of valid categories for the attribute. By
                default, an empty set is used
        '''
        super(CategoricalAttributeSchema, self).__init__(name)
        self.categories = set(categories or [])

    def is_valid_value(self, value):
        '''Returns True/False if value is valid for the attribute.'''
        return value in self.categories

    def add_attribute(self, attr):
        '''Incorporates the given CategoricalAttribute into the schema.'''
        self.validate_type(attr)
        self.categories.add(attr.value)

    def merge_schema(self, schema):
        '''Merges the given CategoricalAttributeSchema into this schema.'''
        self.categories.update(schema.categories)

    @staticmethod
    def get_kwargs(d):
        '''Extracts the relevant keyword arguments for this schema from the
        JSON dictionary.
        '''
        return {"categories": d.get("categories", None)}


class NumericAttributeSchema(AttributeSchema):
    '''Class that encapsulates the schema of numeric attributes.'''

    def __init__(self, name, range=None):
        '''Creates a NumericAttributeSchema instance.

        Args:
            name: the name of the attribute
            range: the (min, max) range for the attribute
        '''
        super(NumericAttributeSchema, self).__init__(name)
        self.range = tuple(range or [])

    def is_valid_value(self, value):
        '''Returns True/False if value is valid for the attribute.'''
        if not self.range:
            return False
        return value >= self.range[0] and value <= self.range[1]

    def add_attribute(self, attr):
        '''Incorporates the given NumericAttribute into the schema.'''
        self.validate_type(attr)
        value = attr.value
        if not self.range:
            self.range = (value, value)
        else:
            self.range = min(self.range[0], value), max(self.range[1], value)

    def merge_schema(self, schema):
        '''Merges the given NumericAttributeSchema into this schema.'''
        if not self.range:
            self.range = schema.range
        else:
            self.range = (
                min(self.range[0], schema.range[0]),
                max(self.range[1], schema.range[1])
            )

    @staticmethod
    def get_kwargs(d):
        '''Extracts the relevant keyword arguments for this schema from the
        JSON dictionary.
        '''
        return {"range": d.get("range", None)}


class BooleanAttributeSchema(AttributeSchema):
    '''Class that encapsulates the schema of boolean attributes.'''

    def __init__(self, name):
        '''Creates a BooleanAttributeSchema instance.

        Args:
            name: the name of the attribute
        '''
        super(BooleanAttributeSchema, self).__init__(name)

    def is_valid_value(self, value):
        '''Returns True/False if value is valid for the attribute.'''
        return isinstance(value, bool)

    def add_attribute(self, attr):
        '''Incorporates the given BooleanAttribute into the schema.'''
        self.validate_type(attr)

    def merge_schema(self, schema):
        '''Merges the given BooleanAttributeSchema into this schema.'''
        pass

    @staticmethod
    def get_kwargs(d):
        '''Extracts the relevant keyword arguments for this schema from the
        JSON dictionary.
        '''
        return {}


class AttributeContainer(Container):
    '''A container for attributes.'''

    _ELE_CLS = Attribute
    _ELE_CLS_FIELD = "_ATTR_CLS"
    # Note: we can't use "attributes" here due to `Serializable.attributes()`
    _ELE_ATTR = "attrs"

    def __init__(self, schema=None, **kwargs):
        '''Creates an AttributeContainer instance.

        Args:
            schema: an optional AttributeContainerSchema to enforce on the
                attributes in this container. By default, no schema is enforced
            **kwargs: valid keyword arguments for Container()

        Raises:
            AttributeContainerSchemaError: if a schema was provided but the
                attributes added to the container violate it
        '''
        super(AttributeContainer, self).__init__(**kwargs)
        self.schema = None
        if schema is not None:
            self.set_schema(schema)

    @property
    def has_schema(self):
        '''Returns True/False whether the container has an enforced schema.'''
        return self.schema is not None

    def add(self, attr):
        '''Adds an attribute to the container.

        Args:
            attr: an Attribute

        Raises:
            AttributeContainerSchemaError: if this container has a schema
                enforced and the given attribute violates it
        '''
        if self.has_schema:
            self._validate_attribute(attr)
        super(AttributeContainer, self).add(attr)

    def add_container(self, container):
        '''Adds the attributes in the given container to this container.

        Args:
            container: an AttributeContainer instance

        Raises:
            AttributeContainerSchemaError: if this container has a schema
                enforced and an attribute in the given container violates it
        '''
        if self.has_schema:
            for attr in container:
                self._validate_attribute(attr)
        super(AttributeContainer, self).add_container(container)

    def sort_by_name(self, reverse=False):
        '''Sorts the attributes in the container by name.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("name", reverse=reverse)

    def filter_by_schema(self, schema):
        '''Removes attributes from this container that are not compliant with
        the given schema.

        Args:
            schema: an AttributeContainerSchema
        '''
        filter_func = lambda attr: schema.is_valid_attribute(attr)
        self.filter_elements([filter_func])

    def get_schema(self):
        '''Gets the current enforced schema for the container, or None if
        no schema is enforced.
        '''
        return self.schema

    def get_active_schema(self):
        '''Returns an AttributeContainerSchema describing the active schema of
        the container.
        '''
        return AttributeContainerSchema.build_active_schema(self)

    def set_schema(self, schema, filter_by_schema=False):
        '''Sets the enforced schema to the given AttributeContainerSchema.

        Args:
            schema: the AttributeContainerSchema to use
            filter_by_schema: whether to filter any invalid values from the
                container after changing the schema. By default, this is False

        Raises:
            AttributeContainerSchemaError: if `filter_by_schema` was False and
                the container contains values that are not compliant with the
                schema
        '''
        self.schema = schema
        if not self.has_schema:
            return

        if filter_by_schema:
            self.filter_by_schema(self.schema)
        else:
            self._validate_schema()

    def freeze_schema(self):
        '''Sets the enforced schema for the container to the current active
        schema.
        '''
        self.set_schema(self.get_active_schema())

    def remove_schema(self):
        '''Removes the enforced schema from the container.'''
        self.schema = None

    def get_attrs_with_name(self, name):
        '''Get a list of all attributes with a given name

        Args:
            name: the Attribute name
        Returns:
            a list of attributes with the given name
        '''
        return [attr for attr in self if attr.name == name]

    def get_attr_with_name(self, name):
        '''Get the single attribute with a given name

        Args:
            name: the Attribute name

        Returns:
            the Attribute

        Raises:
            ValueError if there is not exactly one Attribute with the name
            `name`
        '''
        attrs = self.get_attrs_with_name(name)
        if len(attrs) != 1:
            raise ValueError("Expected 1 attr with name '%s' but there are %d"
                             % (name, len(attrs)))
        return attrs[0]

    def get_attr_values_with_name(self, name):
        '''Get a list of values for all attributes with a given name

        Args:
            name: the Attribute name

        Returns:
            a list of attributes values with the given name
        '''
        return [attr.value for attr in self.get_attrs_with_name(name)]

    def get_attr_value_with_name(self, name):
        '''Get the value of the single attribute with a given name

        Args:
            name: the Attribute name

        Returns:
            the Attribute value

        Raises:
            ValueError if there is not exactly one Attribute with the name
            `name`
        '''
        return self.get_attr_with_name(name).value

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.'''
        _attrs = []
        if self.has_schema:
            _attrs.append("schema")
        _attrs += super(AttributeContainer, self).attributes()
        return _attrs

    def _validate_attribute(self, attr):
        if self.has_schema:
            self.schema.validate_attribute(attr)

    def _validate_schema(self):
        if self.has_schema:
            for attr in self:
                self._validate_attribute(attr)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an AttributeContainer from a JSON dictionary.'''
        container = super(AttributeContainer, cls).from_dict(d)
        schema = d.get("schema", None)
        if schema is not None:
            container.set_schema(AttributeContainerSchema.from_dict(schema))
        return container


class AttributeContainerSchema(Serializable):
    '''A schema for an AttributeContainer.'''

    def __init__(self, schema=None):
        '''Creates an AttributeContainerSchema instance.

        Args:
            schema: a dictionary mapping attribute names to AttributeSchema
                instances. By default, an empty schema is created
        '''
        self.schema = schema or {}

    def has_attribute(self, name):
        '''Returns True/False if the schema has an attribute `name`.'''
        return name in self.schema

    def get_attribute_class(self, name):
        '''Gets the class of the Attribute with the given name.

        Raises:
            AttributeContainerSchemaError: if the schema does not have an
                attribute with the given name
        '''
        if not self.has_attribute(name):
            raise AttributeContainerSchemaError(
                "Attribute '%s' is not allowed by the schema" % name)
        return self.schema[name].get_attribute_class()

    def add_attribute(self, attr):
        '''Incorporates the given Attribute into the schema.'''
        name = attr.name
        if name not in self.schema:
            schema_cls = attr.get_schema_cls()
            self.schema[name] = schema_cls(name)
        self.schema[name].add_attribute(attr)

    def add_attributes(self, attrs):
        '''Incorporates the given AttributeContainer into the schema.'''
        for attr in attrs:
            self.add_attribute(attr)

    def merge_schema(self, schema):
        '''Merges the given AttributeContainerSchema into the schema.'''
        for name, attr_schema in iteritems(schema.schema):
            if name not in self.schema:
                self.schema[name] = attr_schema
            else:
                self.schema[name].merge_schema(attr_schema)

    def validate_attribute(self, attr):
        '''Validates that the attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Raises:
            AttributeContainerSchemaError: if the attribute violates the
                schema
        '''
        if not self.has_attribute(attr.name):
            raise AttributeContainerSchemaError(
                "Attribute '%s' is not allowed by the schema" % attr.name)

        try:
            self.schema[attr.name].validate_attribute(attr)
        except AttributeSchemaError as e:
            raise AttributeContainerSchemaError(e)

    def is_valid_attribute(self, attr):
        '''Returns True/False if the Attribute is compliant with the schema.'''
        return (
            self.has_attribute(attr.name) and
            self.schema[attr.name].is_valid_attribute(attr))

    @classmethod
    def build_active_schema(cls, attrs):
        '''Builds an AttributeContainerSchema that describes the active schema
        of the given attributes.

        Args:
            attrs: an AttributeContainer

        Returns:
            an AttributeContainerSchema describing the active schema of the
                attributes
        '''
        schema = cls()
        schema.add_attributes(attrs)
        return schema

    @classmethod
    def from_dict(cls, d):
        '''Constructs an AttributeContainerSchema from a JSON dictionary.'''
        schema = d.get("schema", None)
        if schema is not None:
            schema = {
                k: AttributeSchema.from_dict(v) for k, v in iteritems(schema)
            }
        return cls(schema=schema)


class AttributeContainerSchemaError(Exception):
    '''Error raised when an AttributeContainerSchema is violated.'''
    pass


class DataFileSequence(Serializable):
    '''Class representing a sequence of data files on disk.

    When a DataFileSequence is created, it must correspond to actual files on
    disk. However, when `immutable_bounds=False`, the `gen_path()` method can
    be used to add files to the beginning or end of the sequence.

    Examples of representable file sequences:
        /path/to/video/%05d.png
        /path/to/objects/%05d.json

    Attributes:
        sequence (str): the sequence pattern
        immutable_bounds (bool): whether the lower and upper bounds of the
            sequence can be modified
        extension (str): the file extension of the pattern
        lower_bound (int): the smallest index in the sequence
        upper_bound (int): the largest index in the sequence (inclusive)
    '''

    def __init__(self, sequence, immutable_bounds=True):
        '''Creates a DataFileSequence instance for the given sequence.

        Args:
            sequence: The printf-style pattern describing the files on disk,
                e.g., `/path/to/frame-%05d.json`
            immutable_bounds: whether the lower and upper bounds of the
                sequence should be immutable. By default, this is True

        Raises:
            DataFileSequenceError: if the sequence did not match any files on
                disk
        '''
        self.sequence = sequence
        self.immutable_bounds = immutable_bounds
        self._extension = os.path.splitext(self.sequence)[1]
        self._lower_bound, self._upper_bound = etau.parse_bounds_from_pattern(
            self.sequence)
        self._iter_index = None

        if self._lower_bound is None or self._upper_bound is None:
            raise DataFileSequenceError(
                "Sequence '%s' did not match any files on disk" % sequence)

    def __getitem__(self, index):
        return self.gen_path(index)

    def __iter__(self):
        self._iter_index = self._lower_bound - 1
        return self

    def __next__(self):
        self._iter_index += 1
        if not self.check_bounds(self._iter_index):
            self._iter_index = None
            raise StopIteration
        return self.gen_path(self._iter_index)

    @property
    def extension(self):
        return self._extension

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @lower_bound.setter
    def lower_bound(self, value):
        if self.immutable_bounds:
            raise DataFileSequenceError(
                "Cannot set bounds of an immutable sequence.")
        self._lower_bound = min(value, self.upper_bound)

    @upper_bound.setter
    def upper_bound(self, value):
        if self.immutable_bounds:
            raise DataFileSequenceError(
                "Cannot set bounds of an immutable sequence.")
        self._upper_bound = max(value, self.lower_bound)

    @property
    def starts_at_zero(self):
        return self._lower_bound == 0

    @property
    def starts_at_one(self):
        return self._lower_bound == 1

    def check_bounds(self, index):
        '''Checks if the index is within the bounds for this sequence.

        Args:
            index: a sequence index

        Returns:
            True/False
        '''
        if index < self.lower_bound or index > self.upper_bound:
            return False
        return True

    def gen_path(self, index):
        '''Generates the path for the file with the given sequence index.

        If the sequence has mutable bounds, the index can extend the sequence
        consecutively (i.e., by one index) above or below the current bounds.

        Args:
            index: a sequence index

        Returns:
            the generated path for the given index
        '''
        if self.immutable_bounds:
            if not self.check_bounds(index):
                raise DataFileSequenceError(
                    "Index %d out of bounds [%d, %d]" %
                    (index, self.lower_bound, self.upper_bound))
        elif index < 0:
            raise DataFileSequenceError("Indices must be nonnegative")
        elif index == self.lower_bound - 1:
            self._lower_bound = index
        elif index == self.upper_bound + 1:
            self._upper_bound = index
        elif not self.check_bounds(index):
            raise DataFileSequenceError(
                "Index %d out of bounds [%d, %d]; mutable sequences can be "
                "extended at most one index above/below." %
                (index, self.lower_bound, self.upper_bound))

        return self.sequence % index

    @classmethod
    def build_for_dir(cls, dir_path):
        '''Builds a `DataFileSequence` for the given directory.'''
        return cls(etau.parse_dir_pattern(dir_path)[0])

    @classmethod
    def from_dict(cls, d):
        '''Builds a `DataFileSequence` from a JSON dictioanry.'''
        return cls(d["sequence"], immutable_bounds=d["immutable_bounds"])


class DataFileSequenceError(Exception):
    '''Error raised when an invalid DataFileSequence is encountered.'''
    pass


class DataRecords(Container):
    '''Container class for data records.

    `DataRecords` is a generic container of records each having a value for
    a certain set of fields. When creating `DataRecords` instances, you must
    provide a `record_cls` that specifies the subclass of `BaseDataRecord`
    that you plan to store in the container.

    When `DataRecords` instances are serialized, they can optionally have their
    reflective `_CLS` and `_RECORD_CLS` attributes set by passing
    `reflective=True`. When this is done, `DataRecords` can be read from disk
    via `DataRecords.from_json("/path/to/records.json")` and the class of the
    records in the container will be inferred while loading.
    '''

    _ELE_CLS = None  # this is set per-instance for DataRecords
    _ELE_CLS_FIELD = "_RECORD_CLS"
    _ELE_ATTR = "records"

    def __init__(self, record_cls, **kwargs):
        '''Creates a `DataRecords` instance.

        Args:
            record_cls: the records class to use for this container
            records: an optional list of records to add to the container
        '''
        self._ELE_CLS = record_cls
        super(DataRecords, self).__init__(**kwargs)

    @property
    def record_cls(self):
        '''Returns the class of records in the container.'''
        return self._ELE_CLS

    def add_dict(self, d, record_cls=None):
        '''Adds the records in the dictionary to the container.

        Args:
            d: a DataRecords dictionary
            record_cls: an optional records class to use when parsing the
                records dictionary. If None, the _ELE_CLS class of this
                instance is used

        Returns:
            the number of elements in the container
        '''
        rc = record_cls or self._ELE_CLS
        self.add_container(self.from_dict(d, record_cls=rc))
        return len(self)

    def add_json(self, json_path, record_cls=None):
        '''Adds the records in the JSON file to the container.

        Args:
            json_path: the path to a DataRecords JSON file
            record_cls: an optional records class to use when parsing the
                records dictionary. If None, the _ELE_CLS class of this
                instance is used

        Returns:
            the number of elements in the container
        '''
        rc = record_cls or self._ELE_CLS
        self.add_container(self.from_json(json_path, record_cls=rc))
        return len(self)

    def build_keyset(self, field):
        '''Returns a list of unique values of `field` across the records in
        the container.
        '''
        keys = set()
        for r in self.__elements__:
            keys.add(getattr(r, field))
        return list(keys)

    def build_lookup(self, field):
        '''Builds a lookup dictionary indexed by `field` whose values are lists
        of indices of the records whose `field` attribute matches the
        corresponding key.
        '''
        lud = defaultdict(list)
        for i, r in enumerate(self.__elements__):
            lud[getattr(r, field)].append(i)
        return dict(lud)

    def build_subsets(self, field):
        '''Builds a dictionary indexed by `field` whose values are lists of
        records whose `field` attribute matches the corresponding key.
        '''
        sss = defaultdict(list)
        for r in self.__elements__:
            sss[getattr(r, field)].append(r)
        return dict(sss)

    def cull(self, field, keep_values=None, remove_values=None):
        '''Cull records from the container based on `field`.

        Args:
            field: the field to process
            keep_values: an optional list of field values to keep
            remove_values: an optional list of field values to remove

        Returns:
            the number of elements in the container
        '''
        lud = self.build_lookup(field)

        # Determine values to keep
        if remove_values:
            keep_values = set(lud.keys()) - set(remove_values)
        if not keep_values:
            raise DataRecordsError(
                "Either keep_values or remove_values must be provided")

        # Cull records
        inds = set()
        for v in keep_values:
            inds.update(lud[v])
        self.keep_inds(inds)

        return len(self)

    def cull_with_function(self, field, func):
        '''Cull records from the container for which `field` returns
        something that evaluates to False when passed through func.

        Args:
            field: the field to process
            func: the test function

        Returns:
            the number of elements in the container
        '''
        lud = self.build_lookup(field)

        # Cull records
        inds = set()
        for v in lud:
            if func(v):
                inds.update(lud[v])
        self.keep_inds(inds)

        return len(self)

    def slice(self, field):
        '''Returns a list of `field` values for the records in the
        container.
        '''
        return [getattr(r, field) for r in self.__elements__]

    def subset_from_indices(self, indices):
        '''Creates a new DataRecords instance containing only the subset of
        records in this container with the specified indices.
        '''
        return self.extract_inds(indices)

    def attributes(self):
        '''Returns a list of class attributes to be serialized.'''
        return [self._ELE_ATTR]

    @classmethod
    def from_dict(cls, d, record_cls=None):
        '''Constructs a DataRecords instance from a dictionary.

        Args:
            d: a DataRecords dictionary
            record_cls: an optional records class to use when parsing the
                records dictionary. If not provided, the DataRecords dictionary
                must have been serialized with `reflective=True`

        Returns:
            a DataRecords instance
        '''
        if record_cls is None:
            record_cls_str = d.get(cls._ELE_CLS_FIELD, None)
            if record_cls_str is None:
                raise DataRecordsError(
                    "Your DataRecords does not have its '%s' attribute "
                    "populated, so you must manually specify the `record_cls` "
                    "to use when loading it" % cls._ELE_CLS_FIELD)
            record_cls = etau.get_class(record_cls_str)

        return DataRecords(
            record_cls=record_cls,
            records=[record_cls.from_dict(r) for r in d[cls._ELE_ATTR]])


class DataRecordsError(Exception):
    '''Exception raised for invalid DataRecords invocations.'''
    pass


DEFAULT_DATA_RECORDS_FILENAME = "records.json"


class BaseDataRecord(Serializable):
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
        kwargs = {k: d[k] for k in cls.required()}  # required
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


class LabeledFileRecord(BaseDataRecord):
    '''A simple DataRecord for a labeled file.

    Attributes:
        file_path: the path to the file
        label: the label of the file
    '''

    def __init__(self, file_path, label):
        '''Creates a new LabeledFileRecord instance.

        Args:
            file_path: the path to the file
            label: the label of the file
        '''
        self.file_path = file_path
        self.label = label
        super(LabeledFileRecord, self).__init__()

    @property
    def filename(self):
        '''Property to support older member access.

        @deprecated Use `file_path` instead
        '''
        logger.info("Deprecated use of filename property.")
        return self.file_path

    @classmethod
    def required(cls):
        return ["file_path", "label"]


class LabeledVideoRecord(LabeledFileRecord):
    '''A simple, reusable DataRecord for a labeled video.

    Args:
        file_path (video_path): the path to the video
        label: the label of the video
        group: an optional group attribute that provides additional information
            about the video. For example, if multiple video clips were sampled
            from a single video, this attribute can be used to specify the
            parent video
    '''

    def __init__(self, video_path, label, group=no_default):
        '''Creates a new LabeledVideoRecord instance.'''
        super(LabeledVideoRecord, self).__init__(video_path, label)
        self.group = group

    @property
    def video_path(self):
        '''Convenience accessor to refer to the file_path.'''
        return self.file_path

    @classmethod
    def optional(cls):
        return ["group"]


class LabeledFeatures(NpzWriteable):
    '''Class representing a feature array `X` and corresponding labels `y`.

    `X` is an n x d array whose rows contain features
    `y` is a length-n array of labels
    '''

    def __init__(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
