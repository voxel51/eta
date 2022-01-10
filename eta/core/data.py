"""
Core tools and data structures for working with data.

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

from collections import defaultdict
import logging
import os

import numpy as np

from eta.core.config import no_default
import eta.core.labels as etal
import eta.core.numutils as etan
import eta.core.serial as etas
import eta.core.utils as etau


logger = logging.getLogger(__name__)


def majority_vote_categorical_attrs(attrs, confidence_weighted=False):
    """Performs majority votes over the given attributes, which are assumed to
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
    """
    if not isinstance(attrs, list):
        attrs = [attrs]

    accums = defaultdict(etan.Accumulator)
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


class Attribute(etal.Labels):
    """Base class for attributes.

    Attributes:
        type: the fully-qualified class name of the attribute
        name: the name of the attribute
        value: the value of the attribute
        confidence: (optional) the confidence of the attribute, in ``[0, 1]``
        constant: whether this attribute is constant, i.e., all attributes of
            the same `name` must be identical to this attribute throughout the
            life of its parent entity
        tags: (optional) a list of tag strings
    """

    def __init__(
        self, name, value, confidence=None, constant=False, tags=None
    ):
        """Initializes the base Attribute instance.

        Args:
            name: the attribute name
            value: the attribute value
            confidence (None): an optional confidence of the value, in
                ``[0, 1]``
            constant (False): whether this attribute is constant, i.e., all
                attributes of the same `name` must be identical to this
                attribute throughout the life of its parent entity. By default,
                this is False
            tags (None): a list of tag strings
        """
        self.type = etau.get_class_name(self)
        self.name = name
        self.value = self.parse_value(value)
        self.confidence = confidence
        self.constant = constant
        self.tags = tags or []

    @classmethod
    def parse_value(cls, value):
        """Parses the attribute value.

        Args:
            value: the value

        Returns:
            the parsed value
        """
        raise NotImplementedError("subclass must implement parse_value()")

    def filter_by_schema(self, schema):
        """Filters the attribute by the given schema.

        Args:
            schema: an AttributeSchema
        """
        pass

    def attributes(self):
        """Returns the list of attributes to serialize.

        Returns:
            a list of attribute names
        """
        _attrs = ["type", "name", "value"]
        if self.confidence is not None:
            _attrs.append("confidence")
        if self.constant:
            _attrs.append("constant")
        if self.tags:
            _attrs.append("tags")

        return _attrs

    @classmethod
    def _from_dict(cls, d):
        """Internal implementation of `from_dict()`.

        Subclasses should implement this method, NOT `from_dict()`.

        Args:
            d: a JSON dictionary

        Returns:
            an Attribute
        """
        confidence = d.get("confidence", None)
        constant = d.get("constant", False)
        tags = d.get("tags", None)
        return cls(
            d["name"],
            d["value"],
            confidence=confidence,
            constant=constant,
            tags=tags,
        )

    @classmethod
    def from_dict(cls, d):
        """Constructs an Attribute from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an Attribute
        """
        attr_cls = etau.get_class(d["type"])
        return attr_cls._from_dict(d)


class CategoricalAttribute(Attribute):
    """Class encapsulating categorical attributes.

    Attributes:
        name: the name of the attribute
        value: the value of the attribute
        confidence: (optional) the confidence of the attribute, in ``[0, 1]``
        top_k_probs: (optional) an optional dictionary mapping values to
            probabilities
        constant: whether this attribute is constant, i.e., all attributes of
            the same `name` must be identical to this attribute throughout the
            life of its parent entity
        tags: (optional) a list of tag strings
    """

    def __init__(
        self,
        name,
        value,
        confidence=None,
        top_k_probs=None,
        constant=False,
        tags=None,
    ):
        """Creates a CategoricalAttribute instance.

        Args:
            name: the attribute name
            value: the attribute value
            confidence (None): an optional confidence of the value, in
                ``[0, 1]``
            top_k_probs (None): an optional dictionary mapping values to
                probabilities. By default, no probabilities are stored
            constant (False): whether this attribute is constant, i.e., all
                attributes of the same `name` must be identical to this
                attribute throughout the life of its parent entity. By default,
                this is False
            tags (None): a list of tag strings
        """
        super(CategoricalAttribute, self).__init__(
            name, value, confidence=confidence, constant=constant, tags=tags
        )
        self.top_k_probs = top_k_probs

    @classmethod
    def parse_value(cls, value):
        """Parses the attribute value.

        Args:
            value: the value

        Returns:
            the parsed value
        """
        return value

    def attributes(self):
        """Returns the list of attributes to serialize.

        Returns:
            the list of attribute names
        """
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
    """Class encapsulating numeric attributes.

    Attributes:
        name: the name of the attribute
        value: the value of the attribute
        confidence: (optional) the confidence of the attribute, in ``[0, 1]``
        constant: whether this attribute is constant, i.e., all attributes of
            the same `name` must be identical to this attribute throughout the
            life of its parent entity
        tags: (optional) a list of tag strings
    """

    @classmethod
    def parse_value(cls, value):
        """Parses the attribute value.

        Args:
            value: the value

        Returns:
            the parsed value
        """
        return value


class BooleanAttribute(Attribute):
    """Class encapsulating boolean attributes.

    Attributes:
        name: the name of the attribute
        value: the value of the attribute
        confidence: (optional) the confidence of the attribute, in ``[0, 1]``
        constant: whether this attribute is constant, i.e., all attributes of
            the same `name` must be identical to this attribute throughout the
            life of its parent entity
        tags: (optional) a list of tag strings
    """

    @classmethod
    def parse_value(cls, value):
        """Parses the attribute value.

        Args:
            value: the value

        Returns:
            the parsed value
        """
        return bool(value)


class AttributeSchema(etal.LabelsSchema):
    """Base class for Attribute schemas.

    Attributes:
        name: the name of the Attribute
        type: the fully-qualified name of the Attribute class
        exclusive: whether at most one attribute with this name may appear in
            an AttributeContainer
        default: an optional default value for the attribute
    """

    def __init__(self, name, exclusive=False, default=None):
        """Initializes the base AttributeSchema instance.

        Args:
            name: the name of the attribute
            exclusive: whether at most one attribute with this name may appear
                in an AttributeContainer. By default, this is False
            default: an optional default value for the attribute. By default,
                no default is stored
        """
        self.name = name
        self.type = etau.get_class_name(self)[: -len("Schema")]
        self.exclusive = exclusive
        self.default = default
        self._attr_cls = etau.get_class(self.type)

    @property
    def is_exclusive(self):
        """Whether this attribute is exclusive."""
        return self.exclusive

    @property
    def has_default_value(self):
        """Whether this attribute has a default value."""
        return self.default is not None

    def get_attribute_class(self):
        """Gets the Attribute class associated with this schema.

        Returns:
            the Attribute class
        """
        return self._attr_cls

    def add_attribute(self, attr):
        """Incorporates the given Attribute into the schema.

        Args:
            attr: an Attribute
        """
        self.add(attr)

    def is_valid_value(self, value):
        """Whether the attribute value is compliant with the schema.

        Args:
            value: the value

        Returns:
            True/False
        """
        raise NotImplementedError("subclass must implement is_valid_value()")

    def is_valid_attribute(self, attr):
        """Whether the attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Returns:
            True/False
        """
        return self.is_valid(attr)

    def validate_schema(self, schema):
        """Validates that the given AttributeSchema has the same class and
        `name` as this schema.

        Args:
            schema: an AttributeSchema

        Raises:
            LabelsSchemaError: if the schema does not match this schema
        """
        if type(schema) is not type(self):
            raise AttributeSchemaError(
                "Expected schema to have type '%s'; found '%s'"
                % (type(self), type(schema))
            )

        if schema.name != self.name:
            raise AttributeSchemaError(
                "Expected schema to have name '%s'; found '%s'"
                % (self.name, schema.name)
            )

    def validate_type(self, attr):
        """Validates that the Attribute is of the correct class.

        Args:
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        if not isinstance(attr, self._attr_cls):
            raise AttributeSchemaError(
                "Expected attribute '%s' to have type '%s'; found '%s'"
                % (attr.name, self.type, etau.get_class_name(attr))
            )

    def validate_attribute(self, attr):
        """Validates that the Attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        self.validate(attr)

    def validate(self, attr):
        """Validates that the Attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        if attr.name != self.name:
            raise AttributeSchemaError(
                "Expected name '%s'; found '%s'" % (self.name, attr.name)
            )

        self.validate_type(attr)

        if not self.is_valid_value(attr.value):
            raise AttributeSchemaError(
                "Value '%s' of attribute '%s' is not allowed by the "
                "schema " % (attr.value, attr.name)
            )

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given schema.

        Args:
            schema: an AttributeSchema

        Raises:
            LabelsSchemaError: if this schema is not a subset of the given
                schema
        """
        self.validate_schema_type(schema)

        if self.name != schema.name:
            raise AttributeSchemaError(
                "Expected name '%s'; found '%s'" % (schema.name, self.name)
            )

        if self.exclusive != schema.exclusive:
            raise AttributeSchemaError(
                "Expected exclusive '%s' for attribute '%s'; found '%s'"
                % (schema.exclusive, self.name, self.exclusive)
            )

        if self.default != schema.default:
            raise AttributeSchemaError(
                "Expected default '%s' for attribute '%s'; found '%s'"
                % (schema.default, self.name, self.default)
            )

    def validate_default_value(self):
        """Validates that the schema's default value (if any) is compliant with
        the schema.

        Raises:
            LabelsSchemaError: if the schema has a default value that is not
                compliant with the schema
        """
        if self.has_default_value:
            if not self.is_valid_value(self.default):
                raise AttributeSchemaError(
                    "Default value '%s' is not compliant with the schema"
                )

    def merge_schema(self, schema):
        """Merges the given AttributeSchema into this schema.

        Args:
            schema: a AttributeSchema
        """
        self.validate_schema(schema)

        if self.exclusive is False:
            self.exclusive = schema.exclusive

        if self.default is None:
            self.default = schema.default

    @staticmethod
    def get_kwargs(d):
        """Extracts the relevant keyword arguments for this schema from the
        JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a dictionary of parsed keyword arguments
        """
        raise NotImplementedError("subclass must implement get_kwargs()")

    def attributes(self):
        """Returns the list of attributes to be serialized.

        Returns:
            the list of attributes
        """
        attrs_ = ["name", "type"]
        if self.exclusive:
            attrs_.append("exclusive")
        if self.default is not None:
            attrs_.append("default")

        return attrs_

    @classmethod
    def from_dict(cls, d):
        """Constructs an AttributeSchema from a JSON dictionary.

        Note that this function reflectively parses the schema type from the
        dictionary, so subclasses do not need to implement this method.

        Args:
            d: a JSON dictionary

        Returns:
            an Attribute
        """
        attr_cls = etau.get_class(d["type"])
        schema_cls = attr_cls.get_schema_cls()

        name = d["name"]
        exclusive = d.get("exclusive", False)
        default = d.get("default", None)
        return schema_cls(
            name,
            exclusive=exclusive,
            default=default,
            **schema_cls.get_kwargs(d)
        )


class AttributeSchemaError(etal.LabelsSchemaError):
    """Error raised when an AttributeSchema is violated."""

    pass


class CategoricalAttributeSchema(AttributeSchema):
    """Schema for `CategoricalAttribute`s.

    Attributes:
        name: the name of the CategoricalAttribute
        type: the fully-qualified name of the CategoricalAttribute class
        categories: the set of valid values for the attribute
        exclusive: whether at most one attribute with this name may appear in
            an AttributeContainer
        default: an optional default value for the attribute
    """

    def __init__(self, name, categories=None, exclusive=False, default=None):
        """Creates a CategoricalAttributeSchema instance.

        Args:
            name: the name of the attribute
            categories: a set of valid values for the attribute. By default, an
                empty set is used
            exclusive: whether at most one attribute with this name may appear
                in an AttributeContainer. By default, this is False
            default: an optional default value for the attribute. By default,
                no default is stored
        """
        super(CategoricalAttributeSchema, self).__init__(
            name, exclusive=exclusive, default=default
        )
        self.categories = set(categories or [])
        self.validate_default_value()

    @property
    def is_empty(self):
        """Whether this schema has no labels of any kind."""
        return not bool(self.categories)

    def is_valid_value(self, value):
        """Whether value is valid for the attribute.

        Args:
            value: the value

        Returns:
            True/False
        """
        return value in self.categories

    def add(self, attr):
        """Incorporates the given CategoricalAttribute into the schema.

        Args:
            attr: a CategoricalAttribute
        """
        self.validate_type(attr)
        self.categories.add(attr.value)

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given schema.

        Args:
            schema: a CategoricalAttributeSchema

        Raises:
            LabelsSchemaError: if this schema is not a subset of the given
                schema
        """
        super(CategoricalAttributeSchema, self).validate_subset_of_schema(
            schema
        )

        if not self.categories.issubset(schema.categories):
            raise AttributeSchemaError(
                "Categories %s are not a subset of %s"
                % (self.categories, schema.categories)
            )

    @classmethod
    def build_active_schema(cls, attr):
        """Builds a CategoricalAttributeSchema that describes the active schema
        of the CategoricalAttribute.

        Args:
            attr: a CategoricalAttribute

        Returns:
            a CategoricalAttributeSchema
        """
        return cls(attr.name, categories={attr.value})

    def merge_schema(self, schema):
        """Merges the given CategoricalAttributeSchema into this schema.

        Args:
            schema: a CategoricalAttributeSchema
        """
        super(CategoricalAttributeSchema, self).merge_schema(schema)
        self.categories.update(schema.categories)

    def attributes(self):
        """Returns the list of attributes to be serialized.

        Returns:
            the list of attributes
        """
        attrs_ = super(CategoricalAttributeSchema, self).attributes()
        attrs_.append("categories")
        return attrs_

    def serialize(self, *args, **kwargs):
        d = super(CategoricalAttributeSchema, self).serialize(*args, **kwargs)

        # Always serialize categories in alphabetical order
        if "categories" in d:
            d["categories"].sort()

        return d

    @staticmethod
    def get_kwargs(d):
        """Extracts the relevant keyword arguments for this schema from the
        JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a dictioanry of parsed keyword arguments
        """
        return {"categories": d.get("categories", None)}


class NumericAttributeSchema(AttributeSchema):
    """Schema for `NumericAttribute`s.

    Attributes:
        name: the name of the NumericAttribute
        type: the fully-qualified name of the NumericAttribute class
        range: the (min, max) range for the attribute
        exclusive: whether at most one attribute with this name may appear in
            an AttributeContainer
        default: an optional default value for the attribute
    """

    def __init__(self, name, range=None, exclusive=False, default=None):
        """Creates a NumericAttributeSchema instance.

        Args:
            name: the name of the attribute
            range: the (min, max) range for the attribute
            exclusive: whether at most one attribute with this name may appear
                in an AttributeContainer. By default, this is False
            default: an optional default value for the attribute. By default,
                no default is stored
        """
        super(NumericAttributeSchema, self).__init__(
            name, exclusive=exclusive, default=default
        )
        self.range = tuple(range or [])
        self.validate_default_value()

    @property
    def is_empty(self):
        """Whether this schema has no labels of any kind."""
        return not bool(self.range)

    def is_valid_value(self, value):
        """Whether value is valid for the attribute.

        Args:
            value: the value

        Returns:
            True/False
        """
        if not self.range:
            return False

        return value >= self.range[0] and value <= self.range[1]

    def add(self, attr):
        """Incorporates the NumericAttribute into the schema.

        Args:
            attr: a NumericAttribute
        """
        self.validate_type(attr)
        value = attr.value
        if not self.range:
            self.range = (value, value)
        else:
            self.range = min(self.range[0], value), max(self.range[1], value)

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given schema.

        Args:
            schema: a NumericAttributeSchema

        Raises:
            LabelsSchemaError: if this schema is not a subset of the given
                schema
        """
        super(NumericAttributeSchema, self).validate_subset_of_schema(schema)

        if self.range and (
            not schema.range
            or self.range[0] < schema.range[0]
            or self.range[1] > schema.range[1]
        ):
            raise AttributeSchemaError(
                "Range %s is not a subset of %s" % (self.range, schema.range)
            )

    @classmethod
    def build_active_schema(cls, attr):
        """Builds a NumericAttributeSchema that describes the active schema of
        the NumericAttribute.

        Args:
            attr: a NumericAttribute

        Returns:
            a NumericAttributeSchema
        """
        return cls(attr.name, range=(attr.value, attr.value))

    def merge_schema(self, schema):
        """Merges the given NumericAttributeSchema into this schema.

        Args:
            schema: a NumericAttributeSchema
        """
        super(NumericAttributeSchema, self).merge_schema(schema)

        if not self.range:
            self.range = schema.range
        else:
            self.range = (
                min(self.range[0], schema.range[0]),
                max(self.range[1], schema.range[1]),
            )

    def attributes(self):
        """Returns the list of attributes to be serialized.

        Returns:
            the list of attributes
        """
        attrs_ = super(NumericAttributeSchema, self).attributes()
        attrs_.append("range")
        return attrs_

    @staticmethod
    def get_kwargs(d):
        """Extracts the relevant keyword arguments for this schema from the
        JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a dictionary of parsed keyword arguments
        """
        return {"range": d.get("range", None)}


class BooleanAttributeSchema(AttributeSchema):
    """Schema for `BooleanAttributeSchema`s.

    Attributes:
        name: the name of the BooleanAttribute
        type: the fully-qualified name of the BooleanAttribute class
        values: the set of valid boolean values for the attribute
        exclusive: whether at most one attribute with this name may appear in
            an AttributeContainer
        default: an optional default value for the attribute. By default,
                no default is stored
    """

    def __init__(self, name, values=None, exclusive=False, default=None):
        """Creates a BooleanAttributeSchema instance.

        Args:
            name: the name of the attribute
            values: a set of valid boolean values for the attribute. By
                default, an empty set is used
            exclusive: whether at most one attribute with this name may appear
                in an AttributeContainer. By default, this is False
            default: an optional default value for the attribute. By default,
                no default is stored
        """
        super(BooleanAttributeSchema, self).__init__(
            name, exclusive=exclusive, default=default
        )
        self.values = set(values or [])
        self.validate_default_value()

    @property
    def is_empty(self):
        """Whether this schema has no labels of any kind."""
        return not bool(self.values)

    def is_valid_value(self, value):
        """Whether value is valid for the attribute.

        Args:
            value: the value

        Returns:
            True/False
        """
        return value in self.values

    def add(self, attr):
        """Incorporates the given BooleanAttribute into the schema.

        Args:
            attr: a BooleanAttribute
        """
        self.validate_type(attr)
        self.values.add(attr.value)

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given schema.

        Args:
            schema: a BooleanAttributeSchema

        Raises:
            LabelsSchemaError: if this schema is not a subset of the given
                schema
        """
        super(BooleanAttributeSchema, self).validate_subset_of_schema(schema)

        if not self.values.issubset(schema.values):
            raise AttributeSchemaError(
                "Values %s are not a subset of %s"
                % (self.values, schema.values)
            )

    @classmethod
    def build_active_schema(cls, attr):
        """Builds a BooleanAttributeSchema that describes the active schema of
        the BooleanAttribute.

        Args:
            attr: a BooleanAttribute

        Returns:
            a BooleanAttributeSchema
        """
        return cls(attr.name, values={attr.value})

    def merge_schema(self, schema):
        """Merges the given BooleanAttributeSchema into this schema.

        Args:
            schema: a BooleanAttributeSchema
        """
        super(BooleanAttributeSchema, self).merge_schema(schema)
        self.values.update(schema.values)

    def attributes(self):
        """Returns the list of attributes to be serialized.

        Returns:
            the list of attributes
        """
        attrs_ = super(BooleanAttributeSchema, self).attributes()
        attrs_.append("values")
        return attrs_

    @staticmethod
    def get_kwargs(d):
        """Extracts the relevant keyword arguments for this schema from the
        JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a dictioanry of parsed keyword arguments
        """
        return {"values": d.get("values", None)}


class AttributeContainer(etal.LabelsContainer):
    """An `eta.core.serial.Container` of `Attribute`s."""

    _ELE_CLS = Attribute
    _ELE_CLS_FIELD = "_ATTR_CLS"
    # Note: we can't use "attributes" here due to `Serializable.attributes()`
    _ELE_ATTR = "attrs"

    def sort_by_name(self, reverse=False):
        """Sorts the `Attribute`s in the container by name.

        Args:
            reverse: whether to sort in descending order. The default is False
        """
        self.sort_by("name", reverse=reverse)

    def has_attr_with_name(self, name):
        """Returns whether or not the container contains an Attribute with
        the given name.

        Args:
            name: the Attribute name

        Returns:
            True/False
        """
        for attr in self:
            if attr.name == name:
                return True

        return False

    def get_attrs_with_name(self, name):
        """Gets all `Attribute`s with the given name.

        Args:
            name: the Attribute name

        Returns:
            an AttributeContainer of attributes with the given name
        """
        return self.get_matches([lambda attr: attr.name == name])

    def get_attr_with_name(self, name, default=no_default):
        """Gets the single Attribute with the given name.

        Args:
            name: the Attribute name
            default: the value to be returned if there is no Attribute with
                the given name. By default, an error is raised in this case

        Returns:
            the Attribute

        Raises:
            ValueError: if there is not exactly one Attribute with the given
                name
        """
        attrs = self.get_attrs_with_name(name)

        if not attrs and default is not no_default:
            return default

        if len(attrs) != 1:
            raise ValueError(
                "Expected one attribute with name '%s' but found %d"
                % (name, len(attrs))
            )

        return attrs[0]

    def get_attr_values_with_name(self, name):
        """Gets a list of values for all `Attribute`s with the given name.

        Args:
            name: the Attribute name

        Returns:
            a list of attributes values with the given name
        """
        return [attr.value for attr in self.get_attrs_with_name(name)]

    def get_attr_value_with_name(self, name, default=no_default):
        """Gets the value of the single Attribute with the given name.

        Args:
            name: the Attribute name
            default: the value to be returned if there is no Attribute with
                the given name. By default, an error is raised in this case.

        Returns:
            the Attribute value

        Raises:
            ValueError: if there is not exactly one Attribute with the given
                name
        """
        try:
            attr = self.get_attr_with_name(name)
            return attr.value
        except ValueError:
            if default is not no_default:
                return default

            raise

    def pop_constant_attrs(self):
        """Pops constant attributes from this container.

        Returns:
            an AttributeContainer with the constant attributes, if any
        """
        return self.pop_elements([lambda attr: attr.constant])

    def filter_by_schema(self, schema, constant_schema=None):
        """Removes attributes from this container that are not compliant with
        the given schema.

        Only the first observation of each exclusive attribute is kept (if
        applicable).

        Args:
            schema: an AttributeContainerSchema
            constant_schema: an AttributeContainerSchema describing a schema
                for constant attributes (those with `constant == True`). If
                provided, `schema` is applied only to non-constant attributes.
                If omitted, `schema` is applied to all attributes
        """
        if constant_schema is None:
            constant_schema = schema

        get_schema = lambda attr: constant_schema if attr.constant else schema

        # Remove attributes with invalid names
        filter_fcn = lambda attr: get_schema(attr).has_attribute(attr.name)
        self.filter_elements([filter_fcn])

        #
        # Filter objects by their schemas
        #

        del_inds = set()
        found_names = set()
        for idx, attr in enumerate(self):
            name = attr.name

            # Remove attributes that violate schema
            attr_schema = get_schema(attr).get_attribute_schema(name)
            if not attr_schema.is_valid_attribute(attr):
                del_inds.add(idx)

            # Enforce exclusivity, if necessary
            is_exclusive = get_schema(attr).is_exclusive_attribute(name)
            if is_exclusive:
                if name in found_names:
                    del_inds.add(idx)
                else:
                    found_names.add(name)

        self.delete_inds(del_inds)

    def get_attribute_counts(self):
        """Returns a dictionary mapping attribute names to their counts in this
        container.

        Returns:
            a dict mapping attribute names to counts
        """
        counts = defaultdict(int)
        for attr in self:
            counts[attr.name] += 1

        return dict(counts)


class AttributeContainerSchema(etal.LabelsContainerSchema):
    """Schema for an AttributeContainer.

    Attributes:
        schema: a dictionary mapping attribute names to AttributeSchema
            instances
    """

    def __init__(self, schema=None):
        """Creates an AttributeContainerSchema instance.

        Args:
            schema: (optional) a dictionary mapping attribute names to
                AttributeSchema instances. By default, an empty schema is
                created
        """
        self.schema = schema or {}

    @property
    def is_empty(self):
        """Whether this schema has no labels of any kind."""
        return not bool(self.schema)

    @property
    def has_exclusive_attributes(self):
        """Whether this schema contains at least one exclusive attribute."""
        return any(schema.is_exclusive for schema in itervalues(self.schema))

    def iter_attributes(self):
        """Returns an iterator over the (name, AttributeSchema) pairs in this
        schema.

        Returns:
            an iterator over (name, AttributeSchema) pairs
        """
        return iteritems(self.schema)

    def has_attribute(self, name):
        """Whether the schema has an Attribute with the given name.

        Args:
            name: the name

        Returns:
            True/False
        """
        return name in self.schema

    def get_attribute_schema(self, name):
        """Gets the AttributeSchema for the Attribute with the given name.

        Args:
            name: the name

        Returns:
            an AttributeSchema
        """
        self.validate_attribute_name(name)
        return self.schema[name]

    def get_attribute_class(self, name):
        """Gets the class of the Attribute with the given name.

        Args:
            name: the name

        Returns:
            the Attribute class

        Raises:
            LabelsSchemaError: if the schema does not have an attribute with
                the given name
        """
        self.validate_attribute_name(name)
        return self.schema[name].get_attribute_class()

    def is_exclusive_attribute(self, name):
        """Whether the Attribute with the given name is exclusive.

        Args:
            name: the name

        Returns:
            True/False
        """
        return self.get_attribute_schema(name).is_exclusive

    def has_default_value(self, name):
        """Whether the Attribute with the given name has a default value.

        Args:
            name: the name

        Returns:
            True/False
        """
        return self.get_attribute_schema(name).has_default_value

    def get_default_value(self, name):
        """Gets the default value for the Attribute with the given name.

        Args:
            name: the name

        Returns:
            the default value, or None if the attribute has no default value
        """
        return self.get_attribute_schema(name).default

    def add_attribute(self, attr):
        """Incorporates the given Attribute into the schema.

        Args:
            attr: an Attribute
        """
        name = attr.name
        if name not in self.schema:
            schema_cls = attr.get_schema_cls()
            self.schema[name] = schema_cls(name)

        self.schema[name].add_attribute(attr)

    def add_attributes(self, attrs):
        """Incorporates the given AttributeContainer into the schema.

        Args:
            attrs: an AttributeContainer
        """
        for attr in attrs:
            self.add_attribute(attr)

    def is_valid_attribute_name(self, name):
        """Whether the schema has an Attribute with the given name.

        Args:
            name: the name

        Returns:
            True/False
        """
        try:
            self.validate_attribute_name(name)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_attribute(self, attr):
        """Whether the Attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Returns:
            True/False
        """
        try:
            self.validate_attribute(attr)
            return True
        except etal.LabelsSchemaError:
            return False

    def validate_attribute_name(self, name):
        """Validates that the schema has an Attribute with the given name.

        Args:
            name: the name

        Raises:
            LabelsSchemaError: if the schema doesn't contain an attribute of
                the given name
        """
        if not self.has_attribute(name):
            raise AttributeContainerSchemaError(
                "Attribute '%s' is not allowed by the schema" % name
            )

    def validate_attribute(self, attr):
        """Validates that the Attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        self.validate_attribute_name(attr.name)
        self.schema[attr.name].validate_attribute(attr)

    def validate(self, attrs):
        """Validates that the AttributeContainer is compliant with the schema.

        Args:
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        """
        # Validate attributes
        for attr in attrs:
            self.validate_attribute(attr)

        # Enforce attribute exclusivity, if necessary
        if self.has_exclusive_attributes:
            counts = attrs.get_attribute_counts()
            for name, count in iteritems(counts):
                if count > 1 and self.is_exclusive_attribute(name):
                    raise AttributeContainerSchemaError(
                        "Attribute '%s' is exclusive but appears %d times in "
                        "this container" % (name, count)
                    )

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given schema.

        Args:
            schema: an AttributeContainerSchema

        Raises:
            LabelsSchemaError: if this schema is not a subset of the given
                schema
        """
        self.validate_schema_type(schema)

        for name, attr_schema in iteritems(self.schema):
            if not schema.has_attribute(name):
                raise AttributeContainerSchemaError(
                    "Attribute '%s' does not appear in schema" % name
                )

            other_attr_schema = schema.get_attribute_schema(name)
            attr_schema.validate_subset_of_schema(other_attr_schema)

    def merge_attribute_schema(self, attr_schema):
        """Merges the given AttributeSchema into the schema.

        Args:
            attr_schema: an AttributeSchema
        """
        name = attr_schema.name
        if name not in self.schema:
            self.schema[name] = attr_schema
        else:
            self.schema[name].merge_schema(attr_schema)

    def merge_schema(self, schema):
        """Merges the given AttributeContainerSchema into the schema.

        Args:
            schema: an AttributeContainerSchema
        """
        for _, attr_schema in schema.iter_attributes():
            self.merge_attribute_schema(attr_schema)

    @classmethod
    def build_active_schema(cls, attrs):
        """Builds an AttributeContainerSchema that describes the active schema
        of the given AttributeContainer.

        Args:
            attrs: an AttributeContainer

        Returns:
            an AttributeContainerSchema describing the active schema of the
                attributes
        """
        schema = cls()
        schema.add_attributes(attrs)
        return schema

    @classmethod
    def from_dict(cls, d):
        """Constructs an AttributeContainerSchema from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an AttributeContainerSchema
        """
        schema = d.get("schema", None)
        if schema is not None:
            schema = {
                attr_name: AttributeSchema.from_dict(asd)
                for attr_name, asd in iteritems(schema)
            }

        return cls(schema=schema)


class AttributeContainerSchemaError(etal.LabelsContainerSchemaError):
    """Error raised when an AttributeContainerSchema is violated."""

    pass


class MaskIndex(etas.Serializable):
    """An index of sementics for the values in a mask."""

    def __init__(self, index=None):
        """Creates a MaskIndex instance.

        Args:
            index: (optional) a dictionary mapping values to `Attribute`s
                describing the semantics of a mask
        """
        self.index = index or {}

    def __contains__(self, value):
        return value in self.index

    def __getitem__(self, value):
        return self.get_attr(value)

    def __setitem__(self, value, attr):
        self.add_value(value, attr)

    def __delitem__(self, value):
        self.delete_value(value)

    def get_attr(self, value):
        """Gets the attribute for the given mask value.

        Args:
            value: the mask value

        Returns:
            an Attribute
        """
        return self.index[value]

    def add_value(self, value, attr):
        """Sets the semantics for the given mask value.

        Args:
            value: the mask value
            attr: an Attribute
        """
        self.index[value] = attr

    def delete_value(self, value):
        """Deteles the given mask value from the index.

        Args:
            value: the mask value
        """
        del self.index[value]

    @classmethod
    def from_labels_map(cls, labels_map):
        """Returns a MaskIndex for the given labels map.

        Args:
            labels_map: a dict mapping indices to label values

        Returns:
            a MaskIndex
        """
        mask_index = cls()
        for index, value in iteritems(labels_map):
            mask_index[index] = CategoricalAttribute("label", value)

        return mask_index

    @classmethod
    def from_dict(cls, d):
        """Constructs a MaskIndex from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a MaskIndex
        """
        index = d.get("index", None)
        if index is not None:
            index = {
                int(value): Attribute.from_dict(ad)
                for value, ad in iteritems(index)
            }

        return cls(index=index)


class DataFileSequence(etas.Serializable):
    """Class representing a sequence of data files on disk.

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
    """

    def __init__(self, sequence, immutable_bounds=True):
        """Creates a DataFileSequence instance for the given sequence.

        Args:
            sequence: The printf-style pattern describing the files on disk,
                e.g., `/path/to/frame-%05d.json`
            immutable_bounds: whether the lower and upper bounds of the
                sequence should be immutable. By default, this is True

        Raises:
            DataFileSequenceError: if the sequence did not match any files on
                disk
        """
        self.sequence = sequence
        self.immutable_bounds = immutable_bounds
        self._extension = os.path.splitext(self.sequence)[1]
        self._lower_bound, self._upper_bound = etau.parse_bounds_from_pattern(
            self.sequence
        )
        self._iter_index = None

        if self._lower_bound is None or self._upper_bound is None:
            raise DataFileSequenceError(
                "Sequence '%s' did not match any files on disk" % sequence
            )

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
                "Cannot set bounds of an immutable sequence."
            )
        self._lower_bound = min(value, self.upper_bound)

    @upper_bound.setter
    def upper_bound(self, value):
        if self.immutable_bounds:
            raise DataFileSequenceError(
                "Cannot set bounds of an immutable sequence."
            )
        self._upper_bound = max(value, self.lower_bound)

    @property
    def starts_at_zero(self):
        return self._lower_bound == 0

    @property
    def starts_at_one(self):
        return self._lower_bound == 1

    def check_bounds(self, index):
        """Checks if the index is within the bounds for this sequence.

        Args:
            index: a sequence index

        Returns:
            True/False
        """
        if index < self.lower_bound or index > self.upper_bound:
            return False
        return True

    def gen_path(self, index):
        """Generates the path for the file with the given sequence index.

        If the sequence has mutable bounds, the index can extend the sequence
        consecutively (i.e., by one index) above or below the current bounds.

        Args:
            index: a sequence index

        Returns:
            the generated path for the given index
        """
        if self.immutable_bounds:
            if not self.check_bounds(index):
                raise DataFileSequenceError(
                    "Index %d out of bounds [%d, %d]"
                    % (index, self.lower_bound, self.upper_bound)
                )
        elif index < 0:
            raise DataFileSequenceError("Indices must be nonnegative")
        elif index == self.lower_bound - 1:
            self._lower_bound = index
        elif index == self.upper_bound + 1:
            self._upper_bound = index
        elif not self.check_bounds(index):
            raise DataFileSequenceError(
                "Index %d out of bounds [%d, %d]; mutable sequences can be "
                "extended at most one index above/below."
                % (index, self.lower_bound, self.upper_bound)
            )

        return self.sequence % index

    @classmethod
    def build_for_dir(cls, dir_path):
        """Builds a DataFileSequence for the given directory."""
        return cls(etau.parse_dir_pattern(dir_path)[0])

    @classmethod
    def from_dict(cls, d):
        """Builds a DataFileSequence from a JSON dictioanry."""
        return cls(d["sequence"], immutable_bounds=d["immutable_bounds"])


class DataFileSequenceError(Exception):
    """Error raised when an invalid DataFileSequence is encountered."""

    pass


class DataRecords(etas.Container):
    """Container class for data records.

    DataRecords is a generic container of records each having a value for
    a certain set of fields. When creating DataRecords instances, you must
    provide a `record_cls` that specifies the subclass of BaseDataRecord
    that you plan to store in the container.

    When DataRecords instances are serialized, they can optionally have their
    reflective `_CLS` and `_RECORD_CLS` attributes set by passing
    `reflective=True`. When this is done, DataRecords can be read from disk
    via `DataRecords.from_json("/path/to/records.json")` and the class of the
    records in the container will be inferred while loading.
    """

    _ELE_CLS = None  # this is set per-instance for DataRecords
    _ELE_CLS_FIELD = "_RECORD_CLS"
    _ELE_ATTR = "records"

    def __init__(self, record_cls, **kwargs):
        """Creates a DataRecords instance.

        Args:
            record_cls: the records class to use for this container
            records: an optional list of records to add to the container
        """
        self._ELE_CLS = record_cls
        super(DataRecords, self).__init__(**kwargs)

    @property
    def record_cls(self):
        """Returns the class of records in the container."""
        return self._ELE_CLS

    def add_dict(self, d, record_cls=None):
        """Adds the records in the dictionary to the container.

        Args:
            d: a DataRecords dictionary
            record_cls: an optional records class to use when parsing the
                records dictionary. If None, the _ELE_CLS class of this
                instance is used

        Returns:
            the number of elements in the container
        """
        rc = record_cls or self._ELE_CLS
        self.add_container(self.from_dict(d, record_cls=rc))
        return len(self)

    def add_json(self, json_path, record_cls=None):
        """Adds the records in the JSON file to the container.

        Args:
            json_path: the path to a DataRecords JSON file
            record_cls: an optional records class to use when parsing the
                records dictionary. If None, the _ELE_CLS class of this
                instance is used

        Returns:
            the number of elements in the container
        """
        rc = record_cls or self._ELE_CLS
        self.add_container(self.from_json(json_path, record_cls=rc))
        return len(self)

    def build_keyset(self, field):
        """Returns a list of unique values of `field` across the records in
        the container.
        """
        keys = set()
        for r in self.__elements__:
            keys.add(getattr(r, field))
        return list(keys)

    def build_lookup(self, field):
        """Builds a lookup dictionary indexed by `field` whose values are lists
        of indices of the records whose `field` attribute matches the
        corresponding key.
        """
        lud = defaultdict(list)
        for i, r in enumerate(self.__elements__):
            lud[getattr(r, field)].append(i)
        return dict(lud)

    def build_subsets(self, field):
        """Builds a dictionary indexed by `field` whose values are lists of
        records whose `field` attribute matches the corresponding key.
        """
        sss = defaultdict(list)
        for r in self.__elements__:
            sss[getattr(r, field)].append(r)
        return dict(sss)

    def cull(self, field, keep_values=None, remove_values=None):
        """Cull records from the container based on `field`.

        Args:
            field: the field to process
            keep_values: an optional list of field values to keep
            remove_values: an optional list of field values to remove

        Returns:
            the number of elements in the container
        """
        lud = self.build_lookup(field)

        # Determine values to keep
        if remove_values:
            keep_values = set(lud.keys()) - set(remove_values)
        if not keep_values:
            raise DataRecordsError(
                "Either keep_values or remove_values must be provided"
            )

        # Cull records
        inds = set()
        for v in keep_values:
            inds.update(lud[v])
        self.keep_inds(inds)

        return len(self)

    def cull_with_function(self, field, func):
        """Cull records from the container for which `field` returns something
        that evaluates to False when passed through func.

        Args:
            field: the field to process
            func: the test function

        Returns:
            the number of elements in the container
        """
        lud = self.build_lookup(field)

        # Cull records
        inds = set()
        for v in lud:
            if func(v):
                inds.update(lud[v])
        self.keep_inds(inds)

        return len(self)

    def slice(self, field):
        """Returns a list of `field` values for the records in the container.
        """
        return [getattr(r, field) for r in self.__elements__]

    def subset_from_indices(self, indices):
        """Creates a new DataRecords instance containing only the subset of
        records in this container with the specified indices.
        """
        return self.extract_inds(indices)

    def attributes(self):
        """Returns a list of class attributes to be serialized."""
        return [self._ELE_ATTR]

    @classmethod
    def from_dict(cls, d, record_cls=None):
        """Constructs a DataRecords instance from a dictionary.

        Args:
            d: a DataRecords dictionary
            record_cls: an optional records class to use when parsing the
                records dictionary. If not provided, the DataRecords dictionary
                must have been serialized with `reflective=True`

        Returns:
            a DataRecords instance
        """
        if record_cls is None:
            record_cls_str = d.get(cls._ELE_CLS_FIELD, None)
            if record_cls_str is None:
                raise DataRecordsError(
                    "Your DataRecords does not have its '%s' attribute "
                    "populated, so you must manually specify the `record_cls` "
                    "to use when loading it" % cls._ELE_CLS_FIELD
                )
            record_cls = etau.get_class(record_cls_str)

        return DataRecords(
            record_cls=record_cls,
            records=[record_cls.from_dict(r) for r in d[cls._ELE_ATTR]],
        )


class DataRecordsError(Exception):
    """Exception raised for invalid DataRecords invocations."""

    pass


DEFAULT_DATA_RECORDS_FILENAME = "records.json"


class BaseDataRecord(etas.Serializable):
    """Base class for all data records.

    Data records are flexible containers that function as dictionary-like
    classes that define the required, optional, and excluded keys that they
    support.

    @todo excluded is redundant. We should only serialize required and optional
    attributes; all others should be excluded by default.
    """

    def __init__(self):
        """Initializes the BaseDataRecord instance."""
        self.clean_optional()

    def __getitem__(self, key):
        return getattr(self, key)

    def attributes(self):
        """Returns the list of attributes of the data record that are to be
        serialized.

        All private attributes (those starting with "_") and attributes in
        `excluded()` are omitted from this list.

        Returns:
            the list of attributes to be serialized
        """
        attr = super(BaseDataRecord, self).attributes()
        return [a for a in attr if a not in self.excluded()]

    def clean_optional(self):
        """Deletes any optional attributes from the data record that are not
        set, i.e., those that are `no_default`.

        Note that `None` is a valid value for an attribute.
        """
        for o in self.optional():
            if hasattr(self, o) and getattr(self, o) is no_default:
                delattr(self, o)

    @classmethod
    def from_dict(cls, d):
        """Constructs a data record from a JSON dictionary. All required
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
        """
        kwargs = {k: d[k] for k in cls.required()}  # required
        kwargs.update({k: d[k] for k in cls.optional() if k in d})  # optional
        return cls(**kwargs)

    @classmethod
    def required(cls):
        """Returns a list of attributes that are required by all instances of
        the data record. By default, an empty list is returned.
        """
        return []

    @classmethod
    def optional(cls):
        """Returns a list of attributes that are optionally included in the
        data record if they are present in the data dictionary. By default,
        an empty list is returned.
        """
        return []

    @classmethod
    def excluded(cls):
        """Return a list of attributes that should always be excluded when the
        data record is serialized. By default, an empty list is returned.
        """
        return []


class LabeledFileRecord(BaseDataRecord):
    """A simple DataRecord for a labeled file.

    Attributes:
        file_path: the path to the file
        label: the label of the file
    """

    def __init__(self, file_path, label):
        """Creates a new LabeledFileRecord instance.

        Args:
            file_path: the path to the file
            label: the label of the file
        """
        self.file_path = file_path
        self.label = label
        super(LabeledFileRecord, self).__init__()

    @property
    def filename(self):
        """The filename of the record."""
        logger.warning("`filename` is deprecated; use `file_path` instead")
        return self.file_path

    @classmethod
    def required(cls):
        return ["file_path", "label"]


class LabeledVideoRecord(LabeledFileRecord):
    """A simple, reusable DataRecord for a labeled video.

    The `group` attribute allows for providing additional information about the
    video. For example, if multiple video clips were sampled from a single
    video, this attribute can be used to specify the p.arent video

    Args:
        video_path: the path to the video
        label: the label of the video
        group: an optional group attribute for the video
    """

    def __init__(self, video_path, label, group=no_default):
        """Creates a LabeledVideoRecord instance.

        Args:
            video_path: the path to the video
            label: the label of the video
            group: an optional group attribute for the video
        """
        super(LabeledVideoRecord, self).__init__(video_path, label)
        self.group = group

    @property
    def video_path(self):
        """Convenience accessor to refer to the file_path."""
        return self.file_path

    @classmethod
    def optional(cls):
        return ["group"]


class LabeledFeatures(etas.NpzWriteable):
    """Class representing a feature array `X` and corresponding labels `y`.

    Attributes:
        x: an n x d array whose rows contain features
        y: a length-n array of labels
    """

    def __init__(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
