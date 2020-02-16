'''
Core data structures for working with labels.

Copyright 2017-2020, Voxel51, Inc.
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

import eta.core.serial as etas
import eta.core.utils as etau


class Labels(etas.Serializable):
    '''Base class for `eta.core.serial.Serializable` classes that hold labels
    representing attributes, objects, frames, events, images, videos, etc.

    Labels classes have associated `Schema` classes that describe the
    ontologies over the labels class.
    '''

    def __bool__(self):
        '''Whether this instance has labels of any kind.'''
        return not self.is_empty

    @property
    def is_empty(self):
        '''Whether this instance has no labels of any kind.'''
        raise NotImplementedError("subclasses must implement is_empty")

    @classmethod
    def get_schema_cls(cls):
        '''Gets the `LabelsSchema` class for the labels.

        Subclasses can override this method, but, by default, this
        implementation assumes the convention that labels class `<Labels>` has
        associated schema class `<Labels>Schema` defined in the same module.

        Returns:
            the LabelsSchema class
        '''
        class_name = etau.get_class_name(cls)
        return etau.get_class(class_name + "Schema")

    def get_active_schema(self):
        '''Returns a `LabelsSchema` that describes the active schema of the
        labels.

        Returns:
            a LabelsSchema
        '''
        schema_cls = self.get_schema_cls()
        return schema_cls.build_active_schema(self)

    def filter_by_schema(self, schema):
        '''Filters the labels by the given schema.

        Args:
            schema: a LabelsSchema
        '''
        raise NotImplementedError(
            "subclasses must implement `filter_by_schema()`")


class LabelsSchema(etas.Serializable):
    '''Base class for schemas of `Labels` classes.'''

    def __bool__(self):
        '''Whether this schema has labels of any kind.'''
        return not self.is_empty

    @property
    def is_empty(self):
        '''Whether this schema has no labels of any kind.'''
        raise NotImplementedError("subclasses must implement is_empty")

    def add(self, labels):
        '''Incorporates the `Labels` into the schema.

        Args:
            label: a Labels instance
        '''
        labels_schema = self.build_active_schema(labels)
        self.merge_schema(labels_schema)

    def add_iterable(self, iterable):
        '''Incorporates the given iterable of `Labels` into the schema.

        Args:
            iterable: an iterable of Labels
        '''
        for labels in iterable:
            self.add(labels)

    def validate(self, labels):
        '''Validates that the `Labels` are compliant with the schema.

        Args:
            labels: a Labels instance

        Raises:
            LabelsSchemaError: if the labels violate the schema
        '''
        raise NotImplementedError("subclasses must implement `validate()`")

    def validate_subset_of_schema(self, schema):
        '''Validates that this schema is a subset of the given LabelsSchema.

        Args:
            schema: a LabelsSchema

        Raises:
            LabelsSchemaError: if this schema is not a subset of the given
                schema
        '''
        raise NotImplementedError(
            "subclasses must implement `validate_subset_of_schema()`")

    def validate_schema_type(self, schema):
        '''Validates that this schema is an instance of same type as the given
        schema.

        Args:
            schema: a LabelsSchema

        Raises:
            LabelsSchemaError: if this schema is not of the same type as the
                given schema
        '''
        if not isinstance(self, type(schema)):
            raise LabelsSchemaError(
                "Expected `self` to match schema type %s; found %s" %
                (type(self), type(schema)))

    def is_valid(self, labels):
        '''Whether the `Labels` are compliant with the schema.

        Args:
            labels: a Labels instance

        Returns:
            True/False
        '''
        try:
            self.validate(labels)
            return True
        except LabelsSchemaError:
            return False

    def is_subset_of_schema(self, schema):
        '''Whether this schema is a subset of the given schema.

        Args:
            schema: a LabelsSchema

        Returns:
            True/False
        '''
        try:
            self.validate_subset_of_schema(schema)
            return True
        except LabelsSchemaError:
            return False

    @classmethod
    def build_active_schema(cls, labels):
        '''Builds a `LabelsSchema` that describes the active schema of the
        labels.

        Args:
            labels: a Labels instance

        Returns:
            a LabelsSchema
        '''
        raise NotImplementedError(
            "subclasses must implement `build_active_schema()`")

    def merge_schema(self, schema):
        '''Merges the given `LabelsSchema` into this schema.

        Args:
            schema: a LabelsSchema
        '''
        raise NotImplementedError("subclasses must implement `merge_schema()`")


class LabelsSchemaError(Exception):
    '''Error raisesd when a `LabelsSchema` is violated.'''
    pass


class HasLabelsSchema(object):
    '''Mixin for `Label` classes that can optionally store and enforce
    `LabelsSchema`s on their labels.

    For efficiency, schemas are not automatically enforced when new labels are
    added to `HasLabelsSchema` instances. Rather, users must manually call
    `validate_schema()` when they would like to validate the schema.
    '''

    def __init__(self, schema=None):
        '''Initializes the `HasLabelsSchema` mixin.

        Args:
            schema: (optional) an optional LabelsSchema to enforce on the
                labels. By default, no schema is enforced
        '''
        self.schema = schema

    @property
    def has_schema(self):
        '''Whether the labels have an enforced schema.'''
        return self.schema is not None

    def get_schema(self):
        '''Gets the current enforced schema for the labels, or None if no
        schema is enforced.

        Returns:
            a LabelsSchema, or None
        '''
        return self.schema

    def set_schema(self, schema, filter_by_schema=False, validate=False):
        '''Sets the enforced schema to the given `LabelsSchema`.

        Args:
            schema: a LabelsSchema to assign
            filter_by_schema: whether to filter labels that are not compliant
                with the schema. By default, this is False
            validate: whether to validate that the labels (after filtering, if
                applicable) are compliant with the new schema. By default, this
                is False

        Raises:
            LabelsSchemaError: if `validate` was `True` and this object
                contains labels that are not compliant with the schema
        '''
        self.schema = schema
        if not self.has_schema:
            return

        if filter_by_schema:
            self.filter_by_schema(self.schema)  # pylint: disable=no-member

        if validate:
            self.validate_schema()

    def validate_schema(self):
        '''Validates that the labels are compliant with the current schema.

        Raises:
            LabelsSchemaError: if this object contains labels that are not
                compliant with the schema
        '''
        if self.has_schema:
            self.schema.validate(self)

    def freeze_schema(self):
        '''Sets the schema for the labels to the current active schema.'''
        self.set_schema(self.get_active_schema())  # pylint: disable=no-member

    def remove_schema(self):
        '''Removes the enforced schema from the labels.'''
        self.set_schema(None)


class LabelsContainer(Labels, HasLabelsSchema, etas.Container):
    '''Base class for `eta.core.serial.Container`s of `Labels`.

    `LabelsContainer`s can optionally store a `LabelsContainerSchema` instance
    that governs the schema of the labels in the container.
    '''

    def __init__(self, schema=None, **kwargs):
        '''Creates a `LabelsContainer` instance.

        Args:
            schema: an optional LabelsContainerSchema to enforce on the labels
                in this container. By default, no schema is enforced
            **kwargs: valid keyword arguments for `eta.core.serial.Container()`

        Raises:
            LabelsSchemaError: if a schema was provided but the labels added to
                the container violate it
        '''
        HasLabelsSchema.__init__(self, schema=schema)
        etas.Container.__init__(self, **kwargs)

    def __bool__(self):
        return etas.Container.__bool__(self)

    @property
    def is_empty(self):
        '''Whether this container has no labels.'''
        return etas.Container.is_empty(self)

    def add_container(self, container):
        '''Appends the labels in the given `LabelContainer` to the container.

        Args:
            container: a LabelsContainer

        Raises:
            LabelsSchemaError: if this container has a schema enforced and any
                labels in the container violate it
        '''
        self.add_iterable(container)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        _attrs = []
        if self.has_schema:
            _attrs.append("schema")

        _attrs += super(LabelsContainer, self).attributes()
        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs a LabelsContainer from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a LabelsContainer
        '''
        schema = d.get("schema", None)
        if schema is not None:
            schema_cls = cls.get_schema_cls()
            schema = schema_cls.from_dict(schema)

        return super(LabelsContainer, cls).from_dict(d, schema=schema)

    def validate_schema(self):
        '''Validates that the labels are compliant with the current schema.

        Raises:
            LabelsSchemaError: if the container has labels that are not
                compliant with the schema
        '''
        if self.has_schema:
            for labels in self:
                self._validate_labels(labels)

    def _validate_labels(self, labels):
        if self.has_schema:
            self.schema.validate(labels)


class LabelsContainerSchema(LabelsSchema):
    '''Base class for schemas of `LabelsContainer`s.'''

    def add(self, labels):
        '''Incorporates the `Labels` into the schema.

        Args:
            label: a Labels instance
        '''
        self.merge_schema(labels.get_active_schema())

    def add_container(self, container):
        '''Incorporates the given `LabelsContainer`s elements into the schema.

        Args:
            container: a LabelsContainer
        '''
        self.add_iterable(container)

    def add_iterable(self, iterable):
        '''Incorporates the given iterable of `Labels` into the schema.

        Args:
            iterable: an iterable of Labels
        '''
        for labels in iterable:
            self.add(labels)

    @classmethod
    def build_active_schema(cls, container):
        '''Builds a `LabelsContainerSchema` describing the active schema of
        the `LabelsContainer`.

        Args:
            container: a LabelsContainer

        Returns:
            a LabelsContainerSchema
        '''
        schema = cls()
        for labels in container:
            schema.add(labels.get_active_schema())

        return schema


class LabelsContainerSchemaError(LabelsSchemaError):
    '''Error raisesd when a `LabelsContainerSchema` is violated.'''
    pass


class LabelsSet(Labels, HasLabelsSchema, etas.Set):
    '''Base class for `eta.core.serial.Set`s of `Labels`.

    `LabelsSet`s can optionally store a `LabelsSchema` instance that governs
    the schemas of the `Labels` in the set.
    '''

    def __init__(self, schema=None, **kwargs):
        '''Creates a `LabelsSet` instance.

        Args:
            schema: an optional LabelsSchema to enforce on each element of the
                set. By default, no schema is enforced
            **kwargs: valid keyword arguments for `eta.core.serial.Set()`

        Raises:
            LabelsSchemaError: if a schema was provided but the labels added to
                the container violate it
        '''
        HasLabelsSchema.__init__(self, schema=schema)
        etas.Set.__init__(self, **kwargs)

    def __bool__(self):
        return etas.Set.__bool__(self)

    @property
    def is_empty(self):
        '''Whether this set has no labels.'''
        return etas.Set.is_empty(self)

    def __getitem__(self, key):
        '''Gets the `Labels` for the given key.

        If the key is not found, an empty `Labels` is created for it, and
        returned.

        Args:
            key: the key

        Returns:
            a Labels instance
        '''
        if key not in self:
            # pylint: disable=not-callable
            labels = self._ELE_CLS(**{self._ELE_KEY_ATTR: key})
            self.add(labels)

        return super(LabelsSet, self).__getitem__(key)

    @classmethod
    def get_schema_cls(cls):
        '''Gets the schema class for the `Labels` in the set.

        Returns:
            the LabelsSchema class
        '''
        return cls._ELE_CLS.get_schema_cls()

    def empty(self):
        '''Returns an empty copy of the `LabelsSet`.

        The schema of the set is preserved, if applicable.

        Returns:
            an empty LabelsSet
        '''
        return self.__class__(schema=self.schema)

    def add_set(self, labels_set):
        '''Adds the labels in the given `LabelSet` to the set.

        Args:
            labels_set: a LabelsSet

        Raises:
            LabelsSchemaError: if this set has a schema enforced and any labels
                in the set violate it
        '''
        self.add_iterable(labels_set)

    def get_active_schema(self):
        '''Gets the `LabelsSchema` describing the active schema of the set.

        Returns:
            a `LabelsSchema`
        '''
        schema_cls = self.get_schema_cls()
        schema = schema_cls()
        for labels in self:
            schema.merge_schema(schema_cls.build_active_schema(labels))

        return schema

    def filter_by_schema(self, schema):
        '''Removes labels from the set that are not compliant with the given
        schema.

        Args:
            schema: a LabelsSchema
        '''
        for labels in self:
            labels.filter_by_schema(schema)

    def set_schema(self, schema, filter_by_schema=False, validate=False):
        '''Sets the enforced schema to the given `LabelsSchema`.

        Args:
            schema: a LabelsSchema to assign
            filter_by_schema: whether to filter labels that are not compliant
                with the schema. By default, this is False
            validate: whether to validate that the labels (after filtering, if
                applicable) are compliant with the new schema. By default, this
                is False

        Raises:
            LabelsSchemaError: if `validate` was `True` and this object
                contains labels that are not compliant with the schema
        '''
        self.schema = schema
        for labels in self:
            labels.set_schema(
                schema, filter_by_schema=filter_by_schema, validate=validate)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        _attrs = []
        if self.has_schema:
            _attrs.append("schema")

        _attrs += super(LabelsSet, self).attributes()
        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs a `LabelsSet` from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a LabelsSet
        '''
        schema = d.get("schema", None)
        if schema is not None:
            schema_cls = cls.get_schema_cls()
            schema = schema_cls.from_dict(schema)

        return super(LabelsSet, cls).from_dict(d, schema=schema)

    @classmethod
    def from_labels_patt(cls, labels_patt):
        '''Creates a `LabelsSet` from a pattern of `Labels` files on disk.

        Args:
             labels_patt: a pattern with one or more numeric sequences for
                Labels files on disk

        Returns:
            a LabelsSet
        '''
        labels_set = cls()
        for labels_path in etau.get_pattern_matches(labels_patt):
            labels_set.add(cls._ELE_CLS.from_json(labels_path))

        return labels_set

    def validate_schema(self):
        '''Validates that the labels in the set are compliant with the current
        schema.

        Raises:
            LabelsSchemaError: if the set has labels that are not compliant
                with the schema
        '''
        if self.has_schema:
            for labels in self:
                self._validate_labels(labels)

    def _validate_labels(self, labels):
        if self.has_schema:
            self.schema.validate(labels)
