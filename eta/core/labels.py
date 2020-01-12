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

from eta.core.serial import Container, Serializable


class Labels(Serializable):
    '''Base class for `eta.core.serial.Serializable` classes that hold labels
    representing attributes, objects, frames, events, images, videos, etc.

    Labels classes have associated `eta.core.labels.Schema` classes that
    describe the ontologies over the labels class.
    '''

    #
    # The `LabelsSchema` class for this labels class
    #
    # Subclasses MUST set this field
    #
    _SCHEMA_CLS = None

    @classmethod
    def get_schema_cls(cls):
        '''Gets the schema class for the labels.

        Returns:
            the LabelsSchema class
        '''
        return cls._SCHEMA_CLS

    def filter_by_schema(self, schema):
        '''Filters the labels by the given schema.

        Args:
            schema: a LabelsSchema
        '''
        raise NotImplementedError(
            "subclasses must implement `filter_by_schema()`")


class LabelsSchema(Serializable):
    '''Base class for schemas of `eta.core.labels.Labels` classes.'''

    #
    # The `Labels` class for this schema
    #
    # Subclasses MUST set this field
    #
    _LABELS_CLS = None

    @classmethod
    def get_labels_cls(cls):
        '''Gets the Labels class for the schema.

        Returns:
            the Labels class
        '''
        return cls._LABELS_CLS

    def merge_schema(self, schema):
        '''Merges the given LabelsSchema into this schema.

        Args:
            schema: a LabelsSchema
        '''
        raise NotImplementedError("subclasses must implement `merge_schema()`")

    def is_valid(self, labels):
        '''Whether the Labels are compliant with the schema.

        Args:
            labels: a Labels instance

        Returns:
            True/False
        '''
        raise NotImplementedError("subclasses must implement `is_valid()`")

    def validate(self, labels):
        '''Validates that the labels are compliant with the schema.

        Args:
            labels: a Labels instance

        Raises:
            LabelsSchemaError: if the labels violate the schema
        '''
        raise NotImplementedError("subclasses must implement `validate()`")

    @classmethod
    def build_active_schema(cls, labels):
        '''Builds an LabelsSchema that describes the active schema of the
        labels.

        Args:
            labels: a Labels instance

        Returns:
            a LabelsSchema
        '''
        raise NotImplementedError(
            "subclasses must implement `build_active_schema()`")


class LabelsSchemaError(Exception):
    '''Error raisesd when a `LabelsSchema` is violated.'''
    pass


class LabelsContainer(Labels, Container):
    '''Base class for `eta.core.serial.Container`s of
    `eta.core.labels.Labels`.
    '''

    def __init__(self, schema=None, **kwargs):
        '''Creates LabelsContainer instance.

        Args:
            schema: an optional LabelsContainerSchema to enforce on the
                elements in this container. By default, no schema is enforced
            **kwargs: valid keyword arguments for `eta.core.serial.Container()`

        Raises:
            LabelsContainerSchemaError: if a schema was provided but the
                elements added to the container violate it
        '''
        super(LabelsContainer, self).__init__(**kwargs)
        self.schema = None
        if schema is not None:
            self.set_schema(schema)

    @property
    def has_schema(self):
        '''Whether the container has an enforced schema.'''
        return self.schema is not None

    def add(self, element):
        '''Appends the element to the container.

        Args:
            element: an instance of `_ELE_CLS`

        Raises:
            LabelsContainerSchemaError: if this container has a schema enforced
                and the element violates it
        '''
        if self.has_schema:
            self._validate_element(element)

        super(LabelsContainer, self).add(element)

    def add_container(self, container):
        '''Appends the given container's elements to the container.

        Args:
            elements: a Container of `_ELE_CLS` objects

        Raises:
            LabelsContainerSchemaError: if this container has a schema enforced
                and an element in the container violates it
        '''
        if self.has_schema:
            for element in container:
                self._validate_element(element)

        super(LabelsContainer, self).add_container(container)

    def add_iterable(self, elements):
        '''Appends the elements in the given iterable to the container.

        Args:
            elements: an iterable of `_ELE_CLS` objects

        Raises:
            LabelsContainerSchemaError: if this container has a schema enforced
                and an element in the container violates it
        '''
        if self.has_schema:
            for element in elements:
                self._validate_element(element)

        super(LabelsContainer, self).add_iterable(elements)

    def get_schema(self):
        '''Gets the current enforced schema for the container, or None if no
        schema is enforced.

        Returns:
            a LabelsContainerSchema
        '''
        return self.schema

    def get_active_schema(self):
        '''Returns a LabelsContainerSchema describing the active schema of the
        container.

        Returns:
            a LabelsContainerSchema
        '''
        schema_cls = self.get_schema_cls()
        return schema_cls.build_active_schema(self)

    def set_schema(self, schema, filter_by_schema=False):
        '''Sets the enforced schema to the given LabelsContainerSchema.

        Args:
            schema: the LabelsContainerSchema
            filter_by_schema: whether to filter any invalid values from the
                container after changing the schema. By default, this is False
                and thus the container must already meet the new schema
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

    def _validate_element(self, element):
        if self.has_schema:
            self.schema.validate_element(element)

    def _validate_schema(self):
        if self.has_schema:
            for element in self:
                self._validate_element(element)


class LabelsContainerSchema(LabelsSchema):
    '''Base class for schemas of `eta.core.labels.LabelsContainer`s.'''

    def validate_element(self, element):
        '''Validates that the element is compliant with the schema.

        Args:
            element: an `_LABELS_CLS._ELE_CLS` instance

        Raises:
            LabelsContainerSchemaError: if the element violates the schema
        '''
        raise NotImplementedError(
            "subclasses must implement `validate_element()`")


class LabelsContainerSchemaError(Exception):
    '''Error raisesd when a `LabelsContainerSchema` is violated.'''
    pass
