'''
Core tools and data structures for working with frames of videos.

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

import eta.core.data as etad
import eta.core.labels as etal
import eta.core.objects as etao


class FrameLabels(etal.Labels):
    '''Class encapsulating labels for a frame, i.e., an image or a specific
    frame of a video.

    Attributes:
        attrs: AttributeContainer describing attributes of the frame
        objects: DetectedObjectContainer describing detected objects in the
            frame
    '''

    def __init__(self, attrs=None, objects=None):
        '''Constructs a FrameLabels instance.

        Args:
            attrs: (optional) AttributeContainer of attributes for the frame.
                By default, an empty AttributeContainer is created
            objects: (optional) DetectedObjectContainer of detected objects for
                the frame. By default, an empty DetectedObjectContainer is
                created
        '''
        self.attrs = attrs or etad.AttributeContainer()
        self.objects = objects or etao.DetectedObjectContainer()

    @property
    def has_attributes(self):
        '''Whether the frame has at least one attribute.'''
        return bool(self.attrs)

    @property
    def has_objects(self):
        '''Whether the frame has at least one object.'''
        return bool(self.objects)

    @property
    def is_empty(self):
        '''Whether the frame has no labels of any kind.'''
        return not self.has_attributes and not self.has_objects

    def add_attribute(self, attr):
        '''Adds the frame-level attribute to the frame.

        Args:
            attr: an Attribute
        '''
        self.attrs.add(attr)

    def add_attributes(self, attrs):
        '''Adds the frame-level attributes to the frame.

        Args:
            attrs: an AttributeContainer
        '''
        self.attrs.add_container(attrs)

    def add_object(self, obj):
        '''Adds the object to the frame.

        Args:
            obj: a DetectedObject
        '''
        self.objects.add(obj)

    def add_objects(self, objs):
        '''Adds the objects to the frame.

        Args:
            objs: a DetectedObjectContainer
        '''
        self.objects.add_container(objs)

    def clear(self):
        '''Removes all labels from the frame.'''
        self.clear_attributes()
        self.clear_objects()

    def clear_attributes(self):
        '''Removes all frame-level attributes from the frame.'''
        self.attrs = etad.AttributeContainer()

    def clear_objects(self):
        '''Removes all objects from the frame.'''
        self.objects = etao.DetectedObjectContainer()

    def merge_labels(self, frame_labels):
        '''Merges the labels into the frame.

        Args:
            frame_labels: a FrameLabels
        '''
        self.add_attributes(frame_labels.attrs)
        self.add_objects(frame_labels.objects)

    def filter_by_schema(self, schema):
        '''Removes objects/attributes from this object that are not compliant
        with the given schema.

        Args:
            schema: a FrameLabelsSchema
        '''
        self.attrs.filter_by_schema(schema.attrs)
        self.objects.filter_by_schema(schema.objects)

    def remove_objects_without_attrs(self, labels=None):
        '''Removes objects from the frame that do not have attributes.

        Args:
            labels: an optional list of object `label` strings to which to
                restrict attention when filtering. By default, all objects are
                processed
        '''
        self.objects.remove_objects_without_attrs(labels=labels)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        _attrs = []
        if self.attrs:
            _attrs.append("attrs")
        if self.objects:
            _attrs.append("objects")
        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs a `FrameLabels` from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a FrameLabels
        '''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainer.from_dict(attrs)

        objects = d.get("objects", None)
        if objects is not None:
            objects = etao.DetectedObjectContainer.from_dict(objects)

        return cls(attrs=attrs, objects=objects)


class FrameLabelsSchema(etal.LabelsSchema):
    '''Schema for FrameLabels.

    Attributes:
        attrs: an AttributeContainerSchema describing the attributes of the
            frame(s)
        objects: an ObjectContainerSchema describing the objects of the
            frame(s)
    '''

    def __init__(self, attrs=None, objects=None):
        '''Creates a FrameLabelsSchema instance.

        Args:
            attrs: (optional) an AttributeContainerSchema describing the
                attributes of the frame(s)
            objects: (optional) an ObjectContainerSchema describing the objects
                of the frame(s)
        '''
        self.attrs = attrs or etad.AttributeContainerSchema()
        self.objects = objects or etao.ObjectContainerSchema()

    @property
    def is_empty(self):
        '''Whether this schema has no labels of any kind.'''
        return not bool(self.attrs) and not bool(self.objects)

    def has_attribute(self, attr_name):
        '''Whether the schema has a frame-level attribute with the given name.

        Args:
            attr_name: an attribute name

        Returns:
            True/False
        '''
        return self.attrs.has_attribute(attr_name)

    def get_attribute_class(self, attr_name):
        '''Gets the `Attribute` class for the frame-level attribute with the
        given name.

        Args:
            attr_name: an attribute name

        Returns:
            the Attribute class
        '''
        return self.attrs.get_attribute_class(attr_name)

    def has_object_label(self, label):
        '''Whether the schema has an object with the given label.

        Args:
            label: an object label

        Returns:
            True/False
        '''
        return self.objects.has_object_label(label)

    def get_object_schema(self, label):
        '''Gets the `ObjectSchema` for the object with the given label.

        Args:
            label: the object label

        Returns:
            the ObjectSchema
        '''
        return self.objects.get_object_schema(label)

    def has_object_attribute(self, label, attr_name):
        '''Whether the schema has an object with the given label with a
        frame-level attribute with the given name.

        Args:
            label: an object label
            attr_name: a frame-level object attribute name

        Returns:
            True/False
        '''
        return self.objects.has_frame_attribute(label, attr_name)

    def get_object_attribute_schema(self, label, attr_name):
        '''Gets the `AttributeSchema` for the frame-level attribute of the
        given name for the object with the given label.

        Args:
            label: the object label
            attr_name: the name of the frame-level object attribute

        Returns:
            the AttributeSchema
        '''
        return self.objects.get_frame_attribute_schema(label, attr_name)

    def get_object_attribute_class(self, label, attr_name):
        '''Gets the `Attribute` class for the frame-level attribute of the
        given name for the object with the given label.

        Args:
            label: an object label
            attr_name: a frame-level object attribute name

        Returns:
            the Attribute class
        '''
        return self.objects.get_frame_attribute_class(label, attr_name)

    def add_attribute(self, attr):
        '''Adds the given frame-level attribute to the schema.

        Args:
            attr: an Attribute
        '''
        self.attrs.add_attribute(attr)

    def add_attributes(self, attrs):
        '''Adds the given frame-level attributes to the schema.

        Args:
            attrs: an AttributeContainer
        '''
        self.attrs.add_attributes(attrs)

    def add_object_label(self, label):
        '''Adds the given object label to the schema.

        Args:
            label: an object label
        '''
        self.objects.add_object_label(label)

    def add_object_attribute(self, label, attr):
        '''Adds the frame-level attribute for the object with the given label
        to the schema.

        Args:
            label: an object label
            attr: a frame-level Attribute
        '''
        self.objects.add_frame_attribute(label, attr)

    def add_object_attributes(self, label, attrs):
        '''Adds the frame-level attributes for the object with the given label
        to the schema.

        Args:
            label: an object label
            attrs: a frame-level AttributeContainer
        '''
        self.objects.add_frame_attributes(label, attrs)

    def add_object(self, obj):
        '''Adds the object to the schema.

        Args:
            obj: a DetectedObject
        '''
        self.objects.add_object(obj)

    def add_objects(self, objects):
        '''Adds the objects to the schema.

        Args:
            objects: a DetectedObjectContainer
        '''
        self.objects.add_objects(objects)

    def add_labels(self, frame_labels):
        '''Adds the FrameLabels to the schema.

        Args:
            frame_labels: a FrameLabels
        '''
        self.add_attributes(frame_labels.attrs)
        self.add_objects(frame_labels.objects)

    def is_valid_attribute(self, attr):
        '''Whether the frame-level attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Returns:
            True/False
        '''
        return self.attrs.is_valid_attribute(attr)

    def is_valid_attributes(self, attrs):
        '''Whether the AttributeContainer of frame-level attributes is
        compliant with the schema.

        Args:
            attrs: an AttributeContainer

        Returns:
            True/False
        '''
        return self.attrs.is_valid_attributes(attrs)

    def is_valid_object_label(self, label):
        '''Whether the object label is compliant with the schema.

        Args:
            label: an object label

        Returns:
            True/False
        '''
        return self.objects.is_valid_object_label(label)

    def is_valid_object_attribute(self, label, attr):
        '''Whether the frame-level attribute for the object with the given
        label is compliant with the schema.

        Args:
            label: an object label
            attr: a frame-level Attribute

        Returns:
            True/False
        '''
        return self.objects.is_valid_frame_attribute(label, attr)

    def is_valid_object_attributes(self, label, attrs):
        '''Whether the AttributeContainer of frame-level attributes for the
        object with the given label is compliant with the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer of frame-level attributes

        Returns:
            True/False
        '''
        return self.objects.is_valid_frame_attributes(label, attrs)

    def is_valid_object(self, obj):
        '''Whether the given object is compliant with the schema.

        Args:
            obj: a DetectedObject

        Returns:
            True/False
        '''
        return self.objects.is_valid_object(obj)

    def validate_attribute(self, attr):
        '''Validates that the frame-level attribute is compliant with the
        schema.

        Args:
            attr: an Attribute

        Raises:
            AttributeContainerSchemaError: if the attribute violates the schema
        '''
        self.attrs.validate_attribute(attr)

    def validate_attributes(self, attrs):
        '''Validates that the AttributeContainer of frame-level attributes is
        compliant with the schema.

        Args:
            attrs: an AttributeContainer

        Raises:
            AttributeContainerSchemaError: if the attributes violate the schema
        '''
        self.attrs.validate(attrs)

    def validate_object_label(self, label):
        '''Validates that the object label is compliant with the schema.

        Args:
            label: an object label

        Raises:
            ObjectContainerSchemaError: if the object label violates the schema
        '''
        self.objects.validate_object_label(label)

    def validate_object_attribute(self, label, attr):
        '''Validates that the frame-level attribute for the object with the
        given label is compliant with the schema.

        Args:
            label: an object label
            attr: a frame-level Attribute

        Raises:
            ObjectContainerSchemaError: if the object label violates the schema
            AttributeContainerSchemaError: if the frame-level attribute
                violates the schema
        '''
        self.objects.validate_frame_attribute(label, attr)

    def validate_object_attributes(self, label, attrs):
        '''Validates that the AttributeContainer of frame-level attributes for
        the object with the given label is compliant with the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer of frame-level attributes

        Raises:
            ObjectContainerSchemaError: if the object label violates the schema
            AttributeContainerSchemaError: if the frame-level attributes
                violate the schema
        '''
        self.objects.validate_frame_attributes(label, attrs)

    def validate_object(self, obj):
        '''Validates that the object is compliant with the schema.

        Args:
            obj: a DetectedObject

        Raises:
            ObjectContainerSchemaError: if the object label violates the schema
            AttributeContainerSchemaError: if any attributes of the object
                violate the schema
        '''
        self.objects.validate_object(obj)

    def validate(self, frame_labels):
        '''Validates that the FrameLabels are compliant with the schema.

        Args:
            frame_labels: a FrameLabels instance

        Raises:
            AttributeContainerSchemaError: if a frame/object attribute
                violates the schema
            ObjectContainerSchemaError: if an object label violates the schema
        '''
        # Validate frame-level attributes
        self.validate_attributes(frame_labels.attrs)

        # Validate DetectedObjects
        for obj in frame_labels.objects:
            self.validate_object(obj)

    def validate_subset_of_schema(self, schema):
        '''Validates that this schema is a subset of the given schema.

        Args:
            schema: a FrameLabelsSchema

        Raises:
            LabelsSchemaError: if this schema is not a subset of the given
                schema
        '''
        self.validate_schema_type(schema)
        self.attrs.validate_subset_of_schema(schema.attrs)
        self.objects.validate_subset_of_schema(schema.objects)

    def merge_schema(self, schema):
        '''Merges the given FrameLabelsSchema into this schema.

        Args:
            schema: a FrameLabelsSchema
        '''
        self.attrs.merge_schema(schema.attrs)
        self.objects.merge_schema(schema.objects)

    @classmethod
    def build_active_schema(cls, frame_labels):
        '''Builds a FrameLabelsSchema that describes the active schema of the
        given FrameLabels.

        Args:
            frame_labels: a FrameLabels instance

        Returns:
            a FrameLabelsSchema
        '''
        schema = cls()
        schema.add_labels(frame_labels)
        return schema

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        _attrs = []
        if self.attrs:
            _attrs.append("attrs")
        if self.objects:
            _attrs.append("objects")

        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs a FrameLabelsSchema from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a FrameLabelsSchema
        '''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainerSchema.from_dict(attrs)

        objects = d.get("objects", None)
        if objects is not None:
            objects = etao.ObjectContainerSchema.from_dict(objects)

        return cls(attrs=attrs, objects=objects)


class FrameLabelsSchemaError(etal.LabelsSchemaError):
    '''Error raised when an `FrameLabelsSchema` is violated.'''
    pass
