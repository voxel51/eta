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
import eta.core.events as etae
import eta.core.labels as etal
import eta.core.objects as etao
import eta.core.serial as etas


class FrameMaskIndex(etad.MaskIndex):
    '''An index of semantics for the values in a frame mask.'''
    pass


class FrameLabels(etal.Labels):
    '''Class encapsulating labels for a frame, i.e., an image or a video frame.

    FrameLabels are spatial concepts that describe a collection of information
    about a specific frame. FrameLabels can have frame-level attributes,
    object detections, event detections, and segmentation masks.

    Attributes:
        frame_number: (optional) the frame number
        mask: (optional) a segmentation mask for the frame
        mask_index: (optional) a FrameMaskIndex describing the semantics of the
            segmentation mask
        attrs: an AttributeContainer of attributes of the frame
        objects: a DetectedObjectContainer of objects in the frame
        events: a DetectedEventContainer of events in the frame
    '''

    def __init__(
            self, frame_number=None, mask=None, mask_index=None, attrs=None,
            objects=None, events=None):
        '''Creates a FrameLabels instance.

        Args:
            frame_number: (optional) a frame number for the labels
            mask: (optional) a segmentation mask for the frame
            mask_index: (optional) a FrameMaskIndex describing the semantics of
                the segmentation mask
            attrs: (optional) an AttributeContainer of attributes for the frame
            objects: (optional) a DetectedObjectContainer of objects for the
                frame
            events: (optional) a DetectedEventContainer of events for the frame
        '''
        self.frame_number = frame_number
        self.mask = mask
        self.mask_index = mask_index
        self.attrs = attrs or etad.AttributeContainer()
        self.objects = objects or etao.DetectedObjectContainer()
        self.events = events or etae.DetectedEventContainer()

    @property
    def is_empty(self):
        '''Whether the frame has no labels of any kind.'''
        return not (
            self.has_mask or self.has_attributes or self.has_objects
            or self.has_events)

    @property
    def has_frame_number(self):
        '''Whether the frame has a frame number.'''
        return self.frame_number is not None

    @property
    def has_mask(self):
        '''Whether this frame has a segmentation mask.'''
        return self.mask is not None

    @property
    def has_mask_index(self):
        '''Whether this frame has a segmentation mask index.'''
        return self.mask_index is not None

    @property
    def has_attributes(self):
        '''Whether the frame has at least one attribute.'''
        return bool(self.attrs)

    @property
    def has_objects(self):
        '''Whether the frame has at least one object.'''
        return bool(self.objects)

    @property
    def has_object_attributes(self):
        '''Whether the frame has at least one object with attributes.'''
        for obj in self.objects:
            if obj.has_attributes:
                return True

        return False

    @property
    def has_events(self):
        '''Whether the frame has at least one event.'''
        return bool(self.events)

    @property
    def has_event_attributes(self):
        '''Whether the frame has at least one event with attributes.'''
        for event in self.events:
            if event.has_attributes:
                return True

        return False

    def iter_attributes(self):
        '''Returns an iterator over the attributes of the frame.

        Returns:
            an iterator over `Attribute`s
        '''
        return iter(self.attrs)

    def iter_objects(self):
        '''Returns an iterator over the objects in the frame.

        Returns:
            an iterator over `DetectedObject`s
        '''
        return iter(self.objects)

    def iter_events(self):
        '''Returns an iterator over the events in the frame.

        Returns:
            an iterator over `DetectedEvent`s
        '''
        return iter(self.events)

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

    def add_event(self, event):
        '''Adds the event to the frame.

        Args:
            event: a DetectedEvent
        '''
        self.events.add(event)

    def add_events(self, events):
        '''Adds the events to the frame.

        Args:
            events: a DetectedEventContainer
        '''
        self.events.add_container(events)

    def clear_attributes(self):
        '''Removes all frame-level attributes from the frame.'''
        self.attrs = etad.AttributeContainer()

    def clear_objects(self):
        '''Removes all objects from the frame.'''
        self.objects = etao.DetectedObjectContainer()

    def clear_events(self):
        '''Removes all events from the frame.'''
        self.events = etae.DetectedEventContainer()

    def clear(self):
        '''Removes all labels from the frame.'''
        self.clear_attributes()
        self.clear_objects()
        self.clear_events()

    def merge_labels(self, frame_labels, reindex=False):
        '''Merges the given FrameLabels into this labels.

        Args:
            frame_labels: a FrameLabels
            reindex: whether to offset the `index` fields of objects and events
                in `frame_labels` before merging so that all indices are
                unique. The default is False
        '''
        if reindex:
            self._reindex_objects(frame_labels)
            self._reindex_events(frame_labels)

        if frame_labels.has_mask:
            self.mask = frame_labels.mask
        if frame_labels.has_mask_index:
            self.mask_index = frame_labels.mask_index

        self.add_attributes(frame_labels.attrs)
        self.add_objects(frame_labels.objects)
        self.add_events(frame_labels.events)

    def filter_by_schema(self, schema):
        '''Filters the frame labels by the given schema.

        Args:
            schema: a FrameLabelsSchema
        '''
        self.attrs.filter_by_schema(schema.attrs)
        self.objects.filter_by_schema(schema.objects)
        self.events.filter_by_schema(schema.events)

    def remove_objects_without_attrs(self, labels=None):
        '''Removes objects from the frame that do not have attributes.

        Args:
            labels: an optional list of object `label` strings to which to
                restrict attention when filtering. By default, all objects are
                processed
        '''
        self.objects.remove_objects_without_attrs(labels=labels)
        self.events.remove_objects_without_attrs(labels=labels)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        _attrs = []
        if self.has_frame_number:
            _attrs.append("frame_number")
        if self.has_mask:
            _attrs.append("mask")
        if self.has_mask_index:
            _attrs.append("mask_index")
        if self.attrs:
            _attrs.append("attrs")
        if self.objects:
            _attrs.append("objects")
        if self.events:
            _attrs.append("events")
        return _attrs

    @classmethod
    def from_dict(cls, d, **kwargs):
        '''Constructs a FrameLabels from a JSON dictionary.

        Args:
            d: a JSON dictionary
            **kwargs: keyword arguments that have already been parsed by a
            subclass

        Returns:
            a FrameLabels
        '''
        frame_number = d.get("frame_number", None)

        mask = d.get("mask", None)
        if mask is not None:
            mask = etas.deserialize_numpy_array(mask)

        mask_index = d.get("mask_index", None)
        if mask_index is not None:
            mask_index = FrameMaskIndex.from_dict(mask_index)

        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainer.from_dict(attrs)

        objects = d.get("objects", None)
        if objects is not None:
            objects = etao.DetectedObjectContainer.from_dict(objects)

        events = d.get("events", None)
        if events is not None:
            events = etae.DetectedEventContainer.from_dict(events)

        return cls(
            frame_number=frame_number, mask=mask, mask_index=mask_index,
            attrs=attrs, objects=objects, events=events, **kwargs)

    def _reindex_objects(self, frame_labels):
        self_indices = self._get_object_indices(self)
        if not self_indices:
            return

        new_indices = self._get_object_indices(frame_labels)
        if not new_indices:
            return

        offset = max(self_indices) + 1 - min(new_indices)
        self._offset_object_indices(frame_labels, offset)

    @staticmethod
    def _get_object_indices(frame_labels):
        obj_indices = set()

        for obj in frame_labels.objects:
            if obj.index is not None:
                obj_indices.add(obj.index)

        for event in frame_labels.events:
            for obj in event.objects:
                if obj.index is not None:
                    obj_indices.add(obj.index)

        return obj_indices

    @staticmethod
    def _offset_object_indices(frame_labels, offset):
        for obj in frame_labels.objects:
            if obj.index is not None:
                obj.index += offset

        for event in frame_labels.events:
            for obj in event.objects:
                if obj.index is not None:
                    obj.index += offset

    def _reindex_events(self, frame_labels):
        self_indices = self._get_event_indices(self)
        if not self_indices:
            return

        new_indices = self._get_event_indices(frame_labels)
        if not new_indices:
            return

        offset = max(self_indices) + 1 - min(new_indices)
        self._offset_event_indices(frame_labels, offset)

    @staticmethod
    def _get_event_indices(frame_labels):
        event_indices = set()

        for event in frame_labels.events:
            if event.index is not None:
                event_indices.add(event.index)

        return event_indices

    @staticmethod
    def _offset_event_indices(frame_labels, offset):
        for event in frame_labels.events:
            if event.index is not None:
                event.index += offset


class FrameLabelsSchema(etal.LabelsSchema):
    '''Schema for FrameLabels.

    Attributes:
        attrs: an AttributeContainerSchema describing attributes of the
            frame(s)
        objects: an ObjectContainerSchema describing the objects in the
            frame(s)
        events: an EventContainerSchema describing the events in the frame(s)
    '''

    def __init__(self, attrs=None, objects=None, events=None):
        '''Creates a FrameLabelsSchema instance.

        Args:
            attrs: (optional) an AttributeContainerSchema describing the
                attributes of the frame(s)
            objects: (optional) an ObjectContainerSchema describing the objects
                in the frame(s)
            events: (optional) an EventContainerSchema describing the events
                in the frame(s)
        '''
        self.attrs = attrs or etad.AttributeContainerSchema()
        self.objects = objects or etao.ObjectContainerSchema()
        self.events = events or etae.EventContainerSchema()

    @property
    def is_empty(self):
        '''Whether this schema has no labels of any kind.'''
        return (
            self.attrs.is_empty and self.objects.is_empty
            and self.events.is_empty)

    def has_attribute(self, attr_name):
        '''Whether the schema has a frame-level attribute with the given name.

        Args:
            attr_name: the attribute name

        Returns:
            True/False
        '''
        return self.attrs.has_attribute(attr_name)

    def get_attribute_class(self, attr_name):
        '''Gets the `Attribute` class for the frame-level attribute with the
        given name.

        Args:
            attr_name: the attribute name

        Returns:
            the Attribute class
        '''
        return self.attrs.get_attribute_class(attr_name)

    def has_object_label(self, label):
        '''Whether the schema has an object with the given label.

        Args:
            label: the object label

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
            label: the object label
            attr_name: a frame-level object attribute name

        Returns:
            True/False
        '''
        return self.objects.has_frame_attribute(label, attr_name)

    def get_object_attribute_schema(self, label, attr_name):
        '''Gets the AttributeSchema for the frame-level attribute of the given
        name for the object with the given label.

        Args:
            label: the object label
            attr_name: the name of the frame-level object attribute

        Returns:
            the AttributeSchema
        '''
        return self.objects.get_frame_attribute_schema(label, attr_name)

    def get_object_attribute_class(self, label, attr_name):
        '''Gets the Attribute class for the frame-level attribute of the given
        name for the object with the given label.

        Args:
            label: the object label
            attr_name: a frame-level object attribute name

        Returns:
            the Attribute class
        '''
        return self.objects.get_frame_attribute_class(label, attr_name)

    def has_event_label(self, label):
        '''Whether the schema has an event with the given label.

        Args:
            label: the event label

        Returns:
            True/False
        '''
        return self.events.has_event_label(label)

    def get_event_schema(self, label):
        '''Gets the EventSchema for the event with the given label.

        Args:
            label: the event label

        Returns:
            the EventSchema
        '''
        return self.events.get_event_schema(label)

    def has_event_attribute(self, label, attr_name):
        '''Whether the schema has an event with the given label with an
        event-level attribute with the given name.

        Args:
            label: an event label
            attr_name: an event-level attribute name

        Returns:
            True/False
        '''
        return self.events.has_event_attribute(label, attr_name)

    def get_event_attribute_schema(self, label, attr_name):
        '''Gets the AttributeSchema for the event-level attribute of the given
        name for the event with the given label.

        Args:
            label: the event label
            attr_name: the name of the event-level attribute

        Returns:
            the AttributeSchema
        '''
        return self.events.get_event_attribute_schema(label, attr_name)

    def get_event_attribute_class(self, label, attr_name):
        '''Gets the Attribute class for the event-level attribute of the given
        name for the event with the given label.

        Args:
            label: the event label
            attr_name: a frame-level object attribute name

        Returns:
            the Attribute class
        '''
        return self.events.get_event_attribute_class(label, attr_name)

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
            attr: an Attribute
        '''
        self.objects.add_frame_attribute(label, attr)

    def add_object_attributes(self, label, attrs):
        '''Adds the frame-level attributes for the object with the given label
        to the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer
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

    def add_event_label(self, label):
        '''Adds the given event label to the schema.

        Args:
            label: an event label
        '''
        self.events.add_event_label(label)

    def add_event_attribute(self, label, attr):
        '''Adds the event-level attribute for the event with the given label to
        the schema.

        Args:
            label: an event label
            attr: an Attribute
        '''
        self.events.add_event_attribute(label, attr)

    def add_event_attributes(self, label, attrs):
        '''Adds the event-level attributes for the event with the given label
        to the schema.

        Args:
            label: an event label
            attrs: an AttributeContainer
        '''
        self.events.add_event_attributes(label, attrs)

    def add_event(self, event):
        '''Adds the event to the schema.

        Args:
            event: a DetectedEvent
        '''
        self.events.add_event(event)

    def add_events(self, events):
        '''Adds the events to the schema.

        Args:
            events: a DetectedEventContainer
        '''
        self.events.add_events(events)

    def add_labels(self, frame_labels):
        '''Adds the labels to the schema.

        Args:
            frame_labels: a FrameLabels
        '''
        self.add_attributes(frame_labels.attrs)
        self.add_objects(frame_labels.objects)
        self.add_events(frame_labels.events)

    def is_valid_attribute(self, attr):
        '''Whether the frame-level attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Returns:
            True/False
        '''
        return self.attrs.is_valid_attribute(attr)

    def is_valid_attributes(self, attrs):
        '''Whether the frame-level attributes is compliant with the schema.

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
        '''Whether the frame-level attributes for the object with the given
        label are compliant with the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer

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

    def is_valid_event_label(self, label):
        '''Whether the event label is compliant with the schema.

        Args:
            label: an event label

        Returns:
            True/False
        '''
        return self.events.is_valid_event_label(label)

    def is_valid_event_attribute(self, label, attr):
        '''Whether the event-level attribute for the event with the given label
        is compliant with the schema.

        Args:
            label: an event label
            attr: an Attribute

        Returns:
            True/False
        '''
        return self.events.is_valid_event_attribute(label, attr)

    def is_valid_event_attributes(self, label, attrs):
        '''Whether the event-level attributes for the event with the given
        label are compliant with the schema.

        Args:
            label: an event label
            attrs: an AttributeContainer

        Returns:
            True/False
        '''
        return self.events.is_valid_event_attributes(label, attrs)

    def is_valid_event(self, event):
        '''Whether the given event is compliant with the schema.

        Args:
            event: a DetectedEvent

        Returns:
            True/False
        '''
        return self.events.is_valid_event(event)

    def validate_attribute(self, attr):
        '''Validates that the frame-level attribute is compliant with the
        schema.

        Args:
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        '''
        self.attrs.validate_attribute(attr)

    def validate_attributes(self, attrs):
        '''Validates that the frame-level attributes are compliant with the
        schema.

        Args:
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        '''
        self.attrs.validate(attrs)

    def validate_object_label(self, label):
        '''Validates that the object label is compliant with the schema.

        Args:
            label: an object label

        Raises:
            LabelsSchemaError: if the object label violates the schema
        '''
        self.objects.validate_object_label(label)

    def validate_object_attribute(self, label, attr):
        '''Validates that the frame-level attribute for the object with the
        given label is compliant with the schema.

        Args:
            label: an object label
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        '''
        self.objects.validate_frame_attribute(label, attr)

    def validate_object_attributes(self, label, attrs):
        '''Validates that the frame-level attributes for the object with the
        given label are compliant with the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        '''
        self.objects.validate_frame_attributes(label, attrs)

    def validate_object(self, obj):
        '''Validates that the object is compliant with the schema.

        Args:
            obj: a DetectedObject

        Raises:
            LabelsSchemaError: if the object violates the schema
        '''
        self.objects.validate_object(obj)

    def validate_event_label(self, label):
        '''Validates that the event label is compliant with the schema.

        Args:
            label: an event label

        Raises:
            LabelsSchemaError: if the event label violates the schema
        '''
        self.events.validate_event_label(label)

    def validate_event_attribute(self, label, attr):
        '''Validates that the event-level attribute for the event with the
        given label is compliant with the schema.

        Args:
            label: an event label
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        '''
        self.events.validate_event_attribute(label, attr)

    def validate_event_attributes(self, label, attrs):
        '''Validates that the event-level attributes for the event with the
        given label are compliant with the schema.

        Args:
            label: an event label
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        '''
        self.events.validate_event_attributes(label, attrs)

    def validate_event(self, event):
        '''Validates that the event is compliant with the schema.

        Args:
            event: a DetectedEvent

        Raises:
            LabelsSchemaError: if the event violates the schema
        '''
        self.events.validate_event(event)

    def validate(self, frame_labels):
        '''Validates that the labels are compliant with the schema.

        Args:
            frame_labels: a FrameLabels

        Raises:
            LabelsSchemaError: if the labels violate the schema
        '''
        self.validate_attributes(frame_labels.attrs)

        for obj in frame_labels.objects:
            self.validate_object(obj)

        for event in frame_labels.events:
            self.validate_event(event)

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
        self.events.validate_subset_of_schema(schema.events)

    def merge_schema(self, schema):
        '''Merges the given FrameLabelsSchema into this schema.

        Args:
            schema: a FrameLabelsSchema
        '''
        self.attrs.merge_schema(schema.attrs)
        self.objects.merge_schema(schema.objects)
        self.events.merge_schema(schema.events)

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

    @classmethod
    def from_video_labels_schema(cls, video_labels_schema):
        '''Creates a FrameLabelsSchema from a VideoLabelsSchema.

        Args:
            video_labels_schema: a VideoLabelsSchema

        Returns:
            a FrameLabelsSchema
        '''
        return cls(
            attrs=video_labels_schema.frames,
            objects=video_labels_schema.objects)

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
        if self.events:
            _attrs.append("events")
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

        events = d.get("events", None)
        if events is not None:
            events = etae.EventContainerSchema.from_dict(events)

        return cls(attrs=attrs, objects=objects, events=events)


class FrameLabelsSchemaError(etal.LabelsSchemaError):
    '''Error raised when an `FrameLabelsSchema` is violated.'''
    pass
