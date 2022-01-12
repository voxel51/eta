"""
Core tools and data structures for working with frames of videos.

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

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import eta.core.data as etad
import eta.core.events as etae
import eta.core.geometry as etag
import eta.core.keypoints as etak
import eta.core.labels as etal
import eta.core.objects as etao
import eta.core.polylines as etap
import eta.core.serial as etas


class FrameLabels(etal.Labels):
    """Class encapsulating labels for a frame, i.e., an image or a video frame.

    FrameLabels are spatial concepts that describe a collection of information
    about a specific frame. FrameLabels can have frame-level attributes,
    object detections, keypoints, polylines, event detections, and
    segmentation masks.

    Attributes:
        frame_number: (optional) the frame number
        mask: (optional) a segmentation mask for the frame
        mask_index: (optional) a MaskIndex describing the semantics of the
            segmentation mask
        attrs: an AttributeContainer of attributes of the frame
        objects: a DetectedObjectContainer of objects in the frame
        keypoints: a KeypointsContainer of keypoints in the frame
        polylines: a PolylineContainer of polylines in the frame
        events: a DetectedEventContainer of events in the frame
        tags: (optional) a list of tag strings
    """

    def __init__(
        self,
        frame_number=None,
        mask=None,
        mask_index=None,
        attrs=None,
        objects=None,
        keypoints=None,
        polylines=None,
        events=None,
        tags=None,
    ):
        """Creates a FrameLabels instance.

        Args:
            frame_number: (optional) a frame number for the labels
            mask: (optional) a segmentation mask for the frame
            mask_index: (optional) a MaskIndex describing the semantics of the
                segmentation mask
            attrs: (optional) an AttributeContainer of attributes for the frame
            objects: (optional) a DetectedObjectContainer of objects for the
                frame
            keypoints: (optional) a KeypointsContainer of keypoints for the
                frame
            polylines: (optional) a PolylineContainer of polylines for the
                frame
            events: (optional) a DetectedEventContainer of events for the frame
            tags: (optional) a list of tag strings
        """
        self.frame_number = frame_number
        self.mask = mask
        self.mask_index = mask_index
        self.attrs = attrs or etad.AttributeContainer()
        self.objects = objects or etao.DetectedObjectContainer()
        self.keypoints = keypoints or etak.KeypointsContainer()
        self.polylines = polylines or etap.PolylineContainer()
        self.events = events or etae.DetectedEventContainer()
        self.tags = tags = []

    @property
    def is_empty(self):
        """Whether the frame has no labels of any kind."""
        return not (
            self.has_mask
            or self.has_attributes
            or self.has_objects
            or self.has_keypoints
            or self.has_polylines
            or self.has_events
        )

    @property
    def has_frame_number(self):
        """Whether the frame has a frame number."""
        return self.frame_number is not None

    @property
    def has_mask(self):
        """Whether this frame has a segmentation mask."""
        return self.mask is not None

    @property
    def has_mask_index(self):
        """Whether this frame has a segmentation mask index."""
        return self.mask_index is not None

    @property
    def has_attributes(self):
        """Whether the frame has at least one attribute."""
        return bool(self.attrs)

    @property
    def has_objects(self):
        """Whether the frame has at least one object."""
        return bool(self.objects)

    @property
    def has_object_attributes(self):
        """Whether the frame has at least one object with attributes."""
        for obj in self.objects:
            if obj.has_attributes:
                return True

        return False

    @property
    def has_keypoints(self):
        """Whether the frame has at least one keypoint."""
        return bool(self.keypoints)

    @property
    def has_keypoints_attributes(self):
        """Whether the frame has at least one keypoints with attributes."""
        for k in self.keypoints:
            if k.has_attributes:
                return True

        return False

    @property
    def has_polylines(self):
        """Whether the frame has at least one polyline."""
        return bool(self.polylines)

    @property
    def has_polylines_attributes(self):
        """Whether the frame has at least one polyline with attributes."""
        for polyline in self.polylines:
            if polyline.has_attributes:
                return True

        return False

    @property
    def has_events(self):
        """Whether the frame has at least one event."""
        return bool(self.events)

    @property
    def has_event_attributes(self):
        """Whether the frame has at least one event with attributes."""
        for event in self.events:
            if event.has_attributes:
                return True

        return False

    @property
    def has_tags(self):
        """Whether the frame has tags."""
        return bool(self.tags)

    def iter_attributes(self):
        """Returns an iterator over the attributes of the frame.

        Returns:
            an iterator over `Attribute`s
        """
        return iter(self.attrs)

    def iter_objects(self):
        """Returns an iterator over the objects in the frame.

        Returns:
            an iterator over `DetectedObject`s
        """
        return iter(self.objects)

    def iter_keypoints(self):
        """Returns an iterator over the keypoints in the frame.

        Returns:
            an iterator over `Keypoints`s
        """
        return iter(self.keypoints)

    def iter_polylines(self):
        """Returns an iterator over the polylines in the frame.

        Returns:
            an iterator over `Polyline`s
        """
        return iter(self.polylines)

    def iter_events(self):
        """Returns an iterator over the events in the frame.

        Returns:
            an iterator over `DetectedEvent`s
        """
        return iter(self.events)

    def get_object_indexes(self):
        """Returns the set of `index`es of all objects in the frame.

        `None` indexes are omitted.

        Returns:
            a set of indexes
        """
        obj_indexes = self.objects.get_indexes()
        obj_indexes.update(self.events.get_object_indexes())
        return obj_indexes

    def offset_object_indexes(self, offset):
        """Adds the given offset to all objects with `index`es in the frame.

        Args:
            offset: the integer offset
        """
        self.objects.offset_indexes(offset)
        self.events.offset_object_indexes(offset)

    def clear_object_indexes(self):
        """Clears the `index` of all objects in the frame."""
        self.objects.clear_indexes()
        self.events.clear_object_indexes()

    def get_keypoint_indexes(self):
        """Returns the set of `index`es of all keypoints in the frame.

        `None` indexes are omitted.

        Returns:
            a set of indexes
        """
        return self.keypoints.get_indexes()

    def offset_keypoint_indexes(self, offset):
        """Adds the given offset to all keypoints with `index`es in the frame.

        Args:
            offset: the integer offset
        """
        self.keypoints.offset_indexes(offset)

    def clear_keypoint_indexes(self):
        """Clears the `index` of all keypoints in the frame."""
        self.keypoints.clear_indexes()

    def get_polyline_indexes(self):
        """Returns the set of `index`es of all polylines in the frame.

        `None` indexes are omitted.

        Returns:
            a set of indexes
        """
        return self.polylines.get_indexes()

    def offset_polyline_indexes(self, offset):
        """Adds the given offset to all polylines with `index`es in the frame.

        Args:
            offset: the integer offset
        """
        self.polylines.offset_indexes(offset)

    def clear_polyline_indexes(self):
        """Clears the `index` of all polylines in the frame."""
        self.polylines.clear_indexes()

    def get_event_indexes(self):
        """Returns the set of `index`es of all events in the frame.

        `None` indexes are omitted.

        Returns:
            a set of indexes
        """
        return self.events.get_indexes()

    def offset_event_indexes(self, offset):
        """Adds the given offset to all events with `index`es in the frame.

        Args:
            offset: the integer offset
        """
        self.events.offset_indexes(offset)

    def clear_event_indexes(self):
        """Clears the `index` of all events in the frame."""
        self.events.clear_indexes()

    def add_attribute(self, attr):
        """Adds the attribute to the frame.

        Args:
            attr: an Attribute
        """
        self.attrs.add(attr)

    def add_attributes(self, attrs):
        """Adds the attributes to the frame.

        Args:
            attrs: an AttributeContainer
        """
        self.attrs.add_container(attrs)

    def add_object(self, obj):
        """Adds the object to the frame.

        Args:
            obj: a DetectedObject
        """
        self.objects.add(obj)

    def add_objects(self, objs):
        """Adds the objects to the frame.

        Args:
            objs: a DetectedObjectContainer
        """
        self.objects.add_container(objs)

    def add_keypoints(self, keypoints):
        """Adds the keypoints to the frame.

        Args:
            keypoints: a Keypoints or KeypointsContainer
        """
        if isinstance(keypoints, etak.KeypointsContainer):
            self.keypoints.add_container(keypoints)
        else:
            self.keypoints.add(keypoints)

    def add_polyline(self, polyline):
        """Adds the polyline to the frame.

        Args:
            polyline: a Polyline
        """
        self.polylines.add(polyline)

    def add_polylines(self, polylines):
        """Adds the polylines to the frame.

        Args:
            polylines: a PolylineContainer
        """
        self.polylines.add_container(polylines)

    def add_event(self, event):
        """Adds the event to the frame.

        Args:
            event: a DetectedEvent
        """
        self.events.add(event)

    def add_events(self, events):
        """Adds the events to the frame.

        Args:
            events: a DetectedEventContainer
        """
        self.events.add_container(events)

    def pop_attributes(self):
        """Pops the frame-level attributes from the frame.

        Returns:
            an AttributeContainer
        """
        attrs = self.attrs
        self.clear_attributes()
        return attrs

    def pop_objects(self):
        """Pops the objects from the frame.

        Returns:
            a DetectedObjectContainer
        """
        objects = self.objects
        self.clear_objects()
        return objects

    def pop_keypoints(self):
        """Pops the keypoints from the frame.

        Returns:
            a KeypointsContainer
        """
        keypoints = self.keypoints
        self.clear_keypoints()
        return keypoints

    def pop_polylines(self):
        """Pops the polylines from the frame.

        Returns:
            a PolylineContainer
        """
        polylines = self.polylines
        self.clear_polylines()
        return polylines

    def pop_events(self):
        """Pops the events from the frame.

        Returns:
            a DetectedEventContainer
        """
        events = self.events
        self.clear_events()
        return events

    def clear_attributes(self):
        """Removes all frame-level attributes from the frame."""
        self.attrs = etad.AttributeContainer()

    def clear_objects(self):
        """Removes all objects from the frame."""
        self.objects = etao.DetectedObjectContainer()

    def clear_keypoints(self):
        """Removes all keypoints from the frame."""
        self.keypoints = etak.KeypointsContainer()

    def clear_polylines(self):
        """Removes all polylines from the frame."""
        self.polylines = etap.PolylineContainer()

    def clear_events(self):
        """Removes all events from the frame."""
        self.events = etae.DetectedEventContainer()

    def clear(self):
        """Removes all labels from the frame."""
        self.mask = None
        self.mask_index = None
        self.clear_attributes()
        self.clear_objects()
        self.clear_keypoints()
        self.clear_polylines()
        self.clear_events()

    def merge_labels(self, frame_labels, reindex=False):
        """Merges the given FrameLabels into this labels.

        Args:
            frame_labels: a FrameLabels
            reindex: whether to offset the `index` fields of objects,
                polylines, keypoints, and events in `frame_labels` before
                merging so that all indices are unique. The default is False
        """
        if reindex:
            self._reindex_objects(frame_labels)
            self._reindex_keypoints(frame_labels)
            self._reindex_polylines(frame_labels)
            self._reindex_events(frame_labels)

        if frame_labels.has_mask:
            self.mask = frame_labels.mask
        if frame_labels.has_mask_index:
            self.mask_index = frame_labels.mask_index

        self.add_attributes(frame_labels.attrs)
        self.add_objects(frame_labels.objects)
        self.add_keypoints(frame_labels.keypoints)
        self.add_polylines(frame_labels.polylines)
        self.add_events(frame_labels.events)

    def filter_by_schema(self, schema):
        """Filters the frame labels by the given schema.

        Args:
            schema: a FrameLabelsSchema
        """
        self.attrs.filter_by_schema(
            schema.frames, constant_schema=schema.attrs
        )
        self.objects.filter_by_schema(schema.objects)
        self.keypoints.filter_by_schema(schema.keypoints)
        self.polylines.filter_by_schema(schema.polylines)
        self.events.filter_by_schema(schema.events)

    def remove_objects_without_attrs(self, labels=None):
        """Removes objects from the frame that do not have attributes.

        Args:
            labels: an optional list of object `label` strings to which to
                restrict attention when filtering. By default, all objects are
                processed
        """
        self.objects.remove_objects_without_attrs(labels=labels)
        self.events.remove_objects_without_attrs(labels=labels)

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        """
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
        if self.keypoints:
            _attrs.append("keypoints")
        if self.polylines:
            _attrs.append("polylines")
        if self.events:
            _attrs.append("events")
        if self.tags:
            _attrs.append("tags")

        return _attrs

    @classmethod
    def from_dict(cls, d, **kwargs):
        """Constructs a FrameLabels from a JSON dictionary.

        Args:
            d: a JSON dictionary
            **kwargs: keyword arguments that have already been parsed by a
                subclass

        Returns:
            a FrameLabels
        """
        frame_number = d.get("frame_number", None)

        mask = d.get("mask", None)
        if mask is not None:
            mask = etas.deserialize_numpy_array(mask)

        mask_index = d.get("mask_index", None)
        if mask_index is not None:
            mask_index = etad.MaskIndex.from_dict(mask_index)

        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainer.from_dict(attrs)

        objects = d.get("objects", None)
        if objects is not None:
            objects = etao.DetectedObjectContainer.from_dict(objects)

        keypoints = d.get("keypoints", None)
        if keypoints is not None:
            keypoints = etak.KeypointsContainer.from_dict(keypoints)

        polylines = d.get("polylines", None)
        if polylines is not None:
            polylines = etap.PolylineContainer.from_dict(polylines)

        events = d.get("events", None)
        if events is not None:
            events = etae.DetectedEventContainer.from_dict(events)

        tags = d.get("tags", None)

        return cls(
            frame_number=frame_number,
            mask=mask,
            mask_index=mask_index,
            attrs=attrs,
            objects=objects,
            keypoints=keypoints,
            polylines=polylines,
            events=events,
            tags=tags,
            **kwargs
        )

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

    def _reindex_keypoints(self, frame_labels):
        self_indices = self._get_keypoints_indices(self)
        if not self_indices:
            return

        new_indices = self._get_keypoints_indices(frame_labels)
        if not new_indices:
            return

        offset = max(self_indices) + 1 - min(new_indices)
        self._offset_keypoints_indices(frame_labels, offset)

    @staticmethod
    def _get_keypoints_indices(frame_labels):
        keypoints_indices = set()

        for keypoints in frame_labels.keypoints:
            if keypoints.index is not None:
                keypoints_indices.add(keypoints.index)

        return keypoints_indices

    @staticmethod
    def _offset_keypoints_indices(frame_labels, offset):
        for keypoints in frame_labels.keypoints:
            if keypoints.index is not None:
                keypoints.index += offset

    def _reindex_polylines(self, frame_labels):
        self_indices = self._get_polyline_indices(self)
        if not self_indices:
            return

        new_indices = self._get_polyline_indices(frame_labels)
        if not new_indices:
            return

        offset = max(self_indices) + 1 - min(new_indices)
        self._offset_polyline_indices(frame_labels, offset)

    @staticmethod
    def _get_polyline_indices(frame_labels):
        polyline_indices = set()

        for polyline in frame_labels.polylines:
            if polyline.index is not None:
                polyline_indices.add(polyline.index)

        return polyline_indices

    @staticmethod
    def _offset_polyline_indices(frame_labels, offset):
        for polyline in frame_labels.polylines:
            if polyline.index is not None:
                polyline.index += offset

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
    """Schema describing the content of one or more FrameLabels.

    Attributes:
        attrs: an AttributeContainerSchema describing the constant attributes
            of the frame(s)
        frames: an AttributeContainerSchema describing the frame-level
            attributes of the frame(s)
        objects: an ObjectContainerSchema describing the objects in the
            frame(s)
        keypoints: a KeypointsContainerSchema describing the polylines in the
            frame(s)
        polylines: a PolylineContainerSchema describing the polylines in the
            frame(s)
        events: an EventContainerSchema describing the events in the frame(s)
    """

    def __init__(
        self,
        attrs=None,
        frames=None,
        objects=None,
        keypoints=None,
        polylines=None,
        events=None,
    ):
        """Creates a FrameLabelsSchema instance.

        Args:
            attrs: (optional) an AttributeContainerSchema describing the
                constant attributes of the frame(s)
            frames: (optional) an AttributeContainerSchema describing the
                frame-level attributes of the frame(s)
            objects: (optional) an ObjectContainerSchema describing the objects
                in the frame(s)
            keypoints: (optional) a KeypointsContainerSchema describing the
                keypoints in the frame(s)
            polylines: (optional) a PolylineContainerSchema describing the
                polylines in the frame(s)
            events: (optional) an EventContainerSchema describing the events
                in the frame(s)
        """
        self.attrs = attrs or etad.AttributeContainerSchema()
        self.frames = frames or etad.AttributeContainerSchema()
        self.objects = objects or etao.ObjectContainerSchema()
        self.keypoints = keypoints or etak.KeypointsContainerSchema()
        self.polylines = polylines or etap.PolylineContainerSchema()
        self.events = events or etae.EventContainerSchema()

    @property
    def is_empty(self):
        """Whether the schema has no labels of any kind."""
        return not (
            self.has_constant_attributes
            or self.has_frame_attributes
            or self.has_objects
            or self.has_keypoints
            or self.has_polylines
            or self.has_events
        )

    @property
    def has_constant_attributes(self):
        """Whether the schema has at least one constant AttributeSchema."""
        return bool(self.attrs)

    @property
    def has_frame_attributes(self):
        """Whether the schema has at least one frame-level AttributeSchema."""
        return bool(self.frames)

    @property
    def has_objects(self):
        """Whether the schema has at least one ObjectSchema."""
        return bool(self.objects)

    @property
    def has_keypoints(self):
        """Whether the schema has at least one KeypointsSchema."""
        return bool(self.keypoints)

    @property
    def has_polylines(self):
        """Whether the schema has at least one PolylineSchema."""
        return bool(self.polylines)

    @property
    def has_events(self):
        """Whether the schema has at least one EventSchema."""
        return bool(self.events)

    def has_constant_attribute(self, attr_name):
        """Whether the schema has a constant frame attribute with the given
        name.

        Args:
            attr_name: the constant frame attribute name

        Returns:
            True/False
        """
        return self.attrs.has_attribute(attr_name)

    def get_constant_attribute_schema(self, attr_name):
        """Gets the AttributeSchema for the constant frame attribute with the
        given name.

        Args:
            attr_name: the constant frame attribute name

        Returns:
            the AttributeSchema
        """
        return self.attrs.get_attribute_schema(attr_name)

    def get_constant_attribute_class(self, attr_name):
        """Gets the Attribute class for the constant frame attribute with the
        given name.

        Args:
            attr_name: the constant frame attribute name

        Returns:
            the Attribute class
        """
        return self.attrs.get_attribute_class(attr_name)

    def has_frame_attribute(self, attr_name):
        """Whether the schema has a frame-level attribute with the given name.

        Args:
            attr_name: the frame-level attribute name

        Returns:
            True/False
        """
        return self.frames.has_attribute(attr_name)

    def get_frame_attribute_schema(self, attr_name):
        """Gets the AttributeSchema for the frame-level attribute with the
        given name.

        Args:
            attr_name: the frame-level attribute name

        Returns:
            the AttributeSchema
        """
        return self.frames.get_attribute_schema(attr_name)

    def get_frame_attribute_class(self, attr_name):
        """Gets the Attribute class for the frame-level attribute with the
        given name.

        Args:
            attr_name: the frame-level attribute name

        Returns:
            the Attribute class
        """
        return self.frames.get_attribute_class(attr_name)

    def has_object_label(self, label):
        """Whether the schema has an object with the given label.

        Args:
            label: the object label

        Returns:
            True/False
        """
        return self.objects.has_object_label(label)

    def get_object_schema(self, label):
        """Gets the ObjectSchema for the object with the given label.

        Args:
            label: the object label

        Returns:
            the ObjectSchema
        """
        return self.objects.get_object_schema(label)

    def has_object_attribute(self, label, attr_name):
        """Whether the schema has an object with the given label with an
        object-level attribute with the given name.

        Args:
            label: the object label
            attr_name: an object-level attribute name

        Returns:
            True/False
        """
        return self.objects.has_object_attribute(label, attr_name)

    def get_object_attribute_schema(self, label, attr_name):
        """Gets the AttributeSchema for the object-level attribute of the given
        name for the object with the given label.

        Args:
            label: the object label
            attr_name: an object-level attribute name

        Returns:
            the AttributeSchema
        """
        return self.objects.get_object_attribute_schema(label, attr_name)

    def get_object_attribute_class(self, label, attr_name):
        """Gets the Attribute class for the object-level attribute of the given
        name for the object with the given label.

        Args:
            label: the object label
            attr_name: an object-level attribute name

        Returns:
            the Attribute class
        """
        return self.objects.get_object_attribute_class(label, attr_name)

    def has_object_frame_attribute(self, label, attr_name):
        """Whether the schema has an object with the given label with a
        frame-level attribute with the given name.

        Args:
            label: the object label
            attr_name: a frame-level object attribute name

        Returns:
            True/False
        """
        return self.objects.has_frame_attribute(label, attr_name)

    def get_object_frame_attribute_schema(self, label, attr_name):
        """Gets the AttributeSchema for the frame-level attribute of the given
        name for the object with the given label.

        Args:
            label: the object label
            attr_name: a frame-level object attribute name

        Returns:
            the AttributeSchema
        """
        return self.objects.get_frame_attribute_schema(label, attr_name)

    def get_object_frame_attribute_class(self, label, attr_name):
        """Gets the Attribute class for the frame-level attribute of the given
        name for the object with the given label.

        Args:
            label: the object label
            attr_name: a frame-level object attribute name

        Returns:
            the Attribute class
        """
        return self.objects.get_frame_attribute_class(label, attr_name)

    def has_keypoints_label(self, label):
        """Whether the schema has a keypoints with the given label.

        Args:
            label: the keypoints label

        Returns:
            True/False
        """
        return self.keypoints.has_keypoints_label(label)

    def get_keypoints_schema(self, label):
        """Gets the KeypointsSchema for the keypoints with the given label.

        Args:
            label: the keypoints label

        Returns:
            the KeypointsSchema
        """
        return self.keypoints.get_keypoints_schema(label)

    def has_keypoints_attribute(self, label, attr_name):
        """Whether the schema has a keypoints with the given label with an
        attribute with the given name.

        Args:
            label: the keypoints label
            attr_name: a keypoints attribute name

        Returns:
            True/False
        """
        return self.keypoints.has_keypoints_attribute(label, attr_name)

    def get_keypoints_attribute_schema(self, label, attr_name):
        """Gets the AttributeSchema for the attribute of the given name for the
        keypoints with the given label.

        Args:
            label: the keypoints label
            attr_name: a keypoints attribute name

        Returns:
            the AttributeSchema
        """
        return self.keypoints.get_keypoints_attribute_schema(label, attr_name)

    def get_keypoints_attribute_class(self, label, attr_name):
        """Gets the Attribute class for the attribute of the given name for the
        keypoints with the given label.

        Args:
            label: the keypoints label
            attr_name: a keypoints attribute name

        Returns:
            the Attribute class
        """
        return self.keypoints.get_keypoints_attribute_class(label, attr_name)

    def has_polyline_label(self, label):
        """Whether the schema has a polyline with the given label.

        Args:
            label: the polyline label

        Returns:
            True/False
        """
        return self.polylines.has_polyline_label(label)

    def get_polyline_schema(self, label):
        """Gets the PolylineSchema for the polyline with the given label.

        Args:
            label: the polyline label

        Returns:
            the PolylineSchema
        """
        return self.polylines.get_polyline_schema(label)

    def has_polyline_attribute(self, label, attr_name):
        """Whether the schema has a polyline with the given label with an
        attribute with the given name.

        Args:
            label: the polyline label
            attr_name: a polyline attribute name

        Returns:
            True/False
        """
        return self.polylines.has_polyline_attribute(label, attr_name)

    def get_polyline_attribute_schema(self, label, attr_name):
        """Gets the AttributeSchema for the attribute of the given name for the
        polyline with the given label.

        Args:
            label: the polyline label
            attr_name: a polyline attribute name

        Returns:
            the AttributeSchema
        """
        return self.polylines.get_polyline_attribute_schema(label, attr_name)

    def get_polyline_attribute_class(self, label, attr_name):
        """Gets the Attribute class for the attribute of the given name for the
        polyline with the given label.

        Args:
            label: the polyline label
            attr_name: a polyline attribute name

        Returns:
            the Attribute class
        """
        return self.polylines.get_polyline_attribute_class(label, attr_name)

    def has_event_label(self, label):
        """Whether the schema has an event with the given label.

        Args:
            label: the event label

        Returns:
            True/False
        """
        return self.events.has_event_label(label)

    def get_event_schema(self, label):
        """Gets the EventSchema for the event with the given label.

        Args:
            label: the event label

        Returns:
            the EventSchema
        """
        return self.events.get_event_schema(label)

    def has_event_attribute(self, label, attr_name):
        """Whether the schema has an event with the given label with an
        event-level attribute with the given name.

        Args:
            label: an event label
            attr_name: an event-level attribute name

        Returns:
            True/False
        """
        return self.events.has_event_attribute(label, attr_name)

    def get_event_attribute_schema(self, label, attr_name):
        """Gets the AttributeSchema for the event-level attribute of the given
        name for the event with the given label.

        Args:
            label: the event label
            attr_name: an event-level attribute name

        Returns:
            the AttributeSchema
        """
        return self.events.get_event_attribute_schema(label, attr_name)

    def get_event_attribute_class(self, label, attr_name):
        """Gets the Attribute class for the event-level attribute of the given
        name for the event with the given label.

        Args:
            label: the event label
            attr_name: an event-level attribute name

        Returns:
            the Attribute class
        """
        return self.events.get_event_attribute_class(label, attr_name)

    def has_event_frame_attribute(self, label, attr_name):
        """Whether the schema has an event with the given label with a
        frame-level attribute with the given name.

        Args:
            label: an event label
            attr_name: a frame-level event attribute name

        Returns:
            True/False
        """
        return self.events.has_frame_attribute(label, attr_name)

    def get_event_frame_attribute_schema(self, label, attr_name):
        """Gets the AttributeSchema for the frame-level attribute of the given
        name for the event with the given label.

        Args:
            label: the event label
            attr_name: a frame-level event attribute name

        Returns:
            the AttributeSchema
        """
        return self.events.get_frame_attribute_schema(label, attr_name)

    def get_event_frame_attribute_class(self, label, attr_name):
        """Gets the Attribute class for the frame-level attribute of the given
        name for the event with the given label.

        Args:
            label: the event label
            attr_name: a frame-level event attribute name

        Returns:
            the Attribute class
        """
        return self.events.get_frame_attribute_class(label, attr_name)

    def has_event_object_label(self, event_label, obj_label):
        """Whether the schema has an event with the given label with an object
        with the given label.

        Args:
            event_label: the event label
            obj_label: the object label

        Returns:
            True/False
        """
        return self.events.has_object_label(event_label, obj_label)

    def get_event_object_schema(self, event_label, obj_label):
        """Gets the ObjectSchema for the object with the given label from the
        event with the given label.

        Args:
            event_label: the event label
            obj_label: the object label

        Returns:
            the ObjectSchema
        """
        return self.events.get_object_schema(event_label, obj_label)

    def has_event_object_attribute(self, event_label, obj_label, attr_name):
        """Whether the schema has an event of the given label with an object
        of the given label with an object-level attribute of the given name.

        Args:
            event_label: the event label
            obj_label: the object label
            attr_name: the object-level attribute name

        Returns:
            True/False
        """
        return self.events.has_object_attribute(
            event_label, obj_label, attr_name
        )

    def get_event_object_attribute_schema(
        self, event_label, obj_label, attr_name
    ):
        """Gets the AttributeSchema for the object-level attribute of the
        given name for the object with the given label from the event with the
        given label.

        Args:
            event_label: the event label
            obj_label: the object label
            attr_name: the object-level attribute name

        Returns:
            the AttributeSchema
        """
        return self.events.get_object_attribute_schema(
            event_label, obj_label, attr_name
        )

    def get_event_object_attribute_class(
        self, event_label, obj_label, attr_name
    ):
        """Gets the Attribute class for the object-level attribute of the
        given name for the object with the given label from the event with the
        given label.

        Args:
            event_label: the event label
            obj_label: the object label
            attr_name: the object-level attribute name

        Returns:
            the Attribute
        """
        return self.events.get_object_attribute_class(
            event_label, obj_label, attr_name
        )

    def has_event_object_frame_attribute(
        self, event_label, obj_label, attr_name
    ):
        """Whether the schema has an event of the given label with an object
        of the given label with a frame-level attribute of the given name.

        Args:
            event_label: the event label
            obj_label: the object label
            attr_name: the frame-level object attribute name

        Returns:
            True/False
        """
        return self.events.has_object_frame_attribute(
            event_label, obj_label, attr_name
        )

    def get_event_object_frame_attribute_schema(
        self, event_label, obj_label, attr_name
    ):
        """Gets the AttributeSchema for the frame-level attribute of the
        given name for the object with the given label from the event with the
        given label.

        Args:
            event_label: the event label
            obj_label: the object label
            attr_name: the frame-level object attribute name

        Returns:
            the AttributeSchema
        """
        return self.events.get_object_frame_attribute_schema(
            event_label, obj_label, attr_name
        )

    def get_event_object_frame_attribute_class(
        self, event_label, obj_label, attr_name
    ):
        """Gets the Attribute class for the frame-level attribute of the
        given name for the object with the given label from the event with the
        given label.

        Args:
            event_label: the event label
            obj_label: the object label
            attr_name: the frame-level object attribute name

        Returns:
            the Attribute
        """
        return self.events.get_object_frame_attribute_class(
            event_label, obj_label, attr_name
        )

    def add_constant_attribute(self, attr):
        """Adds the given constant attribute to the schema.

        Args:
            attr: an Attribute
        """
        self.attrs.add_attribute(attr)

    def add_constant_attributes(self, attrs):
        """Adds the given constant attributes to the schema.

        Args:
            attrs: an AttributeContainer
        """
        self.attrs.add_attributes(attrs)

    def add_frame_attribute(self, attr):
        """Adds the given frame-level attribute to the schema.

        Args:
            attr: an Attribute
        """
        self.frames.add_attribute(attr)

    def add_frame_attributes(self, attrs):
        """Adds the given frame-level attributes to the schema.

        Args:
            attrs: an AttributeContainer
        """
        self.frames.add_attributes(attrs)

    def add_object_label(self, label):
        """Adds the given object label to the schema.

        Args:
            label: an object label
        """
        self.objects.add_object_label(label)

    def add_object_attribute(self, label, attr):
        """Adds the object-level attribute for the object with the given label
        to the schema.

        Args:
            label: an object label
            attr: an Attribute
        """
        self.objects.add_object_attribute(label, attr)

    def add_object_attributes(self, label, attrs):
        """Adds the object-level attributes for the object with the given label
        to the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer
        """
        self.objects.add_object_attributes(label, attrs)

    def add_object_frame_attribute(self, label, attr):
        """Adds the frame-level attribute for the object with the given label
        to the schema.

        Args:
            label: an object label
            attr: an Attribute
        """
        self.objects.add_frame_attribute(label, attr)

    def add_object_frame_attributes(self, label, attrs):
        """Adds the frame-level attributes for the object with the given label
        to the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer
        """
        self.objects.add_frame_attributes(label, attrs)

    def add_object(self, obj):
        """Adds the object to the schema.

        Args:
            obj: a VideoObject or DetectedObject
        """
        self.objects.add_object(obj)

    def add_objects(self, objects):
        """Adds the objects to the schema.

        Args:
            objects: a VideoObjectContainer or DetectedObjectContainer
        """
        self.objects.add_objects(objects)

    def add_keypoints_label(self, label):
        """Adds the given keypoints label to the schema.

        Args:
            label: a keypoints label
        """
        self.keypoints.add_keypoints_label(label)

    def add_keypoints_attribute(self, label, attr):
        """Adds the attribute for the keypoints with the given label to the
        schema.

        Args:
            label: a keypoints label
            attr: an Attribute
        """
        self.keypoints.add_keypoints_attribute(label, attr)

    def add_keypoints_attributes(self, label, attrs):
        """Adds the attributes for the keypoints with the given label to the
        schema.

        Args:
            label: a keypoints label
            attrs: an AttributeContainer
        """
        self.keypoints.add_keypoints_attributes(label, attrs)

    def add_keypoints(self, keypoints):
        """Adds the keypoints to the schema.

        Args:
            keypoints: a Keypoints or KeypointsContainer
        """
        self.keypoints.add_keypoints(keypoints)

    def add_polyline_label(self, label):
        """Adds the given polyline label to the schema.

        Args:
            label: a polyline label
        """
        self.polylines.add_polyline_label(label)

    def add_polyline_attribute(self, label, attr):
        """Adds the attribute for the polyline with the given label to the
        schema.

        Args:
            label: a polyline label
            attr: an Attribute
        """
        self.polylines.add_polyline_attribute(label, attr)

    def add_polyline_attributes(self, label, attrs):
        """Adds the attributes for the polyline with the given label to the
        schema.

        Args:
            label: a polyline label
            attrs: an AttributeContainer
        """
        self.polylines.add_polyline_attributes(label, attrs)

    def add_polyline(self, polyline):
        """Adds the polyline to the schema.

        Args:
            polyline: a Polyline
        """
        self.polylines.add_polyline(polyline)

    def add_polylines(self, polylines):
        """Adds the polylines to the schema.

        Args:
            polylines: a PolylineContainer
        """
        self.polylines.add_polylines(polylines)

    def add_event_label(self, label):
        """Adds the given event label to the schema.

        Args:
            label: an event label
        """
        self.events.add_event_label(label)

    def add_event_attribute(self, label, attr):
        """Adds the event-level attribute for the event with the given label to
        the schema.

        Args:
            label: an event label
            attr: an Attribute
        """
        self.events.add_event_attribute(label, attr)

    def add_event_attributes(self, label, attrs):
        """Adds the event-level attributes for the event with the given label
        to the schema.

        Args:
            label: an event label
            attrs: an AttributeContainer
        """
        self.events.add_event_attributes(label, attrs)

    def add_event_frame_attribute(self, label, attr):
        """Adds the frame-level attribute for the event with the given label to
        the schema.

        Args:
            label: an event label
            attr: an Attribute
        """
        self.events.add_frame_attribute(label, attr)

    def add_event_frame_attributes(self, label, attrs):
        """Adds the frame-level attributes for the event with the given label
        to the schema.

        Args:
            label: an event label
            attrs: an AttributeContainer
        """
        self.events.add_frame_attributes(label, attrs)

    def add_event_object_label(self, event_label, obj_label):
        """Adds the object label for the event with the given label to the
        schema.

        Args:
            event_label: an event label
            obj_label: an object label
        """
        self.events.add_object_label(event_label, obj_label)

    def add_event_object_attribute(self, event_label, obj_label, attr):
        """Adds the object-level attribute for the object with the given label
        to the event with the given label to the schema.

        Args:
            event_label: an event label
            obj_label: an object label
            attr: an Attribute
        """
        self.events.add_object_attribute(event_label, obj_label, attr)

    def add_event_object_attributes(self, event_label, obj_label, attrs):
        """Adds the AttributeContainer of object-level attributes for the
        object with the given label to the event with the given label to the
        schema.

        Args:
            event_label: an event label
            obj_label: an object label
            attrs: an AttributeContainer
        """
        self.events.add_object_attributes(event_label, obj_label, attrs)

    def add_event_object_frame_attribute(self, event_label, obj_label, attr):
        """Adds the frame-level attribute for the object with the given label
        to the event with the given label to the schema.

        Args:
            event_label: an event label
            obj_label: an object label
            attr: an Attribute
        """
        self.events.add_object_frame_attribute(event_label, obj_label, attr)

    def add_event_object_frame_attributes(self, event_label, obj_label, attrs):
        """Adds the AttributeContainer of frame-level attributes for the
        object with the given label to the event with the given label to the
        schema.

        Args:
            event_label: an event label
            obj_label: an object label
            attrs: an AttributeContainer
        """
        self.events.add_object_frame_attributes(event_label, obj_label, attrs)

    def add_event_object(self, event_label, obj):
        """Adds the object to the event with the given label to the schema.

        Args:
            event_label: an event label
            obj: a VideoObject or DetectedObject
        """
        self.events.add_object(event_label, obj)

    def add_event_objects(self, event_label, objects):
        """Adds the objects to the event with the given label to the schema.

        Args:
            event_label: an event label
            objects: a VideoObjectContainer or DetectedObjectContainer
        """
        self.events.add_objects(event_label, objects)

    def add_event(self, event):
        """Adds the event to the schema.

        Args:
            event: a VideoEvent or DetectedEvent
        """
        self.events.add_event(event)

    def add_events(self, events):
        """Adds the events to the schema.

        Args:
            events: a VideoEventContainer or DetectedEventContainer
        """
        self.events.add_events(events)

    def add_frame_labels(self, frame_labels):
        """Adds the frame labels to the schema.

        Args:
            frame_labels: a FrameLabels
        """
        for attr in frame_labels.attrs:
            if attr.constant:
                self.add_constant_attribute(attr)
            else:
                self.add_frame_attribute(attr)

        self.add_objects(frame_labels.objects)
        self.add_keypoints(frame_labels.keypoints)
        self.add_polylines(frame_labels.polylines)
        self.add_events(frame_labels.events)

    def add_image_labels(self, image_labels):
        """Adds the image labels to the schema.

        Args:
            image_labels: an ImageLabels
        """
        self.add_frame_labels(image_labels)

    def add_video_labels(self, video_labels):
        """Adds the video labels to the schema.

        Args:
            video_labels: a VideoLabels
        """
        self.add_constant_attributes(video_labels.attrs)
        self.add_objects(video_labels.objects)
        self.add_events(video_labels.events)
        for frame_labels in video_labels.iter_frames():
            self.add_frame_labels(frame_labels)

    def is_valid_constant_attribute(self, attr):
        """Whether the constant attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Returns:
            True/False
        """
        return self.attrs.is_valid_attribute(attr)

    def is_valid_constant_attributes(self, attrs):
        """Whether the constant attributes are compliant with the schema.

        Args:
            attrs: an AttributeContainer

        Returns:
            True/False
        """
        return self.attrs.is_valid_attributes(attrs)

    def is_valid_frame_attribute(self, attr):
        """Whether the frame-level attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Returns:
            True/False
        """
        return self.frames.is_valid_attribute(attr)

    def is_valid_frame_attributes(self, attrs):
        """Whether the frame-level attributes are compliant with the schema.

        Args:
            attrs: an AttributeContainer

        Returns:
            True/False
        """
        return self.frames.is_valid_attributes(attrs)

    def is_valid_object_label(self, label):
        """Whether the object label is compliant with the schema.

        Args:
            label: an object label

        Returns:
            True/False
        """
        return self.objects.is_valid_object_label(label)

    def is_valid_object_attribute(self, label, attr):
        """Whether the object-level attribute for the object with the given
        label is compliant with the schema.

        Args:
            label: an object label
            attr: an Attribute

        Returns:
            True/False
        """
        return self.objects.is_valid_object_attribute(label, attr)

    def is_valid_object_attributes(self, label, attrs):
        """Whether the object-level attributes for the object with the given
        label are compliant with the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer

        Returns:
            True/False
        """
        return self.objects.is_valid_object_attributes(label, attrs)

    def is_valid_object_frame_attribute(self, label, attr):
        """Whether the frame-level attribute for the object with the given
        label is compliant with the schema.

        Args:
            label: an object label
            attr: an Attribute

        Returns:
            True/False
        """
        return self.objects.is_valid_frame_attribute(label, attr)

    def is_valid_object_frame_attributes(self, label, attrs):
        """Whether the frame-level attributes for the object with the given
        label are compliant with the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer

        Returns:
            True/False
        """
        return self.objects.is_valid_frame_attributes(label, attrs)

    def is_valid_object(self, obj):
        """Whether the given object is compliant with the schema.

        Args:
            obj: a VideoObject or DetectedObject

        Returns:
            True/False
        """
        return self.objects.is_valid_object(obj)

    def is_valid_objects(self, objects):
        """Whether the given objects are compliant with the schema.

        Args:
            objects: a VideoObjectContainer or DetectedObjectContainer

        Returns:
            True/False
        """
        return self.objects.is_valid(objects)

    def is_valid_keypoints_label(self, label):
        """Whether the keypoints label is compliant with the schema.

        Args:
            label: a keypoints label

        Returns:
            True/False
        """
        return self.keypoints.is_valid_keypoints_label(label)

    def is_valid_keypoints_attribute(self, label, attr):
        """Whether the attribute for the keypoints with the given label is
        compliant with the schema.

        Args:
            label: a keypoints label
            attr: an Attribute

        Returns:
            True/False
        """
        return self.keypoints.is_valid_keypoints_attribute(label, attr)

    def is_valid_keypoints_attributes(self, label, attrs):
        """Whether the attributes for the keypoints with the given label are
        compliant with the schema.

        Args:
            label: a keypoints label
            attrs: an AttributeContainer

        Returns:
            True/False
        """
        return self.keypoints.is_valid_keypoints_attributes(label, attrs)

    def is_valid_keypoints(self, keypoints):
        """Whether the given keypoints are compliant with the schema.

        Args:
            keypoints: a Keypoints or KeypointsContainer

        Returns:
            True/False
        """
        if isinstance(keypoints, etak.KeypointsContainer):
            return self.keypoints.is_valid(keypoints)

        return self.keypoints.is_valid_keypoints(keypoints)

    def is_valid_polyline_label(self, label):
        """Whether the polyline label is compliant with the schema.

        Args:
            label: a polyline label

        Returns:
            True/False
        """
        return self.polylines.is_valid_polyline_label(label)

    def is_valid_polyline_attribute(self, label, attr):
        """Whether the attribute for the polyline with the given label is
        compliant with the schema.

        Args:
            label: a polyline label
            attr: an Attribute

        Returns:
            True/False
        """
        return self.polylines.is_valid_polyline_attribute(label, attr)

    def is_valid_polyline_attributes(self, label, attrs):
        """Whether the attributes for the polyline with the given label are
        compliant with the schema.

        Args:
            label: a polyline label
            attrs: an AttributeContainer

        Returns:
            True/False
        """
        return self.polylines.is_valid_polyline_attributes(label, attrs)

    def is_valid_polyline(self, polyline):
        """Whether the given polyline is compliant with the schema.

        Args:
            polyline: a Polyline

        Returns:
            True/False
        """
        return self.polylines.is_valid_polyline(polyline)

    def is_valid_polylines(self, polylines):
        """Whether the given polylines are compliant with the schema.

        Args:
            polylines: a PolylineContainer

        Returns:
            True/False
        """
        return self.polylines.is_valid(polylines)

    def is_valid_event_label(self, label):
        """Whether the event label is compliant with the schema.

        Args:
            label: an event label

        Returns:
            True/False
        """
        return self.events.is_valid_event_label(label)

    def is_valid_event_attribute(self, label, attr):
        """Whether the event-level attribute for the event with the given label
        is compliant with the schema.

        Args:
            label: an event label
            attr: an Attribute

        Returns:
            True/False
        """
        return self.events.is_valid_event_attribute(label, attr)

    def is_valid_event_attributes(self, label, attrs):
        """Whether the event-level attributes for the event with the given
        label are compliant with the schema.

        Args:
            label: an event label
            attrs: an AttributeContainer

        Returns:
            True/False
        """
        return self.events.is_valid_event_attributes(label, attrs)

    def is_valid_event_frame_attribute(self, label, attr):
        """Whether the frame-level attribute for the event with the given label
        is compliant with the schema.

        Args:
            label: an event label
            attr: an Attribute

        Returns:
            True/False
        """
        return self.events.is_valid_frame_attribute(label, attr)

    def is_valid_event_frame_attributes(self, label, attrs):
        """Whether the frame-level attributes for the event with the given
        label are compliant with the schema.

        Args:
            label: an event label
            attrs: an AttributeContainer

        Returns:
            True/False
        """
        return self.events.is_valid_frame_attributes(label, attrs)

    def is_valid_event_object_label(self, event_label, obj_label):
        """Whether the object label for the event with the given label is
        compliant with the schema.

        Args:
            event_label: an event label
            obj_label: an object label

        Returns:
            True/False
        """
        return self.events.is_valid_object_label(event_label, obj_label)

    def is_valid_event_object_attribute(self, event_label, obj_label, attr):
        """Whether the object-level attribute for the object with the given
        label for the event with the given label is compliant with the schema.

        Args:
            event_label: an event label
            obj_label: an object label
            attr: an Attribute

        Returns:
            True/False
        """
        return self.events.is_valid_object_attribute(
            event_label, obj_label, attr
        )

    def is_valid_event_object_attributes(self, event_label, obj_label, attrs):
        """Whether the AttributeContainer of object-level attributes for the
        object with the given label for the event with the given label is
        compliant with the schema.

        Args:
            event_label: an event label
            obj_label: an object label
            attrs: an AttributeContainer

        Returns:
            True/False
        """
        return self.events.is_valid_object_attributes(
            event_label, obj_label, attrs
        )

    def is_valid_event_object_frame_attribute(
        self, event_label, obj_label, attr
    ):
        """Whether the frame-level attribute for the object with the given
        label for the event with the given label is compliant with the schema.

        Args:
            event_label: an event label
            obj_label: an object label
            attr: an Attribute

        Returns:
            True/False
        """
        return self.events.is_valid_object_frame_attribute(
            event_label, obj_label, attr
        )

    def is_valid_event_object_frame_attributes(
        self, event_label, obj_label, attrs
    ):
        """Whether the AttributeContainer of frame-level attributes for the
        object with the given label for the event with the given label is
        compliant with the schema.

        Args:
            event_label: an event label
            obj_label: an object label
            attrs: an AttributeContainer

        Returns:
            True/False
        """
        return self.events.is_valid_object_frame_attributes(
            event_label, obj_label, attrs
        )

    def is_valid_event_object(self, event_label, obj):
        """Whether the object for the event with the given label is compliant
        with the schema.

        Args:
            event_label: an event label
            obj: a VideoObject or DetectedObject

        Returns:
            True/False
        """
        return self.events.is_valid_object(event_label, obj)

    def is_valid_event_objects(self, event_label, objects):
        """Whether the objects for the event with the given label are compliant
        with the schema.

        Args:
            event_label: an event label
            objects: a VideoObjectContainer or DetectedObjectContainer

        Returns:
            True/False
        """
        return self.events.is_valid_objects(event_label, objects)

    def is_valid_event(self, event):
        """Whether the given event is compliant with the schema.

        Args:
            event: a VideoEvent or DetectedEvent

        Returns:
            True/False
        """
        return self.events.is_valid_event(event)

    def is_valid_events(self, events):
        """Whether the given events are compliant with the schema.

        Args:
            event: a VideoEventContainer or DetectedEventContainer

        Returns:
            True/False
        """
        return self.events.is_valid(events)

    def is_valid_frame_labels(self, frame_labels):
        """Whether the given frame labels are compliant with the schema.

        Args:
            frame_labels: a FrameLabels

        Returns:
            True/False
        """
        try:
            self.validate_frame_labels(frame_labels)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_image_labels(self, image_labels):
        """Whether the given image labels are compliant with the schema.

        Args:
            image_labels: an ImageLabels

        Returns:
            True/False
        """
        try:
            self.validate_image_labels(image_labels)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_video_labels(self, video_labels):
        """Whether the given video labels are compliant with the schema.

        Args:
            video_labels: a VideoLabels

        Returns:
            True/False
        """
        try:
            self.validate_video_labels(video_labels)
            return True
        except etal.LabelsSchemaError:
            return False

    def validate_constant_attribute(self, attr):
        """Validates that the constant attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        self.attrs.validate_attribute(attr)

    def validate_constant_attributes(self, attrs):
        """Validates that the constant attributes are compliant with the
        schema.

        Args:
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        """
        self.attrs.validate(attrs)

    def validate_frame_attribute(self, attr):
        """Validates that the frame-level attribute is compliant with the
        schema.

        Args:
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        self.frames.validate_attribute(attr)

    def validate_frame_attributes(self, attrs):
        """Validates that the frame-level attributes are compliant with the
        schema.

        Args:
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        """
        self.frames.validate(attrs)

    def validate_object_label(self, label):
        """Validates that the object label is compliant with the schema.

        Args:
            label: an object label

        Raises:
            LabelsSchemaError: if the object label violates the schema
        """
        self.objects.validate_object_label(label)

    def validate_object_attribute(self, label, attr):
        """Validates that the object-level attribute for the object with the
        given label is compliant with the schema.

        Args:
            label: an object label
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        self.objects.validate_object_attribute(label, attr)

    def validate_object_attributes(self, label, attrs):
        """Validates that the object-level attributes for the object with the
        given label are compliant with the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        """
        self.objects.validate_object_attributes(label, attrs)

    def validate_object_frame_attribute(self, label, attr):
        """Validates that the frame-level attribute for the object with the
        given label is compliant with the schema.

        Args:
            label: an object label
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        self.objects.validate_frame_attribute(label, attr)

    def validate_object_frame_attributes(self, label, attrs):
        """Validates that the frame-level attributes for the object with the
        given label are compliant with the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        """
        self.objects.validate_frame_attributes(label, attrs)

    def validate_object(self, obj):
        """Validates that the object is compliant with the schema.

        Args:
            obj: a VideoObject or DetectedObject

        Raises:
            LabelsSchemaError: if the object violates the schema
        """
        self.objects.validate_object(obj)

    def validate_objects(self, objects):
        """Validates that the objects are compliant with the schema.

        Args:
            objects: a VideoObjectContainer or DetectedObjectContainer

        Raises:
            LabelsSchemaError: if the objects violate the schema
        """
        self.objects.validate(objects)

    def validate_keypoints_label(self, label):
        """Validates that the keypoints label is compliant with the schema.

        Args:
            label: a keypoints label

        Raises:
            LabelsSchemaError: if the keypoints label violates the schema
        """
        self.keypoints.validate_keypoints_label(label)

    def validate_keypoints_attribute(self, label, attr):
        """Validates that the attribute for the keypoints with the given label
        is compliant with the schema.

        Args:
            label: a keypoints label
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        self.keypoints.validate_keypoints_attribute(label, attr)

    def validate_keypoints_attributes(self, label, attrs):
        """Validates that the attributes for the keypoints with the given label
        are compliant with the schema.

        Args:
            label: a keypoints label
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        """
        self.keypoints.validate_keypoints_attributes(label, attrs)

    def validate_keypoints(self, keypoints):
        """Validates that the keypoints are compliant with the schema.

        Args:
            keypoints: a Keypoints or KeypointsContainer

        Raises:
            LabelsSchemaError: if the keypoints violate the schema
        """
        if isinstance(keypoints, etak.KeypointsContainer):
            self.keypoints.validate(keypoints)
        else:
            self.keypoints.validate_keypoints(keypoints)

    def validate_polyline_label(self, label):
        """Validates that the polyline label is compliant with the schema.

        Args:
            label: a polyline label

        Raises:
            LabelsSchemaError: if the polyline label violates the schema
        """
        self.polylines.validate_polyline_label(label)

    def validate_polyline_attribute(self, label, attr):
        """Validates that the attribute for the polyline with the given label
        is compliant with the schema.

        Args:
            label: a polyline label
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        self.polylines.validate_polyline_attribute(label, attr)

    def validate_polyline_attributes(self, label, attrs):
        """Validates that the attributes for the polyline with the given label
        are compliant with the schema.

        Args:
            label: a polyline label
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        """
        self.polylines.validate_polyline_attributes(label, attrs)

    def validate_polyline(self, polyline):
        """Validates that the polyline is compliant with the schema.

        Args:
            polyline: a Polyline

        Raises:
            LabelsSchemaError: if the polyline violates the schema
        """
        self.polylines.validate_polyline(polyline)

    def validate_polylines(self, polylines):
        """Validates that the polylines are compliant with the schema.

        Args:
            polylines: a PolylineContainer

        Raises:
            LabelsSchemaError: if the polylines violate the schema
        """
        self.polylines.validate(polylines)

    def validate_event_label(self, label):
        """Validates that the event label is compliant with the schema.

        Args:
            label: an event label

        Raises:
            LabelsSchemaError: if the event label violates the schema
        """
        self.events.validate_event_label(label)

    def validate_event_attribute(self, label, attr):
        """Validates that the event-level attribute for the event with the
        given label is compliant with the schema.

        Args:
            label: an event label
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        self.events.validate_event_attribute(label, attr)

    def validate_event_attributes(self, label, attrs):
        """Validates that the event-level attributes for the event with the
        given label are compliant with the schema.

        Args:
            label: an event label
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        """
        self.events.validate_event_attributes(label, attrs)

    def validate_event_frame_attribute(self, label, attr):
        """Validates that the frame-level attribute for the event with the
        given label is compliant with the schema.

        Args:
            label: an event label
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        self.events.validate_frame_attribute(label, attr)

    def validate_event_frame_attributes(self, label, attrs):
        """Validates that the frame-level attributes for the event with the
        given label are compliant with the schema.

        Args:
            label: an event label
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        """
        self.events.validate_frame_attributes(label, attrs)

    def validate_event_object_label(self, event_label, obj_label):
        """Validates that the object label for the event with the given label
        is compliant with the schema.

        Args:
            event_label: an event label
            obj_label: an object label

        Raises:
            LabelsSchemaError: if the obect label is not compliant with the
                schema
        """
        self.events.validate_object_label(event_label, obj_label)

    def validate_event_object_attribute(self, event_label, obj_label, attr):
        """Validates that the object-level attribute for the object with the
        given label for the event with the given label is compliant with the
        schema.

        Args:
            event_label: an event label
            obj_label: an object label
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute is not compliant with the
                schema
        """
        self.events.validate_object_attribute(event_label, obj_label, attr)

    def validate_event_object_attributes(self, event_label, obj_label, attrs):
        """Validates that the AttributeContainer of object-level attributes for
        the object with the given label for the event with the given label is
        compliant with the schema.

        Args:
            event_label: an event label
            obj_label: an object label
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes are not compliant with the
                schema
        """
        self.events.validate_object_attributes(event_label, obj_label, attrs)

    def validate_event_object_frame_attribute(
        self, event_label, obj_label, attr
    ):
        """Validates that the frame-level attribute for the object with the
        given label for the event with the given label is compliant with the
        schema.

        Args:
            event_label: an event label
            obj_label: an object label
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute is not compliant with the
                schema
        """
        self.events.validate_object_frame_attribute(
            event_label, obj_label, attr
        )

    def validate_event_object_frame_attributes(
        self, event_label, obj_label, attrs
    ):
        """Validates that the AttributeContainer of frame-level attributes for
        the object with the given label for the event with the given label is
        compliant with the schema.

        Args:
            event_label: an event label
            obj_label: an object label
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attribute is not compliant with the
                schema
        """
        self.events.validate_object_frame_attributes(
            event_label, obj_label, attrs
        )

    def validate_event_object(self, event_label, obj):
        """Validates that the object for the given event label is compliant
        with the schema.

        Args:
            event_label: an event label
            obj: a VideoObject or DetectedObject

        Raises:
            LabelsSchemaError: if the object is not compliant with the schema
        """
        self.events.validate_object(event_label, obj)

    def validate_event_objects(self, event_label, objects):
        """Validates that the objects for the given event label are compliant
        with the schema.

        Args:
            event_label: an event label
            objects: a VideoObjectContainer or DetectedObjectContainer

        Raises:
            LabelsSchemaError: if the object is not compliant with the schema
        """
        self.events.validate_objects(event_label, objects)

    def validate_event(self, event):
        """Validates that the event is compliant with the schema.

        Args:
            event: a VideoEvent or DetectedEvent

        Raises:
            LabelsSchemaError: if the event violates the schema
        """
        self.events.validate_event(event)

    def validate_events(self, events):
        """Validates that the events are compliant with the schema.

        Args:
            events: a VideoEventContainer or DetectedEventContainer

        Raises:
            LabelsSchemaError: if the events violate the schema
        """
        self.events.validate(events)

    def validate_frame_labels(self, frame_labels):
        """Validates that the frame labels are compliant with the schema.

        Args:
            frame_labels: a FrameLabels

        Raises:
            LabelsSchemaError: if the labels violate the schema
        """
        for attr in frame_labels.attrs:
            if attr.constant:
                self.validate_constant_attribute(attr)
            else:
                self.validate_frame_attribute(attr)

        self.validate_objects(frame_labels.objects)
        self.validate_keypoints(frame_labels.keypoints)
        self.validate_polylines(frame_labels.polylines)
        self.validate_events(frame_labels.events)

    def validate_image_labels(self, image_labels):
        """Validates that the image labels are compliant with the schema.

        Args:
            image_labels: an ImageLabels

        Raises:
            LabelsSchemaError: if the labels violate the schema
        """
        self.validate_frame_labels(image_labels)

    def validate_video_labels(self, video_labels):
        """Validates that the video labels are compliant with the schema.

        Args:
            video_labels: a VideoLabels

        Raises:
            LabelsSchemaError: if the labels violate the schema
        """
        self.validate_constant_attributes(video_labels.attrs)
        self.validate_objects(video_labels.objects)
        self.validate_events(video_labels.events)
        for frame_labels in video_labels.iter_frames():
            self.validate_frame_labels(frame_labels)

    def validate(self, frame_labels):
        """Validates that the labels are compliant with the schema.

        Args:
            frame_labels: a FrameLabels

        Raises:
            LabelsSchemaError: if the labels violate the schema
        """
        self.validate_frame_labels(frame_labels)

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given schema.

        Args:
            schema: a FrameLabelsSchema

        Raises:
            LabelsSchemaError: if this schema is not a subset of the given
                schema
        """
        self.validate_schema_type(schema)
        self.attrs.validate_subset_of_schema(schema.attrs)
        self.frames.validate_subset_of_schema(schema.frames)
        self.objects.validate_subset_of_schema(schema.objects)
        self.keypoints.validate_subset_of_schema(schema.keypoints)
        self.polylines.validate_subset_of_schema(schema.polylines)
        self.events.validate_subset_of_schema(schema.events)

    def merge_schema(self, schema):
        """Merges the given FrameLabelsSchema into this schema.

        Args:
            schema: a FrameLabelsSchema
        """
        self.attrs.merge_schema(schema.attrs)
        self.frames.merge_schema(schema.frames)
        self.objects.merge_schema(schema.objects)
        self.keypoints.merge_schema(schema.keypoints)
        self.polylines.merge_schema(schema.polylines)
        self.events.merge_schema(schema.events)

    @classmethod
    def build_active_schema_for_object(cls, obj):
        """Builds a FrameLabelsSchema that describes the active schema of the
        given object.

        Args:
            obj: a VideoObject or DetectedObject

        Returns:
            a FrameLabelsSchema
        """
        schema = cls()
        schema.add_object(obj)
        return schema

    @classmethod
    def build_active_schema_for_objects(cls, objects):
        """Builds a FrameLabelsSchema that describes the active schema of the
        given objects.

        Args:
            objects: a VideoObjectContainer or DetectedObjectContainer

        Returns:
            a FrameLabelsSchema
        """
        schema = cls()
        schema.add_objects(objects)
        return schema

    @classmethod
    def build_active_schema_for_keypoints(cls, keypoints):
        """Builds a FrameLabelsSchema that describes the active schema of the
        given keypoints.

        Args:
            keypoints: a Keypoints or KeypointsContainer

        Returns:
            a FrameLabelsSchema
        """
        schema = cls()
        schema.add_keypoints(keypoints)
        return schema

    @classmethod
    def build_active_schema_for_polyline(cls, polyline):
        """Builds a FrameLabelsSchema that describes the active schema of the
        given polyline.

        Args:
            polyline: a Polyline

        Returns:
            a FrameLabelsSchema
        """
        schema = cls()
        schema.add_polyline(polyline)
        return schema

    @classmethod
    def build_active_schema_for_polylines(cls, polylines):
        """Builds a FrameLabelsSchema that describes the active schema of the
        given polylines.

        Args:
            polylines: a PolylineContainer

        Returns:
            a FrameLabelsSchema
        """
        schema = cls()
        schema.add_polylines(polylines)
        return schema

    @classmethod
    def build_active_schema_for_event(cls, event):
        """Builds a FrameLabelsSchema that describes the active schema of the
        given event.

        Args:
            event: a VideoEvent or DetectedEvent

        Returns:
            a FrameLabelsSchema
        """
        schema = cls()
        schema.add_event(event)
        return schema

    @classmethod
    def build_active_schema_for_events(cls, events):
        """Builds a FrameLabelsSchema that describes the active schema of the
        given events.

        Args:
            events: a VideoEventContainer or DetectedEventContainer

        Returns:
            a FrameLabelsSchema
        """
        schema = cls()
        schema.add_events(events)
        return schema

    @classmethod
    def build_active_schema_for_frame_labels(cls, frame_labels):
        """Builds a FrameLabelsSchema that describes the active schema of the
        given frame labels.

        Args:
            frame_labels: a FrameLabels

        Returns:
            a FrameLabelsSchema
        """
        schema = cls()
        schema.add_frame_labels(frame_labels)
        return schema

    @classmethod
    def build_active_schema_for_image_labels(cls, image_labels):
        """Builds a FrameLabelsSchema that describes the active schema of the
        given image labels.

        Args:
            image_labels: an ImageLabels

        Returns:
            a FrameLabelsSchema
        """
        schema = cls()
        schema.add_image_labels(image_labels)
        return schema

    @classmethod
    def build_active_schema_for_video_labels(cls, video_labels):
        """Builds a FrameLabelsSchema that describes the active schema of the
        given video labels.

        Args:
            video_labels: a VideoLabels

        Returns:
            a FrameLabelsSchema
        """
        schema = cls()
        schema.add_video_labels(video_labels)
        return schema

    @classmethod
    def build_active_schema(cls, frame_labels):
        """Builds a FrameLabelsSchema that describes the active schema of the
        given FrameLabels.

        Args:
            frame_labels: a FrameLabels

        Returns:
            a FrameLabelsSchema
        """
        return cls.build_active_schema_for_frame_labels(frame_labels)

    @classmethod
    def from_frame_labels_schema(cls, frame_labels_schema):
        """Creates a FrameLabelsSchema from another FrameLabelsSchema.

        Args:
            frame_labels_schema: a FrameLabelsSchema

        Returns:
            a FrameLabelsSchema
        """
        return cls(
            attrs=frame_labels_schema.attrs,
            frames=frame_labels_schema.frames,
            objects=frame_labels_schema.objects,
            keypoints=frame_labels_schema.keypoints,
            polylines=frame_labels_schema.polylines,
            events=frame_labels_schema.events,
        )

    @classmethod
    def from_image_labels_schema(cls, image_labels_schema):
        """Creates a FrameLabelsSchema from an ImageLabelsSchema.

        Args:
            image_labels_schema: an ImageLabelsSchema

        Returns:
            a FrameLabelsSchema
        """
        return cls.from_frame_labels_schema(image_labels_schema)

    @classmethod
    def from_video_labels_schema(cls, video_labels_schema):
        """Creates a FrameLabelsSchema from a VideoLabelsSchema.

        Args:
            video_labels_schema: a VideoLabelsSchema

        Returns:
            a FrameLabelsSchema
        """
        return cls.from_frame_labels_schema(video_labels_schema)

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        """
        _attrs = []
        if self.attrs:
            _attrs.append("attrs")
        if self.frames:
            _attrs.append("frames")
        if self.objects:
            _attrs.append("objects")
        if self.keypoints:
            _attrs.append("keypoints")
        if self.polylines:
            _attrs.append("polylines")
        if self.events:
            _attrs.append("events")
        return _attrs

    @classmethod
    def from_dict(cls, d):
        """Constructs a FrameLabelsSchema from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a FrameLabelsSchema
        """
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainerSchema.from_dict(attrs)

        frames = d.get("frames", None)
        if frames is not None:
            frames = etad.AttributeContainerSchema.from_dict(frames)

        objects = d.get("objects", None)
        if objects is not None:
            objects = etao.ObjectContainerSchema.from_dict(objects)

        keypoints = d.get("keypoints", None)
        if keypoints is not None:
            keypoints = etak.KeypointsContainerSchema.from_dict(keypoints)

        polylines = d.get("polylines", None)
        if polylines is not None:
            polylines = etap.PolylineContainerSchema.from_dict(polylines)

        events = d.get("events", None)
        if events is not None:
            events = etae.EventContainerSchema.from_dict(events)

        return cls(
            attrs=attrs,
            frames=frames,
            objects=objects,
            keypoints=keypoints,
            polylines=polylines,
            events=events,
        )
