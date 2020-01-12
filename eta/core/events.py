'''
Core tools and data structures for working with events in videos.

Copyright 2017-2020, Voxel51, Inc.
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
from future.utils import iteritems, itervalues
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from eta.core.data import AttributeContainer, AttributeContainerSchema
from eta.core.frames import FrameLabels, FrameRanges
from eta.core.objects import ObjectContainerSchema
from eta.core.serial import Container, Serializable


class EventFrameLabels(FrameLabels):
    '''FrameLabels for a specific frame of an event.

    Attributes:
        frame_number: frame number
        attrs: AttributeContainer describing attributes of the frame
        objects: DetectedObjectContainer describing detected objects in the
            frame
    '''
    pass


class Event(Serializable):
    '''An event in a video.

    `Event`s are temporal concepts that describe a collection of information
    about an event in a video. `Event`s can have labels with confidences,
    event-level attributes that apply to the entire event, frame-level
    attributes and object detections, as well as child objects and events.

    Attributes:
        label: (optional) the event label
        confidence: (optional) the label confidence in [0, 1]
        support: a FrameRanges instance describing the frames in the event
        index: (optional) an index assigned to the event
        uuid: (optional) a UUID assigned to the event
        attrs: an AttributeContainer of event-level attributes
        frames: dictionary mapping frame numbers to EventFrameLabels
        child_objects: a set of UUIDs of child `Object`s
        child_events: a set of UUIDs of child `Event`s
    '''

    def __init__(
            self, label=None, confidence=None, support=None, index=None,
            uuid=None, attrs=None, frames=None, child_objects=None,
            child_events=None):
        '''Creates an Event instance.

        Args:
            label: (optional) the event label
            confidence: (optional) the label confidence in [0, 1]
            support: (optional) a FrameRanges instance describing the frames in
                the event. If omitted, the support is inferred from the frames
                and children of the event
            index: (optional) a index assigned to the event
            uuid: (optional) a UUID assigned to the event
            attrs: (optional) an AttributeContainer of event-level attributes
            frames: (optional) dictionary mapping frame numbers to
                EventFrameLabels instances
            child_objects: (optional) a set of UUIDs of child `Object`s
            child_events: (optional) a set of UUIDs of child `Event`s
        '''
        self.label = label
        self.confidence = confidence
        self.index = index
        self.uuid = uuid
        self.attrs = attrs or AttributeContainer()
        self.frames = frames or {}
        self.child_objects = set(child_objects or [])
        self.child_events = set(child_events or [])

        self._support = support

    @property
    def support(self):
        '''A FrameRanges instance describing the frames in which this event
        exists.

        If the event has an explicit `support`, it is returned. Otherwise, the
        support is inferred from the frames with EventFrameLabels. Note that
        the latter excludes child objects and events.
        '''
        if self._support is not None:
            return self._support

        return FrameRanges.from_iterable(self.frames.keys())

    def iter_frames(self):
        '''Returns an iterator over the EventFrameLabels in the event.

        Returns:
            an iterator over EventFrameLabels
        '''
        return itervalues(self.frames)

    @property
    def has_attributes(self):
        '''Whether the event has attributes of any kind.'''
        return self.has_event_attributes or self.has_frame_attributes

    @property
    def has_event_attributes(self):
        '''Whether the event has event-level attributes.'''
        return bool(self.attrs)

    @property
    def has_frame_attributes(self):
        '''Whether the event has frame-level attributes.'''
        for frame_labels in self.iter_frames():
            if frame_labels.has_frame_attributes:
                return True

        return False

    @property
    def has_detected_objects(self):
        '''Whether the event has at least one DetectedObject.'''
        for frame_labels in self.iter_frames():
            if frame_labels.has_objects:
                return True

        return False

    @property
    def has_child_objects(self):
        '''Whether the event has at least one child Object.'''
        return bool(self.child_objects)

    @property
    def has_child_events(self):
        '''Whether the event has at least one child Event.'''
        return bool(self.child_events)

    def add_event_attribute(self, event_attr):
        '''Adds the event-level Attribute to the event.

        Args:
            event_attr: an Attribute
        '''
        self.attrs.add(event_attr)

    def add_event_attributes(self, event_attrs):
        '''Adds the AttributeContainer of event-level attributes to the event.

        Args:
            event_attrs: an AttributeContainer
        '''
        self.attrs.add_container(event_attrs)

    def add_frame_attribute(self, frame_attr, frame_number):
        '''Adds the frame attribute to the event.

        Args:
            frame_attr: an Attribute
            frame_number: the frame number
        '''
        self._ensure_frame(frame_number)
        self.frames[frame_number].add_frame_attribute(frame_attr)

    def add_frame_attributes(self, frame_attrs, frame_number):
        '''Adds the given frame attributes to the event.

        Args:
            frame_attrs: an AttributeContainer
            frame_number: the frame number
        '''
        self._ensure_frame(frame_number)
        self.frames[frame_number].add_frame_attributes(frame_attrs)

    def add_detected_object(self, obj, frame_number=None):
        '''Adds the DetectedObject to the event.

        Args:
            obj: a DetectedObject
            frame_number: an optional frame number. If omitted,
                `obj.frame_number` will be used
        '''
        if frame_number is not None:
            obj.frame_number = frame_number
        elif obj.frame_number is None:
            raise ValueError(
                "Expected `frame_number` or the DetectedObject to have its "
                "`frame_number` set")

        self.frames[obj.frame_number].add_object(obj)

    def add_detected_objects(self, objects):
        '''Adds the DetectedObjects to the video.

        The DetectedObjects must have their `frame_number`s set.

        Args:
            objects: a DetectedObjectContainer
        '''
        for obj in objects:
            self.add_detected_object(obj)

    def clear_attributes(self):
        '''Removes all attributes of any kind from the event.'''
        self.clear_event_attributes()
        self.clear_frame_attributes()

    def clear_event_attributes(self):
        '''Removes all event-level attributes from the event.'''
        self.attrs = AttributeContainer()

    def clear_frame_attributes(self):
        '''Removes all frame attributes from the event.'''
        for frame_labels in self.iter_frames():
            frame_labels.clear_frame_attributes()

    def clear_detected_objects(self):
        '''Removes all DetectedObjects from the event.'''
        for frame_labels in self.iter_frames():
            frame_labels.clear_objects()

    def remove_objects_without_attrs(self, labels=None):
        '''Removes objects that do not have attributes from this container.

        Args:
            labels: an optional list of object `label` strings to which to
                restrict attention when filtering. By default, all objects are
                processed
        '''
        for frame_labels in self.iter_frames():
            frame_labels.remove_objects_without_attrs(labels=labels)

    def add_child_object(self, obj):
        '''Adds the Object as a child of this event.

        Args:
            obj: an Object, which must have its `uuid` set
        '''
        if obj.uuid is None:
            raise ValueError("Object must have its `uuid` set")

        self.child_objects.add(obj.uuid)

    def clear_child_objects(self):
        '''Removes all child objects from the event.'''
        self.child_objects = set()

    def add_child_event(self, event):
        '''Adds the Event as a child of this event.

        Args:
            event: an Event, which must have its `uuid` set
        '''
        if event.uuid is None:
            raise ValueError("Event must have its `uuid` set")

        self.child_events.add(event.uuid)

    def clear_child_events(self):
        '''Removes all child events from the event.'''
        self.child_events = set()

    def attributes(self):
        '''Returns the list of attributes to serialize.

        Returns:
            a list of attrinutes
        '''
        _attrs = []
        if self.label is not None:
            _attrs.append("label")
        if self.confidence is not None:
            _attrs.append("confidence")
        _attrs.append("support")
        if self.index is not None:
            _attrs.append("index")
        if self.uuid is not None:
            _attrs.append("uuid")
        if self.attrs:
            _attrs.append("attrs")
        if self.frames:
            _attrs.append("frames")
        if self.child_objects:
            _attrs.append("child_objects")
        if self.child_events:
            _attrs.append("child_events")
        return _attrs

    @staticmethod
    def build_simple(
            first, last, label, confidence=None, index=None, uuid=None):
        '''Creates a simple contiguous `Event`.

        Args:
            first: the first frame of the event
            last: the last frame of the event
            label: the event label
            confidence: (optional) confidence in [0, 1]
            index: (optional) an index for the event
            uuid: (optional) a UUID for the event

        Returns:
             an Event
        '''
        support = FrameRanges.build_simple(first, last)
        return Event(
            label=label, confidence=confidence, support=support, index=index,
            uuid=uuid)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an Event from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an Event
        '''
        support = d.get("support", None)
        if support is not None:
            support = FrameRanges.from_dict(support)

        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = AttributeContainer.from_dict(attrs)

        frames = d.get("frames", None)
        if frames is not None:
            frames = {
                int(fn): EventFrameLabels.from_dict(do)
                for fn, do in iteritems(frames)
            }

        return cls(
            label=d.get("label", None),
            confidence=d.get("confidence", None),
            support=support,
            index=d.get("index", None),
            uuid=d.get("uuid", None),
            attrs=attrs,
            frames=frames,
            child_objects=d.get("child_objects", None),
            child_events=d.get("child_events", None),
        )

    def _ensure_frame(self, frame_number):
        if not frame_number in self.frames:
            self.frames[frame_number] = EventFrameLabels(frame_number)


class EventSchema(Serializable):
    '''Schema for `Event`s.

    Attributes:
        label: the event label
        attrs: an AttributeContainerSchema describing the event-level
            attributes of the event
        frames: an AttributeContainerSchema describing the frame-level
            attributes of the event
        objects: an ObjectContainerSchema describing the objects of the event
    '''

    def __init__(self, label, attrs=None, frames=None, objects=None):
        '''Creates an EventSchema instance.

        Args:
            label: the event label
            attrs: (optional) an AttributeContainerSchema describing the
                video-level attributes of the event
            frames: (optional) an AttributeContainerSchema describing the frame
                attributes of the event
            objects: (optional) an ObjectContainerSchema describing the objects
                of the event
        '''
        self.label = label
        self.attrs = attrs or AttributeContainerSchema()
        self.frames = frames or AttributeContainerSchema()
        self.objects = objects or ObjectContainerSchema()

    def has_label(self, label):
        '''Whether the schema has the given event label.

        Args:
            label: the event label

        Returns:
            True/False
        '''
        return label == self.label

    def get_label(self):
        '''Gets the event label for the schema.

        Returns:
            the event label
        '''
        return self.label

    def has_event_attribute(self, event_attr_name):
        '''Whether the schema has an event-level attribute with the given name.

        Args:
            event_attr_name: the name

        Returns:
            True/False
        '''
        return self.attrs.has_attribute(event_attr_name)

    def get_event_attribute_schema(self, event_attr_name):
        '''Gets the AttributeSchema for the event-level attribute with the
        given name.

        Args:
            event_attr_name: the name

        Returns:
            the AttributeSchema
        '''
        return self.attrs.get_attribute_schema(event_attr_name)

    def get_event_attribute_class(self, event_attr_name):
        '''Gets the Attribute class for the event-level attribute with the
        given name.

        Args:
            event_attr_name: the name

        Returns:
            the Attribute class
        '''
        return self.attrs.get_attribute_class(event_attr_name)

    def has_frame_attribute(self, frame_attr_name):
        '''Whether the schema has a frame attribute with the given name.

        Args:
            frame_attr_name: the name of the frame attribute

        Returns:
            True/False
        '''
        return self.frames.has_attribute(frame_attr_name)

    def get_frame_attribute_schema(self, frame_attr_name):
        '''Gets the AttributeSchema for the frame attribute with the given
        name.

        Args:
            frame_attr_name: the name

        Returns:
            the AttributeSchema
        '''
        return self.frames.get_attribute_schema(frame_attr_name)

    def get_frame_attribute_class(self, frame_attr_name):
        '''Gets the Attribute class for the frame attribute with the given
        name.

        Args:
            frame_attr_name: the name of the frame attribute

        Returns:
            the Attribute
        '''
        return self.frames.get_attribute_class(frame_attr_name)

    def has_object_label(self, label):
        '''Whether the schema has an object with the given label.

        Args:
            label: the object label

        Returns:
            True/False
        '''
        return self.objects.has_object_label(label)

    def has_object_attribute(self, label, obj_attr_name):
        '''Whether the schema has an object with the given label with an
        attribute of the given name.

        Args:
            label: the object label
            obj_attr_name: the name of the object attribute

        Returns:
            True/False
        '''
        return self.objects.has_object_attribute(label, obj_attr_name)

    def get_object_schema(self, label):
        '''Gets the ObjectSchema for the object with the given label.

        Args:
            label: the object label

        Returns:
            the ObjectSchema
        '''
        return self.objects.get_object_schema(label)

    def get_object_attribute_schema(self, label, obj_attr_name):
        '''Gets the AttributeSchema for the attribute of the given name for the
        object with the given label.

        Args:
            label: the object label
            obj_attr_name: the name of the object attribute

        Returns:
            the AttributeSchema
        '''
        return self.objects.get_object_attribute_schema(label, obj_attr_name)

    def get_object_attribute_class(self, label, obj_attr_name):
        '''Gets the Attribute class for the attribute of the given name for the
        object with the given label.

        Args:
            label: the object label
            obj_attr_name: the name of the object attribute

        Returns:
            the Attribute
        '''
        return self.objects.get_object_attribute_class(label, obj_attr_name)

    def add_event_attribute(self, event_attr):
        '''Adds the given event-level attribute to the schema.

        Args:
            event_attr: an Attribute
        '''
        self.attrs.add_attribute(event_attr)

    def add_event_attributes(self, event_attrs):
        '''Adds the given event-level attributes to the schema.

        Args:
            event_attrs: an AttributeContainer
        '''
        self.attrs.add_attributes(event_attrs)

    def add_frame_attribute(self, frame_attr):
        '''Adds the given frame attribute to the schema.

        Args:
            frame_attr: an Attribute
        '''
        self.frames.add_attribute(frame_attr)

    def add_frame_attributes(self, frame_attrs):
        '''Adds the given frame attributes to the schema.

        Args:
            frame_attrs: an AttributeContainer
        '''
        self.frames.add_attributes(frame_attrs)

    def add_object_label(self, label):
        '''Adds the given object label to the schema.

        ArgsL:
            label: an object label
        '''
        self.objects.add_object_label(label)

    def add_object_attribute(self, label, obj_attr):
        '''Adds the Attribute for the object with the given label to the
        schema.

        Args:
            label: an object label
            obj_attr: an Attribute
        '''
        self.objects.add_object_attribute(label, obj_attr)

    def add_object_attributes(self, label, obj_attrs):
        '''Adds the AttributeContainer for the object with the given label to
        the schema.

        Args:
            label: an object label
            obj_attrs: an AttributeContainer
        '''
        self.objects.add_object_attributes(label, obj_attrs)

    def add_object(self, obj):
        '''Adds the Object or DetectedObject to the schema.

        Args:
            obj: an Object or DetectedObject
        '''
        self.objects.add_object(obj)

    def add_objects(self, objects):
        '''Adds the ObjectContainer or DetectedObjectContainer to the schema.

        Args:
            objects: an ObjectContainer or DetectedObjectContainer
        '''
        self.objects.add_objects(objects)

    def add_frame_labels(self, frame_labels):
        '''Adds the FrameLabels to the schema.

        Args:
            frame_labels: a FrameLabels
        '''
        self.add_frame_attributes(frame_labels.attrs)
        self.add_objects(frame_labels.objects)

    def add_event(self, event):
        '''Adds the Event to the schema.

        Args:
            event: an Event
        '''
        self.validate_label(event.label)
        self.add_event_attributes(event.attrs)
        for frame_labels in event.iter_frames():
            self.add_frame_labels(frame_labels)

    def add_events(self, events):
        '''Adds the EventContainer to the schema.

        Args:
            events: an EventContainer
        '''
        for event in events:
            self.add_event(event)

    def merge_schema(self, schema):
        '''Merges the given EventSchema into this schema.

        Args:
            schema: an EventSchema
        '''
        self.validate_label(schema.label)
        self.attrs.merge_schema(schema.attrs)
        self.frames.merge_schema(schema.frames)
        self.objects.merge_schema(schema.objects)

    def is_valid_event_attribute(self, event_attr):
        '''Whether the event-level attribute is compliant with the schema.

        Args:
            event_attr: an Attribute

        Returns:
            True/False
        '''
        return self.attrs.is_valid_attribute(event_attr)

    def is_valid_frame_attribute(self, frame_attr):
        '''Whether the frame attribute is compliant with the schema.

        Args:
            frame_attr: an Attribute

        Returns:
            True/False
        '''
        return self.frames.is_valid_attribute(frame_attr)

    def is_valid_object_label(self, label):
        '''Whether the object label is compliant with the schema.

        Args:
            label: an object label

        Returns:
            True/False
        '''
        return self.objects.is_valid_object_label(label)

    def is_valid_object_attribute(self, label, obj_attr):
        '''Whether the object attribute for the object with the given label is
        compliant with the schema.

        Args:
            label: an object label
            obj_attr: an Attribute

        Returns:
            True/False
        '''
        return self.objects.is_valid_object_attribute(label, obj_attr)

    def is_valid_object(self, obj):
        '''Whether the Object or DetectedObject is compliant with the schema.

        Args:
            obj: an Object or DetectedObject

        Returns:
            True/False
        '''
        return self.objects.is_valid_object(obj)

    def validate_label(self, label):
        '''Validates that the event label is compliant with the schema.

        Args:
            label: the label

        Raises:
            EventSchemaError: if the label violates the schema
        '''
        if label != self.label:
            raise EventSchemaError(
                "Label '%s' does not match event schema" % label)

    def validate_event_attribute_name(self, event_attr_name):
        '''Validates that the schema contains an event-level attribute with the
        given name.

        Args:
            event_attr_name: the name

        Raises:
            AttributeContainerSchemaError: if the schema does not contain the
                attribute
        '''
        self.attrs.validate_attribute_name(event_attr_name)

    def validate_event_attribute(self, event_attr):
        '''Validates that the event-level attribute is compliant with the
        schema.

        Args:
            event_attr: an Attribute

        Raises:
            AttributeContainerSchemaError: if the attribute violates the schema
        '''
        self.attrs.validate_attribute(event_attr)

    def validate_frame_attribute_name(self, frame_attr_name):
        '''Validates that the schema contains a frame attribute with the given
        name.

        Args:
            frame_attr_name: the name

        Raises:
            AttributeContainerSchemaError: if the schema does not contain the
                attribute
        '''
        self.frames.validate_attribute_name(frame_attr_name)

    def validate_frame_attribute(self, frame_attr):
        '''Validates that the frame attribute is compliant with the schema.

        Args:
            frame_attr: an Attribute

        Raises:
            AttributeContainerSchemaError: if the attribute violates the schema
        '''
        self.frames.validate_attribute(frame_attr)

    def validate_object_label(self, label):
        '''Validates that the object label is compliant with the schema.

        Args:
            label: an object label

        Raises:
            ObjectContainerSchemaError: if the object label violates the schema
        '''
        self.objects.validate_object_label(label)

    def validate_object_attribute(self, label, obj_attr):
        '''Validates that the object attribute for the given label is compliant
        with the schema.

        Args:
            label: an object label
            obj_attr: an Attribute

        Raises:
            ObjectContainerSchemaError: if the object label violates the schema
            AttributeContainerSchemaError: if the object attribute violates the
                schema
        '''
        self.objects.validate_object_attribute(label, obj_attr)

    def validate_object(self, obj):
        '''Validates that the object is compliant with the schema.

        Args:
            obj: an Object or DetectedObject

        Raises:
            ObjectContainerSchemaError: if the object label violates the schema
            AttributeContainerSchemaError: if any attributes of the object
                violate the schema
        '''
        self.objects.validate_object(obj)

    def validate_event(self, event):
        '''Validates that the event is compliant with the schema.

        Args:
            event: an Event

        Raises:
            EventSchemaError: if the event label violates the schema
            ObjectContainerSchemaError: if an object label violates the schema
            AttributeContainerSchemaError: if any event/frame/object attribute
                violates the schema
        '''
        # Validate event label
        self.validate_label(event.label)

        # Validate event attributes
        for event_attr in event.attrs:
            self.validate_event_attribute(event_attr)

        # Validate frame-level detections
        for frame_labels in event.iter_frames():
            # Frame-level attributes
            for frame_attr in frame_labels.attrs:
                self.validate_frame_attribute(frame_attr)

            # Frame-level objects
            for obj in frame_labels.objects:
                self.validate_object(obj)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Args:
            a list of attribute names
        '''
        return ["label", "attrs", "frames", "objects"]

    @classmethod
    def build_active_schema(cls, event):
        '''Builds an EventSchema that describes the active schema of the given
        Event.

        Args:
            event: an Event

        Returns:
            an EventSchema
        '''
        schema = cls(event.label)
        schema.add_event(event)
        return schema

    @classmethod
    def from_dict(cls, d):
        '''Constructs an EventSchema from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an EventSchema
        '''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = AttributeContainerSchema.from_dict(attrs)

        frames = d.get("frames", None)
        if frames is not None:
            frames = AttributeContainerSchema.from_dict(frames)

        objects = d.get("objects", None)
        if objects is not None:
            objects = ObjectContainerSchema.from_dict(objects)

        return cls(d["label"], attrs=attrs, frames=frames, objects=objects)


class EventSchemaError(Exception):
    '''Error raised when an EventSchema is violated.'''
    pass


class EventContainer(Container):
    '''A `Container` of `Event`s.'''

    _ELE_CLS = Event
    _ELE_CLS_FIELD = "_EVENT_CLS"
    _ELE_ATTR = "events"

    def __init__(self, schema=None, **kwargs):
        '''Creates an EventContainer instance.

        Args:
            schema: an optional EventContainerSchema to enforce on the events
                in this container. By default, no schema is enforced
            **kwargs: valid keyword arguments for Container()

        Raises:
            EventContainerSchemaError: if a schema was provided but the events
                added to the container violate it
        '''
        super(EventContainer, self).__init__(**kwargs)
        self.schema = None
        if schema is not None:
            self.set_schema(schema)

    @property
    def has_schema(self):
        '''Whether the container has an enforced schema.'''
        return self.schema is not None

    def add(self, event):
        '''Adds an event to the container.

        Args:
            event: an Event

        Raises:
            EventContainerSchemaError: if this container has a schema enforced
                and the given event violates it
        '''
        if self.has_schema:
            self._validate_event(event)

        super(EventContainer, self).add(event)

    def add_container(self, events):
        '''Adds the events in the given container to this container.

        Args:
            events: an EventContainer instance

        Raises:
            EventContainerSchemaError: if this container has a schema enforced
                and an event in the given container violates it
        '''
        if self.has_schema:
            for event in events:
                self._validate_event(event)

        super(EventContainer, self).add_container(events)

    def get_labels(self):
        '''Returns a set containing the labels of the `Event`s.

        Returns:
            a set of labels
        '''
        return set(event.label for event in self)

    def sort_by_confidence(self, reverse=False):
        '''Sorts the `Event`s by confidence.

        `Event`s whose confidence is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("confidence", reverse=reverse)

    def sort_by_index(self, reverse=False):
        '''Sorts the `Event`s by index.

        `Event`s whose index is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("index", reverse=reverse)

    def filter_by_schema(self, schema):
        '''Filters the events/attributes from this container that are not
        compliant with the given schema.

        Args:
            schema: an EventContainerSchema
        '''
        # Filter by event label
        filter_func = lambda event: schema.has_event_label(event.label)
        self.filter_elements([filter_func])

        # Filter events
        for event in self:
            event_schema = schema.get_event_schema(event.label)
            event.filter_by_schema(event_schema)

    def remove_objects_without_attrs(self, labels=None):
        '''Removes objects that do not have attributes from all events in this
        container.

        Args:
            labels: an optional list of object `label` strings to which to
                restrict attention when filtering. By default, all objects are
                processed
        '''
        for event in self:
            event.remove_objects_without_attrs(labels=labels)

    def get_schema(self):
        '''Gets the current enforced schema for the container, or None if no
        schema is enforced.

        Returns:
            an EventContainerSchema
        '''
        return self.schema

    def get_active_schema(self):
        '''Returns an EventContainerSchema describing the active schema of the
        container.

        Returns:
            an EventContainerSchema
        '''
        return EventContainerSchema.build_active_schema(self)

    def set_schema(self, schema, filter_by_schema=False):
        '''Sets the enforced schema to the given EventContainerSchema.

        Args:
            schema: the EventContainerSchema to use
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
        _attrs += super(EventContainer, self).attributes()
        return _attrs

    def _validate_event(self, event):
        if self.has_schema:
            self.schema.validate_event(event)

    def _validate_schema(self):
        if self.has_schema:
            for event in self:
                self._validate_event(event)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an EventContainer from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an EventContainer
        '''
        events = super(EventContainer, cls).from_dict(d)

        schema = d.get("schema", None)
        if schema is not None:
            events.set_schema(EventContainerSchema.from_dict(schema))

        return events


class EventContainerSchema(Serializable):
    '''Schema for `EventContainers`s.

    Attributes:
        schema: a dictionary mapping event labels to EventSchema instances
    '''

    def __init__(self, schema=None):
        '''Creates an EventContainerSchema instance.

        Args:
            schema: a dictionary mapping event labels to EventSchema instances.
                By default, an empty schema is created
        '''
        self.schema = schema or {}

    def has_event_label(self, label):
        '''Whether the schema has an event with the given label.

        Args:
            label: the event label

        Returns:
            True/False
        '''
        return label in self.schema

    def get_event_schema(self, label):
        '''Gets the EventSchema for the event with the given label.

        Args:
            label: the object event

        Returns:
            an EventSchema
        '''
        self.validate_event_label(label)
        return self.schema[label]

    def has_event_attribute(self, label, event_attr_name):
        '''Whether the schema has an event with the given label with an
        event-level attribute of the given name.

        Args:
            label: the event label
            event_attr_name: the name of the event-level attribute

        Returns:
            True/False
        '''
        if not self.has_event_label(label):
            return False

        return self.schema[label].has_event_attribute(event_attr_name)

    def get_event_attribute_schema(self, label, event_attr_name):
        '''Gets the AttributeSchema for the event-level attribute of the given
        name for the event with the given label.

        Args:
            label: the event label
            event_attr_name: the name of the event-level attribute

        Returns:
            the AttributeSchema
        '''
        event_schema = self.get_event_schema(label)
        return event_schema.get_event_attribute_schema(event_attr_name)

    def get_event_attribute_class(self, label, event_attr_name):
        '''Gets the `Attribute` class for the event-level attribute of the
        given name for the event with the given label.

        Args:
            label: the event label
            event_attr_name: the name of the event-level attribute

        Returns:
            the Attribute subclass
        '''
        self.validate_event_label(label)
        return self.schema[label].get_event_attribute_class(event_attr_name)

    def has_frame_attribute(self, label, frame_attr_name):
        '''Whether the schema has an event with the given label with a
        frame-level attribute of the given name.

        Args:
            label: the event label
            frame_attr_name: the name of the frame-level attribute

        Returns:
            True/False
        '''
        if not self.has_event_label(label):
            return False

        return self.schema[label].has_frame_attribute(frame_attr_name)

    def get_frame_attribute_schema(self, label, frame_attr_name):
        '''Gets the AttributeSchema for the frame-level attribute of the given
        name for the event with the given label.

        Args:
            label: the event label
            frame_attr_name: the name of the frame-level attribute

        Returns:
            the AttributeSchema
        '''
        event_schema = self.get_event_schema(label)
        return event_schema.get_frame_attribute_schema(frame_attr_name)

    def get_frame_attribute_class(self, label, frame_attr_name):
        '''Gets the `Attribute` class for the frame-level attribute of the
        given name for the event with the given label.

        Args:
            label: the event label
            frame_attr_name: the name of the frame-level attribute

        Returns:
            the Attribute subclass
        '''
        self.validate_event_label(label)
        return self.schema[label].get_frame_attribute_class(frame_attr_name)

    def has_object_label(self, event_label, obj_label):
        '''Whether the schema has an event of the given label with an object
        of the given label.

        Args:
            event_label: the event label
            obj_label: the object label

        Returns:
            True/False
        '''
        self.validate_event_label(event_label)
        return self.schema[event_label].has_object_label(obj_label)

    def has_object_attribute(self, event_label, obj_label, obj_attr_name):
        '''Whether the schema has an event of the given label with an object
        of the given label with an attribute of the given name.

        Args:
            event_label: the event label
            obj_label: the object label
            obj_attr_name: the name of the object attribute

        Returns:
            True/False
        '''
        self.validate_event_label(event_label)
        return self.schema[event_label].has_object_attribute(
            obj_label, obj_attr_name)

    def get_object_schema(self, event_label, obj_label):
        '''Gets the ObjectSchema for the object with the given label from the
        event with the given label.

        Args:
            event_label: the event label
            obj_label: the object label

        Returns:
            the ObjectSchema
        '''
        self.validate_event_label(event_label)
        return self.schema[event_label].get_object_schema(obj_label)

    def get_object_attribute_schema(
            self, event_label, obj_label, obj_attr_name):
        '''Gets the AttributeSchema for the attribute of the given name for the
        object with the given label from the event with the given label.

        Args:
            event_label: the event label
            obj_label: the object label
            obj_attr_name: the name of the object attribute

        Returns:
            the AttributeSchema
        '''
        self.validate_event_label(event_label)
        return self.schema[event_label].get_object_attribute_schema(
            obj_label, obj_attr_name)

    def get_object_attribute_class(
            self, event_label, obj_label, obj_attr_name):
        '''Gets the Attribute class for the attribute of the given name for the
        object with the given label from the event with the given label.

        Args:
            event_label: the event label
            obj_label: the object label
            obj_attr_name: the name of the object attribute

        Returns:
            the Attribute
        '''
        self.validate_event_label(event_label)
        return self.schema[event_label].get_object_attribute_class(
            obj_label, obj_attr_name)

    def add_event_label(self, label):
        '''Adds the given event label to the schema.

        ArgsL:
            label: an event label
        '''
        self._ensure_has_event_label(label)

    def add_event_attribute(self, label, event_attr):
        '''Adds the event-level Attribute for the event with the given label to
        the schema.

        Args:
            label: an event label
            event_attr: an Attribute
        '''
        self._ensure_has_event_label(label)
        self.schema[label].add_event_attribute(event_attr)

    def add_event_attributes(self, label, event_attrs):
        '''Adds the AttributeContainer of event-level attributes for the event
        with the given label to the schema.

        Args:
            label: an event label
            event_attrs: an AttributeContainer
        '''
        self._ensure_has_event_label(label)
        self.schema[label].add_event_attributes(event_attrs)

    def add_frame_attribute(self, label, frame_attr):
        '''Adds the frame-level Attribute for the event with the given label to
        the schema.

        Args:
            label: an event label
            frame_attr: an Attribute
        '''
        self._ensure_has_event_label(label)
        self.schema[label].add_frame_attribute(frame_attr)

    def add_frame_attributes(self, label, frame_attrs):
        '''Adds the AttributeContainer of frame-level attributes for the event
        with the given label to the schema.

        Args:
            label: an event label
            frame_attrs: an AttributeContainer
        '''
        self._ensure_has_event_label(label)
        self.schema[label].add_frame_attributes(frame_attrs)



    def add_object_label(self, event_label, obj_label):
        '''Adds the given object label for the event with the given label to
        the schema.

        Args:
            event_label: an event label
            obj_label: an object label
        '''
        self._ensure_has_event_label(event_label)
        self.schema[event_label].add_object_label(obj_label)

    def add_object_attribute(self, event_label, obj_label, obj_attr):
        '''Adds the Attribute for the object with the given label to the event
        with the given label to the schema.

        Args:
            event_label: an event label
            obj_label: an object label
            obj_attr: an Attribute
        '''
        self._ensure_has_event_label(event_label)
        self.schema[event_label].add_object_attribute(obj_label, obj_attr)

    def add_object_attributes(self, event_label, obj_label, obj_attrs):
        '''Adds the AttributeContainer for the object with the given label to
        the event with the given label to the schema.

        Args:
            event_label: an event label
            obj_label: an object label
            obj_attrs: an AttributeContainer
        '''
        self._ensure_has_event_label(event_label)
        self.schema[event_label].add_object_attributes(obj_label, obj_attrs)

    def add_object(self, event_label, obj):
        '''Adds the Object or DetectedObject to the event with the given label
        to the schema.

        Args:
            event_label: an event label
            obj: an Object or DetectedObject
        '''
        self._ensure_has_event_label(event_label)
        self.schema[event_label].add_object(obj)

    def add_objects(self, event_label, objects):
        '''Adds the ObjectContainer or DetectedObjectContainer to the event
        with the given label to the schema.

        Args:
            event_label: an event label
            objects: an ObjectContainer or DetectedObjectContainer
        '''
        self._ensure_has_event_label(event_label)
        self.schema[event_label].add_objects(objects)

    def add_event(self, event):
        '''Adds the Event to the schema.

        Args:
            event: an Event
        '''
        self._ensure_has_event_label(event.label)
        self.schema[event.label].add_event(event)

    def add_events(self, events):
        '''Adds the EventContainer to the schema.

        Args:
            events: an EventContainer
        '''
        for event in events:
            self.add_event(event)

    def merge_schema(self, schema):
        '''Merges the given EventContainerSchema into this schema.

        Args:
            schema: an EventContainerSchema
        '''
        for label, event_schema in iteritems(schema.schema):
            self._ensure_has_event_label(label)
            self.schema[label].merge_schema(event_schema)

    def is_valid_event_label(self, label):
        '''Whether the event label is compliant with the schema.

        Args:
            label: an event label

        Returns:
            True/False
        '''
        try:
            self.validate_event_label(label)
            return True
        except:
            return False

    def is_valid_event_attribute(self, label, event_attr):
        '''Whether the event-level attribute for the event with the given label
        is compliant with the schema.

        Args:
            label: an event label
            event_attr: an Attribute

        Returns:
            True/False
        '''
        try:
            self.validate_event_attribute(label, event_attr)
            return True
        except:
            return False

    def is_valid_frame_attribute(self, label, frame_attr):
        '''Whether the frame-level attribute for the event with the given label
        is compliant with the schema.

        Args:
            label: an event label
            frame_attr: an Attribute

        Returns:
            True/False
        '''
        try:
            self.validate_frame_attribute(label, frame_attr)
            return True
        except:
            return False

    def is_valid_object_label(self, event_label, obj_label):
        '''Whether the object label for the event with the given label is
        compliant with the schema.

        Args:
            event_label: an event label
            obj_label: an object label

        Returns:
            True/False
        '''
        try:
            self.validate_object_label(event_label, obj_label)
            return True
        except:
            return False

    def is_valid_object_attribute(self, event_label, obj_label, obj_attr):
        '''Whether the attribute for the object with the given label for the
        event with the given label is compliant with the schema.

        Args:
            event_label: an event label
            obj_label: an object label
            obj_attr: an Attribute

        Returns:
            True/False
        '''
        try:
            self.validate_object_attribute(event_label, obj_label, obj_attr)
            return True
        except:
            return False

    def is_valid_object(self, event_label, obj):
        '''Whether the object for the event with the given label is compliant
        with the schema.

        Args:
            event_label: an event label
            obj: an `Object` or `DetectedObject`

        Returns:
            True/False
        '''
        try:
            self.validate_object(event_label, obj)
            return True
        except:
            return False

    def is_valid_event(self, event):
        '''Whether the `Event` is compliant with the schema.

        Args:
            event: an Event

        Returns:
            True/False
        '''
        try:
            self.validate_event(event)
            return True
        except:
            return False

    def validate_event_label(self, label):
        '''Validates that the event label is compliant with the schema.

        Args:
            label: an event label

        Raises:
            EventContainerSchemaError: if the event label violates the schema
        '''
        if label not in self.schema:
            raise EventContainerSchemaError(
                "Event label '%s' is not allowed by the schema" % label)

    def validate_event_attribute(self, label, event_attr):
        '''Validates that the event-level attribute for the given label is
        compliant with the schema.

        Args:
            label: an event label
            event_attr: an Attribute

        Raises:
            EventContainerSchemaError: if the event label violates the schema
            AttributeContainerSchemaError: if the event attribute violates the
                schema
        '''
        self.validate_event_label(label)
        self.schema[label].validate_event_attribute(event_attr)

    def validate_frame_attribute(self, label, frame_attr):
        '''Validates that the frame-level attribute for the given label is
        compliant with the schema.

        Args:
            label: an event label
            frame_attr: an Attribute

        Raises:
            EventContainerSchemaError: if the event label violates the schema
            AttributeContainerSchemaError: if the frame attribute violates the
                schema
        '''
        self.validate_event_label(label)
        self.schema[label].validate_frame_attribute(frame_attr)

    def validate_object_label(self, event_label, obj_label):
        '''Validates that the object label for the event with the given label
        is compliant with the schema.

        Args:
            event_label: an event label
            obj_label: an object label

        Raises:
            EventContainerSchemaError: if the event label violates the schema
            ObjectContainerSchemaError: if the object label violates the schema
        '''
        self.validate_event_label(event_label)
        self.schema[event_label].validate_object_label(obj_label)

    def validate_object_attribute(self, event_label, obj_label, obj_attr):
        '''Validates that the object attribute for the object with the given
        label for the event with the given label is compliant with the schema.

        Args:
            event_label: an event label
            obj_label: an object label
            obj_attr: an Attribute

        Raises:
            EventContainerSchemaError: if the event label violates the schema
            ObjectContainerSchemaError: if the object label violates the schema
            AttributeContainerSchemaError: if the object attribute violates the
                schema
        '''
        self.validate_event_label(event_label)
        self.schema[event_label].validate_object_attribute(obj_label, obj_attr)

    def validate_object(self, event_label, obj):
        '''Validates that the object for the given event label is compliant
        with the schema.

        Args:
            event_label: an event label
            obj: an Object or DetectedObject

        Raises:
            EventContainerSchemaError: if the event label violates the schema
            ObjectContainerSchemaError: if the object label violates the schema
            AttributeContainerSchemaError: if any attributes of the object
                violate the schema
        '''
        self.validate_event_label(event_label)
        self.schema[event_label].validate_object(obj)

    def validate_event(self, event):
        '''Validates that the `Event` is compliant with the schema.

        Args:
            event: an Event

        Raises:
            EventContainerSchemaError: if the event label violates the schema
            ObjectContainerSchemaError: if an object label violates the schema
            AttributeContainerSchemaError: if any event/frame/object attribute
                violates the schema
        '''
        self.validate_event_label(event.label)
        self.schema[event.label].validate_event(event)

    @classmethod
    def build_active_schema(cls, events):
        '''Builds an EventContainerSchema that describes the active schema of
        the events.

        Args:
            events: an EventContainer

        Returns:
            an EventContainerSchema
        '''
        schema = cls()
        schema.add_events(events)
        return schema

    @classmethod
    def from_dict(cls, d):
        '''Constructs an EventContainerSchema from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an EventContainerSchema
        '''
        schema = d.get("schema", None)
        if schema is not None:
            schema = {
                label: EventSchema.from_dict(esd)
                for label, esd in iteritems(schema)
            }

        return cls(schema=schema)

    def _ensure_has_event_label(self, label):
        if not self.has_event_label(label):
            self.schema[label] = EventSchema(label)


class EventContainerSchemaError(Exception):
    '''Error raised when an EventContainerSchema is violated.'''
    pass
