'''
Core tools and data structures for working with objects in images and videos.

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
from future.utils import iteritems, itervalues
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict

from eta.core.data import AttributeContainer, AttributeContainerSchema
from eta.core.frames import FrameRanges
from eta.core.geometry import BoundingBox, HasBoundingBox
from eta.core.serial import Container, Serializable, deserialize_numpy_array


class DetectedObject(Serializable, HasBoundingBox):
    '''A detected object in an image or frame of a video.

    `DetectedObject`s are spatial concepts that describe information about an
    object in a particular image or a particular frame of a video.
    `DetectedObject`s can have labels with confidences, bounding boxes,
    instance masks, and one or more additional attributes describing their
    properties.

    Attributes:
        label: (optional) object label
        bounding_box: (optional) a BoundingBox around the object
        mask: (optional) a mask for the object within its bounding box
        confidence: (optional) the label confidence, in [0, 1]
        top_k_probs: (optional) dictionary mapping labels to probabilities
        index: (optional) an index assigned to the object
        score: (optional) a multipurpose score for the object
        frame_number: (optional) the frame number in which the object was
            detected
        index_in_frame: (optional) the index of the object in the frame where
            it was detected
        eval_type: (optional) an EvaluationType value
        event_indices: (optional) a set of a Event indices to which the object
            belongs
        attrs: (optional) an AttributeContainer of attributes for the object
    '''

    def __init__(
            self, label=None, bounding_box=None, mask=None, confidence=None,
            top_k_probs=None, index=None, score=None, frame_number=None,
            index_in_frame=None, eval_type=None, event_indices=None,
            attrs=None):
        '''Creates a DetectedObject instance.

        Args:
            label: (optional) object label
            bounding_box: (optional) a BoundingBox around the object
            mask: (optional) a numpy array describing the mask for the object
                within its bounding box
            confidence: (optional) the label confidence, in [0, 1]
            top_k_probs: (optional) dictionary mapping labels to probabilities
            index: (optional) an index assigned to the object
            score: (optional) an optional score for the object
            frame_number: (optional) the frame number in the this object was
                detected
            index_in_frame: (optional) the index of the object in the frame
                where it was detected
            eval_type: (optional) an EvaluationType value
            event_indices: (optional) a set of indices indicating `Events` to
                which the object belongs
            attrs: (optional) an AttributeContainer of attributes for the
                object
        '''
        self.label = label
        self.bounding_box = bounding_box
        self.mask = mask
        self.confidence = confidence
        self.top_k_probs = top_k_probs
        self.index = index
        self.score = score
        self.frame_number = frame_number
        self.index_in_frame = index_in_frame
        self.eval_type = eval_type
        self.event_indices = set(event_indices or [])
        self.attrs = attrs or AttributeContainer()
        self._meta = None  # Usable by clients to store temporary metadata

    @property
    def has_attributes(self):
        '''Whether this object has attributes.'''
        return bool(self.attrs)

    @property
    def has_mask(self):
        '''Whether this object has a segmentation mask.'''
        return self.mask is not None

    def clear_attributes(self):
        '''Removes all attributes from the object.'''
        self.attrs = AttributeContainer()

    def add_attribute(self, attr):
        '''Adds the Attribute to the object.

        Args:
            attr: an Attribute
        '''
        self.attrs.add(attr)

    def add_attributes(self, attrs):
        '''Adds the AttributeContainer of attributes to the object.

        Args:
            attrs: an AttributeContainer
        '''
        self.attrs.add_container(attrs)

    def get_bounding_box(self):
        '''Returns the BoundingBox for the object.

        Returns:
             a BoundingBox
        '''
        return self.bounding_box

    def attributes(self):
        '''Returns the list of attributes to serialize.

        Returns:
            a list of attribute names
        '''
        _attrs = []

        _noneable_attrs = [
            "label", "bounding_box", "mask", "confidence", "top_k_probs",
            "index", "score", "frame_number", "index_in_frame", "eval_type"]
        _attrs.extend(
            [a for a in _noneable_attrs if getattr(self, a) is not None])

        if self.event_indices:
            _attrs.append("event_indices")
        if self.attrs:
            _attrs.append("attrs")

        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs a DetectedObject from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a DetectedObject
        '''
        bounding_box = d.get("bounding_box", None)
        if bounding_box is not None:
            bounding_box = BoundingBox.from_dict(bounding_box)

        mask = d.get("mask", None)
        if mask is not None:
            mask = deserialize_numpy_array(mask)

        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = AttributeContainer.from_dict(attrs)

        return cls(
            label=d.get("label", None),
            bounding_box=bounding_box,
            mask=mask,
            confidence=d.get("confidence", None),
            top_k_probs=d.get("top_k_probs", None),
            index=d.get("index", None),
            score=d.get("score", None),
            frame_number=d.get("frame_number", None),
            index_in_frame=d.get("index_in_frame", None),
            attrs=attrs,
            eval_type=d.get("eval_type", None),
            event_indices=set(d.get("event_indices", []))
        )


class DetectedObjectContainer(Container):
    '''A `Container` of `DetectedObjects`.'''

    _ELE_CLS = DetectedObject
    _ELE_CLS_FIELD = "_OBJ_CLS"
    _ELE_ATTR = "objects"

    def __init__(self, schema=None, **kwargs):
        '''Creates a DetectedObjectContainer instance.

        Args:
            schema: an optional ObjectContainerSchema to enforce on the objects
                in this container. By default, no schema is enforced
            **kwargs: valid keyword arguments for Container()

        Raises:
            ObjectContainerSchemaError: if a schema was provided but the
                objects added to the container violate it
        '''
        super(DetectedObjectContainer, self).__init__(**kwargs)
        self.schema = None
        if schema is not None:
            self.set_schema(schema)

    @property
    def has_schema(self):
        '''Whether the container has an enforced schema.'''
        return self.schema is not None

    def add(self, obj):
        '''Adds an object to the container.

        Args:
            obj: a DetectedObject

        Raises:
            ObjectContainerSchemaError: if this container has a schema enforced
                and the given object violates it
        '''
        if self.has_schema:
            self._validate_object(obj)

        super(DetectedObjectContainer, self).add(obj)

    def add_container(self, objects):
        '''Adds the objects in the given container to this container.

        Args:
            objects: a DetectedObjectContainer instance

        Raises:
            ObjectContainerSchemaError: if this container has a schema enforced
                and an object in the given container violates it
        '''
        if self.has_schema:
            for obj in objects:
                self._validate_object(obj)

        super(DetectedObjectContainer, self).add_container(objects)

    def get_labels(self):
        '''Returns a set containing the labels of the DetectedObjects.

        Returns:
            a set of labels
        '''
        return set(obj.label for obj in self)

    def sort_by_confidence(self, reverse=False):
        '''Sorts the `DetectedObject`s by confidence.

        `DetectedObject`s whose confidence is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("confidence", reverse=reverse)

    def sort_by_index(self, reverse=False):
        '''Sorts the `DetectedObject`s by index.

        `DetectedObject`s whose index is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("index", reverse=reverse)

    def sort_by_score(self, reverse=False):
        '''Sorts the `DetectedObject`s by score.

        `DetectedObject`s whose score is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("score", reverse=reverse)

    def sort_by_frame_number(self, reverse=False):
        '''Sorts the `DetectedObject`s by frame number

        `DetectedObject`s whose frame number is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("frame_number", reverse=reverse)

    def filter_by_schema(self, schema):
        '''Filters the objects/attributes from this container that are not
        compliant with the given schema.

        Args:
            schema: an ObjectContainerSchema
        '''
        # Filter by object label
        filter_func = lambda obj: schema.has_object_label(obj.label)
        self.filter_elements([filter_func])

        # Filter object attributes
        for obj in self:
            if obj.has_attributes:
                obj_schema = schema.get_object_schema(obj.label)
                obj.attrs.filter_by_schema(obj_schema)

    def remove_objects_without_attrs(self, labels=None):
        '''Removes objects from this container that do not have attributes.

        Args:
            labels: an optional list of object `label` strings to which to
                restrict attention when filtering. By default, all objects are
                processed
        '''
        filter_func = lambda obj: (
            (labels is not None and obj.label not in labels)
            or obj.has_attributes)
        self.filter_elements([filter_func])

    def get_schema(self):
        '''Gets the current enforced schema for the container, or None if no
        schema is enforced.

        Returns:
            an ObjectContainerSchema
        '''
        return self.schema

    def get_active_schema(self):
        '''Returns an ObjectContainerSchema describing the active schema of the
        container.

        Returns:
            an ObjectContainerSchema
        '''
        return ObjectContainerSchema.build_active_schema(self)

    def set_schema(self, schema, filter_by_schema=False):
        '''Sets the enforced schema to the given ObjectContainerSchema.

        Args:
            schema: the ObjectContainerSchema to use
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
        _attrs += super(DetectedObjectContainer, self).attributes()
        return _attrs

    def _validate_object(self, obj):
        if self.has_schema:
            self.schema.validate_object(obj)

    def _validate_schema(self):
        if self.has_schema:
            for obj in self:
                self._validate_object(obj)

    @classmethod
    def from_dict(cls, d):
        '''Constructs a DetectedObjectContainer from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a DetectedObjectContainer
        '''
        objects = super(DetectedObjectContainer, cls).from_dict(d)

        schema = d.get("schema", None)
        if schema is not None:
            objects.set_schema(ObjectContainerSchema.from_dict(schema))

        return objects


class Object(Serializable):
    '''A spatiotemporal object in a video.

    `Object`s are spatiotemporal concepts that describe information about an
    object over multiple frames in a video. `Object`s can have labels with
    confidences, object-level attributes that apply to the object over all
    frames, frame-level attributes such as bounding boxes and attributes that
    apply to individual frames, and child objects.

    Attributes:
        label: (optional) the object label
        confidence: (optional) label confidence in [0, 1]
        support: a FrameRanges instance describing the frames for which the
            object exists
        index: (optional) an index assigned to the object
        uuid: (optional) a UUID assigned to the object
        attrs: (optional) AttributeContainer of object-level attributes of the
            object
        frames: dictionary mapping frame numbers to DetectedObject instances
            describing the frame-level attributes of the object
        child_objects: (optional) a set of UUIDs of child `Object`s
    '''

    def __init__(
            self, label=None, confidence=None, support=None, index=None,
            uuid=None, attrs=None, frames=None, child_objects=None):
        '''Creates an Object instance.

        Args:
            label: (optional) the object label
            confidence: (optional) the label confidence in [0, 1]
            support: (optional) a FrameRanges instance describing the frames
                for which the object exists. If omitted, the support is
                inferred from the frames and children of the object
            index: (optional) an index assigned to the object
            uuid: (optional) a UUID assigned to the object
            attrs: (optional) an AttributeContainer of object-level attributes
            frames: (optional) dictionary mapping frame numbers to
                DetectedObject instances
            child_objects: (optional) a set of UUIDs of child `Object`s
        '''
        self.label = label
        self.confidence = confidence
        self.index = index
        self.uuid = uuid
        self.attrs = attrs or AttributeContainer()
        self.frames = frames or {}
        self.child_objects = set(child_objects or [])

        self._support = support

    @property
    def support(self):
        '''A FrameRanges instance describing the frames for which this object
        exists.

        If the object has an explicit `support`, it is returned. Otherwise, the
        support is inferred from the frames with DetectedObjects. Note that
        the latter excludes child objects.
        '''
        if self._support is not None:
            return self._support

        return FrameRanges.from_iterable(self.frames.keys())

    def iter_detections(self):
        '''Returns an iterator over the DetectedObjects in the object.

        Returns:
            an iterator over DetectedObjects
        '''
        return itervalues(self.frames)

    @property
    def has_attributes(self):
        '''Whether the object has attributes of any kind.'''
        return self.has_object_attributes or self.has_frame_attributes

    @property
    def has_object_attributes(self):
        '''Whether the object has object-level attributes.'''
        return bool(self.attrs)

    @property
    def has_detections(self):
        '''Whether the object has frame-level detections.'''
        return bool(self.frames)

    @property
    def has_frame_attributes(self):
        '''Whether the object has frame-level attributes.'''
        for obj in self.iter_detections():
            if obj.has_attributes:
                return True

        return False

    def clear_attributes(self):
        '''Removes all attributes of any kind from the object.'''
        self.clear_object_attributes()
        self.clear_frame_attributes()

    def clear_object_attributes(self):
        '''Removes all object-level attributes from the object.'''
        self.attrs = AttributeContainer()

    def clear_frame_attributes(self):
        '''Removes all frame-level attributes from the object.'''
        for obj in self.iter_detections():
            obj.clear_attributes()

    def add_object_attribute(self, attr):
        '''Adds the object-level attribute to the object.

        Args:
            attr: an Attribute
        '''
        self.attrs.add(attr)

    def add_object_attributes(self, attrs):
        '''Adds the AttributeContainer of object-level attributes to the
        object.

        Args:
            attrs: an AttributeContainer
        '''
        self.attrs.add_container(attrs)

    def add_detection(self, obj, frame_number=None):
        '''Adds the DetectedObject to the object.

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

        self.frames[obj.frame_number] = obj

    def add_detections(self, objects):
        '''Adds the DetectedObjects to the video.

        The DetectedObjects must have their `frame_number`s set.

        Args:
            objects: a DetectedObjectContainer
        '''
        for obj in objects:
            self.add_detection(obj)

    def add_child_object(self, obj):
        '''Adds the Object as a child of this object.

        Args:
            obj: an Object, which must have its `uuid` set
        '''
        if obj.uuid is None:
            raise ValueError("Object must have its `uuid` set")

        self.child_objects.add(obj.uuid)

    def clear_child_objects(self):
        '''Removes all child objects from the event.'''
        self.child_objects = set()

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
        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs an Object from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an Object
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
                int(fn): DetectedObject.from_dict(do)
                for fn, do in iteritems(frames)
            }

        return cls(
            label=d.get("label", None),
            confidence=d.get("confidence", None),
            support=support,
            index=d.get("index", None),
            uuid=d.get("uuid", None),
            frames=frames,
            attrs=attrs,
            child_objects=d.get("child_objects", None),
        )


class ObjectContainer(Container):
    '''A `Container` of `Object`s.'''

    _ELE_CLS = Object
    _ELE_CLS_FIELD = "_OBJ_CLS"
    _ELE_ATTR = "objects"

    def __init__(self, schema=None, **kwargs):
        '''Creates an ObjectContainer instance.

        Args:
            schema: an optional ObjectContainerSchema to enforce on the objects
                in this container. By default, no schema is enforced
            **kwargs: valid keyword arguments for Container()

        Raises:
            ObjectContainerSchemaError: if a schema was provided but the
                objects added to the container violate it
        '''
        super(ObjectContainer, self).__init__(**kwargs)
        self.schema = None
        if schema is not None:
            self.set_schema(schema)

    @property
    def has_schema(self):
        '''Whether the container has an enforced schema.'''
        return self.schema is not None

    def add(self, obj):
        '''Adds an object to the container.

        Args:
            obj: an Object

        Raises:
            ObjectContainerSchemaError: if this container has a schema enforced
                and the given object violates it
        '''
        if self.has_schema:
            self._validate_object(obj)

        super(ObjectContainer, self).add(obj)

    def add_container(self, objects):
        '''Adds the objects in the given container to this container.

        Args:
            objects: an ObjectContainer instance

        Raises:
            ObjectContainerSchemaError: if this container has a schema enforced
                and an object in the given container violates it
        '''
        if self.has_schema:
            for obj in objects:
                self._validate_object(obj)

        super(ObjectContainer, self).add_container(objects)

    def get_labels(self):
        '''Returns a set containing the labels of the `Object`s.

        Returns:
            a set of labels
        '''
        return set(obj.label for obj in self)

    def sort_by_confidence(self, reverse=False):
        '''Sorts the `Object`s by confidence.

        `Object`s whose confidence is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("confidence", reverse=reverse)

    def sort_by_index(self, reverse=False):
        '''Sorts the `Object`s by index.

        `Object`s whose index is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("index", reverse=reverse)

    def filter_by_schema(self, schema):
        '''Filters the objects/attributes from this container that are not
        compliant with the given schema.

        Args:
            schema: an ObjectContainerSchema
        '''
        # Filter by `Object` label
        filter_func = lambda obj: schema.has_object_label(obj.label)
        self.filter_elements([filter_func])

        # Iterate over `Object`s
        for obj in self:
            # Filter static attributes
            if obj.has_object_attributes:
                obj_schema = schema.get_object_schema(obj.label)
                obj.attrs.filter_by_schema(obj_schema)

            # Iterate over `DetectedObject`s
            del_frames = set()
            for frame_number, dobj in iteritems(obj.frames):
                if dobj.label is not None and dobj.label not in schema.objects:
                    # This `DetectedObject` has a disallowed `label`, delete it
                    del_frames.add(frame_number)
                elif dobj.has_attributes:
                    # Filter dynamic attributes
                    dlabel = (
                        dobj.label if dobj.label is not None
                        else obj.label)
                    obj_schema = schema.get_object_schema(dlabel)
                    dobj.attrs.filter_by_schema(obj_schema)

            # Delete `DetectedObject`s with disallowed labels
            if del_frames:
                obj.frames = {
                    fn: obj for fn, obj in iteritems(obj.frames)
                    if fn not in del_frames
                }

    def remove_objects_without_attrs(self, labels=None):
        '''Removes objects from this container that do not have attributes.

        Args:
            labels: an optional list of object `label` strings to which to
                restrict attention when filtering. By default, all objects are
                processed
        '''
        filter_func = lambda obj: (
            (labels is not None and obj.label not in labels)
            or obj.has_attributes)
        self.filter_elements([filter_func])

    def get_schema(self):
        '''Gets the current enforced schema for the container, or None if no
        schema is enforced.

        Returns:
            an ObjectContainerSchema
        '''
        return self.schema

    def get_active_schema(self):
        '''Returns an ObjectContainerSchema describing the active schema of the
        container.

        Returns:
            an ObjectContainerSchema
        '''
        return ObjectContainerSchema.build_active_schema(self)

    def set_schema(self, schema, filter_by_schema=False):
        '''Sets the enforced schema to the given ObjectContainerSchema.

        Args:
            schema: the ObjectContainerSchema to use
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
        _attrs += super(ObjectContainer, self).attributes()
        return _attrs

    def _validate_object(self, obj):
        if self.has_schema:
            self.schema.validate_object(obj)

    def _validate_schema(self):
        if self.has_schema:
            for obj in self:
                self._validate_object(obj)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ObjectContainer from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a ObjectContainer
        '''
        objects = super(ObjectContainer, cls).from_dict(d)

        schema = d.get("schema", None)
        if schema is not None:
            objects.set_schema(ObjectContainerSchema.from_dict(schema))

        return objects


class ObjectContainerSchema(Serializable):
    '''Schema for `ObjectContainer`s and `DetectedObjectContainer`s.'''

    def __init__(self, schema=None):
        '''Creates an ObjectContainerSchema instance.

        Args:
            schema: a dictionary mapping object labels to
                AttributeContainerSchema instances. By default, an empty schema
                is created
        '''
        self.schema = defaultdict(AttributeContainerSchema)
        if schema is not None:
            self.schema.update(schema)

    def has_object_label(self, label):
        '''Whether the schema has an object with the given label.

        Args:
            label: the object label

        Returns:
            True/False
        '''
        return label in self.schema

    def has_object_attribute(self, label, obj_attr_name):
        '''Whether the schema has an object with the given label with an
        attribute of the given name.

        Args:
            label: the object label
            obj_attr_name: the name of the object attribute

        Returns:
            True/False
        '''
        if not self.has_object_label(label):
            return False

        return self.schema[label].has_attribute(obj_attr_name)

    def get_object_schema(self, label):
        '''Gets the AttributeContainerSchema for the object with the given
        label.

        Args:
            label: the object label

        Returns:
            an AttributeContainerSchema
        '''
        self.validate_object_label(label)
        return self.schema[label]

    def get_object_attribute_schema(self, label, obj_attr_name):
        '''Gets the AttributeSchema for the attribute of the given name for the
        object with the given label.

        Args:
            label: the object label
            obj_attr_name: the name of the object attribute

        Returns:
            the AttributeSchema
        '''
        obj_attrs_schema = self.get_object_schema(label)
        return obj_attrs_schema.get_attribute_schema(obj_attr_name)

    def get_object_attribute_class(self, label, obj_attr_name):
        '''Gets the `Attribute` class for the attribute of the given name for
        the object with the given label.

        Args:
            label: the object label
            obj_attr_name: the name of the object attribute

        Returns:
            the Attribute
        '''
        self.validate_object_label(label)
        return self.schema[label].get_attribute_class(obj_attr_name)

    def add_object_label(self, label):
        '''Adds the given object label to the schema.

        ArgsL:
            label: an object label
        '''
        self.schema[label]  # adds key to defaultdict #pylint: disable=W0104

    def add_object_attribute(self, label, obj_attr):
        '''Adds the Attribute for the object with the given label to the
        schema.

        Args:
            label: an object label
            obj_attr: an Attribute
        '''
        self.schema[label].add_attribute(obj_attr)

    def add_object_attributes(self, label, obj_attrs):
        '''Adds the AttributeContainer for the object with the given label to
        the schema.

        Args:
            label: an object label
            obj_attrs: an AttributeContainer
        '''
        self.schema[label].add_attributes(obj_attrs)

    def add_object(self, obj):
        '''Adds the Object or DetectedObject to the schema.

        Args:
            obj: an Object or DetectedObject
        '''
        if isinstance(obj, Object):
            self._add_object(obj)
        else:
            self._add_detected_object(obj)

    def add_objects(self, objects):
        '''Adds the ObjectContainer or DetectedObjectContainer to the schema.

        Args:
            objects: an ObjectContainer or DetectedObjectContainer
        '''
        for obj in objects:
            self.add_object(obj)

    def merge_schema(self, schema):
        '''Merges the given ObjectContainerSchema into this schema.

        Args:
            schema: an ObjectContainerSchema
        '''
        for k, v in iteritems(schema.schema):
            self.schema[k].merge_schema(v)

    def is_valid_object_label(self, label):
        '''Whether the object label is compliant with the schema.

        Args:
            label: an object label

        Returns:
            True/False
        '''
        try:
            self.validate_object_label(label)
            return True
        except:
            return False

    def is_valid_object_attribute(self, label, obj_attr):
        '''Whether the object attribute for the object with the given label is
        compliant with the schema.

        Args:
            label: an object label
            obj_attr: an Attribute

        Returns:
            True/False
        '''
        try:
            self.validate_object_attribute(label, obj_attr)
            return True
        except:
            return False

    def is_valid_object(self, obj):
        '''Whether the `Object` or `DetectedObject` is compliant with the
        schema.

        Args:
            obj: an Object or DetectedObject

        Returns:
            True/False
        '''
        try:
            self.validate_object(obj)
            return True
        except:
            return False

    def validate_object_label(self, label, allow_none_label=False):
        '''Validates that the object label is compliant with the schema.

        Args:
            label: an object label
            allow_none: whether to allow `label == None`. By default, this is
                False

        Raises:
            ObjectContainerSchemaError: if the object label violates the schema
        '''
        if label is None and not allow_none_label:
            raise ObjectContainerSchemaError(
                "None object label is not allowed by the schema")

        if label not in self.schema:
            raise ObjectContainerSchemaError(
                "Object label '%s' is not allowed by the schema" % label)

    def validate_object_attribute(self, label, obj_attr):
        '''Validates that the object attribute for the given label is compliant
        with the schema.

        Args:
            label: an object label
            obj_attr: an Attribute

        Raises:
            ObjectContainerSchemaError: if the object label violates the schema
            AttributeContainerSchemaError: if the object attribute violates
                the schema
        '''
        self.validate_object_label(label)
        self.schema[label].validate_attribute(obj_attr)

    def validate_object(self, obj, allow_none_label=False):
        '''Validates that the `Object` or `DetectedObject` is compliant with
        the schema.

        Args:
            obj: an Object or DetectedObject
            allow_none: whether to allow `label == None`. By default, this is
                False. Objects with a top-level label are always allowed to
                have detections with no label set

        Raises:
            ObjectContainerSchemaError: if the object's label violates the
                schema
            AttributeContainerSchemaError: if any attributes of the object
                violate the schema
        '''
        if isinstance(obj, Object):
            self._validate_object(obj, allow_none_label)
        else:
            self._validate_detected_object(obj, allow_none_label)

    @classmethod
    def build_active_schema(cls, objects):
        '''Builds an ObjectContainerSchema that describes the active schema of
        the objects.

        Args:
            objects: an ObjectContainer or DetectedObjectContainer

        Returns:
            an ObjectContainerSchema
        '''
        schema = cls()
        schema.add_objects(objects)
        return schema

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ObjectContainerSchema from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an ObjectContainerSchema
        '''
        schema = d.get("schema", None)
        if schema is not None:
            schema = {
                k: AttributeContainerSchema.from_dict(v)
                for k, v in iteritems(schema)
            }

        return cls(schema=schema)

    def _add_detected_object(self, dobj, label=None):
        # Add label
        if dobj.label is not None:
            label = dobj.label
            self.add_object_label(dobj.label)

        # Add attributes
        self.add_object_attributes(label, dobj.attrs)

    def _add_object(self, obj):
        # Add label
        self.add_object_label(obj.label)

        # Add attributes
        self.add_object_attributes(obj.label, obj.attrs)

        # Add DetectedObjects
        for dobj in obj.iter_detections():
            self._add_detected_object(dobj, label=obj.label)

    def _validate_detected_object(self, dobj, allow_none_label):
        # Validate label
        self.validate_object_label(
            dobj.label, allow_none_label=allow_none_label)

        # Validate attributes
        for obj_attr in dobj.attrs:
            self.validate_object_attribute(dobj.label, obj_attr)

    def _validate_object(self, obj, allow_none_label):
        # Validate label
        self.validate_object_label(
            obj.label, allow_none_label=allow_none_label)

        # Validate attributes
        for obj_attr in obj.attrs:
            self.validate_object_attribute(obj.label, obj_attr)

        # If the `Object` has a top-level `label`, its okay for the
        # `DetectedObject`s to have no `label`
        allow_none_label |= obj.label is not None

        # Validate DetectedObjects
        for dobj in obj.iter_detections():
            self._validate_detected_object(dobj, allow_none_label)


class ObjectContainerSchemaError(Exception):
    '''Error raised when an ObjectContainerSchema is violated.'''
    pass


class ObjectCount(Serializable):
    '''The number of instances of an object found in an image.'''

    def __init__(self, label, count):
        '''Creates an ObjectCount instance.

        Args:
            label: the label
            count: the count
        '''
        self.label = label
        self.count = count

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ObjectCount from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an ObjectCount
        '''
        return ObjectCount(d["label"], d["count"])


class ObjectCounts(Container):
    '''Container for counting objects in an image.'''

    _ELE_CLS = ObjectCount
    _ELE_ATTR = "counts"


class EvaluationType(object):
    '''Enumeration representing the type of evaluation an object label is
    intended for. This enables evaluation of false negatives on a subset of
    the labels used for evaluating false positives.

    Attributes:
        RECALL: this object is part of the subset that MUST be detected. If it
            is not, it is considered a false negative
        PRECISION: this object MAY be detected, and if so, is marked as a true
            positive, however, if it is not, it is NOT considered a false
            negative
    '''

    RECALL = "RECALL"
    PRECISION = "PRECISION"
