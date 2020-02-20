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
from future.utils import iteritems
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from copy import deepcopy

import eta.core.data as etad
import eta.core.frameutils as etaf
import eta.core.geometry as etag
import eta.core.labels as etal
import eta.core.serial as etas
import eta.core.utils as etau


class DetectedObject(etal.Labels, etag.HasBoundingBox):
    '''A detected object in an image or frame of a video.

    `DetectedObject`s are spatial concepts that describe information about an
    object in a particular image or a particular frame of a video.
    `DetectedObject`s can have labels with confidences, bounding boxes,
    instance masks, and one or more additional attributes describing their
    properties.

    Attributes:
        type: the fully-qualified class name of the object
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
        event_uuids: (optional) a set of a Event uuids to which the object
            belongs
        attrs: (optional) an AttributeContainer of attributes for the object
    '''

    def __init__(
            self, label=None, bounding_box=None, mask=None, confidence=None,
            top_k_probs=None, index=None, score=None, frame_number=None,
            index_in_frame=None, eval_type=None, event_uuids=None,
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
            frame_number: (optional) the frame number in which this object was
                detected
            index_in_frame: (optional) the index of the object in the frame
                where it was detected
            eval_type: (optional) an EvaluationType value
            event_uuids: (optional) a set of Event uuids to which the object
                belongs
            attrs: (optional) an AttributeContainer of attributes for the
                object
        '''
        self.type = etau.get_class_name(self)
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
        self.event_uuids = set(event_uuids or [])
        self.attrs = attrs or etad.AttributeContainer()
        self._meta = None  # Usable by clients to store temporary metadata

    @property
    def is_empty(self):
        '''Whether this object has no labels of any kind.'''
        return not (
            self.has_label or self.has_bounding_box or self.has_mask or
            self.has_confidence or self.has_top_k_probs or self.has_index or
            self.has_frame_number or self.has_attributes)

    @property
    def has_label(self):
        '''Whether this object has a label.'''
        return self.label is not None

    @property
    def has_bounding_box(self):
        '''Whether this object has a bounding box.'''
        return self.bounding_box is not None

    @property
    def has_mask(self):
        '''Whether this object has a segmentation mask.'''
        return self.mask is not None

    @property
    def has_confidence(self):
        '''Whether this object has a confidence.'''
        return self.confidence is not None

    @property
    def has_top_k_probs(self):
        '''Whether this object has top-k probabilities.'''
        return self.top_k_probs is not None

    @property
    def has_index(self):
        '''Whether this object has an index.'''
        return self.index is not None

    @property
    def has_frame_number(self):
        '''Whether this object has a frame number.'''
        return self.frame_number is not None

    @property
    def has_attributes(self):
        '''Whether this object has attributes.'''
        return bool(self.attrs)

    @classmethod
    def get_schema_cls(cls):
        '''Gets the schema class for `DetectedObject`s.

        Returns:
            the LabelsSchema class
        '''
        return ObjectSchema

    def get_bounding_box(self):
        '''Returns the BoundingBox for the object.

        Returns:
             a BoundingBox
        '''
        return self.bounding_box

    def clear_attributes(self):
        '''Removes all attributes from the object.'''
        self.attrs = etad.AttributeContainer()

    def add_attribute(self, attr):
        '''Adds the attribute to the object.

        Args:
            attr: an Attribute
        '''
        self.attrs.add(attr)

    def add_attributes(self, attrs):
        '''Adds the attributes to the object.

        Args:
            attrs: an AttributeContainer
        '''
        self.attrs.add_container(attrs)

    def filter_by_schema(self, schema, allow_none_label=False):
        '''Filters the object by the given schema.

        The `label` of the DetectedObject must match the provided schema. Or,
        it can be `None` when `allow_none_label == True`.

        Args:
            schema: an ObjectSchema
            allow_none_label: whether to allow the object label to be `None`.
                By default, this is False

        Raises:
            LabelsSchemaError: if the object label does not match the schema
        '''
        if self.label is None:
            if not allow_none_label:
                raise ObjectSchemaError(
                    "None object label is not allowed by the schema")
        elif self.label != schema.get_label():
            raise ObjectSchemaError(
                "Label '%s' does not match object schema" % self.label)

        self.attrs.filter_by_schema(schema.frames)

    def attributes(self):
        '''Returns the list of attributes to serialize.

        Returns:
            a list of attribute names
        '''
        _attrs = ["type"]
        _noneable_attrs = [
            "label", "bounding_box", "mask", "confidence", "top_k_probs",
            "index", "score", "frame_number", "index_in_frame", "eval_type"]
        _attrs.extend(
            [a for a in _noneable_attrs if getattr(self, a) is not None])
        if self.event_uuids:
            _attrs.append("event_uuids")
        if self.attrs:
            _attrs.append("attrs")
        return _attrs

    @classmethod
    def _from_dict(cls, d):
        '''Internal implementation of `from_dict()`.

        Subclasses should implement this method, NOT `from_dict()`.

        Args:
            d: a JSON dictionary

        Returns:
            a DetectedObject
        '''
        bounding_box = d.get("bounding_box", None)
        if bounding_box is not None:
            bounding_box = etag.BoundingBox.from_dict(bounding_box)

        mask = d.get("mask", None)
        if mask is not None:
            mask = etas.deserialize_numpy_array(mask)

        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainer.from_dict(attrs)

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
            event_uuids=set(d.get("event_uuids", []))
        )

    @classmethod
    def from_dict(cls, d):
        '''Constructs a DetectedObject from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a DetectedObject
        '''
        if "type" in d:
            obj_cls = etau.get_class(d["type"])
        else:
            obj_cls = cls

        return obj_cls._from_dict(d)


class DetectedObjectContainer(etal.LabelsContainer):
    '''An `eta.core.serial.Container` of `DetectedObject`s.'''

    _ELE_CLS = DetectedObject
    _ELE_CLS_FIELD = "_OBJ_CLS"
    _ELE_ATTR = "objects"

    def get_labels(self):
        '''Returns a set containing the labels of the `DetectedObject`s.

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
        '''Filters the objects in the container by the given schema.

        Args:
            schema: an ObjectContainerSchema
        '''
        # Remove objects with invalid labels
        filter_func = lambda obj: schema.has_object_label(obj.label)
        self.filter_elements([filter_func])

        # Filter objects by their schemas
        for obj in self:
            obj_schema = schema.get_object_schema(obj.label)
            obj.filter_by_schema(obj_schema)

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


class VideoObject(etal.Labels, etal.HasLabelsSupport, etal.HasFramewiseView):
    '''A spatiotemporal object in a video.

    `VideoObject`s are spatiotemporal concepts that describe information about
    an object over multiple frames in a video. `VideoObject`s can have labels
    with confidences, object-level attributes that apply to the object over all
    frames, frame-level attributes such as bounding boxes and attributes that
    apply to individual frames, and child objects.

    Attributes:
        type: the fully-qualified class name of the object
        label: (optional) the object label
        confidence: (optional) label confidence in [0, 1]
        support: a FrameRanges instance describing the support of the object
        index: (optional) an index assigned to the object
        uuid: (optional) a UUID assigned to the object
        attrs: (optional) AttributeContainer of object-level attributes of the
            object
        frames: dictionary mapping frame numbers to DetectedObject instances
            describing the frame-level attributes of the object
        child_objects: (optional) a set of UUIDs of child `VideoObject`s
    '''

    def __init__(
            self, label=None, confidence=None, support=None, index=None,
            uuid=None, attrs=None, frames=None, child_objects=None):
        '''Creates a VideoObject instance.

        Args:
            label: (optional) the object label
            confidence: (optional) the label confidence in [0, 1]
            support: (optional) a FrameRanges instance describing the frozen
                support of the object
            index: (optional) an index assigned to the object
            uuid: (optional) a UUID assigned to the object
            attrs: (optional) an AttributeContainer of object-level attributes
            frames: (optional) a dictionary mapping frame numbers to
                DetectedObject instances
            child_objects: (optional) a set of UUIDs of child `VideoObject`s
        '''
        self.type = etau.get_class_name(self)
        self.label = label
        self.confidence = confidence
        self.index = index
        self.uuid = uuid
        self.attrs = attrs or etad.AttributeContainer()
        self.frames = frames or {}
        self.child_objects = set(child_objects or [])
        etal.HasLabelsSupport.__init__(self, support=support)

    @property
    def is_empty(self):
        '''Whether this instance has no labels of any kind.'''
        return False

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

    @property
    def has_child_objects(self):
        '''Whether the object has at least one child VideoObject.'''
        return bool(self.child_objects)

    def iter_attributes(self):
        '''Returns an iterator over the object-level attributes of the object.

        Returns:
            an iterator over `Attribute`s
        '''
        return iter(self.attrs)

    def iter_detections(self):
        '''Returns an iterator over the `DetectedObject`s for each frame of the
        object.

        The frames are traversed in sorted order.

        Returns:
            an iterator over `DetectedObject`s
        '''
        for frame_number in sorted(self.frames):
            yield self.frames[frame_number]

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

    def add_detection(self, obj, frame_number=None, clean=True):
        '''Adds the detection to the object.

        Args:
            obj: a DetectedObject
            frame_number: a frame number. If omitted, the DetectedObject must
                have its `frame_number` set
            clean: whether to set the `label` and `index` fields of the
                DetectedObject to `None`. By default, this is True
        '''
        self._add_detected_object(obj, frame_number, clean)

    def add_detections(self, objects, clean=True):
        '''Adds the detections to the object.

        The `DetectedObject`s must have their `frame_number`s set.

        Args:
            objects: a DetectedObjectContainer
            clean: whether to set the `label` and `index` fields of the
                `DetectedObject`s to `None`. By default, this is True
        '''
        self._add_detected_objects(objects, clean)

    def add_child_object(self, obj):
        '''Adds the VideoObject as a child of this object.

        Args:
            obj: a VideoObject, which must have its `uuid` set
        '''
        if obj.uuid is None:
            raise ValueError("VideoObject must have its `uuid` set")

        self.child_objects.add(obj.uuid)

    def clear_attributes(self):
        '''Removes all attributes of any kind from the object.'''
        self.clear_object_attributes()
        self.clear_frame_attributes()

    def clear_object_attributes(self):
        '''Removes all object-level attributes from the object.'''
        self.attrs = etad.AttributeContainer()

    def clear_frame_attributes(self):
        '''Removes all frame-level attributes from the object.'''
        for obj in self.iter_detections():
            obj.clear_attributes()

    def clear_detections(self):
        '''Removes all `DetectedObject`s from the object.'''
        self.frames = {}

    def clear_child_objects(self):
        '''Removes all child objects from the object.'''
        self.child_objects = set()

    def render_framewise_labels(self):
        '''Renders a framewise copy of the object.

        Returns:
            a VideoObject whose labels are all contained in `DetectedObject`s
        '''
        renderer = VideoObjectFrameRenderer(self)
        frames = renderer.render_all_frames()
        return VideoObject(frames=frames)

    def filter_by_schema(self, schema, objects=None):
        '''Filters the object by the given schema.

        Args:
            schema: an ObjectSchema
            objects: an optional dictionary mapping uuids to `VideoObject`s. If
                provided, child objects will be filtered by their respective
                schemas

        Raises:
            LabelsSchemaError: if the object label does not match the schema
        '''
        schema.validate_label(self.label)
        self.attrs.filter_by_schema(schema.attrs)
        for dobj in self.iter_detections():
            dobj.filter_by_schema(schema, allow_none_label=True)

        # @todo children...
        '''
        # Filter child objects
        if objects:
            for uuid in self.child_objects:
                if uuid in objects:
                    child_obj = objects[uuid]
                    if not schema.has_label(child_obj.label):
                        self.child_objects.remove(uuid)
                    else:
                        child_obj.filter_by_schema(
                            schema.get_object_schema(child_obj.label))
        '''

    def attributes(self):
        '''Returns the list of attributes to serialize.

        Returns:
            a list of attrinutes
        '''
        _attrs = ["type"]
        if self.label is not None:
            _attrs.append("label")
        if self.confidence is not None:
            _attrs.append("confidence")
        if self.is_support_frozen:
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
    def _from_dict(cls, d):
        '''Internal implementation of `from_dict()`.

        Subclasses should implement this method, NOT `from_dict()`.

        Args:
            d: a JSON dictionary

        Returns:
            a VideoObject
        '''
        support = d.get("support", None)
        if support is not None:
            support = etaf.FrameRanges.from_dict(support)

        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainer.from_dict(attrs)

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

    @classmethod
    def from_dict(cls, d):
        '''Constructs a VideoObject from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a VideoObject
        '''
        if "type" in d:
            obj_cls = etau.get_class(d["type"])
        else:
            obj_cls = cls

        return obj_cls._from_dict(d)

    def _add_detected_object(self, obj, frame_number, clean):
        if frame_number is None:
            if not obj.has_frame_number:
                raise ValueError(
                    "Either `frame_number` must be provided or the "
                    "DetectedObject must have its `frame_number` set")

            frame_number = obj.frame_number

        if clean:
            obj.label = None
            obj.index = None

        obj.frame_number = frame_number
        self.frames[frame_number] = obj

    def _add_detected_objects(self, objects, clean):
        for obj in objects:
            self._add_detected_object(obj, None, clean)

    def _compute_support(self):
        return etaf.FrameRanges.from_iterable(self.frames.keys())


class VideoObjectContainer(etal.LabelsContainer):
    '''An `eta.core.serial.Container` of `VideoObject`s.'''

    _ELE_CLS = VideoObject
    _ELE_CLS_FIELD = "_OBJ_CLS"
    _ELE_ATTR = "objects"

    def get_labels(self):
        '''Returns a set containing the labels of the `VideoObject`s.

        Returns:
            a set of labels
        '''
        return set(obj.label for obj in self)

    def sort_by_confidence(self, reverse=False):
        '''Sorts the `VideoObject`s by confidence.

        `VideoObject`s whose confidence is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("confidence", reverse=reverse)

    def sort_by_index(self, reverse=False):
        '''Sorts the `VideoObject`s by index.

        `VideoObject`s whose index is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("index", reverse=reverse)

    def filter_by_schema(self, schema, objects=None):
        '''Filters the objects in the container by the given schema.

        Args:
            schema: an ObjectContainerSchema
            objects: an optional dictionary mapping uuids to `VideoObject`s. If
                provided, child objects will be filtered by their respective
                schemas
        '''
        # Remove objects with invalid labels
        filter_func = lambda obj: schema.has_object_label(obj.label)
        self.filter_elements([filter_func])

        # Filter objects by their schemas
        for obj in self:
            obj_schema = schema.get_object_schema(obj.label)
            obj.filter_by_schema(obj_schema, objects=objects)

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


class ObjectSchema(etal.LabelsSchema):
    '''Schema for `VideoObject`s and `DetectedObject`s.

    Attributes:
        label: the object label
        attrs: an AttributeContainerSchema describing the object-level
            attributes of the object
        frames: an AttributeContainerSchema describing the frame-level
            attributes of the object
    '''

    def __init__(self, label, attrs=None, frames=None):
        '''Creates an ObjectSchema instance.

        Args:
            label: the object label
            attrs: (optional) an AttributeContainerSchema describing the
                object-level attributes of the object
            frames: (optional) an AttributeContainerSchema describing the
                frame-level attributes of the object
        '''
        self.label = label
        self.attrs = attrs or etad.AttributeContainerSchema()
        self.frames = frames or etad.AttributeContainerSchema()

    @property
    def is_empty(self):
        '''Whether this schema has no labels of any kind.'''
        return False

    def has_label(self, label):
        '''Whether the schema has the given object label.

        Args:
            label: the object label

        Returns:
            True/False
        '''
        return label == self.label

    def get_label(self):
        '''Gets the object label for the schema.

        Returns:
            the object label
        '''
        return self.label

    def has_object_attribute(self, attr_name):
        '''Whether the schema has an object-level Attribute of the given
        name.

        Args:
            attr_name: the name

        Returns:
            True/False
        '''
        return self.attrs.has_attribute(attr_name)

    def has_frame_attribute(self, attr_name):
        '''Whether the schema has a frame-level Attribute of the given name.

        Args:
            attr_name: the name

        Returns:
            True/False
        '''
        return self.frames.has_attribute(attr_name)

    def get_object_attribute_schema(self, attr_name):
        '''Gets the AttributeSchema for the object-level attribute of the given
        name.

        Args:
            attr_name: the name

        Returns:
            the AttributeSchema
        '''
        return self.attrs.get_attribute_schema(attr_name)

    def get_frame_attribute_schema(self, attr_name):
        '''Gets the AttributeSchema for the frame-level attribute of the given
        name.

        Args:
            attr_name: the name

        Returns:
            the AttributeSchema
        '''
        return self.frames.get_attribute_schema(attr_name)

    def get_object_attribute_class(self, attr_name):
        '''Gets the Attribute class for the object-level attribute of the given
        name.

        Args:
            attr_name: the name

        Returns:
            the Attribute
        '''
        return self.attrs.get_attribute_class(attr_name)

    def get_frame_attribute_class(self, attr_name):
        '''Gets the Attribute class for the frame-level attribute of the given
        name.

        Args:
            attr_name: the name

        Returns:
            the Attribute
        '''
        return self.frames.get_attribute_class(attr_name)

    def add_object_attribute(self, attr):
        '''Adds the object-level Attribute to the schema.

        Args:
            attr: an Attribute
        '''
        self.attrs.add_attribute(attr)

    def add_frame_attribute(self, attr):
        '''Adds the frame-level Attribute to the schema.

        Args:
            attr: an Attribute
        '''
        self.frames.add_attribute(attr)

    def add_object_attributes(self, attrs):
        '''Adds the AttributeContainer of object-level attributes to the
        schema.

        Args:
            attrs: an AttributeContainer
        '''
        self.attrs.add_attributes(attrs)

    def add_frame_attributes(self, attrs):
        '''Adds the AttributeContainer of frame-level attributes to the schema.

        Args:
            attrs: an AttributeContainer
        '''
        self.frames.add_attributes(attrs)

    def add_object(self, obj):
        '''Adds the object to the schema.

        Args:
            obj: a VideoObject or DetectedObject
        '''
        if isinstance(obj, DetectedObject):
            self._add_detected_object(obj)
        else:
            self._add_video_object(obj)

    def add_objects(self, objects):
        '''Adds the VideoObjectContainer or DetectedObjectContainer to the
        schema.

        Args:
            objects: an VideoObjectContainer or DetectedObjectContainer
        '''
        for obj in objects:
            self.add_object(obj)

    def is_valid_label(self, label):
        '''Whether the object label is compliant with the schema.

        Args:
            label: an object label

        Returns:
            True/False
        '''
        try:
            self.validate_label(label)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_object_attribute(self, attr):
        '''Whether the object-level attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Returns:
            True/False
        '''
        try:
            self.validate_object_attribute(attr)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_object_attributes(self, attrs):
        '''Whether the AttributeContainer of object-level attributes is
        compliant with the schema.

        Args:
            attrs: an AttributeContainer

        Returns:
            True/False
        '''
        try:
            self.validate_object_attributes(attrs)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_frame_attribute(self, attr):
        '''Whether the frame-level attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Returns:
            True/False
        '''
        try:
            self.validate_frame_attribute(attr)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_frame_attributes(self, attrs):
        '''Whether the AttributeContainer of frame-level attributes is
        compliant with the schema.

        Args:
            attrs: an AttributeContainer

        Returns:
            True/False
        '''
        try:
            self.validate_frame_attributes(attrs)
            return True
        except etal.LabelsSchemaError:
            return False

    def validate_label(self, label):
        '''Validates that the object label is compliant with the schema.

        Args:
            label: the label

        Raises:
            LabelsSchemaError: if the label violates the schema
        '''
        if label != self.label:
            raise ObjectSchemaError(
                "Label '%s' does not match object schema" % label)

    def validate_object_attribute(self, attr):
        '''Validates that the object-level attribute is compliant with the
        schema.

        Args:
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        '''
        self.attrs.validate_attribute(attr)

    def validate_object_attributes(self, attrs):
        '''Validates that the AttributeContainer of object-level attributes is
        compliant with the schema.

        Args:
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        '''
        self.attrs.validate(attrs)

    def validate_frame_attribute(self, attr):
        '''Validates that the frame-level attribute is compliant with the
        schema.

        Args:
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        '''
        self.frames.validate_attribute(attr)

    def validate_frame_attributes(self, attrs):
        '''Validates that the AttributeContainer of frame-level attributes is
        compliant with the schema.

        Args:
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        '''
        self.frames.validate(attrs)

    def validate(self, obj):
        '''Validates that the object is compliant with the schema.

        Args:
            obj: a VideoObject or DetectedObject

        Raises:
            LabelsSchemaError: if the object violates the schema
        '''
        if isinstance(obj, DetectedObject):
            self._validate_detected_object(obj)
        else:
            self._validate_video_object(obj)

    def validate_subset_of_schema(self, schema):
        '''Validates that this schema is a subset of the given schema.

        Args:
            schema: an ObjectSchema

        Raises:
            LabelsSchemaError: if this schema is not a subset of the given
                schema
        '''
        self.validate_schema_type(schema)

        if self.label != schema.label:
            raise ObjectSchemaError(
                "Expected object label '%s'; found '%s'" %
                (schema.label, self.label))

        self.attrs.validate_subset_of_schema(schema.attrs)
        self.frames.validate_subset_of_schema(schema.frames)

    def merge_schema(self, schema):
        '''Merges the given ObjectSchema into this schema.

        Args:
            schema: an ObjectSchema
        '''
        self.validate_label(schema.label)
        self.attrs.merge_schema(schema.attrs)
        self.frames.merge_schema(schema.frames)

    @classmethod
    def build_active_schema(cls, obj, objects=None):
        '''Builds an ObjectSchema that describes the active schema of the
        object.

        Args:
            obj: a VideoObject or DetectedObject
            objects: an optional dictionary mapping uuids to `VideoObject`s. If
                provided, the child objects of this object will be incorporated
                into the schema

        Returns:
            an ObjectSchema
        '''
        schema = cls(obj.label)
        schema.add_object(obj)

        # @todo children...
        '''
        # Child objects
        if objects:
            for uuid in obj.child_objects:
                if uuid in objects:
                    schema.add_object(objects[uuid])
        '''

        return schema

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Args:
            a list of attribute names
        '''
        _attrs = ["label"]
        if self.attrs:
            _attrs.append("attrs")
        if self.frames:
            _attrs.append("frames")
        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ObjectSchema from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an ObjectSchema
        '''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainerSchema.from_dict(attrs)

        frames = d.get("frames", None)
        if frames is not None:
            frames = etad.AttributeContainerSchema.from_dict(frames)

        return cls(d["label"], attrs=attrs, frames=frames)

    def _add_detected_object(self, dobj, validate_label=True):
        if validate_label:
            self.validate_label(dobj.label)

        self.add_frame_attributes(dobj.attrs)

    def _add_video_object(self, obj):
        self.validate_label(obj.label)
        self.add_object_attributes(obj.attrs)
        for dobj in obj.iter_detections():
            self._add_detected_object(dobj, validate_label=False)

    def _validate_detected_object(self, dobj, validate_label=True):
        if validate_label:
            self.validate_label(dobj.label)

        self.validate_frame_attributes(dobj.attrs)

    def _validate_video_object(self, obj):
        self.validate_label(obj.label)
        self.validate_object_attributes(obj.attrs)
        for dobj in obj.iter_detections():
            self._validate_detected_object(dobj, validate_label=False)

        # @todo children...
        '''
        # Validate child objects
        for child_object in obj.child_objects:
            self.validate(child_object)
        '''


class ObjectSchemaError(etal.LabelsSchemaError):
    '''Error raised when an ObjectSchema is violated.'''
    pass


class ObjectContainerSchema(etal.LabelsContainerSchema):
    '''Schema for `VideoObjectContainer`s and `DetectedObjectContainer`s.

    Attributes:
        schema: a dictionary mapping object labels to ObjectSchema instances
    '''

    def __init__(self, schema=None):
        '''Creates an ObjectContainerSchema instance.

        Args:
            schema: (optional) a dictionary mapping object labels to
                ObjectSchema instances
        '''
        self.schema = schema or {}

    @property
    def is_empty(self):
        '''Whether this schema has no labels of any kind.'''
        return not bool(self.schema)

    def iter_object_labels(self):
        '''Returns an iterator over the object labels in this schema.

        Returns:
            an iterator over object labels
        '''
        return iter(self.schema)

    def iter_objects(self):
        '''Returns an iterator over the (label, ObjectSchema) pairs in this
        schema.

        Returns:
            an iterator over (label, ObjectSchema) pairs
        '''
        return iteritems(self.schema)

    def has_object_label(self, label):
        '''Whether the schema has an object with the given label.

        Args:
            label: the object label

        Returns:
            True/False
        '''
        return label in self.schema

    def has_object_attribute(self, label, attr_name):
        '''Whether the schema has an object with the given label with an
        object-level attribute of the given name.

        Args:
            label: the object label
            attr_name: the name of the object-level attribute

        Returns:
            True/False
        '''
        if not self.has_object_label(label):
            return False

        return self.schema[label].has_object_attribute(attr_name)

    def has_frame_attribute(self, label, attr_name):
        '''Whether the schema has an object with the given label with a
        frame-level attribute of the given name.

        Args:
            label: the object label
            attr_name: the name of the frame-level attribute

        Returns:
            True/False
        '''
        if not self.has_object_label(label):
            return False

        return self.schema[label].has_frame_attribute(attr_name)

    def get_object_schema(self, label):
        '''Gets the ObjectSchema for the object with the given label.

        Args:
            label: the object label

        Returns:
            an ObjectSchema
        '''
        self.validate_object_label(label)
        return self.schema[label]

    def get_object_attribute_schema(self, label, attr_name):
        '''Gets the AttributeSchema for the object-level attribute of the given
        name for the object with the given label.

        Args:
            label: the object label
            attr_name: the name of the object-level attribute

        Returns:
            the AttributeSchema
        '''
        obj_schema = self.get_object_schema(label)
        return obj_schema.get_object_attribute_schema(attr_name)

    def get_frame_attribute_schema(self, label, attr_name):
        '''Gets the AttributeSchema for the frame-level attribute of the given
        name for the object with the given label.

        Args:
            label: the object label
            attr_name: the name of the frame-level attribute

        Returns:
            the AttributeSchema
        '''
        obj_schema = self.get_object_schema(label)
        return obj_schema.get_frame_attribute_schema(attr_name)

    def get_object_attribute_class(self, label, attr_name):
        '''Gets the Attribute class for the object-level attribute of the given
        name for the object with the given label.

        Args:
            label: the object label
            attr_name: the name of the object-level attribute

        Returns:
            the Attribute
        '''
        self.validate_object_label(label)
        return self.schema[label].get_object_attribute_class(attr_name)

    def get_frame_attribute_class(self, label, attr_name):
        '''Gets the Attribute class for the frame-level attribute of the given
        name for the object with the given label.

        Args:
            label: the object label
            attr_name: the name of the frame-level attribute

        Returns:
            the Attribute
        '''
        self.validate_object_label(label)
        return self.schema[label].get_frame_attribute_class(attr_name)

    def add_object_label(self, label):
        '''Adds the given object label to the schema.

        Args:
            label: an object label
        '''
        self._ensure_has_object_label(label)

    def add_object_attribute(self, label, attr):
        '''Adds the object-level Attribute for the object with the given label
        to the schema.

        Args:
            label: an object label
            attr: an object-level Attribute
        '''
        self._ensure_has_object_label(label)
        self.schema[label].add_object_attribute(attr)

    def add_frame_attribute(self, label, attr):
        '''Adds the frame-level Attribute for the object with the given label
        to the schema.

        Args:
            label: an object label
            attr: a frame-level Attribute
        '''
        self._ensure_has_object_label(label)
        self.schema[label].add_frame_attribute(attr)

    def add_object_attributes(self, label, attrs):
        '''Adds the AttributeContainer of object-level attributes for the
        object with the given label to the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer of object-level attributes
        '''
        self._ensure_has_object_label(label)
        self.schema[label].add_object_attributes(attrs)

    def add_frame_attributes(self, label, attrs):
        '''Adds the AttributeContainer of frame-level attributes for the object
        with the given label to the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer of frame-level attributes
        '''
        self._ensure_has_object_label(label)
        self.schema[label].add_frame_attributes(attrs)

    def add_object(self, obj):
        '''Adds the object to the schema.

        Args:
            obj: a VideoObject or DetectedObject
        '''
        self._ensure_has_object_label(obj.label)
        self.schema[obj.label].add_object(obj)

    def add_objects(self, objects):
        '''Adds the objects to the schema.

        Args:
            objects: a VideoObjectContainer or DetectedObjectContainer
        '''
        for obj in objects:
            self.add_object(obj)

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
        except etal.LabelsSchemaError:
            return False

    def is_valid_object_attribute(self, label, attr):
        '''Whether the object-level attribute for the object with the given
        label is compliant with the schema.

        Args:
            label: an object label
            attr: an object-level Attribute

        Returns:
            True/False
        '''
        try:
            self.validate_object_attribute(label, attr)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_object_attributes(self, label, attrs):
        '''Whether the object-level attributes for the object with the given
        label are compliant with the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer of object-level attributes

        Returns:
            True/False
        '''
        try:
            self.validate_object_attributes(label, attrs)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_frame_attribute(self, label, attr):
        '''Whether the frame-level attribute for the object with the given
        label is compliant with the schema.

        Args:
            label: an object label
            attr: a frame-level Attribute

        Returns:
            True/False
        '''
        try:
            self.validate_frame_attribute(label, attr)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_frame_attributes(self, label, attrs):
        '''Whether the frame-level attributes for the object with the given
        label are compliant with the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer of frame-level attributes

        Returns:
            True/False
        '''
        try:
            self.validate_frame_attributes(label, attrs)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_object(self, obj):
        '''Whether the object is compliant with the schema.

        Args:
            obj: a VideoObject or DetectedObject

        Returns:
            True/False
        '''
        try:
            self.validate_object(obj)
            return True
        except etal.LabelsSchemaError:
            return False

    def validate_object_label(self, label):
        '''Validates that the object label is compliant with the schema.

        Args:
            label: an object label

        Raises:
            LabelsSchemaError: if the object label violates the schema
        '''
        if label not in self.schema:
            raise ObjectContainerSchemaError(
                "Object label '%s' is not allowed by the schema" % label)

    def validate_object_attribute(self, label, attr):
        '''Validates that the object-level Attribute for the object with the
        given label is compliant with the schema.

        Args:
            label: an object label
            attr: an object-level Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        '''
        self.validate_object_label(label)
        self.schema[label].validate_object_attribute(attr)

    def validate_object_attributes(self, label, attrs):
        '''Validates that the AttributeContainer of object-level attributes for
        the object with the given label is compliant with the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer of object-level attributes

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        '''
        self.validate_object_label(label)
        self.schema[label].validate_object_attributes(attrs)

    def validate_frame_attribute(self, label, attr):
        '''Validates that the frame-level Attribute for the object with the
        given label is compliant with the schema.

        Args:
            label: an object label
            attr: a frame-level Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        '''
        self.validate_object_label(label)
        self.schema[label].validate_frame_attribute(attr)

    def validate_frame_attributes(self, label, attrs):
        '''Validates that the AttributeContainer of frame-level attributes for
        the object with the given label is compliant with the schema.

        Args:
            label: an object label
            attrs: an AttributeContainer of frame-level attributes

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        '''
        self.validate_object_label(label)
        self.schema[label].validate_frame_attributes(attrs)

    def validate_object(self, obj):
        '''Validates that the object is compliant with the schema.

        Args:
            obj: a VideoObject or DetectedObject

        Raises:
            LabelsSchemaError: if the object violates the schema
        '''
        self.validate_object_label(obj.label)
        self.schema[obj.label].validate(obj)

    def validate(self, objects):
        '''Validates that the objects are compliant with the schema.

        Args:
            objects: a VideoObjectContainer or DetectedObjectContainer

        Raises:
            LabelsSchemaError: if the objects violate the schema
        '''
        for obj in objects:
            self.validate_object(obj)

    def validate_subset_of_schema(self, schema):
        '''Validates that this schema is a subset of the given schema.

        Args:
            schema: an ObjectContainerSchema

        Raises:
            LabelsSchemaError: if this schema is not a subset of the given
                schema
        '''
        self.validate_schema_type(schema)

        for label, obj_schema in iteritems(self.schema):
            if not schema.has_object_label(label):
                raise ObjectContainerSchemaError(
                    "Object label '%s' does not appear in schema" % label)

            other_obj_schema = schema.get_object_schema(label)
            obj_schema.validate_subset_of_schema(other_obj_schema)

    def merge_schema(self, schema):
        '''Merges the given ObjectContainerSchema into this schema.

        Args:
            schema: an ObjectContainerSchema
        '''
        for label, obj_schema in schema.iter_objects():
            self._ensure_has_object_label(label)
            self.schema[label].merge_schema(obj_schema)

    @classmethod
    def build_active_schema(cls, objects):
        '''Builds an ObjectContainerSchema that describes the active schema
        of the objects.

        Args:
            objects: a VideoObjectContainer or DetectedObjectContainer

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
                label: ObjectSchema.from_dict(osd)
                for label, osd in iteritems(schema)
            }

        return cls(schema=schema)

    def _ensure_has_object_label(self, label):
        if not self.has_object_label(label):
            self.schema[label] = ObjectSchema(label)


class ObjectContainerSchemaError(etal.LabelsContainerSchemaError):
    '''Error raised when an ObjectContainerSchema is violated.'''
    pass


class VideoObjectFrameRenderer(etal.LabelsFrameRenderer):
    '''Class for rendering labels for a VideoObject at the frame-level.'''

    def __init__(self, obj):
        '''Creates an VideoObjectFrameRenderer instance.

        Args:
            obj: a VideoObject
        '''
        self._obj = obj

    def render_frame(self, frame_number):
        '''Renders the VideoObject for the given frame.

        Args:
            frame_number: the frame number

        Returns:
            a DetectedObject, or None if no labels exist for the given frame
        '''
        if frame_number not in self._obj.support:
            return None

        obj_attrs = self._get_object_attrs()
        return self._render_frame(frame_number, obj_attrs)

    def render_all_frames(self):
        '''Renders the VideoObject for all possible frames.

        Returns:
            a dictionary mapping frame numbers to DetectedObject instances
        '''
        obj_attrs = self._get_object_attrs()

        dobjs_map = {}
        for frame_number in self._obj.support:
            dobjs_map[frame_number] = self._render_frame(
                frame_number, obj_attrs)

        return dobjs_map

    def _render_frame(self, frame_number, obj_attrs):
        # Base DetectedObject
        if frame_number in self._obj.frames:
            dobj = deepcopy(self._obj.frames[frame_number])
        else:
            dobj = DetectedObject(frame_number=frame_number)

        # Render object-level attributes
        if obj_attrs is not None:
            # Prepend object-level attributes
            dobj.attrs.prepend_container(obj_attrs)

        # Inherit available object-level metadata
        if self._obj.label is not None:
            dobj.label = self._obj.label
        if self._obj.confidence is not None:
            dobj.confidence = self._obj.confidence
        if self._obj.index is not None:
            dobj.index = self._obj.index

        return dobj

    def _get_object_attrs(self):
        if not self._obj.has_object_attributes:
            return None

        return deepcopy(self._obj.attrs)


class VideoObjectContainerFrameRenderer(etal.LabelsContainerFrameRenderer):
    '''Class for rendering labels for a VideoObjectContainer at the
    frame-level.
    '''

    _FRAME_CONTAINER_CLS = DetectedObjectContainer
    _ELEMENT_RENDERER_CLS = VideoObjectFrameRenderer


class ObjectCount(etas.Serializable):
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
        return cls(d["label"], d["count"])


class ObjectCounts(etas.Container):
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
