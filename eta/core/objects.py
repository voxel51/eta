"""
Core tools and data structures for working with objects in images and videos.

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
from copy import deepcopy
import logging

import eta.core.data as etad
import eta.core.frameutils as etaf
import eta.core.geometry as etag
import eta.core.labels as etal
import eta.core.serial as etas
import eta.core.utils as etau


logger = logging.getLogger(__name__)


class DetectedObject(etal.Labels, etag.HasBoundingBox):
    """A detected object in an image or frame of a video.

    :class:`DetectedObject` is a spatial concept that describes information
    about an object in a particular image or a particular frame of a video.
    A :class:`DetectedObject` can have a name (i.e., a class), a label with a
    confidence, a bounding box, an instance mask, and one or more additional
    attributes describing its properties.

    Attributes:
        type: the fully-qualified class name of the object
        label: (optional) object label
        bounding_box: (optional) a :class:`eta.core.geometry.BoundingBox`
            around the object
        mask: (optional) a mask for the object within its bounding box
        confidence: (optional) a confidence for the object, in ``[0, 1]``
        name: (optional) the name of the object, e.g., ``ground_truth`` or the
            name of the model that produced it
        top_k_probs: (optional) dictionary mapping labels to probabilities
        index: (optional) an index assigned to the object
        score: (optional) a multipurpose score for the object
        frame_number: (optional) the frame number in which the object was
            detected
        index_in_frame: (optional) the index of the object in the frame where
            it was detected
        eval_type: (optional) an :class:`EvaluationType` value
        attrs: (optional) an :class:`eta.core.data.AttributeContainer` of
            attributes for the object
        tags: (optional) a list of tag strings

    Args:
        label (None): the object label
        bounding_box (None): a :class:`eta.core.geometry.BoundingBox` for the
            object
        mask (None): a numpy array describing the mask for the object within
            its bounding box
        confidence (None): a confidence for the detection, in ``[0, 1]``
        name (None): a name for the object, e.g., ``ground_truth`` or the name
            of the model that produced it
        top_k_probs (None): a dict mapping labels to probabilities
        index (None): an integer index assigned to the object
        score (None): a score for the object
        frame_number (None): the frame number in which this object was detected
        index_in_frame (None): the index of the object in the frame where it
            was detected
        eval_type (None): an :class:`EvaluationType` for the object
        attrs (None): an :class:`eta.core.data.AttributeContainer` of
            attributes for the object
        tags (None): a list of tag strings
    """

    def __init__(
        self,
        label=None,
        bounding_box=None,
        mask=None,
        confidence=None,
        name=None,
        top_k_probs=None,
        index=None,
        score=None,
        frame_number=None,
        index_in_frame=None,
        eval_type=None,
        attrs=None,
        tags=None,
    ):
        self.type = etau.get_class_name(self)
        self.label = label
        self.bounding_box = bounding_box
        self.mask = mask
        self.confidence = confidence
        self.name = name
        self.top_k_probs = top_k_probs
        self.index = index
        self.score = score
        self.frame_number = frame_number
        self.index_in_frame = index_in_frame
        self.eval_type = eval_type
        self.attrs = attrs or etad.AttributeContainer()
        self.tags = tags or []
        self._meta = None  # Usable by clients to store temporary metadata

    @property
    def is_empty(self):
        """Whether the object has no labels of any kind."""
        return not (
            self.has_label
            or self.has_bounding_box
            or self.has_mask
            or self.has_attributes
        )

    @property
    def has_label(self):
        """Whether the object has a ``label``."""
        return self.label is not None

    @property
    def has_bounding_box(self):
        """Whether the object has a ``bounding_box``."""
        return self.bounding_box is not None

    @property
    def has_mask(self):
        """Whether the object has a ``mask``."""
        return self.mask is not None

    @property
    def has_confidence(self):
        """Whether the object has a ``confidence``."""
        return self.confidence is not None

    @property
    def has_name(self):
        """Whether the object has a ``name``."""
        return self.name is not None

    @property
    def has_top_k_probs(self):
        """Whether the object has ``top_k_probs``."""
        return self.top_k_probs is not None

    @property
    def has_index(self):
        """Whether the object has an ``index``."""
        return self.index is not None

    @property
    def has_frame_number(self):
        """Whether the object has a ``frame_number``."""
        return self.frame_number is not None

    @property
    def has_attributes(self):
        """Whether the object has attributes."""
        return bool(self.attrs)

    @property
    def has_tags(self):
        """Whether the object has tags."""
        return bool(self.tags)

    @classmethod
    def get_schema_cls(cls):
        """Gets the schema class for :class:`DetectedObject`.

        Returns:
            the :class:`eta.core.labels.LabelsSchema` class
        """
        return ObjectSchema

    def get_bounding_box(self):
        """Returns the :class:`eta.core.geometry.BoundingBox` for the object.

        Returns:
             a :class:`eta.core.geometry.BoundingBox`
        """
        return self.bounding_box

    def get_index(self):
        """Returns the ``index`` of the object.

        Returns:
            the index, or None if the object has no index
        """
        return self.index

    def offset_index(self, offset):
        """Adds the given offset to the object's ``index``.

        If the object has no index, this does nothing.

        Args:
            offset: the integer offset
        """
        if self.has_index:
            self.index += offset

    def clear_index(self):
        """Clears the ``index`` of the object."""
        self.index = None

    def add_attribute(self, attr):
        """Adds the attribute to the object.

        Args:
            attr: an :class:`eta.core.data.Attribute`
        """
        self.attrs.add(attr)

    def add_attributes(self, attrs):
        """Adds the attributes to the object.

        Args:
            attrs: an :class:`eta.core.data.AttributeContainer`
        """
        self.attrs.add_container(attrs)

    def pop_attributes(self):
        """Pops the attributes from the object.

        Returns:
            an :class:`eta.core.data.AttributeContainer`
        """
        attrs = self.attrs
        self.clear_attributes()
        return attrs

    def clear_attributes(self):
        """Removes all attributes from the object."""
        self.attrs = etad.AttributeContainer()

    def filter_by_schema(self, schema, allow_none_label=False):
        """Filters the object by the given schema.

        The ``label`` of the :class:`DetectedObject` must match the provided
        schema. Or, it can be ``None`` when ``allow_none_label == True``.

        Args:
            schema: an :class:`ObjectSchema`
            allow_none_label (False): whether to allow the object label to be
                ``None``

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the object label
            does not match the schema
        """
        if self.label is None:
            if not allow_none_label:
                raise ObjectSchemaError(
                    "None object label is not allowed by the schema"
                )
        elif self.label != schema.get_label():
            raise ObjectSchemaError(
                "Label '%s' does not match object schema" % self.label
            )

        self.attrs.filter_by_schema(
            schema.frames, constant_schema=schema.attrs
        )

    def attributes(self):
        """Returns the list of attributes to serialize.

        Returns:
            a list of attribute names
        """
        _attrs = ["type"]
        _noneable_attrs = [
            "label",
            "bounding_box",
            "mask",
            "confidence",
            "name",
            "top_k_probs",
            "index",
            "score",
            "frame_number",
            "index_in_frame",
            "eval_type",
            "tags",
        ]
        _attrs.extend(
            [a for a in _noneable_attrs if getattr(self, a) is not None]
        )
        if self.attrs:
            _attrs.append("attrs")
        return _attrs

    @classmethod
    def _from_dict(cls, d):
        """Internal implementation of :meth:`DetectedObject.from_dict`.

        Subclasses should implement this method, not
        :meth:`DetectedObject.from_dict`.

        Args:
            d: a JSON dictionary

        Returns:
            a :class:`DetectedObject`
        """
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
            name=d.get("name", None),
            top_k_probs=d.get("top_k_probs", None),
            index=d.get("index", None),
            score=d.get("score", None),
            frame_number=d.get("frame_number", None),
            index_in_frame=d.get("index_in_frame", None),
            attrs=attrs,
            eval_type=d.get("eval_type", None),
            tags=d.get("tags", None),
        )

    @classmethod
    def from_dict(cls, d):
        """Constructs a :class:`DetectedObject` from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a :class:`DetectedObject`
        """
        if "type" in d:
            obj_cls = etau.get_class(d["type"])
        else:
            obj_cls = cls

        return obj_cls._from_dict(d)


class DetectedObjectContainer(etal.LabelsContainer):
    """An :class:`eta.core.serial.Container` of :class:`DetectedObject`
    instances.
    """

    _ELE_CLS = DetectedObject
    _ELE_CLS_FIELD = "_OBJ_CLS"
    _ELE_ATTR = "objects"

    def get_labels(self):
        """Returns the set of ``label`` values of all objects in the container.

        Returns:
            a set of labels
        """
        return set(dobj.label for dobj in self)

    def get_indexes(self):
        """Returns the set of ``index`` values of all objects in the container.

        ``None`` indexes are omitted.

        Returns:
            a set of indexes
        """
        return set(dobj.index for dobj in self if dobj.has_index)

    def offset_indexes(self, offset):
        """Adds the given offset to all objects with ``index`` values.

        Args:
            offset: the integer offset
        """
        for dobj in self:
            dobj.offset_index(offset)

    def clear_indexes(self):
        """Clears the ``index`` of all objects in the container.
        """
        for dobj in self:
            dobj.clear_index()

    def sort_by_confidence(self, reverse=False):
        """Sorts the :class:`DetectedObject` instances by confidence.

        Objects whose confidence is ``None`` are always put last.

        Args:
            reverse (False): whether to sort in descending order
        """
        self.sort_by("confidence", reverse=reverse)

    def sort_by_index(self, reverse=False):
        """Sorts the :class:`DetectedObject` instances by index.

        Objects whose index is ``None`` are always put last.

        Args:
            reverse (False): whether to sort in descending order
        """
        self.sort_by("index", reverse=reverse)

    def sort_by_score(self, reverse=False):
        """Sorts the :class:`DetectedObject` instances by score.

        Objects whose score is ``None`` are always put last.

        Args:
            reverse (False): whether to sort in descending order
        """
        self.sort_by("score", reverse=reverse)

    def sort_by_frame_number(self, reverse=False):
        """Sorts the :class:`DetectedObject` instances by frame number.

        Objects whose frame number is ``None`` are always put last.

        Args:
            reverse (False): whether to sort in descending order
        """
        self.sort_by("frame_number", reverse=reverse)

    def filter_by_schema(self, schema):
        """Filters the objects in the container by the given schema.

        Args:
            schema: an :class:`ObjectContainerSchema`
        """
        # Remove objects with invalid labels
        filter_func = lambda obj: schema.has_object_label(obj.label)
        self.filter_elements([filter_func])

        # Filter objects by their schemas
        for obj in self:
            obj_schema = schema.get_object_schema(obj.label)
            obj.filter_by_schema(obj_schema)

    def remove_objects_without_attrs(self, labels=None):
        """Removes objects from this container that do not have attributes.

        Args:
            labels (None): an optional list of object
                :class:`DetectedObject.label` strings to which to restrict
                attention when filtering. By default, all objects are processed
        """
        filter_func = lambda obj: (
            (labels is not None and obj.label not in labels)
            or obj.has_attributes
        )
        self.filter_elements([filter_func])


class VideoObject(
    etal.Labels,
    etal.HasLabelsSupport,
    etal.HasFramewiseView,
    etal.HasSpatiotemporalView,
):
    """A spatiotemporal object in a video.

    :class:`VideoObject` is a spatiotemporal concept that describes information
    about an object over multiple frames in a video. A :class:`VideoObject` can
    have a label with a confidence, a name (i.e., a class), object-level
    attributes that apply to the object over all frames, and frame-level
    attributes such as bounding boxes and attributes that apply to individual
    frames.

    Note that the :class:`VideoObject` class implements the
    :class:`eta.core.labels.HasFramewiseView` and
    :class:`eta.core.labels.HasSpatiotemporalView` mixins. This means that all
    :class:`VideoObject` instances can be rendered in both *framewise* and
    *spatiotemporal* format. Converting between these formats is guaranteed to
    be lossless and idempotent.

    In framewise format, a :class:`VideoObject` stores all information at the
    frame-level in :class:`DetectedObject` instances. In particular, the
    following invariants will hold:

        - The :attr:`DetectedObject.attrs` field will be empty. All
          object-level attributes will be stored as frame-level
          :class:`eta.core.data.Attribute` instances within
          :class:`DetectedObject` instances with ``constant == True``

    In spatiotemporal format, a :class:`VideoObject` store all possible
    information in the highest-available video construct. In particular, the
    following invariants will hold:

        - The :attr:`DetectedObject.attrs` fields of all
          :class:`DetectedObject` instances will contain only non-constant
          :class:`eta.core.data.Attribute` instances. All constant attributes
          will be upgraded to object-level attributes in the top-level
          :attr:`DetectedObject.attrs` field

    Attributes:
        type: the fully-qualified class name of the object
        label: (optional) the object label
        confidence: (optional) a confidence for the object, in ``[0, 1]``
        name: (optional) the name of the object, e.g., ``ground_truth`` or the
            name of the model that generated it
        index: (optional) an index assigned to the object
        support: a :class:`eta.core.frameutils.FrameRanges` instance describing
            the support of the object
        attrs: (optional) an :class:`eta.core.data.AttributeContainer` of
            object-level attributes of the object
        frames: a dict mapping frame numbers to :class:`DetectedObject`
            instances describing the frame-level attributes of the object

    Args:
        label (None): a label for the object
        confidence (None): the label confidence in ``[0, 1]``
        name (None): a name for the object, e.g., ``ground_truth`` or the name
            of the model that generated it
        index (None): an index assigned to the object
        support (None): a :class:`eta.core.frameutils.FrameRanges` instance
            describing the frozen support of the object
        attrs (None): an :class:`eta.core.data.AttributeContainer` of
            object-level attributes
        frames (None): a dict mapping frame numbers to :class:`DetectedObject`
            instances
    """

    def __init__(
        self,
        label=None,
        confidence=None,
        name=None,
        index=None,
        support=None,
        attrs=None,
        frames=None,
    ):
        self.type = etau.get_class_name(self)
        self.label = label
        self.confidence = confidence
        self.name = name
        self.index = index
        self.attrs = attrs or etad.AttributeContainer()
        self.frames = frames or {}
        etal.HasLabelsSupport.__init__(self, support=support)

    @property
    def is_empty(self):
        """Whether the object has no labels of any kind."""
        return not (
            self.has_label or self.has_object_attributes or self.has_detections
        )

    @property
    def has_label(self):
        """Whether the object has a ``label``."""
        return self.label is not None

    @property
    def has_confidence(self):
        """Whether the object has a ``confidence``."""
        return self.confidence is not None

    @property
    def has_name(self):
        """Whether the object has a ``name``."""
        return self.name is not None

    @property
    def has_index(self):
        """Whether the object has an ``index``."""
        return self.index is not None

    @property
    def has_object_attributes(self):
        """Whether the object has object-level attributes."""
        return bool(self.attrs)

    @property
    def has_frame_attributes(self):
        """Whether the object has frame-level attributes."""
        for obj in self.iter_detections():
            if obj.has_attributes:
                return True

        return False

    @property
    def has_attributes(self):
        """Whether the object has object- or frame-level attributes."""
        return self.has_object_attributes or self.has_frame_attributes

    @property
    def has_detections(self):
        """Whether the object has at least one non-empty DetectedObject."""
        return any(not dobj.is_empty for dobj in itervalues(self.frames))

    def iter_attributes(self):
        """Returns an iterator over the object-level attributes of the object.

        Returns:
            an iterator over :class:`eta.core.dataAttribute` instances
        """
        return iter(self.attrs)

    def iter_detections(self):
        """Returns an iterator over the :class:`DetectedObject` instances for
        each frame of the object.

        The frames are traversed in sorted order.

        Returns:
            an iterator over :class:`DetectedObject` instances
        """
        for frame_number in sorted(self.frames):
            yield self.frames[frame_number]

    @property
    def framewise_renderer_cls(self):
        """The :class:`eta.core.labels.LabelsFrameRenderer` used by this class.
        """
        return VideoObjectFrameRenderer

    @property
    def spatiotemporal_renderer_cls(self):
        """The :class:`eta.core.labels.LabelsSpatiotemporalRenderer` used by
        this class.
        """
        return VideoObjectSpatiotemporalRenderer

    def get_index(self):
        """Returns the ``index`` of the object.

        Returns:
            the index, or None if the object has no index
        """
        return self.index

    def offset_index(self, offset):
        """Adds the given offset to the object's ``index``.

        If the object has no index, this does nothing.

        Args:
            offset: the integer offset
        """
        if self.has_index:
            self.index += offset
            for dobj in self.iter_detections():
                dobj.offset_index(offset)

    def clear_index(self):
        """Clears the ``index`` of the object."""
        self.index = None
        for dobj in self.iter_detections():
            dobj.clear_index()

    def has_detection(self, frame_number):
        """Whether the object has a detection on the given frame number.

        Args:
            frame_number: the frame number

        Returns:
            True/False
        """
        return frame_number in self.frames

    def get_detection(self, frame_number):
        """Gets the detection for the given frame number, if available.

        Args:
            frame_number: the frame number

        Returns:
            a :class:`DetectedObject`, or ``None``
        """
        return self.frames.get(frame_number, None)

    def add_object_attribute(self, attr):
        """Adds the object-level attribute to the object.

        Args:
            attr: an :class:`eta.core.data.Attribute`
        """
        self.attrs.add(attr)

    def add_object_attributes(self, attrs):
        """Adds the :class:`eta.core.data.AttributeContainer` of object-level
        attributes to the object.

        Args:
            attrs: an :class:`eta.core.data.AttributeContainer`
        """
        self.attrs.add_container(attrs)

    def add_detection(self, obj, frame_number=None):
        """Adds the detection to the object.

        The detection will have its ``label`` and ``index`` scrubbed.

        Args:
            obj: a :class:`DetectedObject`
            frame_number: a frame number. If omitted, the
                :class:`DetectedObject` must have its ``frame_number`` set
        """
        self._add_detected_object(obj, frame_number)

    def add_detections(self, objects):
        """Adds the detections to the object.

        The :class:`DetectedObject` instances must have their ``frame_number``
        set, and they will have their ``label`` and ``index`` scrubbed.

        Args:
            objects: a :class:`DetectedObjectContainer`
        """
        self._add_detected_objects(objects)

    def remove_empty_frames(self):
        """Removes all empty :class:`DetectedObject` instances from this
        object.
        """
        self.frames = {
            fn: dobj
            for fn, dobj in iteritems(self.frames)
            if not dobj.is_empty
        }

    def clear_attributes(self):
        """Removes all attributes of any kind from the object."""
        self.clear_object_attributes()
        self.clear_frame_attributes()

    def clear_object_attributes(self):
        """Removes all object-level attributes from the object."""
        self.attrs = etad.AttributeContainer()

    def clear_frame_attributes(self):
        """Removes all frame-level attributes from the object."""
        for obj in self.iter_detections():
            obj.clear_attributes()

    def clear_detections(self):
        """Removes all :class:`DetectedObject` instance from the object."""
        self.frames = {}

    def filter_by_schema(self, schema):
        """Filters the object by the given schema.

        Args:
            schema: an :class:`ObjectSchema`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the object label
            does not match the schema
        """
        schema.validate_label(self.label)
        self.attrs.filter_by_schema(schema.attrs)
        for dobj in self.iter_detections():
            dobj.filter_by_schema(schema, allow_none_label=True)

    def attributes(self):
        """Returns the list of attributes to serialize.

        Returns:
            a list of attrinutes
        """
        _attrs = ["type"]
        if self.label is not None:
            _attrs.append("label")
        if self.confidence is not None:
            _attrs.append("confidence")
        if self.name is not None:
            _attrs.append("name")
        if self.is_support_frozen:
            _attrs.append("support")
        if self.index is not None:
            _attrs.append("index")
        if self.attrs:
            _attrs.append("attrs")
        if self.frames:
            _attrs.append("frames")
        return _attrs

    @classmethod
    def from_detections(cls, objects):
        """Builds a :class:`VideoObject` from a container of
        :class:`DetectedObject` instances.

        The :class:`DetectedObject` instances must have their ``frame_number``
        set, and they must all have the same ``label``, ``name``, and ``index``
        (the latter two may be ``None``).

        The input objects are modified in-place and passed by reference to the
        :class:`VideoObject`.

        Args:
            objects: a :class:``DetectedObjectContainer``

        Returns:
            a :class:`VideoObject`
        """
        if not objects:
            return cls()

        label = objects[0].label
        name = objects[0].name
        index = objects[0].index

        for dobj in objects:
            if dobj.label != label:
                raise ValueError(
                    "Object label '%s' does not match first label '%s'"
                    % (dobj.label, label)
                )

            if dobj.name != name:
                raise ValueError(
                    "Object name '%s' does not match first name '%s'"
                    % (dobj.name, name)
                )

            if dobj.index != index:
                raise ValueError(
                    "Object index '%s' does not match first index '%s'"
                    % (dobj.index, index)
                )

        # Strip constant attributes
        obj_attrs = strip_spatiotemporal_content_from_objects(objects)

        # Remove empty detections
        objects.remove_empty_labels()

        # Build VideoObject
        obj = cls(label=label, name=name, index=index, attrs=obj_attrs)
        obj.add_detections(objects)
        return obj

    @classmethod
    def _from_dict(cls, d):
        """Internal implementation of :class:`VideoObject.from_dict`.

        Subclasses should implement this method, not
        :class:`VideoObject.from_dict`.

        Args:
            d: a JSON dictionary

        Returns:
            a :class:`VideoObject`
        """
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
            name=d.get("name", None),
            index=d.get("index", None),
            support=support,
            frames=frames,
            attrs=attrs,
        )

    @classmethod
    def from_dict(cls, d):
        """Constructs a :class:`VideoObject` from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a :class:`VideoObject`
        """
        if "type" in d:
            obj_cls = etau.get_class(d["type"])
        else:
            obj_cls = cls

        return obj_cls._from_dict(d)

    def _add_detected_object(self, dobj, frame_number):
        if frame_number is None:
            if not dobj.has_frame_number:
                raise ValueError(
                    "Either `frame_number` must be provided or the "
                    "DetectedObject must have its `frame_number` set"
                )

            frame_number = dobj.frame_number

        if dobj.label is not None and dobj.label != self.label:
            logger.warning(
                "DetectedObject label '%s' does not match VideoObject label "
                "'%s'",
                dobj.label,
                self.label,
            )

        if dobj.name is not None and dobj.name != self.name:
            logger.warning(
                "DetectedObject name '%s' does not match VideoObject name "
                "'%s'",
                dobj.name,
                self.name,
            )

        if dobj.index is not None and dobj.index != self.index:
            logger.warning(
                "DetectedObject index '%s' does not match VideoObject index "
                "'%s'",
                dobj.index,
                self.index,
            )

        dobj.label = None
        dobj.name = None
        dobj.index = None
        dobj.frame_number = frame_number
        self.frames[frame_number] = dobj

    def _add_detected_objects(self, objects):
        for dobj in objects:
            self._add_detected_object(dobj, None)

    def _compute_support(self):
        return etaf.FrameRanges.from_iterable(self.frames.keys())


class VideoObjectContainer(
    etal.LabelsContainer, etal.HasFramewiseView, etal.HasSpatiotemporalView
):
    """An :class:`eta.core.serial.Container` of :class:`VideoObject`
    instances.
    """

    _ELE_CLS = VideoObject
    _ELE_CLS_FIELD = "_OBJ_CLS"
    _ELE_ATTR = "objects"

    @property
    def framewise_renderer_cls(self):
        """The :class:`eta.core.labels.LabelsFrameRenderer` used by this class.
        """
        return VideoObjectContainerFrameRenderer

    @property
    def spatiotemporal_renderer_cls(self):
        """The :class:`eta.core.labels.LabelsSpatiotemporalRenderer` used by
        this class.
        """
        return VideoObjectContainerSpatiotemporalRenderer

    def get_labels(self):
        """Returns the set of ``label`` values of all objects in the container.

        Returns:
            a set of labels
        """
        return set(obj.label for obj in self)

    def get_indexes(self):
        """Returns the set of ``index`` values of all objects in the container.

        ``None`` indexes are omitted.

        Returns:
            a set of indexes
        """
        return set(obj.index for obj in self if obj.has_index)

    def offset_indexes(self, offset):
        """Adds the given offset to all objects with ``index`` values.

        Args:
            offset: the integer offset
        """
        for obj in self:
            obj.offset_index(offset)

    def clear_indexes(self):
        """Clears the ``index`` of all objects in the container."""
        for obj in self:
            obj.clear_index()

    def sort_by_confidence(self, reverse=False):
        """Sorts the :class:`VideoObject` instances by confidence.

        Objects whose confidence is ``None`` are always put last.

        Args:
            reverse (False): whether to sort in descending order
        """
        self.sort_by("confidence", reverse=reverse)

    def sort_by_index(self, reverse=False):
        """Sorts the :class:`VideoObject` instances by index.

        Objects whose index is ``None`` are always put last.

        Args:
            reverse (False): whether to sort in descending order
        """
        self.sort_by("index", reverse=reverse)

    def filter_by_schema(self, schema):
        """Filters the objects in the container by the given schema.

        Args:
            schema: an :class:`ObjectContainerSchema`
        """
        # Remove objects with invalid labels
        filter_func = lambda obj: schema.has_object_label(obj.label)
        self.filter_elements([filter_func])

        # Filter objects by their schemas
        for obj in self:
            obj_schema = schema.get_object_schema(obj.label)
            obj.filter_by_schema(obj_schema)

    def remove_objects_without_attrs(self, labels=None):
        """Removes objects from this container that do not have attributes.

        Args:
            labels (None): an optional list of object ``label`` strings to
                which to restrict attention when filtering. By default, all
                objects are processed
        """
        filter_func = lambda obj: (
            (labels is not None and obj.label not in labels)
            or obj.has_attributes
        )
        self.filter_elements([filter_func])

    @classmethod
    def from_detections(cls, objects):
        """Builds a :class:`VideoObjectContainer` from a
        :class:`DetectedObjectContainer` of objects by constructing
        :class:`VideoObject` instances from the collections of detections
        constructed by partitioning into ``(label, index)`` groups.

        The :class:`DetectedObject` instances must have their ``frame_number``
        set, and each instance without an ``index`` is treated as a separate
        object.

        The input objects may be modified in-place and are passed by reference
        to the :class:`VideoObject` instances.

        Args:
            objects: a :class:`DetectedObjectContainer`

        Returns:
            a :class:`VideoObjectContainer`
        """
        # Group objects by (label, index)
        objects_map = defaultdict(DetectedObjectContainer)
        single_objects = []
        max_index = 0
        for obj in objects:
            if obj.index is not None:
                max_index = max(max_index, obj.index)
                objects_map[(obj.label, obj.index)].add(obj)
            else:
                single_objects.append(obj)

        # Give objects with no `index` their own groups
        for obj in single_objects:
            max_index += 1
            objects_map[(obj.label, max_index)].add(obj)

        # Build VideoObjects
        video_objects = cls()
        for dobjects in itervalues(objects_map):
            video_objects.add(VideoObject.from_detections(dobjects))

        return video_objects


class ObjectSchema(etal.LabelsSchema):
    """Schema for :class:`VideoObject` and :class:`DetectedObject`.

    Attributes:
        label: the object label
        attrs: an :class:`eta.core.data.AttributeContainerSchema` describing
            the object-level attributes of the object
        frames: an :class:`eta.core.data.AttributeContainerSchema` describing
            the frame-level attributes of the object

    Args:
        label: the object label
        attrs (None): an :class:`eta.core.data.AttributeContainerSchema`
            describing the object-level attributes of the object
        frames (None): an :class:`eta.core.data.AttributeContainerSchema`
            describing the frame-level attributes of the object
    """

    def __init__(self, label, attrs=None, frames=None):
        self.label = label
        self.attrs = attrs or etad.AttributeContainerSchema()
        self.frames = frames or etad.AttributeContainerSchema()

    @property
    def is_empty(self):
        """Whether this schema has no labels of any kind."""
        return False

    def has_label(self, label):
        """Whether the schema has the given object label.

        Args:
            label: the object label

        Returns:
            True/False
        """
        return label == self.label

    def get_label(self):
        """Gets the object label for the schema.

        Returns:
            the object label
        """
        return self.label

    def has_object_attribute(self, attr_name):
        """Whether the schema has an object-level
        :class:`eta.core.data.Attribute` of the given name.

        Args:
            attr_name: the name

        Returns:
            True/False
        """
        return self.attrs.has_attribute(attr_name)

    def get_object_attribute_schema(self, attr_name):
        """Gets the :class:`eta.core.data.AttributeSchema` for the object-level
        attribute of the given name.

        Args:
            attr_name: the name

        Returns:
            the `eta.core.data.AttributeSchema`
        """
        return self.attrs.get_attribute_schema(attr_name)

    def get_object_attribute_class(self, attr_name):
        """Gets the `eta.core.data.Attribute` class for the object-level
        attribute of the given name.

        Args:
            attr_name: the name

        Returns:
            the `eta.core.data.Attribute`
        """
        return self.attrs.get_attribute_class(attr_name)

    def has_frame_attribute(self, attr_name):
        """Whether the schema has a frame-level `eta.core.data.Attribute` of
        the given name.

        Args:
            attr_name: the name

        Returns:
            True/False
        """
        return self.frames.has_attribute(attr_name)

    def get_frame_attribute_schema(self, attr_name):
        """Gets the `eta.core.data.AttributeSchema` for the frame-level
        attribute of the given name.

        Args:
            attr_name: the name

        Returns:
            the `eta.core.data.AttributeSchema`
        """
        return self.frames.get_attribute_schema(attr_name)

    def get_frame_attribute_class(self, attr_name):
        """Gets the `eta.core.data.Attribute` class for the frame-level
        attribute of the given name.

        Args:
            attr_name: the name

        Returns:
            the `eta.core.data.Attribute`
        """
        return self.frames.get_attribute_class(attr_name)

    def add_object_attribute(self, attr):
        """Adds the object-level `eta.core.data.Attribute` to the schema.

        Args:
            attr: an `eta.core.data.Attribute`
        """
        self.attrs.add_attribute(attr)

    def add_object_attributes(self, attrs):
        """Adds the `eta.core.data.AttributeContainer` of object-level
        attributes to the schema.

        Args:
            attrs: an `eta.core.data.AttributeContainer`
        """
        self.attrs.add_attributes(attrs)

    def add_frame_attribute(self, attr):
        """Adds the frame-level `eta.core.data.Attribute` to the schema.

        Args:
            attr: an `eta.core.data.Attribute`
        """
        self.frames.add_attribute(attr)

    def add_frame_attributes(self, attrs):
        """Adds the `eta.core.data.AttributeContainer` of frame-level
        attributes to the schema.

        Args:
            attrs: an `eta.core.data.AttributeContainer`
        """
        self.frames.add_attributes(attrs)

    def add_object(self, obj):
        """Adds the object to the schema.

        Args:
            obj: a :class:`VideoObject` or :class:`DetectedObject`
        """
        if isinstance(obj, DetectedObject):
            self._add_detected_object(obj)
        else:
            self._add_video_object(obj)

    def add_objects(self, objects):
        """Adds the :class:`VideoObjectContainer` or
        :class:`DetectedObjectContainer` to the schema.

        Args:
            objects: an :class:`VideoObjectContainer` or
                :class:`DetectedObjectContainer`
        """
        for obj in objects:
            self.add_object(obj)

    def is_valid_label(self, label):
        """Whether the object label is compliant with the schema.

        Args:
            label: an object label

        Returns:
            True/False
        """
        try:
            self.validate_label(label)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_object_attribute(self, attr):
        """Whether the object-level attribute is compliant with the schema.

        Args:
            attr: an `eta.core.data.Attribute`

        Returns:
            True/False
        """
        try:
            self.validate_object_attribute(attr)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_object_attributes(self, attrs):
        """Whether the `eta.core.data.AttributeContainer` of object-level
        attributes is compliant with the schema.

        Args:
            attrs: an `eta.core.data.AttributeContainer`

        Returns:
            True/False
        """
        try:
            self.validate_object_attributes(attrs)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_frame_attribute(self, attr):
        """Whether the frame-level attribute is compliant with the schema.

        Args:
            attr: an `eta.core.data.Attribute`

        Returns:
            True/False
        """
        try:
            self.validate_frame_attribute(attr)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_frame_attributes(self, attrs):
        """Whether the `eta.core.data.AttributeContainer` of frame-level
        attributes is compliant with the schema.

        Args:
            attrs: an `eta.core.data.AttributeContainer`

        Returns:
            True/False
        """
        try:
            self.validate_frame_attributes(attrs)
            return True
        except etal.LabelsSchemaError:
            return False

    def validate_label(self, label):
        """Validates that the object label is compliant with the schema.

        Args:
            label: the label

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the label violates
            the schema
        """
        if label != self.label:
            raise ObjectSchemaError(
                "Label '%s' does not match object schema" % label
            )

    def validate_object_attribute(self, attr):
        """Validates that the object-level attribute is compliant with the
        schema.

        Args:
            attr: an `eta.core.data.Attribute`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attribute
            violates the schema
        """
        self.attrs.validate_attribute(attr)

    def validate_object_attributes(self, attrs):
        """Validates that the `eta.core.data.AttributeContainer` of
        object-level attributes is compliant with the schema.

        Args:
            attrs: an `eta.core.data.AttributeContainer`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attributes
            violate the schema
        """
        self.attrs.validate(attrs)

    def validate_frame_attribute(self, attr):
        """Validates that the frame-level attribute is compliant with the
        schema.

        Args:
            attr: an `eta.core.data.Attribute`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attribute
            violates the schema
        """
        self.frames.validate_attribute(attr)

    def validate_frame_attributes(self, attrs):
        """Validates that the `eta.core.data.AttributeContainer` of frame-level
        attributes is compliant with the schema.

        Args:
            attrs: an `eta.core.data.AttributeContainer`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attributes
            violate the schema
        """
        self.frames.validate(attrs)

    def validate(self, obj):
        """Validates that the object is compliant with the schema.

        Args:
            obj: a :class:`VideoObject` or :class:`DetectedObject`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the object violates
            the schema
        """
        if isinstance(obj, DetectedObject):
            self._validate_detected_object(obj)
        else:
            self._validate_video_object(obj)

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given schema.

        Args:
            schema: an :class:`ObjectSchema`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if this schema is not a
            subset of the given schema
        """
        self.validate_schema_type(schema)

        if self.label != schema.label:
            raise ObjectSchemaError(
                "Expected object label '%s'; found '%s'"
                % (schema.label, self.label)
            )

        self.attrs.validate_subset_of_schema(schema.attrs)
        self.frames.validate_subset_of_schema(schema.frames)

    def merge_schema(self, schema):
        """Merges the given :class:`ObjectSchema` into this schema.

        Args:
            schema: an :class:`ObjectSchema`
        """
        self.validate_label(schema.label)
        self.attrs.merge_schema(schema.attrs)
        self.frames.merge_schema(schema.frames)

    @classmethod
    def build_active_schema(cls, obj):
        """Builds an :class:`ObjectSchema` that describes the active schema of
        the object.

        Args:
            obj: a :class:`VideoObject` or :class:`DetectedObject`

        Returns:
            an :class:`ObjectSchema`
        """
        schema = cls(obj.label)
        schema.add_object(obj)
        return schema

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Args:
            a list of attribute names
        """
        _attrs = ["label"]
        if self.attrs:
            _attrs.append("attrs")
        if self.frames:
            _attrs.append("frames")
        return _attrs

    @classmethod
    def from_dict(cls, d):
        """Constructs an :class:`ObjectSchema` from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an :class:`ObjectSchema`
        """
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

        for attr in dobj.attrs:
            if attr.constant:
                self.add_object_attribute(attr)
            else:
                self.add_frame_attribute(attr)

    def _add_video_object(self, obj):
        self.validate_label(obj.label)
        self.add_object_attributes(obj.attrs)
        for dobj in obj.iter_detections():
            self._add_detected_object(dobj, validate_label=False)

    def _validate_detected_object(self, dobj, validate_label=True):
        if validate_label:
            self.validate_label(dobj.label)

        for attr in dobj.attrs:
            if attr.constant:
                self.validate_object_attribute(attr)
            else:
                self.validate_frame_attribute(attr)

    def _validate_video_object(self, obj):
        self.validate_label(obj.label)
        self.validate_object_attributes(obj.attrs)
        for dobj in obj.iter_detections():
            self._validate_detected_object(dobj, validate_label=False)


class ObjectSchemaError(etal.LabelsSchemaError):
    """Error raised when an :class:`ObjectSchema` is violated."""

    pass


class ObjectContainerSchema(etal.LabelsContainerSchema):
    """Schema for :class:`VideoObjectContainer` and
    :class:`DetectedObjectContainer`.

    Attributes:
        schema: a dict mapping object labels to :class:`ObjectSchema` instances

    Args:
        schema (None): a dict mapping object labels to :class:`ObjectSchema`
            instances
    """

    def __init__(self, schema=None):
        self.schema = schema or {}

    @property
    def is_empty(self):
        """Whether this schema has no labels of any kind."""
        return not bool(self.schema)

    def iter_object_labels(self):
        """Returns an iterator over the object labels in this schema.

        Returns:
            an iterator over object labels
        """
        return iter(self.schema)

    def iter_objects(self):
        """Returns an iterator over the (label, :class:`ObjectSchema`) pairs in
        this schema.

        Returns:
            an iterator over (label, :class:`ObjectSchema`) pairs
        """
        return iteritems(self.schema)

    def has_object_label(self, label):
        """Whether the schema has an object with the given label.

        Args:
            label: the object label

        Returns:
            True/False
        """
        return label in self.schema

    def has_object_attribute(self, label, attr_name):
        """Whether the schema has an object with the given label with an
        object-level attribute of the given name.

        Args:
            label: the object label
            attr_name: the object-level attribute name

        Returns:
            True/False
        """
        if not self.has_object_label(label):
            return False

        return self.schema[label].has_object_attribute(attr_name)

    def has_frame_attribute(self, label, attr_name):
        """Whether the schema has an object with the given label with a
        frame-level attribute of the given name.

        Args:
            label: the object label
            attr_name: the frame-level object attribute name

        Returns:
            True/False
        """
        if not self.has_object_label(label):
            return False

        return self.schema[label].has_frame_attribute(attr_name)

    def get_object_schema(self, label):
        """Gets the :class:`ObjectSchema` for the object with the given label.

        Args:
            label: the object label

        Returns:
            an :class:`ObjectSchema`
        """
        self.validate_object_label(label)
        return self.schema[label]

    def get_object_attribute_schema(self, label, attr_name):
        """Gets the :class:`eta.core.data.AttributeSchema` for the object-level
        attribute of the given name for the object with the given label.

        Args:
            label: the object label
            attr_name: the object-level attribute name

        Returns:
            the :class:`eta.core.data.AttributeSchema`
        """
        obj_schema = self.get_object_schema(label)
        return obj_schema.get_object_attribute_schema(attr_name)

    def get_object_attribute_class(self, label, attr_name):
        """Gets the :class:`eta.core.data.Attribute` class for the object-level
        attribute of the given name for the object with the given label.

        Args:
            label: the object label
            attr_name: the object-level attribute name

        Returns:
            the :class:`eta.core.data.Attribute`
        """
        self.validate_object_label(label)
        return self.schema[label].get_object_attribute_class(attr_name)

    def get_frame_attribute_schema(self, label, attr_name):
        """Gets the :class:`eta.core.data.AttributeSchema` for the frame-level
        attribute of the given name for the object with the given label.

        Args:
            label: the object label
            attr_name: the frame-level object attribute name

        Returns:
            the :class:`eta.core.data.AttributeSchema`
        """
        obj_schema = self.get_object_schema(label)
        return obj_schema.get_frame_attribute_schema(attr_name)

    def get_frame_attribute_class(self, label, attr_name):
        """Gets the :class:`eta.core.data.Attribute` class for the frame-level
        attribute of the given name for the object with the given label.

        Args:
            label: the object label
            attr_name: the frame-level object attribute name

        Returns:
            the :class:`eta.core.data.Attribute`
        """
        self.validate_object_label(label)
        return self.schema[label].get_frame_attribute_class(attr_name)

    def add_object_label(self, label):
        """Adds the given object label to the schema.

        Args:
            label: an object label
        """
        self._ensure_has_object_label(label)

    def add_object_attribute(self, label, attr):
        """Adds the object-level :class:`eta.core.data.Attribute` for the
        object with the given label to the schema.

        Args:
            label: an object label
            attr: an :class:`eta.core.data.Attribute`
        """
        self._ensure_has_object_label(label)
        self.schema[label].add_object_attribute(attr)

    def add_object_attributes(self, label, attrs):
        """Adds the :class:`eta.core.data.AttributeContainer` of object-level
        attributes for the object with the given label to the schema.

        Args:
            label: an object label
            attrs: an :class:`eta.core.data.AttributeContainer`
        """
        self._ensure_has_object_label(label)
        self.schema[label].add_object_attributes(attrs)

    def add_frame_attribute(self, label, attr):
        """Adds the frame-level :class:`eta.core.data.Attribute` for the object
        with the given label to the schema.

        Args:
            label: an object label
            attr: an :class:`eta.core.data.Attribute`
        """
        self._ensure_has_object_label(label)
        self.schema[label].add_frame_attribute(attr)

    def add_frame_attributes(self, label, attrs):
        """Adds the :class:`eta.core.data.AttributeContainer` of frame-level
        attributes for the object with the given label to the schema.

        Args:
            label: an object label
            attrs: an :class:`eta.core.data.AttributeContainer`
        """
        self._ensure_has_object_label(label)
        self.schema[label].add_frame_attributes(attrs)

    def add_object(self, obj):
        """Adds the object to the schema.

        Args:
            obj: a :class:`VideoObject` or :class:`DetectedObject`
        """
        self._ensure_has_object_label(obj.label)
        self.schema[obj.label].add_object(obj)

    def add_objects(self, objects):
        """Adds the objects to the schema.

        Args:
            objects: a :class:`VideoObjectContainer` or
                :class:`DetectedObjectContainer`
        """
        for obj in objects:
            self.add_object(obj)

    def is_valid_object_label(self, label):
        """Whether the object label is compliant with the schema.

        Args:
            label: an object label

        Returns:
            True/False
        """
        try:
            self.validate_object_label(label)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_object_attribute(self, label, attr):
        """Whether the object-level attribute for the object with the given
        label is compliant with the schema.

        Args:
            label: an object label
            attr: an :class:`eta.core.data.Attribute`

        Returns:
            True/False
        """
        try:
            self.validate_object_attribute(label, attr)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_object_attributes(self, label, attrs):
        """Whether the object-level attributes for the object with the given
        label are compliant with the schema.

        Args:
            label: an object label
            attrs: an :class:`eta.core.data.AttributeContainer`

        Returns:
            True/False
        """
        try:
            self.validate_object_attributes(label, attrs)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_frame_attribute(self, label, attr):
        """Whether the frame-level attribute for the object with the given
        label is compliant with the schema.

        Args:
            label: an object label
            attr: an :class:`eta.core.data.Attribute`

        Returns:
            True/False
        """
        try:
            self.validate_frame_attribute(label, attr)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_frame_attributes(self, label, attrs):
        """Whether the frame-level attributes for the object with the given
        label are compliant with the schema.

        Args:
            label: an object label
            attrs: an :class:`eta.core.data.AttributeContainer`

        Returns:
            True/False
        """
        try:
            self.validate_frame_attributes(label, attrs)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_object(self, obj):
        """Whether the object is compliant with the schema.

        Args:
            obj: a :class:`VideoObject` or :class:`DetectedObject`

        Returns:
            True/False
        """
        try:
            self.validate_object(obj)
            return True
        except etal.LabelsSchemaError:
            return False

    def validate_object_label(self, label):
        """Validates that the object label is compliant with the schema.

        Args:
            label: an object label

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the object label
            violates the schema
        """
        if label not in self.schema:
            raise ObjectContainerSchemaError(
                "Object label '%s' is not allowed by the schema" % label
            )

    def validate_object_attribute(self, label, attr):
        """Validates that the object-level :class:`eta.core.data.Attribute` for
        the object with the given label is compliant with the schema.

        Args:
            label: an object label
            attr: an :class:`eta.core.data.Attribute`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attribute
            violates the schema
        """
        self.validate_object_label(label)
        self.schema[label].validate_object_attribute(attr)

    def validate_object_attributes(self, label, attrs):
        """Validates that the :class:`eta.core.data.AttributeContainer` of
        object-level attributes for the object with the given label is
        compliant with the schema.

        Args:
            label: an object label
            attrs: an :class:`eta.core.data.AttributeContainer`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attributes
            violate the schema
        """
        self.validate_object_label(label)
        self.schema[label].validate_object_attributes(attrs)

    def validate_frame_attribute(self, label, attr):
        """Validates that the frame-level :class:`eta.core.data.Attribute` for
        the object with the given label is compliant with the schema.

        Args:
            label: an object label
            attr: an :class:`eta.core.data.Attribute`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attribute
            violates the schema
        """
        self.validate_object_label(label)
        self.schema[label].validate_frame_attribute(attr)

    def validate_frame_attributes(self, label, attrs):
        """Validates that the :class:`eta.core.data.AttributeContainer` of
        frame-level attributes for the object with the given label is compliant
        with the schema.

        Args:
            label: an object label
            attrs: an :class:`eta.core.data.AttributeContainer`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attributes
            violate the schema
        """
        self.validate_object_label(label)
        self.schema[label].validate_frame_attributes(attrs)

    def validate_object(self, obj):
        """Validates that the object is compliant with the schema.

        Args:
            obj: a :class:`VideoObject` or :class:`DetectedObject`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the object violates
            the schema
        """
        self.validate_object_label(obj.label)
        self.schema[obj.label].validate(obj)

    def validate(self, objects):
        """Validates that the objects are compliant with the schema.

        Args:
            objects: a :class:`VideoObjectContainer` or
                :class:`DetectedObjectContainer`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the objects violate
            the schema
        """
        for obj in objects:
            self.validate_object(obj)

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given schema.

        Args:
            schema: an :class:`ObjectContainerSchema`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if this schema is not a
            subset of the given schema
        """
        self.validate_schema_type(schema)

        for label, obj_schema in iteritems(self.schema):
            if not schema.has_object_label(label):
                raise ObjectContainerSchemaError(
                    "Object label '%s' does not appear in schema" % label
                )

            other_obj_schema = schema.get_object_schema(label)
            obj_schema.validate_subset_of_schema(other_obj_schema)

    def merge_object_schema(self, obj_schema):
        """Merges the given :class:`ObjectSchema` into the schema.

        Args:
            obj_schema: an :class:`ObjectSchema`
        """
        label = obj_schema.label
        self._ensure_has_object_label(label)
        self.schema[label].merge_schema(obj_schema)

    def merge_schema(self, schema):
        """Merges the given :class:`ObjectContainerSchema` into this schema.

        Args:
            schema: an :class:`ObjectContainerSchema`
        """
        for _, obj_schema in schema.iter_objects():
            self.merge_object_schema(obj_schema)

    @classmethod
    def build_active_schema(cls, objects):
        """Builds an :class:`ObjectContainerSchema` that describes the active
        schema of the objects.

        Args:
            objects: a :class:`VideoObjectContainer` or
                :class:`DetectedObjectContainer`

        Returns:
            an :class:`ObjectContainerSchema`
        """
        schema = cls()
        schema.add_objects(objects)
        return schema

    @classmethod
    def from_dict(cls, d):
        """Constructs an :class:`ObjectContainerSchema` from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an :class:`ObjectContainerSchema`
        """
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
    """Error raised when an :class:`ObjectContainerSchema` is violated."""

    pass


class VideoObjectFrameRenderer(etal.LabelsFrameRenderer):
    """Class for rendering a :class:`VideoObject` at the frame-level.

    See the :class:`VideoObject` class docstring for the framewise format spec.

    Args:
        obj: a :class:`VideoObject`
    """

    _LABELS_CLS = VideoObject
    _FRAME_LABELS_CLS = DetectedObject

    def __init__(self, obj):
        self._obj = obj

    def render(self, in_place=False):
        """Renders the :class:`VideoObject` in framewise format.

        Args:
            in_place (False): whether to perform the rendering in-place

        Returns:
            a :class:`VideoObject`
        """
        obj = self._obj
        frames = self.render_all_frames(in_place=in_place)

        if in_place:
            # Render in-place
            obj.clear_object_attributes()
            obj.clear_detections()
            obj.frames = frames
            return obj

        # Render new copy of object
        label = deepcopy(obj.label)
        confidence = deepcopy(obj.confidence)
        index = deepcopy(obj.index)
        if obj.is_support_frozen:
            support = deepcopy(obj.support)
        else:
            support = None

        return VideoObject(
            label=label,
            confidence=confidence,
            index=index,
            support=support,
            frames=frames,
        )

    def render_frame(self, frame_number, in_place=False):
        """Renders the :class:`VideoObject` for the given frame.

        Args:
            frame_number: the frame number
            in_place (False): whether to perform the rendering in-place

        Returns:
            a :class:`DetectedObject`, or ``None`` if no labels exist for the
            given frame
        """
        if frame_number not in self._obj.support:
            return None

        obj_attrs = self._get_object_attrs()
        return self._render_frame(frame_number, obj_attrs, in_place)

    def render_all_frames(self, in_place=False):
        """Renders the :class:`VideoObject` for all possible frames.

        Args:
            in_place (False): whether to perform the rendering in-place

        Returns:
            a dict mapping frame numbers to :class:`DetectedObject` instances
        """
        obj_attrs = self._get_object_attrs()

        dobjs_map = {}
        for frame_number in self._obj.support:
            dobjs_map[frame_number] = self._render_frame(
                frame_number, obj_attrs, in_place
            )

        return dobjs_map

    def _render_frame(self, frame_number, obj_attrs, in_place):
        # Base DetectedObject
        if frame_number in self._obj.frames:
            dobj = self._obj.frames[frame_number]
            if not in_place:
                dobj = deepcopy(dobj)
        else:
            dobj = DetectedObject(frame_number=frame_number)

        # Render object-level attributes
        if obj_attrs is not None:
            #
            # Prepend object-level attributes
            #
            # We cannot avoid `deepcopy` here because object-level attributes
            # must be embedded in each frame
            #
            dobj.attrs.prepend_container(deepcopy(obj_attrs))

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

        # There's no need to avoid `deepcopy` here when `in_place == True`
        # because copies of object-level attributes must be made for each frame
        obj_attrs = deepcopy(self._obj.attrs)
        for attr in obj_attrs:
            attr.constant = True

        return obj_attrs


class VideoObjectContainerFrameRenderer(etal.LabelsContainerFrameRenderer):
    """Class for rendering labels for a :class:`VideoObjectContainer` at the
    frame-level.
    """

    _LABELS_CLS = VideoObjectContainer
    _FRAME_LABELS_CLS = DetectedObjectContainer
    _ELEMENT_RENDERER_CLS = VideoObjectFrameRenderer


class VideoObjectSpatiotemporalRenderer(etal.LabelsSpatiotemporalRenderer):
    """Class for rendering a :class:`VideoObject` in spatiotemporal format.

    See the :class:`VideoObject` class docstring for the spatiotemporal format
    spec.

    Args:
        obj: a :class:`VideoObject`
    """

    _LABELS_CLS = VideoObject

    def __init__(self, obj):
        self._obj = obj

    def render(self, in_place=False):
        """Renders the :class:`VideoObject` in spatiotemporal format.

        Args:
            in_place (False): whether to perform the rendering in-place

        Returns:
            a :class:`VideoObject`
        """
        obj = self._obj
        if not in_place:
            obj = deepcopy(obj)

        # Upgrade spatiotemporal elements from frames
        attrs = strip_spatiotemporal_content_from_objects(
            obj.iter_detections()
        )
        obj.add_object_attributes(attrs)
        obj.remove_empty_frames()

        return obj


class VideoObjectContainerSpatiotemporalRenderer(
    etal.LabelsContainerSpatiotemporalRenderer
):
    """Class for rendering labels for a :class:`VideoObjectContainer` in
    spatiotemporal format.
    """

    _LABELS_CLS = VideoObjectContainer
    _ELEMENT_RENDERER_CLS = VideoObjectSpatiotemporalRenderer


def strip_spatiotemporal_content_from_objects(objects):
    """Strips the spatiotemporal content from the given iterable of
    :class:`DetectedObject` instances.

    The input objects are modified in-place.

    Args:
        objects: an iterable of :class:`DetectedObject` instances

    Returns:
        an :class:`eta.core.data.AttributeContainer` of constant object
        attributes. By convention, the returned attributes are no longer marked
        as constant, as this is assumed to be implicit
    """
    # Extract spatiotemporal content from objects
    attrs_map = {}
    for dobj in objects:
        dobj.label = None
        dobj.index = None

        for const_attr in dobj.attrs.pop_constant_attrs():
            # @todo could verify here that duplicate constant attributes are
            # exactly equal, as they should be
            attrs_map[const_attr.name] = const_attr

    # Store object-level attributes in a container with `constant == False`
    attrs = etad.AttributeContainer()
    for attr in itervalues(attrs_map):
        attr.constant = False
        attrs.add(attr)

    return attrs


class ObjectCount(etas.Serializable):
    """The number of instances of an object found in an image.

    Args:
        label: the label
        count: the count
    """

    def __init__(self, label, count):
        self.label = label
        self.count = count

    @classmethod
    def from_dict(cls, d):
        """Constructs an :class:`ObjectCount` from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an :class:`ObjectCount`
        """
        return cls(d["label"], d["count"])


class ObjectCounts(etas.Container):
    """Container for counting objects in an image."""

    _ELE_CLS = ObjectCount
    _ELE_ATTR = "counts"


class EvaluationType(object):
    """Enumeration representing the type of evaluation an object label is
    intended for. This enables evaluation of false negatives on a subset of
    the labels used for evaluating false positives.

    Attributes:
        RECALL: this object is part of the subset that **must** be detected. If
            it is not, it is considered a false negative
        PRECISION: this object **may** be detected, and if so, is marked as a
            true positive, however, if it is not, it is **not** considered a
            false negative
    """

    RECALL = "RECALL"
    PRECISION = "PRECISION"
