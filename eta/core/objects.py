'''
Core data structures for working with detected objects in images and videos.

Copyright 2017-2019, Voxel51, Inc.
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

from eta.core.data import AttributeContainer
from eta.core.geometry import BoundingBox, HasBoundingBox
from eta.core.serial import Container, Serializable, deserialize_numpy_array


class DetectedObject(Serializable, HasBoundingBox):
    '''A detected object in an image.

    Attributes:
        label: object label
        bounding_box: a BoundingBox around the object
        mask: (optional) a mask for the object within its bounding box
        confidence: (optional) the detection confidence, in [0, 1]
        index: (optional) an index assigned to the object
        score: (optional) an optional score for the object
        frame_number: (optional) the frame number in which this object was
            detected
        index_in_frame: (optional) the index of this object in the frame
            where it was detected
        attrs: (optional) an AttributeContainer describing additional
            attributes of the object
        eval_type: (optional) an EvaluationType value
        event_indices: (optional) a set of a DetectedEvent indices to which
            the object belongs
    '''

    def __init__(
            self, label, bounding_box, mask=None, confidence=None, index=None,
            score=None, frame_number=None, index_in_frame=None, attrs=None,
            eval_type=None, event_indices=set()):
        '''Creates a DetectedObject instance.

        Args:
            label: object label string
            bounding_box: a BoundingBox around the object
            mask: (optional) a numpy array describing the mask for the object
                within its bounding box
            confidence: (optional) the detection confidence, in [0, 1]
            index: (optional) an index assigned to the object
            score: (optional) an optional score for the object
            frame_number: (optional) the frame number in which this object was
                detected
            index_in_frame: (optional) the index of this object in the frame
                where it was detected
            attrs: (optional) an AttributeContainer describing additional
                attributes of the object
            eval_type: (optional) an EvaluationType value
            event_indices: (optional) a set of a DetectedEvent indices to which
                the object belongs
        '''
        self.label = label
        self.bounding_box = bounding_box
        self.mask = mask
        self.confidence = confidence
        self.index = index
        self.score = score
        self.frame_number = frame_number
        self.index_in_frame = index_in_frame
        self.eval_type = eval_type
        self.event_indices = set(event_indices)
        self.attrs = attrs or AttributeContainer()
        self._meta = None  # Usable by clients to store temporary metadata

    @property
    def has_attributes(self):
        '''Returns True/False if this object has attributes.'''
        return bool(self.attrs)

    @property
    def has_mask(self):
        '''Returns True/False if this object has a segmentation mask.'''
        return self.mask is not None

    def clear_attributes(self):
        '''Removes all attributes from the object.'''
        self.attrs = AttributeContainer()

    def add_attribute(self, attr):
        '''Adds the Attribute to the object.'''
        self.attrs.add(attr)

    def add_attributes(self, attrs):
        '''Adds the AttributeContainer of attributes to the object.'''
        self.attrs.add_container(attrs)

    def get_bounding_box(self):
        '''Returns the bounding box for the object.'''
        return self.bounding_box

    def attributes(self):
        '''Returns the list of attributes to serialize.'''
        _attrs = ["label", "bounding_box"]

        _optional_attrs = [
            "mask", "confidence", "index", "score", "frame_number",
            "index_in_frame", "eval_type"]
        _attrs.extend(
            [a for a in _optional_attrs if getattr(self, a) is not None])

        _iff_attrs = ["event_indices", "attrs"]
        _attrs.extend([a for a in _iff_attrs if getattr(self, a)])

        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs a DetectedObject from a JSON dictionary.'''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = AttributeContainer.from_dict(attrs)

        mask = d.get("mask", None)
        if mask is not None:
            mask = deserialize_numpy_array(mask)

        return cls(
            d["label"],
            BoundingBox.from_dict(d["bounding_box"]),
            mask=mask,
            confidence=d.get("confidence", None),
            index=d.get("index", None),
            score=d.get("score", None),
            frame_number=d.get("frame_number", None),
            index_in_frame=d.get("index_in_frame", None),
            attrs=attrs,
            eval_type=d.get("eval_type", None),
            event_indices=set(d.get("event_indices", []))
        )


class DetectedObjectContainer(Container):
    '''Base class for containers that store lists of `DetectedObject`s.'''

    _ELE_CLS = DetectedObject
    _ELE_CLS_FIELD = "_OBJ_CLS"
    _ELE_ATTR = "objects"

    def get_labels(self):
        '''Returns a set containing the labels of the DetectedObjects.'''
        return set(obj.label for obj in self)

    def sort_by_confidence(self, reverse=False):
        '''Sorts the object list by confidence.

        Objects whose confidence is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("confidence", reverse=reverse)

    def sort_by_index(self, reverse=False):
        '''Sorts the object list by index.

        Objects whose index is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("index", reverse=reverse)

    def sort_by_score(self, reverse=False):
        '''Sorts the object list by score.

        Objects whose score is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("score", reverse=reverse)

    def sort_by_frame_number(self, reverse=False):
        '''Sorts the object list by frame number

        Objects whose frame number is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("frame_number", reverse=reverse)

    def filter_by_schema(self, schema):
        '''Filters the objects/attributes from this container that are not
        compliant with the given schema.

        Args:
            schema: an ImageLabelsSchema or VideoLabelsSchema
        '''
        filter_func = lambda obj: obj.label in schema.objects
        self.filter_elements([filter_func])
        for obj in self:
            if obj.has_attributes:
                obj.attrs.filter_by_schema(schema.objects[obj.label])

    def remove_objects_without_attrs(self, labels=None):
        '''Filters the objects from this container that do not have attributes.

        Args:
            labels: an optional list of DetectedObject label strings to which
                to restrict attention when filtering. By default, all objects
                are processed
        '''
        filter_func = lambda obj: (
            (labels is not None and obj.label not in labels)
            or obj.has_attributes
        )
        self.filter_elements([filter_func])


class ObjectCount(Serializable):
    '''The number of instances of an object found in an image.'''

    def __init__(self, label, count):
        self.label = label
        self.count = count

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ObjectCount from a JSON dictionary.'''
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
