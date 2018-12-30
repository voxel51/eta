'''
Core data structures for working with detected objects in images and videos.

Copyright 2017-2018, Voxel51, Inc.
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

from eta.core.data import DataContainer, AttributeContainer
from eta.core.data import SpatiotemporalLike
from eta.core.geometry import BoundingBox, HasBoundingBox
from eta.core.serial import Serializable


class DetectedObject(Serializable, HasBoundingBox):
    '''A detected object in an image.

    Attributes:
        label: object label
        bounding_box: a BoundingBox around the object
        confidence: (optional) the detection confidence, in [0, 1]
        index: (optional) an index assigned to the object
        score: (optional) an optional score for the object
        frame_number: (optional) the frame number in which this object was
            detected
        index_in_frame: (optional) the index of this object in the frame
            where it was detected
        attrs: (optional) an AttributeContainer describing additional
            attributes of the object
    '''

    def __init__(
            self, label, bounding_box, confidence=None, index=None, score=None,
            frame_number=None, index_in_frame=None, attrs=None):
        '''Constructs a DetectedObject.

        Args:
            label: object label string
            bounding_box: a BoundingBox around the object
            confidence: (optional) the detection confidence, in [0, 1]
            index: (optional) an index assigned to the object
            score: (optional) an optional score for the object
            frame_number: (optional) the frame number in which this object was
                detected
            index_in_frame: (optional) the index of this object in the frame
                where it was detected
            attrs: (optional) an AttributeContainer describing additional
                attributes of the object
        '''
        self.label = label
        self.bounding_box = bounding_box
        self.confidence = confidence
        self.index = index
        self.score = score
        self.frame_number = frame_number
        self.index_in_frame = index_in_frame
        self.attrs = attrs
        self._meta = None  # Usable by clients to store temporary metadata

    @property
    def has_attributes(self):
        '''Returns True/False if this object has attributes.'''
        return self.attrs is not None

    def get_bounding_box(self):
        '''Returns the bounding box for the object.'''
        return self.bounding_box

    def add_attribute(self, attr):
        '''Adds the Attribute to the object.'''
        if not self.has_attributes:
            self.attrs = AttributeContainer()
        self.attrs.add(attr)

    def attributes(self):
        '''Returns the list of attributes to serialize.

        Optional attributes that were not provided (e.g. are None) are omitted
        from this list.
        '''
        _attrs = [
            "label", "bounding_box", "confidence", "index", "score",
            "frame_number", "index_in_frame", "attrs"
        ]
        # Exclude attributes that are None
        return [a for a in _attrs if getattr(self, a) is not None]

    @classmethod
    def from_dict(cls, d):
        '''Constructs a DetectedObject from a JSON dictionary.'''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = AttributeContainer.from_dict(attrs)

        return cls(
            d["label"],
            BoundingBox.from_dict(d["bounding_box"]),
            confidence=d.get("confidence", None),
            index=d.get("index", None),
            score=d.get("score", None),
            frame_number=d.get("frame_number", None),
            index_in_frame=d.get("index_in_frame", None),
            attrs=attrs,
        )


class DetectedObjectContainer(DataContainer, SpatiotemporalLike):
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
        return self._sort_by_attr("confidence", reverse=reverse)

    def sort_by_index(self, reverse=False):
        '''Sorts the object list by index.

        Objects whose index is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        return self._sort_by_attr("index", reverse=reverse)

    def sort_by_score(self, reverse=False):
        '''Sorts the object list by score.

        Objects whose score is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        return self._sort_by_attr("score", reverse=reverse)

    def sort_by_frame_number(self, reverse=False):
        '''Sorts the object list by frame number

        Objects whose frame number is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        return self._sort_by_attr("frame_number", reverse=reverse)


class ObjectCount(Serializable):
    '''The number of instances of an object found in an image.'''

    def __init__(self, label, count):
        self.label = label
        self.count = count

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ObjectCount from a JSON dictionary.'''
        return ObjectCount(d["label"], d["count"])


class ObjectCounts(DataContainer):
    '''Container for counting objects in an image.'''

    _ELE_CLS = ObjectCount
    _ELE_ATTR = "counts"
