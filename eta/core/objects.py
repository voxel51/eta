'''
Core data structures for working with detected objects in images.

Copyright 2017-2018, Voxel51, LLC
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

from eta.core.data import DataContainer
from eta.core.geometry import BoundingBox
from eta.core.serial import Serializable


class ObjectContainer(DataContainer):
    '''Base class for containers that store lists of objects.

    Subclasses must set the `_ELE_CLS` attribute.
    '''

    _ELE_CLS = None
    _ELE_CLS_FIELD = "_OBJ_CLS"
    _ELE_ATTR = "objects"


class ObjectAttribute(Serializable):
    '''Base class for object attributes.'''

    def __init__(self, category=None, label=None, confidence=None):
        '''Constructs an ObjectAttribute.

        Args:
            category: (optional) the attribute category
            label: (optional) the attribute label
            confidence: (optional) the confidence of the label, in [0, 1]
        '''
        self.category = category
        self.label = label
        self.confidence = float(confidence)

    def attributes(self):
        '''Returns the list of attributes to serialize.

        Optional attributes that were not provided (e.g. are None) are omitted
        from this list.
        '''
        _attrs = []
        if self.category is not None:
            _attrs.append("category")
        if self.label is not None:
            _attrs.append("label")
        if self.confidence is not None:
            _attrs.append("confidence")
        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ObjectAttribute from a JSON dictionary.'''
        return cls(
            category=d.get("category", None),
            label=d.get("label", None),
            confidence=d.get("confidence", None),
        )


class ObjectAttributeContainer(DataContainer):
    '''A generic container for `ObjectAttribute`s or its subclasses.'''

    _ELE_CLS = ObjectAttribute
    _ELE_CLS_FIELD = "_ATTR_CLS"
    # Note: we can't use "attributes" here due to `Serialiable.attributes()`
    _ELE_ATTR = "attrs"

    @classmethod
    def _validate(cls):
        '''Validates that an ObjectAttributeContainer definition is valid.

        `ObjectAttributeContainer`s must only contain `ObjectAttribute`s or
        subclasses of them.
        '''
        super(ObjectAttributeContainer, cls)._validate()
        if not issubclass(cls._ELE_CLS, ObjectAttribute):
            raise ObjectAttributeContainerError(
                "%s is not an ObjectAttribute subclass" % cls._ELE_CLS)


class ObjectAttributeContainerError(Exception):
    '''Exception raised when an invalid ConfigContainer is encountered.'''
    pass


class DetectedObject(Serializable):
    '''A detected object in an image.

    Attributes:
        label: object label
        confidence: detection confidence
        bounding_box: a BoundingBox around the object
        index: (optional) an index assigned to the object
        frame_number: (optional) the frame number in which this object was
            detected
        attrs: (optional) an `ObjectAttributeContainer` describing additional
            attributes of the object
    '''

    def __init__(
            self, label, confidence, bounding_box, index=None,
            frame_number=None, attrs=None):
        '''Constructs a DetectedObject.

        Args:
            label: object label string
            confidence: detection confidence, in [0, 1]
            bounding_box: a BoundingBox around the object
            index: (optional) an index assigned to the object
            frame_number: (optional) the frame number in which this object was
                detected
            attrs: (optional) an `ObjectAttributeContainer` describing
                additional attributes of the object
        '''
        self.label = str(label)
        self.confidence = float(confidence)
        self.bounding_box = bounding_box
        self.index = index
        self.frame_number = frame_number
        self.attrs = attrs

    def extract_from(self, img, force_square=False):
        '''Extracts the subimage containing this object from the image.

        Args:
            img: an image
            force_square: whether to (minimally) manipulate the object bounding
                box during extraction so that the returned subimage is square
        '''
        return self.bounding_box.extract_from(img, force_square=force_square)

    def attributes(self):
        '''Returns the list of attributes to serialize.

        Optional attributes that were not provided (e.g. are None) are omitted
        from this list.
        '''
        _attrs = ["label", "confidence", "bounding_box"]
        if self.index is not None:
            _attrs.append("index")
        if self.frame_number is not None:
            _attrs.append("frame_number")
        if self.attrs is not None:
            _attrs.append("attrs")
        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs a DetectedObject from a JSON dictionary.'''
        return cls(
            d["label"],
            d["confidence"],
            BoundingBox.from_dict(d["bounding_box"]),
        )


class Frame(ObjectContainer):
    '''Container for detected objects in a frame.'''

    _ELE_CLS = DetectedObject

    def label_set(self):
        '''Returns a set containing the labels of the DetectedObjects.'''
        return set(obj.label for obj in self.objects)


class ObjectCount(Serializable):
    '''The number of instances of an object found in an image.'''

    def __init__(self, label, count):
        self.label = str(label)
        self.count = int(count)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ObjectCount from a JSON dictionary.'''
        return ObjectCount(d["label"], d["count"])


class ObjectCounts(DataContainer):
    '''Container for counting objects in an image.'''

    _ELE_CLS = ObjectCount
    _ELE_ATTR = "counts"


class ScoredObject(Serializable):
    '''A DetectedObject decorated with a score.

    Attributes:
        detected_object: a DetectedObject instance
        score: the score of the object
    '''

    def __init__(self, detected_object, score=None, index=None):
        '''Constructs a ScoredObject.'''
        self.detected_object = detected_object
        self.score = score
        self.index = index
        self._meta = None  # used by clients to store temporary metadata

    def extract_from(self, img, force_square=False):
        '''Extracts the subimage containing this object from the image.

        Args:
            img: an image
            force_square: whether to (minimally) manipulate the object bounding
                box during extraction so that the returned subimage is square
        '''
        return self.detected_object.extract_from(
            img, force_square=force_square)

    @classmethod
    def from_dict(cls, d):
        '''Constructs a ScoredObject from a JSON dictionary.'''
        return cls(
            DetectedObject.from_dict(d["detected_object"]),
            d["score"],
            d["index"],
        )


class ScoredObjects(ObjectContainer):
    '''Container for scored objects.'''

    _ELE_CLS = ScoredObject

    def sort(self):
        '''Sorts the current object list in ascending order by score.'''
        setattr(
            self, self._ELE_ATTR,
            sorted(self._data, key=lambda obj: obj.score)
        )
