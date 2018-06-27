'''
Core data structures for working with objects in images.

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

from collections import OrderedDict

from eta.core.data import DataContainer
from eta.core.geometry import BoundingBox
from eta.core.serial import Serializable
import eta.core.utils as etau


class ObjectContainer(DataContainer):
    '''Base class for containers that store lists of objects.

    This class should not be instantiated directly. Instead a subclass should
    be created for each type of object to be stored.

    By default, ObjectContainer subclasses embed their class names and
    underlying object class names in their JSON representations, so object
    containers can be read reflectively from disk.

    Examples:
        ```
        frame = Frame(...)
        frame.write_json("frame.json")
        frame2 = ObjectContainer.from_json("frame.json")
        print(frame2.__class__)  # Frame, not ObjectContainer
        ```

    Attributes:
        objects: a list of objects
    '''

    # The class of the objects stored in the container
    _DATA_CLS = None
    _DATA_ATTR = "objects"

    @classmethod
    def get_object_class(cls):
        '''Gets the class of object stored in this container.'''
        return cls._DATA_CLS

    @property
    def num_objects(self):
        '''The number of objects in the container.'''
        return len(self.objects)

    @classmethod
    def _validate(cls):
        if cls._DATA_CLS is None:
            raise ValueError(
                "_DATA_CLS is None; note that you cannot instantiate "
                "ObjectContainer directly."
            )


class DetectedObject(Serializable):
    '''A detected object in an image.

    Attributes:
        label: object label
        confidence: detection confidence
        bounding_box: a BoundingBox around the object
    '''

    def __init__(self, label, confidence, bounding_box):
        '''Constructs a DetectedObject.

        Args:
            label: object label string
            confidence: detection confidence, in [0, 1]
            bounding_box: a BoundingBox around the object
        '''
        self.label = str(label)
        self.confidence = float(confidence)
        self.bounding_box = bounding_box

    def extract_from(self, img, force_square=False):
        '''Extracts the subimage containing this object from the image.

        Args:
            img: an image
            force_square: whether to (minimally) manipulate the object bounding
                box during extraction so that the returned subimage is square
        '''
        return self.bounding_box.extract_from(img, force_square=force_square)

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

    _DATA_CLS = DetectedObject

    def label_set(self):
        '''Returns a set containing the labels of the DetectedObjects.'''
        return set(obj.label for obj in self.objects)


class ObjectCounts(Serializable):
    '''Container for counting objects in an image.'''

    def __init__(self, counts=None):
        '''Constructs an ObjectCounts container.

        Args:
            counts: optional list of ObjectCount objects
        '''
        self.counts = counts or []

    def add(self, count):
        '''Adds an ObjectCount to the container.'''
        self.counts.append(count)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ObjectCounts from a JSON dictionary.'''
        return ObjectCounts(
            counts=[ObjectCount.from_dict(dc) for dc in d["counts"]]
        )


class ObjectCount(Serializable):
    '''The number of instances of an object found in an image.'''

    def __init__(self, label, count):
        self.label = str(label)
        self.count = int(count)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ObjectCount from a JSON dictionary.'''
        return ObjectCount(d["label"], d["count"])


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

    _DATA_CLS = ScoredObject

    def sort(self):
        '''Sorts the current object list in ascending order by score.'''
        self.objects = sorted(self.objects, key=lambda o: o.score)
