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
import os

from eta.core.geometry import BoundingBox
import eta.core.image as etai
from eta.core.serial import Serializable
import eta.core.utils as etau


class ObjectContainer(Serializable):
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
    _OBJ_CLS = None

    def __init__(self, objects=None):
        '''Constructs an ObjectContainer.

        Args:
            objects: optional list of objects to store.
        '''
        self._validate()
        self.objects = objects or []

    def __iter__(self):
        return iter(self.objects)

    def serialize(self):
        '''Custom serialization implementation for ObjectContainers that embeds
        the class name and the object class name in the JSON to enable
        reflective parsing when reading from disk.
        '''
        d = OrderedDict()
        d["_CLS"] = etau.get_class_name(self)
        d["_OBJ_CLS"] = etau.get_class_name(self._OBJ_CLS)
        d["objects"] = [o.serialize() for o in self.objects]
        return d

    @classmethod
    def get_object_class(cls):
        '''Gets the class of object stored in this container.'''
        return cls._OBJ_CLS

    @property
    def num_objects(self):
        '''The number of objects in the container.'''
        return len(self.objects)

    def add(self, obj):
        '''Adds an object to the container.

        Args:
            obj: an object instance
        '''
        self.objects.append(obj)

    def get_matches(self, filters, match=any):
        '''Returns an object container containing only objects that match the
        filters.

        Args:
            filters: a list of functions that accept objects and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
        '''
        return self.__class__(
            objects=list(filter(
                lambda o: match(f(o) for f in filters),
                self.objects,
            )),
        )

    def count_matches(self, filters, match=any):
        '''Counts number of objects that match the filters.

        Args:
            filters: a list of functions that accept objects and return
                True/False
            match: a function (usually `any` or `all`) that accepts an iterable
                and returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                `any`
        '''
        return len(self.get_matches(filters, match=match).objects)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ObjectContainer from a JSON dictionary.

        If the JSON contains the reflective `_CLS` and `_OBJ_CLS` fields, they
        are used to infer the underlying object classes, and this method can
        be invoked as `ObjectContainer.from_dict`. Otherwise, this method must
        be called on a concrete subclass of `ObjectContainer`.
        '''
        if "_CLS" in d and "_OBJ_CLS" in d:
            # Parse reflectively
            cls = etau.get_class(d["_CLS"])
            obj_cls = etau.get_class(d["_OBJ_CLS"])
        else:
            # Parse using provided class
            cls._validate()
            obj_cls = cls._OBJ_CLS
        return cls(objects=[obj_cls.from_dict(do) for do in d["objects"]])

    @classmethod
    def _validate(cls):
        if cls._OBJ_CLS is None:
            raise ValueError(
                "_OBJ_CLS is None; note that you cannot instantiate "
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

    _OBJ_CLS = DetectedObject

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

    _OBJ_CLS = ScoredObject

    def sort(self):
        '''Sorts the current object list in ascending order by score.'''
        self.objects = sorted(self.objects, key=lambda o: o.score)
