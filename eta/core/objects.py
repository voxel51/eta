'''
Core data structures for working with objects in images.

Copyright 2017, Voxel51, LLC
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

from eta.core.geometry import BoundingBox
from eta.core.serial import Serializable


class Frame(Serializable):
    '''Container for detected objects in an image.'''

    def __init__(self, objects=None):
        '''Constructs a Frame.

        Args:
            objects: optional list of DetectedObjects in the frame.
        '''
        self.objects = objects or []

    def add(self, obj):
        '''Adds a DetectedObject to the frame.

        Args:
            obj: A DetectedObject instance
        '''
        self.objects.append(obj)

    def label_set(self):
        '''Returns a set containing the labels of objects in this frame.'''
        return set(obj.label for obj in self.objects)

    def get_matches(self, filters, match=any):
        '''Returns a Frame containing only objects that match the filters.

        Args:
            filters: a list of functions that accept DetectedObjects and return
                True/False
            match: a function (usually any or all) that accepts an iterable and
                returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                any
        '''
        return Frame(
            objects=list(filter(
                lambda o: match(f(o) for f in filters),
                self.objects,
            )),
        )

    def count_matches(self, filters, match=any):
        '''Counts number of detected objects that match the filters.

        Args:
            filters: a list of functions that accept DetectedObjects and return
                True/False
            match: a function (usually any or all) that accepts an iterable and
                returns True/False. Used to aggregate the outputs of each
                filter to decide whether a match has occurred. The default is
                any
        '''
        return len(self.get_matches(filters, match=match).objects)

    @classmethod
    def from_dict(cls, d):
        '''Constructs a Frame from a JSON dictionary.'''
        return Frame(objects=[
            DetectedObject.from_dict(do) for do in d["objects"]
        ])


class DetectedObject(Serializable):
    '''A detected object in an image.

    Attributes:
        label: object label
        confidence: detection confidence
        bounding_box: A BoundingBox around the object
    '''

    def __init__(self, label, confidence, bounding_box):
        '''Constructs a DetectedObject.

        Args:
            label: object label string
            confidence: detection confidence, in [0, 1]
            bounding_box: A BoundingBox around the object
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
        return ObjectCounts(counts=[
            ObjectCount.from_dict(dc) for dc in d["counts"]
        ])


class ObjectCount(Serializable):
    '''The number of instances of an object found in an image.'''

    def __init__(self, label, count):
        self.label = str(label)
        self.count = int(count)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ObjectCount from a JSON dictionary.'''
        return ObjectCount(d["label"], d["count"])
