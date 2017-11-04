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

    def matches_any(self, filters):
        '''Keep detected objects that match at least one filter.

        Args:
            filters: a list of functions that accept DetectedObjects and return
                True/False
        '''
        self.objects = list(filter(
            lambda o: any(f(o) for f in filters),
            self.objects,
        ))

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
