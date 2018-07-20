'''
Core data structures for working with geometric concepts like points,
bounding boxes, etc.

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
import eta.core.image as etai
import eta.core.numutils as etan
from eta.core.serial import Serializable


class BoundingBox(Serializable):
    '''A bounding box in an image.

    The bounding box is represented as two RelativePoint instances describing
    the top-left and bottom-right corners of the box, respectively.
    '''

    def __init__(self, top_left, bottom_right):
        '''Constructs a BoundingBox from two RelativePoint instances.'''
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __str__(self):
        return "%s x %s" % (self.top_left, self.bottom_right)

    def __eq__(self, bbox):
        return (
            self.top_left == bbox.top_left and
            self.bottom_right == bbox.bottom_right
        )

    def coords_in(self, **kwargs):
        '''Returns the coordinates of the bounding box in the specified image.

        Args:
            **kwargs: a valid argument for etai.to_frame_size()

        Returns:
            box: a (top-left-x, top-left-y, width, height) tuple describing the
                bounding box
        '''
        tl = self.top_left.coords_in(**kwargs)
        br = self.bottom_right.coords_in(**kwargs)
        return tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]

    def extract_from(self, img, force_square=False):
        '''Extracts the subimage defined by this bounding box from the image.

        Args:
            img: an image
            force_square: whether to (minimally) manipulate the bounding box
                during extraction so that the returned subimage is square

        Returns:
            the extracted subimage
        '''
        x1, y1 = self.top_left.coords_in(img=img)
        x2, y2 = self.bottom_right.coords_in(img=img)
        x = slice(x1, x2)
        y = slice(y1, y2)
        if force_square:
            h, w = img.shape[:2]
            x, y = _make_square(x, y, w, h)
        return img[y, x, ...]

    def pad_relative(self, alpha):
        '''Returns a bounding box whose length and width are expanded (or
        shrunk, when alpha < 0) by (100 * alpha)%.

        The coordinates are clamped to [0, 1] x [0, 1] if necessary.

        Args:
            alpha: the desired padding relative to the size of this
                bounding box; a float in [-1, \inf)

        Returns:
            the padded bounding box
        '''
        w = self.bottom_right.x - self.top_left.x
        h = self.bottom_right.y - self.top_left.y

        alpha = max(alpha, -1)
        wpad = 0.5 * alpha * w
        hpad = 0.5 * alpha * h

        tlx, tly = RelativePoint.clamp(
            self.top_left.x - wpad, self.top_left.y - hpad)
        brx, bry = RelativePoint.clamp(
            self.bottom_right.x + wpad, self.bottom_right.y + hpad)

        return BoundingBox(RelativePoint(tlx, tly), RelativePoint(brx, bry))

    def area(self):
        '''Computes the area of the bounding box, in [0, 1].'''
        w = self.bottom_right.x - self.top_left.x
        h = self.bottom_right.y - self.top_left.y
        return w * h

    def get_intersection(self, bbox):
        '''Returns the bounding box describing the intersection of this
        bounding box with the given bounding box.

        If the bounding boxes do not intersect, an empty bounding box is
        returned.

        Args:
            bbox: a BoundingBox

        Returns:
            a bounding box describing the intersection
        '''
        tlx = max(self.top_left.x, bbox.top_left.x)
        tly = max(self.top_left.y, bbox.top_left.y)
        brx = min(self.bottom_right.x, bbox.bottom_right.x)
        bry = min(self.bottom_right.y, bbox.bottom_right.y)

        if (brx - tlx < 0) or (bry - tly < 0):
            return BoundingBox.empty()

        return BoundingBox(RelativePoint(tlx, tly), RelativePoint(brx, bry))

    def contains_box(self, bbox):
        '''Determines if this bounding box contains the given bounding box.

        Args:
            bbox: a BoundingBox

        Returns:
            True/False
        '''
        return self.get_intersection(bbox) == bbox

    def overlap_ratio(self, bbox):
        '''Computes the overlap ratio with the given bounding box, defined as
        the area of the intersection divided by the area of the union of the
        two bounding boxes.

        Args:
            bbox: a BoundingBox

        Returns:
            the overlap ratio, in [0, 1]
        '''
        inter_area = self.get_intersection(bbox).area()
        union_area = self.area() + bbox.area() - inter_area
        try:
            return inter_area / union_area
        except ZeroDivisionError:
            return 0.0

    @classmethod
    def empty(cls):
        '''Returns an empty bounding box.'''
        return cls(RelativePoint.origin(), RelativePoint.origin())

    @classmethod
    def from_dict(cls, d):
        '''Constructs a BoundingBox from a JSON dictionary.'''
        return cls(
            RelativePoint.from_dict(d["top_left"]),
            RelativePoint.from_dict(d["bottom_right"]),
        )


class HasBoundingBox(object):
    '''Mixin to explicitly indicate that an instance has a bounding box.'''

    def get_bounding_box(self):
        raise NotImplementedError(
            "Classes implementing HasBoundingBox need to implement the "
            "get_bounding_box method.")


class RelativePoint(Serializable):
    '''A point in an image, represented as (x, y) coordinates in [0, 1].'''

    def __init__(self, x, y):
        '''Construct a relative point from (x, y) coordinates.

        Args:
            x: a number in [0, 1]
            y: a number in [0, 1]
        '''
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return "(%.3f, %.3f)" % (self.x, self.y)

    def __eq__(self, rel_point):
        return (
            etan.is_close(self.x, rel_point.x) and
            etan.is_close(self.y, rel_point.y)
        )

    def coords_in(self, **kwargs):
        '''Returns the absolute (x, y) coordinates of this point in the
        specified image.

        Args:
            **kwargs: a valid argument for etai.to_frame_size()

        Returns:
            (x, y): the absolute x and y coordinates of this point
        '''
        w, h = etai.to_frame_size(**kwargs)
        return int(w * 1.0 * self.x), int(h * 1.0 * self.y)

    @staticmethod
    def clamp(x, y):
        '''Clamps the (x, y) coordinates to [0, 1].'''
        return max(0, min(x, 1)), max(0, min(y, 1))

    @classmethod
    def from_abs(cls, x, y, **kwargs):
        '''Constructs a RelativePoint from absolute (x, y) pixel coordinates.

        Args:
            **kwargs: a valid argument for etai.to_frame_size()
        '''
        w, h = etai.to_frame_size(**kwargs)
        x /= 1.0 * w
        y /= 1.0 * h
        return cls(x, y)

    @classmethod
    def origin(cls):
        '''Returns a relative point at the origin.'''
        return cls(0, 0)

    @classmethod
    def from_dict(cls, d):
        '''Constructs a RelativePoint from a JSON dictionary.'''
        return cls(d["x"], d["y"])


class LabeledPoint(Serializable):
    '''A relative point that has an associated label.

    Attributes:
        label: object label
        relative_point: a RelativePoint instance
    '''

    def __init__(self, label, relative_point):
        '''Constructs a LabeledPoint.

        Args:
            label: label string
            relative_point: a RelativePoint instance
        '''
        self.label = str(label)
        self.relative_point = relative_point

    @classmethod
    def from_dict(cls, d):
        '''Constructs a LabeledPoint from a JSON dictionary.'''
        return cls(
            d["label"],
            RelativePoint.from_dict(d["relative_point"]),
        )


class LabeledPointContainer(DataContainer):
    '''Container for points in an image that each have an associated label.'''

    _ELE_CLS = LabeledPoint
    _ELE_ATTR = "points"

    def label_set(self):
        '''Returns a set containing the labels of the LabeledPoints.'''
        return set(p.label for p in self.points)


def _make_square(x, y, w, h):
    '''Force the x, y slices into a square by expanding the smaller dimension.

    If the smaller dimension can't be expanded enough and still fit
    in the maximum allowed size, the larger dimension is contracted as needed.

    Args:
        x, y: slice objects
        w, h: the (width, height) of the maximum allowed size

    Returns:
        x and y slices that define a square
    '''
    ws = x.stop - x.start
    hs = y.stop - y.start
    dx = hs - ws
    if dx < 0:
        return _make_square(y, x, h, w)[::-1]

    # subimage is now always skinny

    def pad(z, dz, zmax):
        dz1 = int(0.5 * dz)
        dz2 = dz - dz1
        ddz = max(0, dz1 - z.start) - max(0, z.stop + dz2 - zmax)
        return slice(z.start - dz1 + ddz, z.stop + dz2 + ddz)

    dy = min(0, w - dx - ws)
    return pad(x, dx + dy, w), pad(y, dy, h)
