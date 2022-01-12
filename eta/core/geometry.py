"""
Core data structures for working with geometric concepts like points and
bounding boxes.

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

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import eta.core.numutils as etan
from eta.core.serial import BigContainer, Container, Serializable, Set, BigSet


def compute_minimal_covering_box(bounding_box, *args):
    """Computes the minimal covering BoundingBox for the given BoundingBoxes.

    Args:
        bounding_box: a BoundingBox
        *args: additional `BoundingBox`s

    Returns:
        the minimal covering BoundingBox
    """
    tlx, tly, brx, bry = bounding_box.to_coords()

    for bbox in args:
        tlx = min(tlx, bbox.top_left.x)
        tly = min(tly, bbox.top_left.y)
        brx = max(brx, bbox.bottom_right.x)
        bry = max(bry, bbox.bottom_right.y)

    return BoundingBox.from_coords(tlx, tly, brx, bry)


class BoundingBox(Serializable):
    """A bounding box in an image.

    The bounding box is represented as two RelativePoint instances describing
    the top-left and bottom-right corners of the box, respectively.

    ETA follows the convention that the top-left corner of the image is [0, 0]
    and the bottom-right corner of the image is [1, 1]. Thus, proper bounding
    boxes satisfy the convention that their bottom-right coordinates are always
    greater than their top-left coordinates.
    """

    def __init__(self, top_left, bottom_right):
        """Creates a BoundingBox instance.

        Args:
            top_left: a RelativePoint describing the top-left corner of the box
            bottom_right: a RelativePoint describing the bottom-right corner of
                the box
        """
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __str__(self):
        return "%s x %s" % (self.top_left, self.bottom_right)

    def __eq__(self, bbox):
        return (
            self.top_left == bbox.top_left
            and self.bottom_right == bbox.bottom_right
        )

    @property
    def top_right(self):
        """Returns a top right RelativePoint"""
        return RelativePoint(self.bottom_right.x, self.top_left.y)

    @property
    def bottom_left(self):
        """Returns a bottom left RelativePoint"""
        return RelativePoint(self.top_left.x, self.bottom_right.y)

    @property
    def corners(self):
        """Returns all four RelativePoint corners in clockwise order starting
        with the top left corner.

        Returns:
            (top_left, top_right, bottom_right, bottom_left) tuple of
            RelativePoints
        """
        return (
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
        )

    @property
    def is_proper(self):
        """Whether the bounding box is proper, i.e., its top-left coordinate
        lies to the left and above its bottom right coordinate.
        """
        return self.height() >= 0 and self.width() >= 0

    def ensure_proper(self):
        """Ensures that the bounding box if proper by swapping its coordinates
        as necessary.
        """
        if self.height() < 0:
            tly, bry = self.bottom_right.y, self.top_left.y
            self.top_left.y = tly
            self.bottom_right.y = bry

        if self.width() < 0:
            tlx, brx = self.bottom_right.x, self.top_left.x
            self.top_left.x = tlx
            self.bottom_right.x = brx

    def coords_in(self, frame_size=None, shape=None, img=None):
        """Returns the coordinates of the bounding box in the specified image.

        Pass *one* keyword argument to this function.

        Args:
            frame_size: the (width, height) of the image
            shape: the (height, width, ...) of the image, e.g. from img.shape
            img: the image itself

        Returns:
            a (top-left-x, top-left-y, width, height) tuple describing the
                bounding box
        """
        tl = self.top_left.coords_in(
            frame_size=frame_size, shape=shape, img=img
        )
        br = self.bottom_right.coords_in(
            frame_size=frame_size, shape=shape, img=img
        )
        return tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]

    def aspect_ratio_in(self, frame_size=None, shape=None, img=None):
        """Returns the aspect ratio of the bounding box in the specified image.

        Pass *one* keyword argument to this function.

        Args:
            frame_size: the (width, height) of the image
            shape: the (height, width, ...) of the image, e.g. from img.shape
            img: the image itself

        Returns:
            the aspect ratio of the box
        """
        w, h = _to_frame_size(frame_size=frame_size, shape=shape, img=img)
        tl = self.top_left
        br = self.bottom_right
        return (br.x - tl.x) * w / (h * (br.y - tl.y))

    def extract_from(self, img, force_square=False):
        """Extracts the subimage defined by this bounding box from the image.

        Args:
            img: an image
            force_square: whether to (minimally) manipulate the bounding box
                during extraction so that the returned subimage is square

        Returns:
            the extracted subimage
        """
        x1, y1 = self.top_left.coords_in(img=img)
        x2, y2 = self.bottom_right.coords_in(img=img)
        x = slice(x1, x2)
        y = slice(y1, y2)
        if force_square:
            h, w = img.shape[:2]
            x, y = _make_square(x, y, w, h)
        return img[y, x, ...]

    def pad_relative(self, alpha):
        """Returns a bounding box whose length and width are expanded (or
        shrunk, when alpha < 0) by (100 * alpha)%.

        The coordinates are clamped to [0, 1] x [0, 1] if necessary.

        Args:
            alpha: the desired padding relative to the size of this bounding
                box; a float in [-1, \\inf)

        Returns:
            the padded BoundingBox
        """
        w = self.bottom_right.x - self.top_left.x
        h = self.bottom_right.y - self.top_left.y

        alpha = max(alpha, -1)
        wpad = 0.5 * alpha * w
        hpad = 0.5 * alpha * h

        tlx, tly = RelativePoint.clamp(
            self.top_left.x - wpad, self.top_left.y - hpad
        )
        brx, bry = RelativePoint.clamp(
            self.bottom_right.x + wpad, self.bottom_right.y + hpad
        )

        return BoundingBox.from_coords(tlx, tly, brx, bry)

    def height(self):
        """Computes the height of the bounding box, in [0, 1].

        Returns:
            the height
        """
        return self.bottom_right.y - self.top_left.y

    def width(self):
        """Computes the width of the bounding box, in [0, 1].

        Returns:
            the width
        """
        return self.bottom_right.x - self.top_left.x

    def area(self):
        """Computes the area of the bounding box, in [0, 1].

        Returns:
            the area
        """
        return self.width() * self.height()

    def centroid(self):
        """Computes the cenroid of the bounding box.

        Returns:
            a RelativePoint
        """
        xc = 0.5 * (self.top_left.x + self.bottom_right.x)
        yc = 0.5 * (self.top_left.y + self.bottom_right.y)
        return RelativePoint(xc, yc)

    def get_intersection(self, bbox):
        """Returns the bounding box describing the intersection of this
        bounding box with the given bounding box.

        If the bounding boxes do not intersect, an empty bounding box is
        returned.

        Args:
            bbox: a BoundingBox

        Returns:
            a bounding box describing the intersection
        """
        tlx = max(self.top_left.x, bbox.top_left.x)
        tly = max(self.top_left.y, bbox.top_left.y)
        brx = min(self.bottom_right.x, bbox.bottom_right.x)
        bry = min(self.bottom_right.y, bbox.bottom_right.y)

        if (brx - tlx < 0) or (bry - tly < 0):
            return BoundingBox.empty()

        return BoundingBox.from_coords(tlx, tly, brx, bry)

    def contains_box(self, bbox):
        """Determines if this bounding box contains the given bounding box.

        Args:
            bbox: a BoundingBox

        Returns:
            True/False
        """
        return self.get_intersection(bbox) == bbox

    def compute_overlap(self, bbox):
        """Computes the proportion of this bounding box that overlaps the given
        bounding box.

        Args:
            bbox: a BoundingBox

        Returns:
            the overlap, in [0, 1]
        """
        try:
            inter_area = self.get_intersection(bbox).area()
            return inter_area / self.area()
        except ZeroDivisionError:
            return 0.0

    def compute_iou(self, bbox):
        """Computes the IoU (intersection over union) of the bounding boxes.

        The IoU is defined as the area of the intersection of the boxes divided
        by the area of their union.

        Args:
            bbox: a BoundingBox

        Returns:
            the IoU, in [0, 1]
        """
        inter_area = self.get_intersection(bbox).area()
        union_area = self.area() + bbox.area() - inter_area
        try:
            return inter_area / union_area
        except ZeroDivisionError:
            return 0.0

    @classmethod
    def empty(cls):
        """Returns a BoundingBox whose top-left and bottom-right corners are
        both at the origin.

        Returns:
            a BoundingBox
        """
        return cls(RelativePoint.origin(), RelativePoint.origin())

    def to_coords(self):
        """Returns a tuple containing the top-left and bottom-right coordinates
        of the bounding box.

        Returns:
            a (tlx, tly, brx, bry) tuple
        """
        return self.top_left.to_tuple() + self.bottom_right.to_tuple()

    @classmethod
    def from_coords(cls, tlx, tly, brx, bry, clamp=True):
        """Constructs a BoundingBox from top-left and bottom-right coordinates.

        Args:
            tlx: the top-left x coordinate
            tly: the top-left y coordinate
            brx: the bottom-right x coordinate
            bry: the bottom-right y coordinate
            clamp: whether to clamp the bounding box to [0, 1] x [0, 1], if
                necessary. By default, this is True

        Returns:
            a BoundingBox
        """
        top_left = RelativePoint.from_coords(tlx, tly, clamp=clamp)
        bottom_right = RelativePoint.from_coords(brx, bry, clamp=clamp)
        return cls(top_left, bottom_right)

    @classmethod
    def from_abs_coords(
        cls,
        tlx,
        tly,
        brx,
        bry,
        clamp=True,
        frame_size=None,
        shape=None,
        img=None,
    ):
        """Constructs a BoundingBox from absolute top-left and bottom-right
        coordinates.

        One of `frame_size`, `shape`, or `img` must be provided.

        Args:
            tlx: the absolute top-left x coordinate
            tly: the absolute top-left y coordinate
            brx: the absolute bottom-right x coordinate
            bry: the absolute bottom-right y coordinate
            clamp: whether to clamp the bounding box to [0, 1] x [0, 1], if
                necessary. By default, this is True
            frame_size: the (width, height) of the image
            shape: the (height, width, ...) of the image, e.g. from img.shape
            img: the image itself

        Returns:
            a BoundingBox
        """
        top_left = RelativePoint.from_abs_coords(
            tlx, tly, clamp=clamp, frame_size=frame_size, shape=shape, img=img
        )
        bottom_right = RelativePoint.from_abs_coords(
            brx, bry, clamp=clamp, frame_size=frame_size, shape=shape, img=img
        )
        return cls(top_left, bottom_right)

    @classmethod
    def from_dict(cls, d):
        """Constructs a BoundingBox from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a BoundingBox
        """
        return cls(
            RelativePoint.from_dict(d["top_left"]),
            RelativePoint.from_dict(d["bottom_right"]),
        )


class HasBoundingBox(object):
    """Mixin to explicitly indicate that an instance has a bounding box."""

    def get_bounding_box(self):
        """Gets the bounding box for the instance.

        Returns:
            a BoundingBox
        """
        raise NotImplementedError(
            "Subclasses must implement get_bounding_box()"
        )


class RelativePoint(Serializable):
    """A point in an image, represented as (x, y) coordinates in
    [0, 1] x [0, 1].
    """

    def __init__(self, x, y):
        """Constructs a RelativePoint instance.

        Args:
            x: a number in [0, 1]
            y: a number in [0, 1]
        """
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return "(%.3f, %.3f)" % (self.x, self.y)

    def __eq__(self, rel_point):
        return etan.is_close(self.x, rel_point.x) and etan.is_close(
            self.y, rel_point.y
        )

    def coords_in(self, frame_size=None, shape=None, img=None):
        """Returns the absolute (x, y) coordinates of this point in the
        specified image.

        Pass *one* keyword argument to this function.

        Args:
            frame_size: the (width, height) of the image
            shape: the (height, width, ...) of the image, e.g. from img.shape
            img: the image itself

        Returns:
            the absolute (x, y) coordinates of this point
        """
        w, h = _to_frame_size(frame_size=frame_size, shape=shape, img=img)
        return int(w * 1.0 * self.x), int(h * 1.0 * self.y)

    @staticmethod
    def clamp(x, y):
        """Clamps the (x, y) coordinates to [0, 1].

        Args:
            x: x coordinate
            y: y coordinate

        Returns:
            (x, y) clamped to [0, 1] x [0, 1]
        """
        return max(0, min(x, 1)), max(0, min(y, 1))

    def to_tuple(self):
        """Returns a tuple representation of the point.

        Returns:
            an (x, y) tuple
        """
        return (self.x, self.y)

    @classmethod
    def from_coords(cls, x, y, clamp=True):
        """Constructs a RelativePoint from (x, y) coordinates.

        Args:
            x: the x coordinate
            y: the y coordinate
            clamp: whether to clamp the point to [0, 1] if necessary. By
                default, this is True

        Returns:
            a RelativePoint
        """
        if clamp:
            x, y = cls.clamp(x, y)

        return cls(x, y)

    @classmethod
    def from_abs_coords(
        cls, x, y, clamp=True, frame_size=None, shape=None, img=None
    ):
        """Constructs a RelativePoint from absolute (x, y) pixel coordinates.

        One of `frame_size`, `shape`, or `img` must be provided.

        Args:
            x: the absolute x coordinate
            y: the absolute y coordinate
            clamp: whether to clamp the point to [0, 1] if necessary. By
                default, this is True
            frame_size: the (width, height) of the image
            shape: the (height, width, ...) of the image, e.g. from img.shape
            img: the image itself

        Returns:
            a RelativePoint instance
        """
        # Convert to relative coordinates
        w, h = _to_frame_size(frame_size=frame_size, shape=shape, img=img)
        x /= 1.0 * w
        y /= 1.0 * h

        return cls.from_coords(x, y, clamp=clamp)

    @classmethod
    def origin(cls):
        """Returns a relative point at the origin.

        Returns:
            a RelativePoint at (0, 0)
        """
        return cls(0, 0)

    @classmethod
    def from_dict(cls, d):
        """Constructs a RelativePoint from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a RelativePoint
        """
        return cls(d["x"], d["y"])


class LabeledPoint(Serializable):
    """A RelativePoint with an associated label."""

    def __init__(self, label, relative_point):
        """Constructs a LabeledPoint instance.

        Args:
            label: label string
            relative_point: a RelativePoint instance
        """
        self.label = label
        self.relative_point = relative_point

    @classmethod
    def from_dict(cls, d):
        """Constructs a LabeledPoint from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a LabeledPoint
        """
        return cls(d["label"], RelativePoint.from_dict(d["relative_point"]))


class LabeledPointContainer(Container):
    """Container for points in an image that each have an associated label."""

    _ELE_CLS = LabeledPoint
    _ELE_ATTR = "points"

    def get_labels(self):
        """Returns a set containing the labels of the LabeledPoints."""
        return set(p.label for p in self)


class BigLabeledPointContainer(LabeledPointContainer, BigContainer):
    """Big container for points in an image that each have an associated label.

    As a BigContainer, each LabeledPoint is stored individually on disk.
    """

    pass


class LabeledPointSet(Set):
    """Set for points in an image that each have an associated label."""

    _ELE_CLS = LabeledPoint
    _ELE_ATTR = "points"
    _ELE_KEY_ATTR = "label"

    def get_labels(self):
        """Returns a set containing the labels of the LabeledPoints."""
        return set(p.label for p in self)


class BigLabeledPointSet(LabeledPointSet, BigSet):
    """Big set for points in an image that each have an associated label.

    As a BigSet, each LabeledPoint is stored individually on disk.
    """

    pass


def _to_frame_size(frame_size=None, shape=None, img=None):
    if img is not None:
        shape = img.shape
    if shape is not None:
        return shape[1], shape[0]
    if frame_size is not None:
        return tuple(frame_size)
    raise TypeError("A valid keyword argument must be provided")


def _make_square(x, y, w, h):
    """Force the x, y slices into a square by expanding the smaller dimension.

    If the smaller dimension can't be expanded enough and still fit
    in the maximum allowed size, the larger dimension is contracted as needed.

    Args:
        x, y: slice objects
        w, h: the (width, height) of the maximum allowed size

    Returns:
        x and y slices that define a square
    """
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
