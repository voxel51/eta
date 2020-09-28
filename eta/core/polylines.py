"""
Core data structures for working with geometric concepts like points and
bounding boxes.

Copyright 2017-2020, Voxel51, Inc.
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

import eta.core.data as etad
from eta.core.serial import Container, Serializable
import eta.core.utils as etau


# @todo make Polyline a subclass of `eta.core.labels.Labels`
class Polyline(Serializable):
    """A list of points describing a polyline or polygon in an image.

    :class:`Polyline` is a spatial concept that describes information about a
    polyline (collection of line segments) or polygon in a particular image or
    a particular frame of a video. A :class:`Polyline` can have a name (i.e.,
    a class), a label, a list of vertices, and one or more additional
    attributes describing its properties. The shape can be closed, filled, or
    neither.

    Attributes:
        type: the fully-qualified class name of the polyline
        name: (optional) the name for the polyline, e.g., ``ground_truth`` or
            the name of the model that produced it
        label: (optional) polyline label
        points: a list of (x, y) points in [0, 1] x [0, 1]
            describing the vertices of the polyline
        closed: whether the polyline is closed, i.e., an edge should be drawn
            from the last vertex to the first vertex
        filled: whether the polyline represents a shape that can be filled when
            rendering it
        attrs: (optional) an :class:`eta.core.data.AttributeContainer` of
            attributes for the object

    Args:
        name (None): a name for the polyline, e.g., ``ground_truth`` or the
            name of the model that produced it
        label (None): a label for the polyline
        points (None): a list of (x, y) points in [0, 1] x [0, 1]
            describing the vertices of a curve
        closed (False): whether the polyline is closed, i.e., an edge
            should be drawn from the last vertex to the first vertex
        filled (False): whether the polyline represents a shape that can
            be filled when rendering it
        attrs (None): an :class:`eta.core.data.AttributeContainer` of
            attributes for the polyline
    """

    def __init__(
        self,
        name=None,
        label=None,
        points=None,
        closed=False,
        filled=False,
        attrs=None,
    ):
        self.type = etau.get_class_name(self)
        self.name = name
        self.label = label
        self.points = points or []
        self.closed = closed
        self.filled = filled
        self.attrs = attrs or etad.AttributeContainer()

    def __bool__(self):
        # @todo remove when Polyline(Labels)
        return not self.is_empty

    def __len__(self):
        return len(self.points)

    @property
    def is_empty(self):
        """Whether the polyline has labels of any kind."""
        return not (
            self.has_label
            or self.has_vertices
            or self.has_name
            or self.has_attributes
        )

    @property
    def has_label(self):
        """Whether the polyline has a ``label``."""
        return self.label is not None

    @property
    def has_vertices(self):
        """Whether the polyline has at least one vertex."""
        return bool(self.points)

    @property
    def has_name(self):
        """Whether the polyline has a ``name``."""
        return self.name is not None

    @property
    def has_attributes(self):
        """Whether the polyline has attributes."""
        return bool(self.attrs)

    def add_attribute(self, attr):
        """Adds the attribute to the polyline.

        Args:
            attr: an :class:`eta.core.data.Attribute`
        """
        self.attrs.add(attr)

    def add_attributes(self, attrs):
        """Adds the attributes to the polyline.

        Args:
            attrs: an :class:`eta.core.data.AttributeContainer`
        """
        self.attrs.add_container(attrs)

    def pop_attributes(self):
        """Pops the attributes from the polyline.

        Returns:
            an :class:`eta.core.data.AttributeContainer`
        """
        attrs = self.attrs
        self.clear_attributes()
        return attrs

    def clear_attributes(self):
        """Removes all attributes from the polyline."""
        self.attrs = etad.AttributeContainer()

    def coords_in(self, frame_size=None, shape=None, img=None):
        """Returns the coordinates of the polyline vertices in the specified
        image.

        Pass *one* keyword argument to this function.

        Args:
            frame_size (None): the ``(width, height)`` of the image
            shape (None): the ``(height, width, ...)`` of the image, e.g. from
                ``img.shape``
            img (None): the image itself

        Returns:
            a list of ``(x, y)`` vertices in pixels
        """
        w, h = _to_frame_size(frame_size=frame_size, shape=shape, img=img)
        return [(int(round(x * w)), int(round(y * h))) for x, y in self.points]

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        """
        _attrs = []
        if self.name:
            _attrs.append("name")
        if self.label:
            _attrs.append("label")
        if self.closed:
            _attrs.append("closed")
        if self.filled:
            _attrs.append("filled")
        if self.attrs:
            _attrs.append("attrs")
        return _attrs + ["points"]

    @classmethod
    def from_abs_coords(
        cls,
        points,
        clamp=True,
        frame_size=None,
        shape=None,
        img=None,
        **kwargs,
    ):
        """Constructs a :class:`Polyline` from absolute pixel coordinates.

        One of ``frame_size``, ``shape``, or ``img`` must be provided.

        Args:
            points: a list of ``(x, y)``` keypoints in ``[0, 1] x [0, 1]``
                describing the vertices of the polyline
            clamp (True): whether to clamp the relative points to
                ``[0, 1] x [0, 1]``, if necessary
            frame_size (None): the ``(width, height)`` of the image
            shape (None): the ``(height, width, ...)`` of the image, e.g. from
                ``img.shape``
            img (None): the image itself
            **kwargs: additional keyword arguments for
                ``Polyline(points=points, **kwargs)``

        Returns:
            a :class:`Polyline`
        """
        w, h = _to_frame_size(frame_size=frame_size, shape=shape, img=img)

        rpoints = []
        for x, y in points:
            xr = x / w
            yr = y / h
            if clamp:
                xr = max(0, min(xr, 1))
                yr = max(0, min(yr, 1))

            rpoints.append((xr, yr))

        return cls(points=rpoints, **kwargs)

    @classmethod
    def from_dict(cls, d):
        """Constructs a :class:`Polyline` from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a :class:`Polyline`
        """
        name = d.get("name", None)
        label = d.get("label", None)
        points = d.get("points", None)
        closed = d.get("closed", False)
        filled = d.get("filled", False)

        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainer.from_dict(attrs)

        return cls(
            name=name,
            label=label,
            points=points,
            closed=closed,
            filled=filled,
            attrs=attrs,
        )


class PolylineContainer(Container):
    """A ``eta.core.serial.Container`` of :class:`Polyline` instances."""

    _ELE_CLS = Polyline
    _ELE_ATTR = "polylines"

    def get_labels(self):
        """Returns the set of ``label`` values of all polylines in the
        container.

        Returns:
            a set of labels
        """
        return set(polyline.label for polyline in self)

    def remove_polylines_without_attrs(self, labels=None):
        """Removes polylines from this container that do not have attributes.

        Args:
            labels (None): an optional list of :class:`Polyline` label strings
                to which to restrict attention when filtering. By default, all
                polylines are processed
        """
        filter_func = lambda polyline: (
            (labels is not None and polyline.label not in labels)
            or polyline.has_attributes
        )
        self.filter_elements([filter_func])


def _to_frame_size(frame_size=None, shape=None, img=None):
    if img is not None:
        shape = img.shape
    if shape is not None:
        return shape[1], shape[0]
    if frame_size is not None:
        return tuple(frame_size)
    raise TypeError("A valid keyword argument must be provided")
