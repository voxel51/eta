"""
Core data structures for working with polylines and polygons.

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
from future.utils import iteritems

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import eta.core.data as etad
import eta.core.labels as etal
import eta.core.utils as etau


class Polyline(etal.Labels):
    """A set of semantically related polylines or polygons in an image.

    :class:`Polyline` is a spatial concept that describes information about a
    semantically related set of polylines (collection of line segments) or
    polygons in a particular image or a particular frame of a video. A
    :class:`Polyline` can have a name (i.e., a class), a label, a list of lists
    of vertices, and one or more additional attributes describing its
    properties. The shapes can be closed, filled, or neither.

    Attributes:
        type: the fully-qualified class name of the polyline
        name: (optional) the name for the polyline, e.g., ``ground_truth`` or
            the name of the model that produced it
        label: (optional) a label for the polyline
        confidence: (optional) a confidence for the polyline, in ``[0, 1]``
        index: (optional) an index assigned to the polyline
        points: a list of lists of ``(x, y)`` points in ``[0, 1] x [0, 1]``
            describing the vertices of each shape in the polyline
        closed: whether the polyline is closed, i.e., an edge should be drawn
            from the last vertex to the first vertex of each shape
        filled: whether the polyline represents polygons, i.e., shapes that
            should be filled when rendering them
        attrs: (optional) an :class:`eta.core.data.AttributeContainer` of
            attributes for the polyline
        tags: (optional) a list of tag strings

    Args:
        name (None): a name for the polyline, e.g., ``ground_truth`` or the
            name of the model that produced it
        label (None): a label for the polyline
        confidence (None): a confidence for the polyline, in ``[0, 1]``
        index (None): an integer index assigned to the polyline
        points (None): a list of lists of ``(x, y)`` points in
            ``[0, 1] x [0, 1]`` describing the vertices of the shapes in the
            polyline
        closed (False): whether the polyline is closed, i.e., an edge
            should be drawn from the last vertex to the first vertex of each
            shape
        filled (False): whether the polyline contains polygons, i.e., shapes
            that should be filled when rendering them
        attrs (None): an :class:`eta.core.data.AttributeContainer` of
            attributes for the polyline
        tags (None): a list of tag strings
    """

    def __init__(
        self,
        name=None,
        label=None,
        confidence=None,
        index=None,
        points=None,
        closed=False,
        filled=False,
        attrs=None,
        tags=None,
    ):
        self.type = etau.get_class_name(self)
        self.name = name
        self.label = label
        self.confidence = confidence
        self.index = index
        self.points = points or []
        self.closed = closed
        self.filled = filled
        self.attrs = attrs or etad.AttributeContainer()
        self.tags = tags or []

    @property
    def is_empty(self):
        """Whether the polyline has labels of any kind."""
        return not (
            self.has_label
            or self.has_name
            or self.has_vertices
            or self.has_attributes
        )

    @property
    def has_label(self):
        """Whether the polyline has a ``label``."""
        return self.label is not None

    @property
    def has_vertices(self):
        """Whether the polyline has at least one vertex."""
        return any(bool(shape) for shape in self.points)

    @property
    def has_name(self):
        """Whether the polyline has a ``name``."""
        return self.name is not None

    @property
    def has_confidence(self):
        """Whether the polyline has a ``confidence``."""
        return self.confidence is not None

    @property
    def has_index(self):
        """Whether the polyline has an ``index``."""
        return self.index is not None

    @property
    def has_attributes(self):
        """Whether the polyline has attributes."""
        return bool(self.attrs)

    @property
    def has_tags(self):
        """Whether the polyline has tags."""
        return bool(self.tags)

    @classmethod
    def get_schema_cls(cls):
        """Gets the schema class for :class:`Polyline`.

        Returns:
            the :class:`eta.core.labels.LabelsSchema` class
        """
        return PolylineSchema

    def get_index(self):
        """Returns the ``index`` of the polyline.

        Returns:
            the index, or None if the polyline has no index
        """
        return self.index

    def offset_index(self, offset):
        """Adds the given offset to the polyline's ``index``.

        If the polyline has no index, this does nothing.

        Args:
            offset: the integer offset
        """
        if self.has_index:
            self.index += offset

    def clear_index(self):
        """Clears the ``index`` of the polyline."""
        self.index = None

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
            a list of lists of ``(x, y)`` vertices in pixels
        """
        w, h = _to_frame_size(frame_size=frame_size, shape=shape, img=img)
        return [
            [(int(round(x * w)), int(round(y * h))) for x, y in shape]
            for shape in self.points
        ]

    def filter_by_schema(self, schema, allow_none_label=False):
        """Filters the polyline by the given schema.

        The ``label`` of the :class:`Polyline` must match the provided
        schema. Or, it can be ``None`` when ``allow_none_label == True``.

        Args:
            schema: a :class:`PolylineSchema`
            allow_none_label (False): whether to allow the polyline label to be
                ``None``

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the polyline label
            does not match the schema
        """
        if self.label is None:
            if not allow_none_label:
                raise PolylineSchemaError(
                    "None polyline label is not allowed by the schema"
                )
        elif self.label != schema.get_label():
            raise PolylineSchemaError(
                "Label '%s' does not match polyline schema" % self.label
            )

        self.attrs.filter_by_schema(
            schema.frames, constant_schema=schema.attrs
        )

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
        if self.confidence:
            _attrs.append("confidence")
        if self.index:
            _attrs.append("index")
        if self.closed:
            _attrs.append("closed")
        if self.filled:
            _attrs.append("filled")
        if self.attrs:
            _attrs.append("attrs")
        if self.tags:
            _attrs.append("tags")
        return _attrs + ["points"]

    @classmethod
    def from_abs_coords(
        cls,
        points,
        clamp=True,
        frame_size=None,
        shape=None,
        img=None,
        **kwargs
    ):
        """Constructs a :class:`Polyline` from absolute pixel coordinates.

        One of ``frame_size``, ``shape``, or ``img`` must be provided.

        Args:
            points: a list of lists of ``(x, y)``` keypoints in
                ``[0, 1] x [0, 1]`` describing the vertices of the polyline
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
        for shape in points:
            rshape = []
            for x, y in shape:
                xr = x / w
                yr = y / h
                if clamp:
                    xr = max(0, min(xr, 1))
                    yr = max(0, min(yr, 1))

                rshape.append((xr, yr))

            rpoints.append(rshape)

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
        confidence = d.get("confidence", None)
        index = d.get("index", None)
        points = d.get("points", None)
        closed = d.get("closed", False)
        filled = d.get("filled", False)
        tags = d.get("tags", None)

        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainer.from_dict(attrs)

        return cls(
            name=name,
            label=label,
            confidence=confidence,
            index=index,
            points=points,
            closed=closed,
            filled=filled,
            attrs=attrs,
            tags=tags,
        )


class PolylineContainer(etal.LabelsContainer):
    """A ``eta.core.serial.Container`` of :class:`Polyline` instances."""

    _ELE_CLS = Polyline
    _ELE_CLS_FIELD = "_POLY_CLS"
    _ELE_ATTR = "polylines"

    def get_labels(self):
        """Returns the set of ``label`` values of all polylines in the
        container.

        Returns:
            a set of labels
        """
        return set(polyline.label for polyline in self)

    def get_indexes(self):
        """Returns the set of ``index`` values of all polylines in the
        container.

        ``None`` indexes are omitted.

        Returns:
            a set of indexes
        """
        return set(poly.index for poly in self if poly.has_index)

    def offset_indexes(self, offset):
        """Adds the given offset to all polylines with ``index`` values.

        Args:
            offset: the integer offset
        """
        for poly in self:
            poly.offset_index(offset)

    def clear_indexes(self):
        """Clears the ``index`` of all polylines in the container.
        """
        for poly in self:
            poly.clear_index()

    def sort_by_confidence(self, reverse=False):
        """Sorts the :class:`Polyline` instances by confidence.

        Polylines whose confidence is ``None`` are always put last.

        Args:
            reverse (False): whether to sort in descending order
        """
        self.sort_by("confidence", reverse=reverse)

    def sort_by_index(self, reverse=False):
        """Sorts the :class:`Polyline` instances by index.

        Polylines whose index is ``None`` are always put last.

        Args:
            reverse (False): whether to sort in descending order
        """
        self.sort_by("index", reverse=reverse)

    def filter_by_schema(self, schema):
        """Filters the polylines in the container by the given schema.

        Args:
            schema: a :class:`PolylineContainerSchema`
        """
        # Remove polylines with invalid labels
        filter_func = lambda poly: schema.has_polyline_label(poly.label)
        self.filter_elements([filter_func])

        # Filter polylines by their schemas
        for poly in self:
            poly_schema = schema.get_polyline_schema(poly.label)
            poly.filter_by_schema(poly_schema)

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


class PolylineSchema(etal.LabelsSchema):
    """Schema for :class:`Polyline` instances.

    Attributes:
        label: the polyline label
        attrs: an :class:`eta.core.data.AttributeContainerSchema` describing
            the attributes of the polyline

    Args:
        label: the polyline label
        attrs (None): an :class:`eta.core.data.AttributeContainerSchema`
            describing the attributes of the polyline
    """

    def __init__(self, label, attrs=None):
        self.label = label
        self.attrs = attrs or etad.AttributeContainerSchema()

    @property
    def is_empty(self):
        """Whether this schema has no labels of any kind."""
        return False

    def has_label(self, label):
        """Whether the schema has the given polyline label.

        Args:
            label: the polyline label

        Returns:
            True/False
        """
        return label == self.label

    def get_label(self):
        """Gets the polyline label for the schema.

        Returns:
            the polyline label
        """
        return self.label

    def has_attribute(self, attr_name):
        """Whether the schema has an :class:`eta.core.data.Attribute` of the
        given name.

        Args:
            attr_name: the name

        Returns:
            True/False
        """
        return self.attrs.has_attribute(attr_name)

    def get_attribute_schema(self, attr_name):
        """Gets the :class:`eta.core.data.AttributeSchema` for the attribute of
        the given name.

        Args:
            attr_name: the name

        Returns:
            the `eta.core.data.AttributeSchema`
        """
        return self.attrs.get_attribute_schema(attr_name)

    def get_attribute_class(self, attr_name):
        """Gets the `eta.core.data.Attribute` class for the attribute of the
        given name.

        Args:
            attr_name: the name

        Returns:
            the `eta.core.data.Attribute`
        """
        return self.attrs.get_attribute_class(attr_name)

    def add_attribute(self, attr):
        """Adds the `eta.core.data.Attribute` to the schema.

        Args:
            attr: an `eta.core.data.Attribute`
        """
        self.attrs.add_attribute(attr)

    def add_attributes(self, attrs):
        """Adds the `eta.core.data.AttributeContainer` of attributes to the
        schema.

        Args:
            attrs: an `eta.core.data.AttributeContainer`
        """
        self.attrs.add_attributes(attrs)

    def add_polyline(self, polyline):
        """Adds the polyline to the schema.

        Args:
            polyline: a :class:`Polyline`
        """
        self.validate_label(polyline.label)
        for attr in polyline.attrs:
            self.add_attribute(attr)

    def add_polylines(self, polylines):
        """Adds the :class:`PolylineContainer` to the schema.

        Args:
            polylines: a :class:`PolylineContainer`
        """
        for polyline in polylines:
            self.add_polyline(polyline)

    def is_valid_label(self, label):
        """Whether the polyline label is compliant with the schema.

        Args:
            label: a polyline label

        Returns:
            True/False
        """
        try:
            self.validate_label(label)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_attribute(self, attr):
        """Whether the attribute is compliant with the schema.

        Args:
            attr: an `eta.core.data.Attribute`

        Returns:
            True/False
        """
        try:
            self.validate_attribute(attr)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_attributes(self, attrs):
        """Whether the `eta.core.data.AttributeContainer` of attributes is
        compliant with the schema.

        Args:
            attrs: an `eta.core.data.AttributeContainer`

        Returns:
            True/False
        """
        try:
            self.validate_attributes(attrs)
            return True
        except etal.LabelsSchemaError:
            return False

    def validate_label(self, label):
        """Validates that the polyline label is compliant with the schema.

        Args:
            label: the label

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the label violates
            the schema
        """
        if label != self.label:
            raise PolylineSchemaError(
                "Label '%s' does not match polyline schema" % label
            )

    def validate_attribute(self, attr):
        """Validates that the attribute is compliant with the schema.

        Args:
            attr: an `eta.core.data.Attribute`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attribute
            violates the schema
        """
        self.attrs.validate_attribute(attr)

    def validate_attributes(self, attrs):
        """Validates that the `eta.core.data.AttributeContainer` of attributes
        is compliant with the schema.

        Args:
            attrs: an `eta.core.data.AttributeContainer`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attributes
            violate the schema
        """
        self.attrs.validate(attrs)

    def validate(self, polyline):
        """Validates that the polyline is compliant with the schema.

        Args:
            polyline: a :class:`Polyline`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the polyline
            violates the schema
        """
        self.validate_label(polyline.label)
        for attr in polyline.attrs:
            self.validate_attribute(attr)

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given schema.

        Args:
            schema: a :class:`PolylineSchema`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if this schema is not a
            subset of the given schema
        """
        self.validate_schema_type(schema)

        if self.label != schema.label:
            raise PolylineSchemaError(
                "Expected polyline label '%s'; found '%s'"
                % (schema.label, self.label)
            )

        self.attrs.validate_subset_of_schema(schema.attrs)

    def merge_schema(self, schema):
        """Merges the given :class:`PolylineSchema` into this schema.

        Args:
            schema: a :class:`PolylineSchema`
        """
        self.validate_label(schema.label)
        self.attrs.merge_schema(schema.attrs)

    @classmethod
    def build_active_schema(cls, polyline):
        """Builds an :class:`PolylineSchema` that describes the active schema
        of the polyline.

        Args:
            polyline: a :class:`Polyline`

        Returns:
            an :class:`PolylineSchema`
        """
        schema = cls(polyline.label)
        schema.add_polyline(polyline)
        return schema

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Args:
            a list of attribute names
        """
        _attrs = ["label"]
        if self.attrs:
            _attrs.append("attrs")
        return _attrs

    @classmethod
    def from_dict(cls, d):
        """Constructs an :class:`PolylineSchema` from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an :class:`PolylineSchema`
        """
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainerSchema.from_dict(attrs)

        return cls(d["label"], attrs=attrs)


class PolylineSchemaError(etal.LabelsSchemaError):
    """Error raised when a :class:`PolylineSchema` is violated."""

    pass


class PolylineContainerSchema(etal.LabelsContainerSchema):
    """Schema for :class:`PolylineContainer`,

    Attributes:
        schema: a dict mapping polyline labels to :class:`PolylineSchema`
            instances

    Args:
        schema (None): a dict mapping polyline labels to
            :class:`PolylineSchema` instances
    """

    def __init__(self, schema=None):
        self.schema = schema or {}

    @property
    def is_empty(self):
        """Whether this schema has no labels of any kind."""
        return not bool(self.schema)

    def iter_polyline_labels(self):
        """Returns an iterator over the polyline labels in this schema.

        Returns:
            an iterator over polyline labels
        """
        return iter(self.schema)

    def iter_polylines(self):
        """Returns an iterator over the (label, :class:`PolylineSchema`) pairs
        in this schema.

        Returns:
            an iterator over (label, :class:`PolylineSchema`) pairs
        """
        return iteritems(self.schema)

    def has_polyline_label(self, label):
        """Whether the schema has a polyline with the given label.

        Args:
            label: the polyline label

        Returns:
            True/False
        """
        return label in self.schema

    def has_polyline_attribute(self, label, attr_name):
        """Whether the schema has a polyline with the given label with an
        attribute of the given name.

        Args:
            label: the polyline label
            attr_name: the attribute name

        Returns:
            True/False
        """
        if not self.has_polyline_label(label):
            return False

        return self.schema[label].has_attribute(attr_name)

    def get_polyline_schema(self, label):
        """Gets the :class:`PolylineSchema` for the polyline with the given
        label.

        Args:
            label: the polyline label

        Returns:
            a :class:`PolylineSchema`
        """
        self.validate_polyline_label(label)
        return self.schema[label]

    def get_polyline_attribute_schema(self, label, attr_name):
        """Gets the :class:`eta.core.data.AttributeSchema` for the attribute of
        the given name for the polyline with the given label.

        Args:
            label: the polyline label
            attr_name: the attribute name

        Returns:
            the :class:`eta.core.data.AttributeSchema`
        """
        poly_schema = self.get_polyline_schema(label)
        return poly_schema.get_attribute_schema(attr_name)

    def get_polyline_attribute_class(self, label, attr_name):
        """Gets the :class:`eta.core.data.Attribute` class for the attribute of
        the given name for the polyline with the given label.

        Args:
            label: the polyline label
            attr_name: the attribute name

        Returns:
            the :class:`eta.core.data.Attribute`
        """
        self.validate_polyline_label(label)
        return self.schema[label].get_attribute_class(attr_name)

    def add_polyline_label(self, label):
        """Adds the given polyline label to the schema.

        Args:
            label: a polyline label
        """
        self._ensure_has_polyline_label(label)

    def add_polyline_attribute(self, label, attr):
        """Adds the :class:`eta.core.data.Attribute` for the polyline with the
        given label to the schema.

        Args:
            label: a polyline label
            attr: an :class:`eta.core.data.Attribute`
        """
        self._ensure_has_polyline_label(label)
        self.schema[label].add_attribute(attr)

    def add_polyline_attributes(self, label, attrs):
        """Adds the :class:`eta.core.data.AttributeContainer` of attributes for
        the polyline with the given label to the schema.

        Args:
            label: a polyline label
            attrs: an :class:`eta.core.data.AttributeContainer`
        """
        self._ensure_has_polyline_label(label)
        self.schema[label].add_attributes(attrs)

    def add_polyline(self, polyline):
        """Adds the polyline to the schema.

        Args:
            polyline: a :class:`Polyline`
        """
        self._ensure_has_polyline_label(polyline.label)
        self.schema[polyline.label].add_polyline(polyline)

    def add_polylines(self, polylines):
        """Adds the polylines to the schema.

        Args:
            polylines: a :class:`PolylineContainer`
        """
        for polyline in polylines:
            self.add_polyline(polyline)

    def is_valid_polyline_label(self, label):
        """Whether the polyline label is compliant with the schema.

        Args:
            label: a polyline label

        Returns:
            True/False
        """
        try:
            self.validate_polyline_label(label)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_polyline_attribute(self, label, attr):
        """Whether the attribute for the polyline with the given label is
        compliant with the schema.

        Args:
            label: a polyline label
            attr: an :class:`eta.core.data.Attribute`

        Returns:
            True/False
        """
        try:
            self.validate_polyline_attribute(label, attr)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_polyline_attributes(self, label, attrs):
        """Whether the attributes for the polyline with the given label are
        compliant with the schema.

        Args:
            label: a polyline label
            attrs: an :class:`eta.core.data.AttributeContainer`

        Returns:
            True/False
        """
        try:
            self.validate_polyline_attributes(label, attrs)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_polyline(self, polyline):
        """Whether the polyline is compliant with the schema.

        Args:
            polyline: a :class:`Polyline`

        Returns:
            True/False
        """
        try:
            self.validate_polyline(polyline)
            return True
        except etal.LabelsSchemaError:
            return False

    def validate_polyline_label(self, label):
        """Validates that the polyline label is compliant with the schema.

        Args:
            label: a polyline label

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the polyline label
            violates the schema
        """
        if label not in self.schema:
            raise PolylineContainerSchemaError(
                "Polyline label '%s' is not allowed by the schema" % label
            )

    def validate_polyline_attribute(self, label, attr):
        """Validates that the :class:`eta.core.data.Attribute` for the polyline
        with the given label is compliant with the schema.

        Args:
            label: a polyline label
            attr: an :class:`eta.core.data.Attribute`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attribute
            violates the schema
        """
        self.validate_polyline_label(label)
        self.schema[label].validate_polyline_attribute(attr)

    def validate_polyline_attributes(self, label, attrs):
        """Validates that the :class:`eta.core.data.AttributeContainer` of
        attributes for the polyline with the given label is compliant with the
        schema.

        Args:
            label: a polyline label
            attrs: an :class:`eta.core.data.AttributeContainer`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attributes
            violate the schema
        """
        self.validate_polyline_label(label)
        self.schema[label].validate_polyline_attributes(attrs)

    def validate_polyline(self, polyline):
        """Validates that the polyline is compliant with the schema.

        Args:
            polyline: a :class:`Polyline`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the polyline
            violates the schema
        """
        self.validate_polyline_label(polyline.label)
        self.schema[polyline.label].validate(polyline)

    def validate(self, polylines):
        """Validates that the polylines are compliant with the schema.

        Args:
            polylines: a :class:`PolylineContainer`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the polyline violate
            the schema
        """
        for polyline in polylines:
            self.validate_polyline(polyline)

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given schema.

        Args:
            schema: an :class:`PolylineContainerSchema`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if this schema is not a
            subset of the given schema
        """
        self.validate_schema_type(schema)

        for label, poly_schema in iteritems(self.schema):
            if not schema.has_polyline_label(label):
                raise PolylineContainerSchemaError(
                    "Polyline label '%s' does not appear in schema" % label
                )

            other_polyline_schema = schema.get_polyline_schema(label)
            poly_schema.validate_subset_of_schema(other_polyline_schema)

    def merge_polyline_schema(self, poly_schema):
        """Merges the given :class:`PolylineSchema` into the schema.

        Args:
            poly_schema: an :class:`PolylineSchema`
        """
        label = poly_schema.label
        self._ensure_has_polyline_label(label)
        self.schema[label].merge_schema(poly_schema)

    def merge_schema(self, schema):
        """Merges the given :class:`PolylineContainerSchema` into this schema.

        Args:
            schema: an :class:`PolylineContainerSchema`
        """
        for _, poly_schema in schema.iter_polylines():
            self.merge_polyline_schema(poly_schema)

    @classmethod
    def build_active_schema(cls, polylines):
        """Builds an :class:`PolylineContainerSchema` that describes the active
        schema of the polylines.

        Args:
            polyliness: a :class:`PolylineContainer`

        Returns:
            an :class:`PolylineContainerSchema`
        """
        schema = cls()
        schema.add_polylines(polylines)
        return schema

    @classmethod
    def from_dict(cls, d):
        """Constructs an :class:`PolylineContainerSchema` from a JSON
        dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an :class:`PolylineContainerSchema`
        """
        schema = d.get("schema", None)
        if schema is not None:
            schema = {
                label: PolylineSchema.from_dict(psd)
                for label, psd in iteritems(schema)
            }

        return cls(schema=schema)

    def _ensure_has_polyline_label(self, label):
        if not self.has_polyline_label(label):
            self.schema[label] = PolylineSchema(label)


class PolylineContainerSchemaError(etal.LabelsContainerSchemaError):
    """Error raised when a :class:`PolylineContainerSchema` is violated."""

    pass


def _to_frame_size(frame_size=None, shape=None, img=None):
    if img is not None:
        shape = img.shape
    if shape is not None:
        return shape[1], shape[0]
    if frame_size is not None:
        return tuple(frame_size)
    raise TypeError("A valid keyword argument must be provided")
