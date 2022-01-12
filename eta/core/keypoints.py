"""
Core data structures for working with keypoints.

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


class Keypoints(etal.Labels):
    """A list of keypoints in an image.

    :class:`Keypoints` is a spatial concept that describes information about
    one or more points in a particular image or particular frame of a video.
    A :class:`Keypoints` can have a name (i.e., a class), a label, and a list
    of points, and one or more additional attributes describing properties of
    the keypoints.

    Attributes:
        type: the fully-qualified class name of the keypoints
        name: (optional) the name for the keypoints, e.g., ``ground_truth`` or
            the name of the model that produced it
        label: (optional) keypoints label
        confidence: (optional) a confidence for the keypoints, in ``[0, 1]``
        index: (optional) an index assigned to the keypoints
        points: a list of ``(x, y)`` keypoints in ``[0, 1] x [0, 1]``
        attrs: (optional) an :class:`eta.core.data.AttributeContainer` of
            attributes for the keypoints
        tags: (optional) a list of tag strings

    Args:
        name (None): a name for the keypoints, e.g., ``ground_truth`` or the
            name of the model that produced it
        label (None): a label for the keypoints
        confidence (None): a confidence for the keypoints, in ``[0, 1]``
        index (None): an integer index assigned to the keypoints
        points (None): a list of ``(x, y)`` keypoints in ``[0, 1] x [0, 1]``
        attrs (None): an :class:`eta.core.data.AttributeContainer` of
            attributes for the keypoints
        tags (None): a list of tag strings
    """

    def __init__(
        self,
        name=None,
        label=None,
        confidence=None,
        index=None,
        points=None,
        attrs=None,
        tags=None,
    ):
        self.type = etau.get_class_name(self)
        self.name = name
        self.label = label
        self.confidence = confidence
        self.index = index
        self.points = points or []
        self.attrs = attrs or etad.AttributeContainer()
        self.tags = tags or []

    @property
    def is_empty(self):
        """Whether the keypoints has labels of any kind."""
        return not (
            self.has_label
            or self.has_name
            or self.has_points
            or self.has_attributes
        )

    @property
    def has_label(self):
        """Whether the keypoints has a ``label``."""
        return self.label is not None

    @property
    def has_confidence(self):
        """Whether the keypoints has a ``confidence``."""
        return self.confidence is not None

    @property
    def has_index(self):
        """Whether the keypoints has an ``index``."""
        return self.index is not None

    @property
    def has_points(self):
        """Whether the keypoints has at least one vertex."""
        return bool(self.points)

    @property
    def has_name(self):
        """Whether the keypoints has a ``name``."""
        return self.name is not None

    @property
    def has_attributes(self):
        """Whether the keypoints has attributes."""
        return bool(self.attrs)

    @property
    def has_tags(self):
        """Whether the keypoints has tags."""
        return bool(self.tags)

    @classmethod
    def get_schema_cls(cls):
        """Gets the schema class for :class:`Keypoints`.

        Returns:
            the :class:`eta.core.labels.LabelsSchema` class
        """
        return KeypointsSchema

    def get_index(self):
        """Returns the ``index`` of the keypoints.

        Returns:
            the index, or None if the keypoints has no index
        """
        return self.index

    def offset_index(self, offset):
        """Adds the given offset to the keypoints' ``index``.

        If the keypoints has no index, this does nothing.

        Args:
            offset: the integer offset
        """
        if self.has_index:
            self.index += offset

    def clear_index(self):
        """Clears the ``index`` of the keypoints."""
        self.index = None

    def add_attribute(self, attr):
        """Adds the attribute to the keypoints.

        Args:
            attr: an :class:`eta.core.data.Attribute`
        """
        self.attrs.add(attr)

    def add_attributes(self, attrs):
        """Adds the attributes to the keypoints.

        Args:
            attrs: an :class:`eta.core.data.AttributeContainer`
        """
        self.attrs.add_container(attrs)

    def pop_attributes(self):
        """Pops the attributes from the keypoints.

        Returns:
            an :class:`eta.core.data.AttributeContainer`
        """
        attrs = self.attrs
        self.clear_attributes()
        return attrs

    def clear_attributes(self):
        """Removes all attributes from the keypoints."""
        self.attrs = etad.AttributeContainer()

    def coords_in(self, frame_size=None, shape=None, img=None):
        """Returns the coordinates of the keypoints in the specified image.

        Pass *one* keyword argument to this function.

        Args:
            frame_size: the (width, height) of the image
            shape: the (height, width, ...) of the image, e.g. from img.shape
            img: the image itself

        Returns:
            a list of (x, y) keypoints in pixels
        """
        w, h = _to_frame_size(frame_size=frame_size, shape=shape, img=img)
        return [(int(round(x * w)), int(round(y * h))) for x, y in self.points]

    def filter_by_schema(self, schema, allow_none_label=False):
        """Filters the keypoints by the given schema.

        The ``label`` of the :class:`Keypoints` must match the provided
        schema. Or, it can be ``None`` when ``allow_none_label == True``.

        Args:
            schema: a :class:`KeypointsSchema`
            allow_none_label (False): whether to allow the keypoints label to
                be ``None``

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the keypoints label
            does not match the schema
        """
        if self.label is None:
            if not allow_none_label:
                raise KeypointsSchemaError(
                    "None keypoints label is not allowed by the schema"
                )
        elif self.label != schema.get_label():
            raise KeypointsSchemaError(
                "Label '%s' does not match keypoints schema" % self.label
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
        **kwargs
    ):
        """Constructs a :class:`Keypoints` from absolute pixel coordinates.

        One of ``frame_size``, ``shape``, or ``img`` must be provided.

        Args:
            points: a list of ``(x, y)``` keypoints in ``[0, 1] x [0, 1]``
                describing the vertices of the keypoints
            clamp (True): whether to clamp the relative points to
                ``[0, 1] x [0, 1]``, if necessary
            frame_size (None): the ``(width, height)`` of the image
            shape (None): the ``(height, width, ...)`` of the image, e.g. from
                ``img.shape``
            img (None): the image itself
            **kwargs: additional keyword arguments for
                ``Keypoints(points=points, **kwargs)``

        Returns:
            a :class:`Keypoints`
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
        """Constructs a :class:`Keypoints` from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a :class:`Keypoints`
        """
        name = d.get("name", None)
        label = d.get("label", None)
        confidence = d.get("confidence", None)
        index = (d.get("index", None),)
        points = d.get("points", None)
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
            attrs=attrs,
            tags=tags,
        )


class KeypointsContainer(etal.LabelsContainer):
    """A ``eta.core.serial.Container`` of :class:`Keypoints` instances."""

    _ELE_CLS = Keypoints
    _ELE_CLS_FIELD = "_KEY_CLS"
    _ELE_ATTR = "keypoints"

    def get_labels(self):
        """Returns the set of ``label`` values of all keypoints in the
        container.

        Returns:
            a set of labels
        """
        return set(k.label for k in self)

    def get_indexes(self):
        """Returns the set of ``index`` values of all keypoints in the container.

        ``None`` indexes are omitted.

        Returns:
            a set of indexes
        """
        return set(k.index for k in self if k.has_index)

    def offset_indexes(self, offset):
        """Adds the given offset to all keypoints with ``index`` values.

        Args:
            offset: the integer offset
        """
        for keypoints in self:
            keypoints.offset_index(offset)

    def clear_indexes(self):
        """Clears the ``index`` of all keypoints in the container.
        """
        for keypoints in self:
            keypoints.clear_index()

    def sort_by_confidence(self, reverse=False):
        """Sorts the :class:`Keypoints` instances by confidence.

        Keypoints whose confidence is ``None`` are always put last.

        Args:
            reverse (False): whether to sort in descending order
        """
        self.sort_by("confidence", reverse=reverse)

    def sort_by_index(self, reverse=False):
        """Sorts the :class:`Keypoints` instances by index.

        Keypoints whose index is ``None`` are always put last.

        Args:
            reverse (False): whether to sort in descending order
        """
        self.sort_by("index", reverse=reverse)

    def filter_by_schema(self, schema):
        """Filters the keypoints in the container by the given schema.

        Args:
            schema: a :class:`KeypointsContainerSchema`
        """
        # Remove keypoints with invalid labels
        filter_func = lambda keypoints: schema.has_keypoints_label(
            keypoints.label
        )
        self.filter_elements([filter_func])

        # Filter keypoints by their schemas
        for keypoints in self:
            keypoints_schema = schema.get_keypoints_schema(keypoints.label)
            keypoints.filter_by_schema(keypoints_schema)

    def remove_keypoints_without_attrs(self, labels=None):
        """Removes keypoints from this container that do not have attributes.

        Args:
            labels (None): an optional list of :class:`Keypoints` label strings
                to which to restrict attention when filtering. By default, all
                keypoints are processed
        """
        filter_func = lambda keypoints: (
            (labels is not None and keypoints.label not in labels)
            or keypoints.has_attributes
        )
        self.filter_elements([filter_func])


class KeypointsSchema(etal.LabelsSchema):
    """Schema for :class:`Keypoints` instances.

    Attributes:
        label: the keypoints label
        attrs: an :class:`eta.core.data.AttributeContainerSchema` describing
            the attributes of the keypoints

    Args:
        label: the keypoints label
        attrs (None): an :class:`eta.core.data.AttributeContainerSchema`
            describing the attributes of the keypoints
    """

    def __init__(self, label, attrs=None):
        self.label = label
        self.attrs = attrs or etad.AttributeContainerSchema()

    @property
    def is_empty(self):
        """Whether this schema has no labels of any kind."""
        return False

    def has_label(self, label):
        """Whether the schema has the given keypoints label.

        Args:
            label: the keypoints label

        Returns:
            True/False
        """
        return label == self.label

    def get_label(self):
        """Gets the keypoints label for the schema.

        Returns:
            the keypoints label
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

    def add_keypoints(self, keypoints):
        """Adds the keypoints to the schema.

        Args:
            keypoints: a :class:`Keypoints` or :class:`KeypointsContainer`
        """
        if isinstance(keypoints, KeypointsContainer):
            for k in keypoints:
                self._add_keypoints(k)
        else:
            self._add_keypoints(keypoints)

    def is_valid_label(self, label):
        """Whether the keypoints label is compliant with the schema.

        Args:
            label: a keypoints label

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
        """Validates that the keypoints label is compliant with the schema.

        Args:
            label: the label

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the label violates
            the schema
        """
        if label != self.label:
            raise KeypointsSchemaError(
                "Label '%s' does not match keypoints schema" % label
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

    def validate(self, keypoints):
        """Validates that the keypoints is compliant with the schema.

        Args:
            keypoints: a :class:`Keypoints`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the keypoints
            violates the schema
        """
        self.validate_label(keypoints.label)
        for attr in keypoints.attrs:
            self.validate_attribute(attr)

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given schema.

        Args:
            schema: a :class:`KeypointsSchema`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if this schema is not a
            subset of the given schema
        """
        self.validate_schema_type(schema)

        if self.label != schema.label:
            raise KeypointsSchemaError(
                "Expected keypoints label '%s'; found '%s'"
                % (schema.label, self.label)
            )

        self.attrs.validate_subset_of_schema(schema.attrs)

    def merge_schema(self, schema):
        """Merges the given :class:`KeypointsSchema` into this schema.

        Args:
            schema: a :class:`KeypointsSchema`
        """
        self.validate_label(schema.label)
        self.attrs.merge_schema(schema.attrs)

    @classmethod
    def build_active_schema(cls, keypoints):
        """Builds an :class:`KeypointsSchema` that describes the active schema
        of the keypoints.

        Args:
            keypoints: a :class:`Keypoints`

        Returns:
            an :class:`KeypointsSchema`
        """
        schema = cls(keypoints.label)
        schema.add_keypoints(keypoints)
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
        """Constructs an :class:`KeypointsSchema` from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an :class:`KeypointsSchema`
        """
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainerSchema.from_dict(attrs)

        return cls(d["label"], attrs=attrs)

    def _add_keypoints(self, keypoints):
        self.validate_label(keypoints.label)
        for attr in keypoints.attrs:
            self.add_attribute(attr)


class KeypointsSchemaError(etal.LabelsSchemaError):
    """Error raised when a :class:`KeypointsSchema` is violated."""

    pass


class KeypointsContainerSchema(etal.LabelsContainerSchema):
    """Schema for :class:`KeypointsContainer`,

    Attributes:
        schema: a dict mapping keypoints labels to :class:`KeypointsSchema`
            instances

    Args:
        schema (None): a dict mapping keypoints labels to
            :class:`KeypointsSchema` instances
    """

    def __init__(self, schema=None):
        self.schema = schema or {}

    @property
    def is_empty(self):
        """Whether this schema has no labels of any kind."""
        return not bool(self.schema)

    def iter_keypoints_labels(self):
        """Returns an iterator over the keypoints labels in this schema.

        Returns:
            an iterator over keypoints labels
        """
        return iter(self.schema)

    def iter_keypoints(self):
        """Returns an iterator over the (label, :class:`KeypointsSchema`) pairs
        in this schema.

        Returns:
            an iterator over (label, :class:`KeypointsSchema`) pairs
        """
        return iteritems(self.schema)

    def has_keypoints_label(self, label):
        """Whether the schema has a keypoints with the given label.

        Args:
            label: the keypoints label

        Returns:
            True/False
        """
        return label in self.schema

    def has_keypoints_attribute(self, label, attr_name):
        """Whether the schema has a keypoints with the given label with an
        attribute of the given name.

        Args:
            label: the keypoints label
            attr_name: the attribute name

        Returns:
            True/False
        """
        if not self.has_keypoints_label(label):
            return False

        return self.schema[label].has_attribute(attr_name)

    def get_keypoints_schema(self, label):
        """Gets the :class:`KeypointsSchema` for the keypoints with the given
        label.

        Args:
            label: the keypoints label

        Returns:
            a :class:`KeypointsSchema`
        """
        self.validate_keypoints_label(label)
        return self.schema[label]

    def get_keypoints_attribute_schema(self, label, attr_name):
        """Gets the :class:`eta.core.data.AttributeSchema` for the attribute of
        the given name for the keypoints with the given label.

        Args:
            label: the keypoints label
            attr_name: the attribute name

        Returns:
            the :class:`eta.core.data.AttributeSchema`
        """
        poly_schema = self.get_keypoints_schema(label)
        return poly_schema.get_attribute_schema(attr_name)

    def get_keypoints_attribute_class(self, label, attr_name):
        """Gets the :class:`eta.core.data.Attribute` class for the attribute of
        the given name for the keypoints with the given label.

        Args:
            label: the keypoints label
            attr_name: the attribute name

        Returns:
            the :class:`eta.core.data.Attribute`
        """
        self.validate_keypoints_label(label)
        return self.schema[label].get_attribute_class(attr_name)

    def add_keypoints_label(self, label):
        """Adds the given keypoints label to the schema.

        Args:
            label: a keypoints label
        """
        self._ensure_has_keypoints_label(label)

    def add_keypoints_attribute(self, label, attr):
        """Adds the :class:`eta.core.data.Attribute` for the keypoints with the
        given label to the schema.

        Args:
            label: a keypoints label
            attr: an :class:`eta.core.data.Attribute`
        """
        self._ensure_has_keypoints_label(label)
        self.schema[label].add_attribute(attr)

    def add_keypoints_attributes(self, label, attrs):
        """Adds the :class:`eta.core.data.AttributeContainer` of attributes for
        the keypoints with the given label to the schema.

        Args:
            label: a keypoints label
            attrs: an :class:`eta.core.data.AttributeContainer`
        """
        self._ensure_has_keypoints_label(label)
        self.schema[label].add_attributes(attrs)

    def add_keypoints(self, keypoints):
        """Adds the keypoints to the schema.

        Args:
            keypoints: a :class:`Keypoints`
        """
        if isinstance(keypoints, KeypointsContainer):
            for k in keypoints:
                self._add_keypoints(k)
        else:
            self._add_keypoints(keypoints)

    def is_valid_keypoints_label(self, label):
        """Whether the keypoints label is compliant with the schema.

        Args:
            label: a keypoints label

        Returns:
            True/False
        """
        try:
            self.validate_keypoints_label(label)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_keypoints_attribute(self, label, attr):
        """Whether the attribute for the keypoints with the given label is
        compliant with the schema.

        Args:
            label: a keypoints label
            attr: an :class:`eta.core.data.Attribute`

        Returns:
            True/False
        """
        try:
            self.validate_keypoints_attribute(label, attr)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_keypoints_attributes(self, label, attrs):
        """Whether the attributes for the keypoints with the given label are
        compliant with the schema.

        Args:
            label: a keypoints label
            attrs: an :class:`eta.core.data.AttributeContainer`

        Returns:
            True/False
        """
        try:
            self.validate_keypoints_attributes(label, attrs)
            return True
        except etal.LabelsSchemaError:
            return False

    def is_valid_keypoints(self, keypoints):
        """Whether the keypoints is compliant with the schema.

        Args:
            keypoints: a :class:`Keypoints`

        Returns:
            True/False
        """
        try:
            self.validate_keypoints(keypoints)
            return True
        except etal.LabelsSchemaError:
            return False

    def validate_keypoints_label(self, label):
        """Validates that the keypoints label is compliant with the schema.

        Args:
            label: a keypoints label

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the keypoints label
            violates the schema
        """
        if label not in self.schema:
            raise KeypointsContainerSchemaError(
                "Keypoints label '%s' is not allowed by the schema" % label
            )

    def validate_keypoints_attribute(self, label, attr):
        """Validates that the :class:`eta.core.data.Attribute` for the
        keypoints with the given label is compliant with the schema.

        Args:
            label: a keypoints label
            attr: an :class:`eta.core.data.Attribute`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attribute
            violates the schema
        """
        self.validate_keypoints_label(label)
        self.schema[label].validate_keypoints_attribute(attr)

    def validate_keypoints_attributes(self, label, attrs):
        """Validates that the :class:`eta.core.data.AttributeContainer` of
        attributes for the keypoints with the given label is compliant with the
        schema.

        Args:
            label: a keypoints label
            attrs: an :class:`eta.core.data.AttributeContainer`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the attributes
            violate the schema
        """
        self.validate_keypoints_label(label)
        self.schema[label].validate_keypoints_attributes(attrs)

    def validate_keypoints(self, keypoints):
        """Validates that the keypoints is compliant with the schema.

        Args:
            keypoints: a :class:`Keypoints`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the keypoints
            violates the schema
        """
        self.validate_keypoints_label(keypoints.label)
        self.schema[keypoints.label].validate(keypoints)

    def validate(self, keypoints):
        """Validates that the keypoints are compliant with the schema.

        Args:
            keypoints: a :class:`KeypointsContainer`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if the keypoints
            violate the schema
        """
        for k in keypoints:
            self.validate_keypoints(k)

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given schema.

        Args:
            schema: an :class:`KeypointsContainerSchema`

        Raises:
            :class:`eta.core.labels.LabelsSchemaError`: if this schema is not a
            subset of the given schema
        """
        self.validate_schema_type(schema)

        for label, poly_schema in iteritems(self.schema):
            if not schema.has_keypoints_label(label):
                raise KeypointsContainerSchemaError(
                    "Keypoints label '%s' does not appear in schema" % label
                )

            other_keypoints_schema = schema.get_keypoints_schema(label)
            poly_schema.validate_subset_of_schema(other_keypoints_schema)

    def merge_keypoints_schema(self, poly_schema):
        """Merges the given :class:`KeypointsSchema` into the schema.

        Args:
            poly_schema: an :class:`KeypointsSchema`
        """
        label = poly_schema.label
        self._ensure_has_keypoints_label(label)
        self.schema[label].merge_schema(poly_schema)

    def merge_schema(self, schema):
        """Merges the given :class:`KeypointsContainerSchema` into this schema.

        Args:
            schema: an :class:`KeypointsContainerSchema`
        """
        for _, poly_schema in schema.iter_keypoints():
            self.merge_keypoints_schema(poly_schema)

    @classmethod
    def build_active_schema(cls, keypoints):
        """Builds an :class:`KeypointsContainerSchema` that describes the
        active schema of the keypoints.

        Args:
            keypointss: a :class:`KeypointsContainer`

        Returns:
            an :class:`KeypointsContainerSchema`
        """
        schema = cls()
        schema.add_keypoints(keypoints)
        return schema

    @classmethod
    def from_dict(cls, d):
        """Constructs an :class:`KeypointsContainerSchema` from a JSON
        dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an :class:`KeypointsContainerSchema`
        """
        schema = d.get("schema", None)
        if schema is not None:
            schema = {
                label: KeypointsSchema.from_dict(psd)
                for label, psd in iteritems(schema)
            }

        return cls(schema=schema)

    def _add_keypoints(self, keypoints):
        self._ensure_has_keypoints_label(keypoints.label)
        self.schema[keypoints.label].add_keypoints(keypoints)

    def _ensure_has_keypoints_label(self, label):
        if not self.has_keypoints_label(label):
            self.schema[label] = KeypointsSchema(label)


class KeypointsContainerSchemaError(etal.LabelsContainerSchemaError):
    """Error raised when a :class:`KeypointsContainerSchema` is violated."""

    pass


def _to_frame_size(frame_size=None, shape=None, img=None):
    if img is not None:
        shape = img.shape
    if shape is not None:
        return shape[1], shape[0]
    if frame_size is not None:
        return tuple(frame_size)
    raise TypeError("A valid keyword argument must be provided")
