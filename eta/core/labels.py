"""
Core data structures for working with labels.

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

from collections import defaultdict
import logging

import eta.core.serial as etas
import eta.core.utils as etau


logger = logging.getLogger(__name__)


class Labels(etas.Serializable):
    """Base class for `eta.core.serial.Serializable` classes that hold labels
    representing attributes, objects, frames, events, images, videos, etc.

    Labels classes have associated Schema classes that describe the
    ontologies over the labels class.
    """

    def __bool__(self):
        """Whether this instance has labels of any kind."""
        return not self.is_empty

    @property
    def is_empty(self):
        """Whether this instance has no labels of any kind."""
        raise NotImplementedError("subclasses must implement is_empty")

    @classmethod
    def get_schema_cls(cls):
        """Gets the LabelsSchema class for the labels.

        Subclasses can override this method, but, by default, this
        implementation assumes the convention that labels class `<Labels>` has
        associated schema class `<Labels>Schema` defined in the same module.

        Returns:
            the LabelsSchema class
        """
        class_name = etau.get_class_name(cls)
        return etau.get_class(class_name + "Schema")

    def get_active_schema(self):
        """Returns a LabelsSchema that describes the active schema of the
        labels.

        Returns:
            a LabelsSchema
        """
        schema_cls = self.get_schema_cls()
        return schema_cls.build_active_schema(self)

    def filter_by_schema(self, schema):
        """Filters the labels by the given schema.

        Args:
            schema: a LabelsSchema
        """
        raise NotImplementedError(
            "subclasses must implement `filter_by_schema()`"
        )


class LabelsSchema(etas.Serializable):
    """Base class for schemas of Labels classes."""

    def __bool__(self):
        """Whether this schema has labels of any kind."""
        return not self.is_empty

    @property
    def is_empty(self):
        """Whether this schema has no labels of any kind."""
        raise NotImplementedError("subclasses must implement is_empty")

    def add(self, labels):
        """Incorporates the Labels into the schema.

        Args:
            label: a Labels instance
        """
        labels_schema = self.build_active_schema(labels)
        self.merge_schema(labels_schema)

    def add_iterable(self, iterable):
        """Incorporates the given iterable of Labels into the schema.

        Args:
            iterable: an iterable of Labels
        """
        for labels in iterable:
            self.add(labels)

    def validate(self, labels):
        """Validates that the Labels are compliant with the schema.

        Args:
            labels: a Labels instance

        Raises:
            LabelsSchemaError: if the labels violate the schema
        """
        raise NotImplementedError("subclasses must implement `validate()`")

    def validate_subset_of_schema(self, schema):
        """Validates that this schema is a subset of the given LabelsSchema.

        Args:
            schema: a LabelsSchema

        Raises:
            LabelsSchemaError: if this schema is not a subset of the given
                schema
        """
        raise NotImplementedError(
            "subclasses must implement `validate_subset_of_schema()`"
        )

    def validate_schema_type(self, schema):
        """Validates that this schema is an instance of same type as the given
        schema.

        Args:
            schema: a LabelsSchema

        Raises:
            LabelsSchemaError: if this schema is not of the same type as the
                given schema
        """
        if not isinstance(self, type(schema)):
            raise LabelsSchemaError(
                "Expected `self` to match schema type %s; found %s"
                % (type(self), type(schema))
            )

    def is_valid(self, labels):
        """Whether the Labels are compliant with the schema.

        Args:
            labels: a Labels instance

        Returns:
            True/False
        """
        try:
            self.validate(labels)
            return True
        except LabelsSchemaError:
            return False

    def is_subset_of_schema(self, schema):
        """Whether this schema is a subset of the given schema.

        Args:
            schema: a LabelsSchema

        Returns:
            True/False
        """
        try:
            self.validate_subset_of_schema(schema)
            return True
        except LabelsSchemaError:
            return False

    @classmethod
    def build_active_schema(cls, labels):
        """Builds a LabelsSchema that describes the active schema of the
        labels.

        Args:
            labels: a Labels instance

        Returns:
            a LabelsSchema
        """
        raise NotImplementedError(
            "subclasses must implement `build_active_schema()`"
        )

    def merge_schema(self, schema):
        """Merges the given LabelsSchema into this schema.

        Args:
            schema: a LabelsSchema
        """
        raise NotImplementedError("subclasses must implement `merge_schema()`")


class LabelsSchemaError(Exception):
    """Error raisesd when a LabelsSchema is violated."""

    pass


class HasLabelsSchema(object):
    """Mixin for Label classes that can optionally store and enforce
    `LabelsSchema`s on their labels.

    For efficiency, schemas are not automatically enforced when new labels are
    added to HasLabelsSchema instances. Rather, users must manually call
    `validate_schema()` when they would like to validate the schema.

    Attributes:
        schema: the enforced LabelsSchema, or None
    """

    def __init__(self, schema=None):
        """Initializes the HasLabelsSchema mixin.

        Args:
            schema: (optional) a LabelsSchema to enforce on the labels. By
                default, no schema is enforced
        """
        self.schema = schema

    @property
    def has_schema(self):
        """Whether the labels have an enforced schema."""
        return self.schema is not None

    def get_schema(self):
        """Gets the current enforced schema for the labels, or None if no
        schema is enforced.

        Returns:
            a LabelsSchema, or None
        """
        return self.schema

    def set_schema(self, schema, filter_by_schema=False, validate=False):
        """Sets the enforced schema to the given LabelsSchema.

        Args:
            schema: a LabelsSchema to assign
            filter_by_schema: whether to filter labels that are not compliant
                with the schema. By default, this is False
            validate: whether to validate that the labels (after filtering, if
                applicable) are compliant with the new schema. By default, this
                is False

        Raises:
            LabelsSchemaError: if `validate` was `True` and this object
                contains labels that are not compliant with the schema
        """
        self.schema = schema
        if not self.has_schema:
            return

        if filter_by_schema:
            self.filter_by_schema(self.schema)  # pylint: disable=no-member

        if validate:
            self.validate_schema()

    def validate_schema(self):
        """Validates that the labels are compliant with the current schema.

        Raises:
            LabelsSchemaError: if this object contains labels that are not
                compliant with the schema
        """
        if self.has_schema:
            self.schema.validate(self)

    def freeze_schema(self):
        """Sets the schema for the labels to the current active schema."""
        self.set_schema(self.get_active_schema())  # pylint: disable=no-member

    def remove_schema(self):
        """Removes the enforced schema from the labels."""
        self.set_schema(None)


class HasLabelsSupport(object):
    """Mixin for Label classes that describe videos and can keep track of
    their own support, i.e., the frames for which they contain labels.

    The support is represented via a `eta.core.frameutils.FrameRanges`
    instance.

    For efficiency, supports should not be automatically updated when new
    labels are added to HasLabelsSupport instances. Rather, the support is
    dynamically computed when the `support` property is accessed.
    Alternatively, the current support can be frozen via `freeze_support()`
    to avoid recomputing it each time `support` is called.
    """

    def __init__(self, support=None):
        """Initializes the HasLabelsSupport mixin.

        Args:
            support: (optional) a FrameRanges instance describing the frozen
                support of the labels. By default, the support is not frozen
        """
        self._support = support

    @property
    def support(self):
        """A FrameRanges instance describing the frames for which this instance
        contains labels.

        If this instance has a frozen support, it is returned. Otherwise, the
        support is dynamically computed via `_compute_support()`.
        """
        if self.is_support_frozen:
            return self._support

        return self._compute_support()

    @property
    def is_support_frozen(self):
        """Whether the support is currently frozen."""
        return self._support is not None

    def set_support(self, support):
        """Sets the support to the given value.

        This action freezes the support for this instance.

        Args:
            support: a FrameRanges
        """
        self._support = support

    def merge_support(self, support):
        """Merges the given support into the current support.

        This action freezes the support for this instance.

        Args:
            support: a FrameRanges
        """
        new_support = self.support.merge(support)
        self.set_support(new_support)

    def freeze_support(self):
        """Freezes the support to the current `support`.

        This optional optimization is useful to avoid recalculating the support
        of the labels each time `support` is called.
        """
        if not self.is_support_frozen:
            self._support = self._compute_support()

    def clear_support(self):
        """Clears the frozen support, if necessary."""
        self._support = None

    def _compute_support(self):
        """Computes the current support of the labels in this instance.

        Returns:
            a FrameRanges
        """
        raise NotImplementedError(
            "subclasses must implement _compute_support()"
        )


class HasFramewiseView(object):
    """Mixin for Label classes that describe videos and can be rendered in
    a framewise view by a LabelsFrameRenderer.
    """

    @property
    def framewise_renderer_cls(self):
        """The LabelsFrameRenderer used by this class."""
        raise NotImplementedError(
            "subclasses must implement framewise_renderer_cls()"
        )

    def render_framewise(self, in_place=False):
        """Renders a framewise version of the labels.

        Args:
            in_place: whether to perform the rendering in-place. By default,
                this is False

        Returns:
            a framewise version of the labels
        """
        renderer = self.framewise_renderer_cls(self)
        return renderer.render(in_place=in_place)


class HasSpatiotemporalView(object):
    """Mixin for Label classes that describe videos and can be rendered in a
    spatiotemporal view by a LabelsSpatiotemporalRenderer.
    """

    @property
    def spatiotemporal_renderer_cls(self):
        """The LabelsSpatiotemporalRenderer used by this class."""
        raise NotImplementedError(
            "subclasses must implement spatiotemporal_renderer_cls()"
        )

    def render_spatiotemporal(self, in_place=False):
        """Renders a spatiotemporal version of the labels.

        Args:
            in_place: whether to perform the rendering in-place. By default,
                this is False

        Returns:
            a spatiotemporal version of the labels
        """
        renderer = self.spatiotemporal_renderer_cls(self)
        return renderer.render(in_place=in_place)


class LabelsContainer(Labels, HasLabelsSchema, etas.Container):
    """Base class for `eta.core.serial.Container`s of Labels.

    `LabelsContainer`s can optionally store a LabelsContainerSchema instance
    that governs the schema of the labels in the container.
    """

    def __init__(self, schema=None, **kwargs):
        """Creates a LabelsContainer instance.

        Args:
            schema: an optional LabelsContainerSchema to enforce on the labels
                in this container. By default, no schema is enforced
            **kwargs: valid keyword arguments for `eta.core.serial.Container()`

        Raises:
            LabelsSchemaError: if a schema was provided but the labels added to
                the container violate it
        """
        HasLabelsSchema.__init__(self, schema=schema)
        etas.Container.__init__(self, **kwargs)

    def __bool__(self):
        return etas.Container.__bool__(self)

    @property
    def is_empty(self):
        """Whether this container has no labels."""
        return etas.Container.is_empty(self)

    def remove_empty_labels(self):
        """Removes all empty Labels from the container."""
        self.filter_elements([lambda labels: not labels.is_empty])

    def add_container(self, container):
        """Appends the labels in the given LabelContainer to the container.

        Args:
            container: a LabelsContainer

        Raises:
            LabelsSchemaError: if this container has a schema enforced and any
                labels in the container violate it
        """
        self.add_iterable(container)

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        """
        _attrs = []
        if self.has_schema:
            _attrs.append("schema")

        _attrs += super(LabelsContainer, self).attributes()
        return _attrs

    @classmethod
    def from_dict(cls, d):
        """Constructs a LabelsContainer from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a LabelsContainer
        """
        schema = d.get("schema", None)
        if schema is not None:
            schema_cls = cls.get_schema_cls()
            schema = schema_cls.from_dict(schema)

        return super(LabelsContainer, cls).from_dict(d, schema=schema)

    def validate_schema(self):
        """Validates that the labels are compliant with the current schema.

        Raises:
            LabelsSchemaError: if the container has labels that are not
                compliant with the schema
        """
        if self.has_schema:
            for labels in self:
                self._validate_labels(labels)

    def _validate_labels(self, labels):
        if self.has_schema:
            self.schema.validate(labels)


class LabelsContainerSchema(LabelsSchema):
    """Base class for schemas of `LabelsContainer`s."""

    def add(self, labels):
        """Incorporates the Labels into the schema.

        Args:
            label: a Labels instance
        """
        self.merge_schema(labels.get_active_schema())

    def add_container(self, container):
        """Incorporates the given `LabelsContainer`s elements into the schema.

        Args:
            container: a LabelsContainer
        """
        self.add_iterable(container)

    def add_iterable(self, iterable):
        """Incorporates the given iterable of Labels into the schema.

        Args:
            iterable: an iterable of Labels
        """
        for labels in iterable:
            self.add(labels)

    @classmethod
    def build_active_schema(cls, container):
        """Builds a LabelsContainerSchema describing the active schema of the
        LabelsContainer.

        Args:
            container: a LabelsContainer

        Returns:
            a LabelsContainerSchema
        """
        schema = cls()
        for labels in container:
            schema.add(labels.get_active_schema())

        return schema


class LabelsContainerSchemaError(LabelsSchemaError):
    """Error raisesd when a LabelsContainerSchema is violated."""

    pass


class LabelsSet(Labels, HasLabelsSchema, etas.Set):
    """Base class for `eta.core.serial.Set`s of Labels.

    `LabelsSet`s can optionally store a LabelsSchema instance that governs
    the schemas of the Labels in the set.
    """

    def __init__(self, schema=None, **kwargs):
        """Creates a LabelsSet instance.

        Args:
            schema: an optional LabelsSchema to enforce on each element of the
                set. By default, no schema is enforced
            **kwargs: valid keyword arguments for `eta.core.serial.Set()`

        Raises:
            LabelsSchemaError: if a schema was provided but the labels added to
                the container violate it
        """
        HasLabelsSchema.__init__(self, schema=schema)
        etas.Set.__init__(self, **kwargs)

    def __getitem__(self, key):
        """Gets the Labels for the given key.

        If the key is not found, an empty Labels is created and returned.

        Args:
            key: the key

        Returns:
            a Labels instance
        """
        if key not in self:
            logger.warning(
                "Key '%s' not found; creating new %s",
                key,
                self._ELE_CLS.__name__,
            )
            # pylint: disable=not-callable
            labels = self._ELE_CLS(**{self._ELE_KEY_ATTR: key})
            self.add(labels)

        return super(LabelsSet, self).__getitem__(key)

    def __bool__(self):
        return etas.Set.__bool__(self)

    @property
    def is_empty(self):
        """Whether this set has no labels."""
        return etas.Set.is_empty(self)

    @classmethod
    def get_schema_cls(cls):
        """Gets the schema class for the Labels in the set.

        Returns:
            the LabelsSchema class
        """
        return cls._ELE_CLS.get_schema_cls()

    def empty(self):
        """Returns an empty copy of the LabelsSet.

        The schema of the set is preserved, if applicable.

        Returns:
            an empty LabelsSet
        """
        return self.__class__(schema=self.schema)

    def remove_empty_labels(self):
        """Removes all empty Labels from the set."""
        self.filter_elements([lambda labels: not labels.is_empty])

    def add_set(self, labels_set):
        """Adds the labels in the given LabelSet to the set.

        Args:
            labels_set: a LabelsSet

        Raises:
            LabelsSchemaError: if this set has a schema enforced and any labels
                in the set violate it
        """
        self.add_iterable(labels_set)

    def get_active_schema(self):
        """Gets the LabelsSchema describing the active schema of the set.

        Returns:
            a LabelsSchema
        """
        schema_cls = self.get_schema_cls()
        schema = schema_cls()
        for labels in self:
            schema.merge_schema(schema_cls.build_active_schema(labels))

        return schema

    def filter_by_schema(self, schema):
        """Removes labels from the set that are not compliant with the given
        schema.

        Args:
            schema: a LabelsSchema
        """
        for labels in self:
            labels.filter_by_schema(schema)

    def set_schema(self, schema, filter_by_schema=False, validate=False):
        """Sets the enforced schema to the given LabelsSchema.

        Args:
            schema: a LabelsSchema to assign
            filter_by_schema: whether to filter labels that are not compliant
                with the schema. By default, this is False
            validate: whether to validate that the labels (after filtering, if
                applicable) are compliant with the new schema. By default, this
                is False

        Raises:
            LabelsSchemaError: if `validate` was `True` and this object
                contains labels that are not compliant with the schema
        """
        self.schema = schema
        for labels in self:
            labels.set_schema(
                schema, filter_by_schema=filter_by_schema, validate=validate
            )

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        """
        _attrs = []
        if self.has_schema:
            _attrs.append("schema")

        _attrs += super(LabelsSet, self).attributes()
        return _attrs

    @classmethod
    def from_dict(cls, d):
        """Constructs a LabelsSet from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a LabelsSet
        """
        schema = d.get("schema", None)
        if schema is not None:
            schema_cls = cls.get_schema_cls()
            schema = schema_cls.from_dict(schema)

        return super(LabelsSet, cls).from_dict(d, schema=schema)

    @classmethod
    def from_labels_patt(cls, labels_patt):
        """Creates a LabelsSet from a pattern of Labels files on disk.

        Args:
             labels_patt: a pattern with one or more numeric sequences for
                Labels files on disk

        Returns:
            a LabelsSet
        """
        labels_set = cls()
        for labels_path in etau.get_pattern_matches(labels_patt):
            labels_set.add(cls._ELE_CLS.from_json(labels_path))

        return labels_set

    def validate_schema(self):
        """Validates that the labels in the set are compliant with the current
        schema.

        Raises:
            LabelsSchemaError: if the set has labels that are not compliant
                with the schema
        """
        if self.has_schema:
            for labels in self:
                self._validate_labels(labels)

    def _validate_labels(self, labels):
        if self.has_schema:
            self.schema.validate(labels)


class LabelsRenderer(object):
    """Interface for classes that render Labels instances in a specified
    format.

    `LabelsRenderer`s must follow the strict convention that, when
    `in_place == False`, they do not modify or pass by reference any components
    of the source labels that they are rendering. In particular, any labels
    they produce are deep copies of the source labels.
    """

    #
    # The Labels class that this renderer takes as input
    #
    # Subclasses MUST set this field
    #
    _LABELS_CLS = None

    @property
    def labels_cls(self):
        """The Labels subclass that this renderer takes as input."""
        return self._LABELS_CLS

    def render(self, in_place=False):
        """Renders the labels in the format specified by the class.

        Args:
            in_place: whether to perform the rendering in-place. By default,
                this is False

        Returns:
            a `labels_cls` instance
        """
        raise NotImplementedError("subclasses must implement render()")


class LabelsContainerRenderer(LabelsRenderer):
    """Base class for rendering labels in `LabelsContainer`s in a specified
    format.

    The only thing that subclasses need to do to implement this interface is
    to define their `_LABELS_CLS` and `_ELEMENT_RENDERER_CLS`.
    """

    #
    # The LabelsRenderer class to use to render elements of the container
    #
    # Subclasses MUST set this field
    #
    _ELEMENT_RENDERER_CLS = None

    def __init__(self, container):
        """Creates a LabelsContainerRenderer instance.

        Args:
            container: a LabelsContainer
        """
        self._container = container

    @property
    def element_renderer_cls(self):
        """The LabelsRenderer class to use to render elements of the container.
        """
        return self._ELEMENT_RENDERER_CLS

    def render(self, in_place=False):
        """Renders the container in the format specified by the class.

        Args:
            in_place: whether to perform the rendering in-place. By default,
                this is False

        Returns:
            a `labels_cls` instance
        """
        if in_place:
            return self._render_in_place()

        return self._render_copy()

    def _render_in_place(self):
        for labels in self._container:
            # pylint: disable=not-callable
            renderer = self.element_renderer_cls(labels)
            renderer.render(in_place=True)

        return self._container

    def _render_copy(self):
        new_container = self._container.empty()
        for labels in self._container:
            # pylint: disable=not-callable
            renderer = self.element_renderer_cls(labels)
            element = renderer.render(in_place=False)
            new_container.add(element)

        return new_container


class LabelsFrameRenderer(LabelsRenderer):
    """Interface for classes that render Labels at the frame-level."""

    #
    # The per-frame Labels class that this renderer outputs
    #
    # Subclasses MUST set this field
    #
    _FRAME_LABELS_CLS = None

    @property
    def frame_labels_cls(self):
        """The per-frame Labels class that this renderer outputs."""
        return self._FRAME_LABELS_CLS

    def render_frame(self, frame_number, in_place=False):
        """Renders the labels for the given frame.

        Args:
            frame_number: the frame number
            in_place: whether to perform the rendering in-place (i.e., without
                deep copying objects). By default, this is False

        Returns:
            a `frame_labels_cls` instance, or None if no labels exist for the
                given frame
        """
        raise NotImplementedError("subclasses must implement render_frame()")

    def render_all_frames(self, in_place=False):
        """Renders the labels for all possible frames.

        Args:
            in_place: whether to perform the rendering in-place (i.e., without
                deep copying objects). By default, this is False

        Returns:
            a dictionary mapping frame numbers to `frame_labels_cls` instances
        """
        raise NotImplementedError(
            "subclasses must implement render_all_frames()"
        )


class LabelsContainerFrameRenderer(
    LabelsFrameRenderer, LabelsContainerRenderer
):
    """Base class for rendering labels in `LabelsContainer`s at the
    frame-level.

    The only thing that subclasses need to do to implement this interface is
    to define their `_LABELS_CLS`, `_FRAME_LABELS_CLS`, and
    `_ELEMENT_RENDERER_CLS`.
    """

    #
    # The LabelsFrameRenderer class to use to render elements of the container
    #
    # Subclasses MUST set this field
    #
    _ELEMENT_RENDERER_CLS = None

    @property
    def element_renderer_cls(self):
        """The LabelsFrameRenderer class to use to render elements of the
        container.
        """
        return self._ELEMENT_RENDERER_CLS

    def render_frame(self, frame_number, in_place=False):
        """Renders the container for the given frame.

        Args:
            frame_number: the frame number
            in_place: whether to perform the rendering in-place (i.e., without
                deep copying objects). By default, this is False

        Returns:
            a `frame_labels_cls` instance, which may be empty if no labels
                exist for the specified frame
        """
        # pylint: disable=not-callable
        frame_elements = self.frame_labels_cls()

        for labels in self._container:
            # pylint: disable=not-callable
            renderer = self.element_renderer_cls(labels)
            frame_element = renderer.render_frame(
                frame_number, in_place=in_place
            )
            if frame_element is not None:
                frame_elements.add(frame_element)

        return frame_elements

    def render_all_frames(self, in_place=False):
        """Renders the container for all possible frames.

        Args:
            in_place: whether to perform the rendering in-place (i.e., without
                deep copying objects). By default, this is False

        Returns:
            a dictionary mapping frame numbers to `frame_labels_cls` instances
        """
        # pylint: disable=not-callable
        frame_elements_map = defaultdict(self.frame_labels_cls)

        for labels in self._container:
            # pylint: disable=not-callable
            renderer = self.element_renderer_cls(labels)
            frames_map = renderer.render_all_frames(in_place=in_place)
            for frame_number, frame_element in iteritems(frames_map):
                frame_elements_map[frame_number].add(frame_element)

        return dict(frame_elements_map)


class LabelsSpatiotemporalRenderer(LabelsRenderer):
    """Interface for classes that render Labels in spatiotemporal format."""

    pass


class LabelsContainerSpatiotemporalRenderer(
    LabelsSpatiotemporalRenderer, LabelsContainerRenderer
):
    """Base class for rendering labels for `LabelsContainer`s in spatiotemporal
    format.

    The only thing that subclasses need to do to implement this interface is
    to define their `_LABELS_CLS` and `_ELEMENT_RENDERER_CLS`.
    """

    pass
