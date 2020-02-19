'''
Core data structures for working with labels.

Copyright 2017-2020, Voxel51, Inc.
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
from future.utils import iteritems
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import OrderedDict, defaultdict
import logging

import eta.core.serial as etas
import eta.core.utils as etau


logger = logging.getLogger(__name__)


class Labels(etas.Serializable):
    '''Base class for `eta.core.serial.Serializable` classes that hold labels
    representing attributes, objects, frames, events, images, videos, etc.

    Labels classes have associated Schema classes that describe the
    ontologies over the labels class.
    '''

    def __bool__(self):
        '''Whether this instance has labels of any kind.'''
        return not self.is_empty

    @property
    def is_empty(self):
        '''Whether this instance has no labels of any kind.'''
        raise NotImplementedError("subclasses must implement is_empty")

    @classmethod
    def get_schema_cls(cls):
        '''Gets the LabelsSchema class for the labels.

        Subclasses can override this method, but, by default, this
        implementation assumes the convention that labels class `<Labels>` has
        associated schema class `<Labels>Schema` defined in the same module.

        Returns:
            the LabelsSchema class
        '''
        class_name = etau.get_class_name(cls)
        return etau.get_class(class_name + "Schema")

    def get_active_schema(self):
        '''Returns a LabelsSchema that describes the active schema of the
        labels.

        Returns:
            a LabelsSchema
        '''
        schema_cls = self.get_schema_cls()
        return schema_cls.build_active_schema(self)

    def filter_by_schema(self, schema):
        '''Filters the labels by the given schema.

        Args:
            schema: a LabelsSchema
        '''
        raise NotImplementedError(
            "subclasses must implement `filter_by_schema()`")


class LabelsSchema(etas.Serializable):
    '''Base class for schemas of Labels classes.'''

    def __bool__(self):
        '''Whether this schema has labels of any kind.'''
        return not self.is_empty

    @property
    def is_empty(self):
        '''Whether this schema has no labels of any kind.'''
        raise NotImplementedError("subclasses must implement is_empty")

    def add(self, labels):
        '''Incorporates the Labels into the schema.

        Args:
            label: a Labels instance
        '''
        labels_schema = self.build_active_schema(labels)
        self.merge_schema(labels_schema)

    def add_iterable(self, iterable):
        '''Incorporates the given iterable of Labels into the schema.

        Args:
            iterable: an iterable of Labels
        '''
        for labels in iterable:
            self.add(labels)

    def validate(self, labels):
        '''Validates that the Labels are compliant with the schema.

        Args:
            labels: a Labels instance

        Raises:
            LabelsSchemaError: if the labels violate the schema
        '''
        raise NotImplementedError("subclasses must implement `validate()`")

    def validate_subset_of_schema(self, schema):
        '''Validates that this schema is a subset of the given LabelsSchema.

        Args:
            schema: a LabelsSchema

        Raises:
            LabelsSchemaError: if this schema is not a subset of the given
                schema
        '''
        raise NotImplementedError(
            "subclasses must implement `validate_subset_of_schema()`")

    def validate_schema_type(self, schema):
        '''Validates that this schema is an instance of same type as the given
        schema.

        Args:
            schema: a LabelsSchema

        Raises:
            LabelsSchemaError: if this schema is not of the same type as the
                given schema
        '''
        if not isinstance(self, type(schema)):
            raise LabelsSchemaError(
                "Expected `self` to match schema type %s; found %s" %
                (type(self), type(schema)))

    def is_valid(self, labels):
        '''Whether the Labels are compliant with the schema.

        Args:
            labels: a Labels instance

        Returns:
            True/False
        '''
        try:
            self.validate(labels)
            return True
        except LabelsSchemaError:
            return False

    def is_subset_of_schema(self, schema):
        '''Whether this schema is a subset of the given schema.

        Args:
            schema: a LabelsSchema

        Returns:
            True/False
        '''
        try:
            self.validate_subset_of_schema(schema)
            return True
        except LabelsSchemaError:
            return False

    @classmethod
    def build_active_schema(cls, labels):
        '''Builds a LabelsSchema that describes the active schema of the
        labels.

        Args:
            labels: a Labels instance

        Returns:
            a LabelsSchema
        '''
        raise NotImplementedError(
            "subclasses must implement `build_active_schema()`")

    def merge_schema(self, schema):
        '''Merges the given LabelsSchema into this schema.

        Args:
            schema: a LabelsSchema
        '''
        raise NotImplementedError("subclasses must implement `merge_schema()`")


class LabelsSchemaError(Exception):
    '''Error raisesd when a LabelsSchema is violated.'''
    pass


class HasLabelsSchema(object):
    '''Mixin for Label classes that can optionally store and enforce
    `LabelsSchema`s on their labels.

    For efficiency, schemas are not automatically enforced when new labels are
    added to HasLabelsSchema instances. Rather, users must manually call
    `validate_schema()` when they would like to validate the schema.

    Attributes:
        schema: the enforced LabelsSchema, or None
    '''

    def __init__(self, schema=None):
        '''Initializes the HasLabelsSchema mixin.

        Args:
            schema: (optional) a LabelsSchema to enforce on the labels. By
                default, no schema is enforced
        '''
        self.schema = schema

    @property
    def has_schema(self):
        '''Whether the labels have an enforced schema.'''
        return self.schema is not None

    def get_schema(self):
        '''Gets the current enforced schema for the labels, or None if no
        schema is enforced.

        Returns:
            a LabelsSchema, or None
        '''
        return self.schema

    def set_schema(self, schema, filter_by_schema=False, validate=False):
        '''Sets the enforced schema to the given LabelsSchema.

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
        '''
        self.schema = schema
        if not self.has_schema:
            return

        if filter_by_schema:
            self.filter_by_schema(self.schema)  # pylint: disable=no-member

        if validate:
            self.validate_schema()

    def validate_schema(self):
        '''Validates that the labels are compliant with the current schema.

        Raises:
            LabelsSchemaError: if this object contains labels that are not
                compliant with the schema
        '''
        if self.has_schema:
            self.schema.validate(self)

    def freeze_schema(self):
        '''Sets the schema for the labels to the current active schema.'''
        self.set_schema(self.get_active_schema())  # pylint: disable=no-member

    def remove_schema(self):
        '''Removes the enforced schema from the labels.'''
        self.set_schema(None)


class HasLabelsSupport(object):
    '''Mixin for Label classes that describe videos and can keep track of
    their own support, i.e., the frames for which they contain labels.

    The support is represented via a `eta.core.frameutils.FrameRanges`
    instance.

    For efficiency, supports should not be automatically updated when new
    labels are added to HasLabelsSupport instances. Rather, the support is
    dynamically computed when the `support` property is accessed.
    Alternatively, the current support can be frozen via `freeze_support()`
    to avoid recomputing it each time `support` is called.
    '''

    def __init__(self, support=None):
        '''Initializes the HasLabelsSupport mixin.

        Args:
            support: (optional) a FrameRanges instance describing the frozen
                support of the labels. By default, the support is not frozen
        '''
        self._support = support

    @property
    def support(self):
        '''A FrameRanges instance describing the frames for which this instance
        contains labels.

        If this instance has a frozen support, it is returned. Otherwise, the
        support is dynamically computed via `_compute_support()`.
        '''
        if self.is_support_frozen:
            return self._support

        return self._compute_support()

    @property
    def is_support_frozen(self):
        '''Whether the support is currently frozen.'''
        return self._support is not None

    def set_support(self, support):
        '''Sets the support to the given value.

        This action freezes the support for this instance.

        Args:
            support: a FrameRanges
        '''
        self._support = support

    def merge_support(self, support):
        '''Merges the given support into the current support.

        This action freezes the support for this instance.

        Args:
            support: a FrameRanges
        '''
        new_support = self.support.merge(support)
        self.set_support(new_support)

    def freeze_support(self):
        '''Freezes the support to the current `support`.

        This optional optimization is useful to avoid recalculating the support
        of the labels each time `support` is called.
        '''
        if not self.is_support_frozen:
            self._support = self._compute_support()

    def clear_support(self):
        '''Clears the frozen support, if necessary.'''
        self._support = None

    def _compute_support(self):
        '''Computes the current support of the labels in this instance.

        Returns:
            a FrameRanges
        '''
        raise NotImplementedError(
            "subclasses must implement _compute_support()")


class HasFramewiseView(object):
    '''Mixin for Label classes that describe videos and can be rendered in
    a framewise view by a LabelsFrameRenderer.
    '''

    def render_framewise_labels(self):
        '''Renders a framewise copy of the labels.

        Returns:
            an framewise copy of the labels
        '''
        raise NotImplementedError(
            "subclasses must implement render_framewise_labels()")


class LabelsContainer(Labels, HasLabelsSchema, etas.Container):
    '''Base class for `eta.core.serial.Container`s of Labels.

    `LabelsContainer`s can optionally store a LabelsContainerSchema instance
    that governs the schema of the labels in the container.
    '''

    def __init__(self, schema=None, **kwargs):
        '''Creates a LabelsContainer instance.

        Args:
            schema: an optional LabelsContainerSchema to enforce on the labels
                in this container. By default, no schema is enforced
            **kwargs: valid keyword arguments for `eta.core.serial.Container()`

        Raises:
            LabelsSchemaError: if a schema was provided but the labels added to
                the container violate it
        '''
        HasLabelsSchema.__init__(self, schema=schema)
        etas.Container.__init__(self, **kwargs)

    def __bool__(self):
        return etas.Container.__bool__(self)

    @property
    def is_empty(self):
        '''Whether this container has no labels.'''
        return etas.Container.is_empty(self)

    def add_container(self, container):
        '''Appends the labels in the given LabelContainer to the container.

        Args:
            container: a LabelsContainer

        Raises:
            LabelsSchemaError: if this container has a schema enforced and any
                labels in the container violate it
        '''
        self.add_iterable(container)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        _attrs = []
        if self.has_schema:
            _attrs.append("schema")

        _attrs += super(LabelsContainer, self).attributes()
        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs a LabelsContainer from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a LabelsContainer
        '''
        schema = d.get("schema", None)
        if schema is not None:
            schema_cls = cls.get_schema_cls()
            schema = schema_cls.from_dict(schema)

        return super(LabelsContainer, cls).from_dict(d, schema=schema)

    def validate_schema(self):
        '''Validates that the labels are compliant with the current schema.

        Raises:
            LabelsSchemaError: if the container has labels that are not
                compliant with the schema
        '''
        if self.has_schema:
            for labels in self:
                self._validate_labels(labels)

    def _validate_labels(self, labels):
        if self.has_schema:
            self.schema.validate(labels)


class LabelsContainerSchema(LabelsSchema):
    '''Base class for schemas of `LabelsContainer`s.'''

    def add(self, labels):
        '''Incorporates the Labels into the schema.

        Args:
            label: a Labels instance
        '''
        self.merge_schema(labels.get_active_schema())

    def add_container(self, container):
        '''Incorporates the given `LabelsContainer`s elements into the schema.

        Args:
            container: a LabelsContainer
        '''
        self.add_iterable(container)

    def add_iterable(self, iterable):
        '''Incorporates the given iterable of Labels into the schema.

        Args:
            iterable: an iterable of Labels
        '''
        for labels in iterable:
            self.add(labels)

    @classmethod
    def build_active_schema(cls, container):
        '''Builds a LabelsContainerSchema describing the active schema of the
        LabelsContainer.

        Args:
            container: a LabelsContainer

        Returns:
            a LabelsContainerSchema
        '''
        schema = cls()
        for labels in container:
            schema.add(labels.get_active_schema())

        return schema


class LabelsContainerSchemaError(LabelsSchemaError):
    '''Error raisesd when a LabelsContainerSchema is violated.'''
    pass


class LabelsSet(Labels, HasLabelsSchema, etas.Set):
    '''Base class for `eta.core.serial.Set`s of Labels.

    `LabelsSet`s can optionally store a LabelsSchema instance that governs
    the schemas of the Labels in the set.
    '''

    def __init__(self, schema=None, **kwargs):
        '''Creates a LabelsSet instance.

        Args:
            schema: an optional LabelsSchema to enforce on each element of the
                set. By default, no schema is enforced
            **kwargs: valid keyword arguments for `eta.core.serial.Set()`

        Raises:
            LabelsSchemaError: if a schema was provided but the labels added to
                the container violate it
        '''
        HasLabelsSchema.__init__(self, schema=schema)
        etas.Set.__init__(self, **kwargs)

    def __bool__(self):
        return etas.Set.__bool__(self)

    @property
    def is_empty(self):
        '''Whether this set has no labels.'''
        return etas.Set.is_empty(self)

    def __getitem__(self, key):
        '''Gets the Labels for the given key.

        If the key is not found, an empty Labels is created and returned.

        Args:
            key: the key

        Returns:
            a Labels instance
        '''
        if key not in self:
            # pylint: disable=not-callable
            labels = self._ELE_CLS(**{self._ELE_KEY_ATTR: key})
            self.add(labels)

        return super(LabelsSet, self).__getitem__(key)

    @classmethod
    def get_schema_cls(cls):
        '''Gets the schema class for the Labels in the set.

        Returns:
            the LabelsSchema class
        '''
        return cls._ELE_CLS.get_schema_cls()

    def empty(self):
        '''Returns an empty copy of the LabelsSet.

        The schema of the set is preserved, if applicable.

        Returns:
            an empty LabelsSet
        '''
        return self.__class__(schema=self.schema)

    def add_set(self, labels_set):
        '''Adds the labels in the given LabelSet to the set.

        Args:
            labels_set: a LabelsSet

        Raises:
            LabelsSchemaError: if this set has a schema enforced and any labels
                in the set violate it
        '''
        self.add_iterable(labels_set)

    def get_active_schema(self):
        '''Gets the LabelsSchema describing the active schema of the set.

        Returns:
            a LabelsSchema
        '''
        schema_cls = self.get_schema_cls()
        schema = schema_cls()
        for labels in self:
            schema.merge_schema(schema_cls.build_active_schema(labels))

        return schema

    def filter_by_schema(self, schema):
        '''Removes labels from the set that are not compliant with the given
        schema.

        Args:
            schema: a LabelsSchema
        '''
        for labels in self:
            labels.filter_by_schema(schema)

    def set_schema(self, schema, filter_by_schema=False, validate=False):
        '''Sets the enforced schema to the given LabelsSchema.

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
        '''
        self.schema = schema
        for labels in self:
            labels.set_schema(
                schema, filter_by_schema=filter_by_schema, validate=validate)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        _attrs = []
        if self.has_schema:
            _attrs.append("schema")

        _attrs += super(LabelsSet, self).attributes()
        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs a LabelsSet from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a LabelsSet
        '''
        schema = d.get("schema", None)
        if schema is not None:
            schema_cls = cls.get_schema_cls()
            schema = schema_cls.from_dict(schema)

        return super(LabelsSet, cls).from_dict(d, schema=schema)

    @classmethod
    def from_labels_patt(cls, labels_patt):
        '''Creates a LabelsSet from a pattern of Labels files on disk.

        Args:
             labels_patt: a pattern with one or more numeric sequences for
                Labels files on disk

        Returns:
            a LabelsSet
        '''
        labels_set = cls()
        for labels_path in etau.get_pattern_matches(labels_patt):
            labels_set.add(cls._ELE_CLS.from_json(labels_path))

        return labels_set

    def validate_schema(self):
        '''Validates that the labels in the set are compliant with the current
        schema.

        Raises:
            LabelsSchemaError: if the set has labels that are not compliant
                with the schema
        '''
        if self.has_schema:
            for labels in self:
                self._validate_labels(labels)

    def _validate_labels(self, labels):
        if self.has_schema:
            self.schema.validate(labels)


class LabelsFrameRenderer(object):
    '''Interface for classes that render Labels at the frame-level.

    `LabelsFrameRenderer`s must follow the strict convention that they do not
    modify or pass by reference any components of the source Labels that they
    are rendering. I.e., any labels they produce are deep copies of the source
    labels.
    '''

    def render_frame(self, frame_number):
        '''Renders the labels for the given frame.

        Args:
            frame_number: the frame number

        Returns:
            a Labels instance, or None if no labels exist for the given frame
        '''
        raise NotImplementedError("subclasses must implement render_frame()")

    def render_all_frames(self):
        '''Renders the labels for all possible frames.

        Returns:
            a dictionary mapping frame numbers to Labels instances
        '''
        raise NotImplementedError(
            "subclasses must implement render_all_frames()")


class LabelsContainerFrameRenderer(LabelsFrameRenderer):
    '''Base class for rendering labels for Containers at the frame-level.'''

    #
    # The Container class in which to store frame elements that are rendered
    #
    # Subclasses MUST set this field
    #
    _FRAME_CONTAINER_CLS = None

    #
    # The LabelsFrameRenderer class to use to render elements of the container
    #
    # Subclasses MUST set this field
    #
    _ELEMENT_RENDERER_CLS = None

    def __init__(self, container):
        '''Creates an LabelsContainerFrameRenderer instance.

        Args:
            container: a Container
        '''
        self._container = container

    def render_frame(self, frame_number):
        '''Renders the Container for the given frame.

        Args:
            frame_number: the frame number

        Returns:
            a `_FRAME_CONTAINER_CLS` instance, which may be empty if no labels
                exist for the specified frame
        '''
        # pylint: disable=not-callable
        frame_elements = self._FRAME_CONTAINER_CLS()

        for element in self._container:
            # pylint: disable=not-callable
            renderer = self._ELEMENT_RENDERER_CLS(element)
            frame_element = renderer.render_frame(frame_number)
            if frame_element is not None:
                frame_elements.add(frame_element)

        return frame_elements

    def render_all_frames(self):
        '''Renders the Container for all possible frames.

        Returns:
            a dictionary mapping frame numbers to `_FRAME_CONTAINER_CLS`
                instances
        '''
        # pylint: disable=not-callable
        frame_elements_map = defaultdict(self._FRAME_CONTAINER_CLS)

        for element in self._container:
            # pylint: disable=not-callable
            renderer = self._ELEMENT_RENDERER_CLS(element)
            frame_map = renderer.render_all_frames()
            for frame_number, frame_element in iteritems(frame_map):
                frame_elements_map[frame_number].add(frame_element)

        return dict(frame_elements_map)


CONDENSED_STRING_CLASS_MAP = {
    # "<image attr>": ImageAttrFilter,
    # "<video attr>": VideoAttrFilter,
    # "<frame attr>": FrameAttrFilter,
    # "<object>": DetectedObjectFilter,
    # "<event>": EventFilter,
    # "<object attr>": DetectedObjectAttrFilter,
    # "<event attr>": EventAttrFilter
}


class LabelsIterator(etas.Serializable):
    '''@todo(Tyler)'''

    @property
    def type(self):
        return self._type

    def __init__(self):
        self._type = etau.get_class_name(self)

    def iter_matches(self, labels):
        raise NotImplementedError("Subclass must implement")

    def attributes(self):
        return super(LabelsIterator, self).attributes() + ["type"]

    @classmethod
    def from_dict(cls, d, *args, **kwargs):
        subcls = etau.get_class(d["type"])
        if not issubclass(subcls, cls):
            raise ValueError(
                "%s not subclass of %s" % (d["type"], etau.get_class_name(cls)))
        return subcls._from_dict(d)

    @classmethod
    def from_condensed_str(cls, s: str):
        '''TODO

        Example inputs:
            "<object attr>:*:<boolean>:occluded:false"
        '''
        parts = s.split(":")
        subcls = CONDENSED_STRING_CLASS_MAP[parts.pop(0)]

        if not issubclass(subcls, cls):
            raise ValueError(
                "%s not subclass of %s"
                % (etau.get_class_name(subcls), etau.get_class_name(cls)))

        return subcls._from_condensed_strings(*parts)

    @classmethod
    def _from_dict(cls, d, *args, **kwargs):
        raise NotImplementedError("Subclass must implement")

    @classmethod
    def _from_condensed_strings(cls, *args, **kwargs):
        raise NotImplementedError("Subclass must implement")



# MANAGER


class LabelsTransformerManager():
    '''
    - keeps track of a series of transforms
    - manages running transforms on Labels, SetLabels, Dataset, Directory or
        List of Labels
    '''

    @property
    def transformers(self):
        return self._transformers

    def __init__(self):
        self._transformers = OrderedDict()

    def get_reports(self):
        reports = {}
        for k, transform in self._transformers.items():
            reports[k] = transform.report
        return reports

    def add_transformer(self, transformer):
        if not isinstance(transformer, LabelsTransformer):
            raise ValueError(
                "Unexpected type: '%s'".format(type(transformer)))

        k = str(len(self._transformers)) + " - " + \
            etau.get_class_name(transformer)

        self._transformers[k] = transformer

    def transform_labels(self, labels, labels_path=None):
        for transformer in self._transformers.values():
            transformer.transform(labels)

        if labels_path:
            labels.write_json(labels_path)

    def transform_set_labels(
            self, set_labels, set_labels_path=None, verbose=20):
        for idx, labels in enumerate(set_labels):
            if verbose and idx % verbose == 0:
                logger.info("%4d/%4d" % (idx, len(set_labels)))

            self.transform_labels(labels, labels_path=None)

        if set_labels_path:
            set_labels.write_json(set_labels_path)

    def transform_dataset(self, dataset, verbose=20):
        for idx, labels_path in enumerate(dataset.iter_labels_paths()):
            if verbose and idx % verbose == 0:
                logger.info("%4d/%4d" % (idx, len(dataset)))

            labels = dataset.read_labels(labels_path)

            self.transform_labels(labels, labels_path=labels_path)

            # break # @todo(Tyler) TEMP


# ABSTRACT CLASS


class LabelsTransformerError(Exception):
    '''Error raised when a LabelsTransformer is violated.'''
    pass


class LabelsTransformer():
    _ERROR_CLS = LabelsTransformerError
    _LABELS_CLS = Labels

    @classmethod
    def get_labels_cls(cls):
        return cls._LABELS_CLS

    @property
    def num_labels_transformed(self):
        return self._num_labels_transformed

    @property
    def report(self):
        return {
            "num_labels_transformed": self.num_labels_transformed,
        }

    def __init__(self):
        self.clear_state()

        if type(self) == LabelsTransformer:
            raise TypeError("Cannot instantiate abstract class %s"
                            % etau.get_class_name(LabelsTransformer))

    def clear_state(self):
        self._num_labels_transformed = 0

    def transform(self, labels):
        etau.validate_type(labels, self.get_labels_cls())
        self._num_labels_transformed += 1


class SyntaxCheckerError(Exception):
    '''Error raised when a LabelsTransformer is violated.'''
    pass


class SyntaxChecker(LabelsTransformer):
    '''Using a target schema, match capitalization and underscores versus spaces
    to match the schema
    '''
    _ERROR_CLS = SyntaxCheckerError
    _SCHEMA_CLS = None

    @property
    def target_schema(self):
        '''The target (desired) _SCHEMA_CLS instance'''
        return self._target_schema

    @property
    def fixable_schema(self):
        '''The fixable _SCHEMA_CLS instance
        If for example:
            - checked labels have object label "Road_Sign"
            - `target_schema` has object label "road sign"
        then `fixable_schema` will contain object label "Road_Sign"
        This schema accumulates from all calls to `check()`
        '''
        return self._fixable_schema

    @property
    def report(self):
        d = super(SyntaxChecker, self).report
        d["target_schema"] = self.target_schema
        d["fixable_schema"] = self.fixable_schema
        d["unfixable_schema"] = self.unfixable_schema
        return d

    @property
    def unfixable_schema(self):
        '''The un-fixable _SCHEMA_CLS instance containing anything not in the
        target_schema and cannot be mapped to the target_schema by
        capitalization and spaces/underscores.
        This schema accumulates from all calls to `check()`
        '''
        return self._unfixable_schema

    def __init__(self, target_schema):
        '''Creates an ImageLabelsSyntaxChecker instance.
        Args:
            target_schema: a _SCHEMA_CLS object with the target (desired) schema
                to check labels against
        '''
        etau.validate_type(target_schema, self._SCHEMA_CLS)
        self._target_schema = target_schema
        super(SyntaxChecker, self).__init__()

    def clear_state(self):
        '''Clear the `fixable_schema` and `unfixable_schema` of any accumulated
        data.
        '''
        super(SyntaxChecker, self).clear_state()

        # child class must instantiate these schemas
        self._fixable_schema = None
        self._unfixable_schema = None

    @staticmethod
    def _standardize(s):
        return str(s).lower().replace("_", " ")
