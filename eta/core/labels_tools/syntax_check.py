'''TODO

Copyright 2017-2020, Voxel51, Inc.
voxel51.com

Tyler Ganter, tyler@voxel51.com
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

import logging

import eta.core.data as etad
import eta.core.image as etai
import eta.core.utils as etau
import eta.core.video as etav


logger = logging.getLogger(__name__)



class ImageLabelsSyntaxCheckerError(Exception):
    '''Error raised when a ImageLabelsSyntaxChecker is violated.'''
    pass


def _skip_non_categorical_attrs(func):
    '''decorator for `ImageLabelsSyntaxChecker._check_thing` that ensures only
    categorical attributes are checked
    '''

    def wrapper(self, thing, *args, **kwargs):
        if isinstance(thing, etad.Attribute):
            # We only care about checking the value for categorical attributes
            if isinstance(thing, etad.BooleanAttribute):
                return

            if isinstance(thing, etad.NumericAttribute):
                return

            if not isinstance(thing, etad.CategoricalAttribute):
                raise self._ERROR_CLS(
                    "Unexpected attribute type: '%s'" % thing.type)

        return func(self, thing, *args, **kwargs)

    return wrapper


def _map_attrs_to_and_from_strings(func):
    ''' decorator for `ImageLabelsSyntaxChecker_map_to_target` that joins attr
    name and value before checking mapping and parses to a CategoricalAttribute
    after receiving the result.
    '''

    def wrapper(self, attr, attr_container_schema, *args, **kwargs):
        if isinstance(attr, etad.Attribute):
            self._validate_type(
                attr_container_schema, etad.AttributeContainerSchema)

            value = "%s:%s" % (attr.name, attr.value)
            target_iterable = [
                ":".join((attr_name, attr_value))
                for attr_schema in attr_container_schema.schema.values()
                if isinstance(attr_schema, etad.CategoricalAttributeSchema)
                for attr_name, attr_value in attr_schema.iter_name_values()
            ]

            result = func(self, value, target_iterable, *args, **kwargs)

            if result:
                attr_name, attr_value = result.split(":")
                return etad.CategoricalAttribute(attr_name, attr_value)
            else:
                return result

        else:
            return func(self, attr, attr_container_schema, *args, **kwargs)


    return wrapper


class ImageLabelsSyntaxChecker(object):
    '''Tool that checks ImageLabels against an ImageLabelsSchema and
    makes capitalization and underscores versus spaces consistent.
    e.g. assuming the target_schema contains "road sign", map "Road Sign" and
    "road_sign" in an ImageLabels object to "road sign"
    The Checker is initialized with a schema, then any number of ImageLabels
    can be checked with it. The Checker modifies the labels in place but does
    NOT write them to disk. The Checker also accumulates fixable and un-fixable
    schemas. Un-fixable labels cannot be inferred and should be fixed by another
    means.
    Example usage:
        ```
        checker = ImageLabelsSyntaxChecker(schema)
        was_modified = checker.check(labels)
        if was_modified:
            print("Was modified!")
            labels.write_json(labels_path)
        print(checker.fixable_schema)
        print(checker.unfixable_schema)
        ```
    Into the weeds:
        This class has one main "worker" function: `_check_thing`, which can
        check strings and attributes against a schema. The other private
        helpers such as:
            `_check_image_attrs`
            `_check_object_label`
            `_check_object_attrs`
        just specify functions used by `_check_thing` for their respective
        tasks.
    '''

    _SCHEMA_CLS = etai.ImageLabelsSchema
    _LABELS_CLS = etai.ImageLabels
    _ERROR_CLS = ImageLabelsSyntaxCheckerError

    def __init__(self, target_schema):
        '''Creates an ImageLabelsSyntaxChecker instance.
        Args:
            target_schema: a _SCHEMA_CLS object with the target (desired) schema
                to check labels against
        '''
        self._validate_type(target_schema, self._SCHEMA_CLS)

        self._target_schema = target_schema

        self.clear_state()

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
    def unfixable_schema(self):
        '''The un-fixable _SCHEMA_CLS instance containing anything not in the
        target_schema and cannot be mapped to the target_schema by
        capitalization and spaces/underscores.
        This schema accumulates from all calls to `check()`
        '''
        return self._unfixable_schema

    def clear_state(self):
        '''Clear the `fixable_schema` and `unfixable_schema` of any accumulated
        data.
        '''
        self._fixable_schema = self._SCHEMA_CLS()
        self._unfixable_schema = self._SCHEMA_CLS()

    def check(self, labels):
        '''Check a labels object against the target_schema and modify in-place
        any "fixable" values.
        Args:
            labels: a _LABELS_CLS instance
        Returns:
            True if the labels were modified (implying at least one fixable
                label was found)
        '''
        self._was_modified = False
        self._validate_type(labels, self._LABELS_CLS)
        self._check(labels)
        return self._was_modified

    def _check(self, labels):
        '''To be overridden by any child class'''
        self._check_image_attrs(labels)
        self._check_objects(labels.objects)

    def _check_image_attrs(self, labels):
        def valid_in_schema(schema, thing):
            return schema.is_valid_image_attribute(thing)

        def add_to_schema(schema, thing):
            schema.add_image_attribute(thing)

        def get_target_iterable():
            return self.target_schema.attrs

        def assign_mapped_value(thing, mapped_thing):
            thing.name = mapped_thing.name
            thing.value = mapped_thing.value

        for attr in labels.attrs:
            self._check_thing(attr, valid_in_schema, add_to_schema,
                              get_target_iterable, assign_mapped_value)

    def _check_objects(self, object_container):
        for obj in object_container:
            self._check_object_label(obj)
            self._check_object_attrs(obj)

    def _check_object_label(self, obj):
        def valid_in_schema(schema, thing):
            return schema.is_valid_object_label(thing)

        def add_to_schema(schema, thing):
            schema.add_object_label(thing)

        def get_target_iterable():
            return self.target_schema.objects.keys()

        def assign_mapped_value(thing, mapped_thing):
            obj.label = mapped_thing

        self._check_thing(obj.label, valid_in_schema, add_to_schema,
                          get_target_iterable, assign_mapped_value)

    def _check_object_attrs(self, obj):
        if not self.target_schema.is_valid_object_label(obj.label):
            return

        def valid_in_schema(schema, thing):
            return schema.is_valid_object_attribute(obj.label, thing)

        def add_to_schema(schema, thing):
            schema.add_object_attribute(obj.label, thing)

        def get_target_iterable():
            return self.target_schema.objects[obj.label]

        def assign_mapped_value(thing, mapped_thing):
            thing.name = mapped_thing.name
            thing.value = mapped_thing.value

        for attr in obj.attrs:
            self._check_thing(attr, valid_in_schema, add_to_schema,
                              get_target_iterable, assign_mapped_value)

    @_skip_non_categorical_attrs
    def _check_thing(self, thing, valid_in_schema, add_to_schema,
                     get_target_iterable, assign_mapped_value):
        # Is the value in the target schema?
        if valid_in_schema(self.target_schema, thing):
            return

        # Is the value in the fixable schema?
        if valid_in_schema(self.fixable_schema, thing):
            mapped_value = self._map_to_target(thing, get_target_iterable())

            if mapped_value is None:
                raise self._ERROR_CLS("Woah this is bad!")

            self._was_modified = True
            assign_mapped_value(thing, mapped_value)

            return

        # Is the value in the unfixable schema?
        if valid_in_schema(self.unfixable_schema, thing):
            return

        mapped_value = self._map_to_target(thing, get_target_iterable())

        if mapped_value is not None:
            add_to_schema(self.fixable_schema, thing)
            self._was_modified = True
            assign_mapped_value(thing, mapped_value)

        else:
            add_to_schema(self.unfixable_schema, thing)

    @_map_attrs_to_and_from_strings
    def _map_to_target(self, value, target_iterable):
        std_value = self._standardize(value)
        std_to_target_map = {self._standardize(x): x for x in target_iterable}

        if std_value in std_to_target_map:
            return std_to_target_map[std_value]

        return None

    def _standardize(self, value):
        return str(value).lower().replace("_", " ")

    def _validate_type(self, obj, expected_type):
        if not isinstance(obj, expected_type):
            raise ValueError(
                "Invalid input type: '%s'. Expected: '%s'"
                % (etau.get_class_name(obj),
                   etau.get_class_name(expected_type))
            )


class VideoLabelsSyntaxCheckerError(ImageLabelsSyntaxCheckerError):
    '''Error raised when a VideoLabelsSyntaxCheckerError is violated.'''
    pass


class VideoLabelsSyntaxChecker(ImageLabelsSyntaxChecker):

    _SCHEMA_CLS = etav.VideoLabelsSchema
    _LABELS_CLS = etav.VideoLabels
    _ERROR_CLS = VideoLabelsSyntaxCheckerError

    def _check(self, labels):
        '''Override of etai.ImageLabelsSyntaxChecker._check'''
        self._check_video_attrs(labels)
        self._check_frames(labels)
        self._check_events(labels)

    def _check_video_attrs(self, labels):
        def valid_in_schema(schema, thing):
            return schema.is_valid_video_attribute(thing)

        def add_to_schema(schema, thing):
            schema.add_video_attribute(thing)

        def get_target_iterable():
            return self.target_schema.attrs

        def assign_mapped_value(thing, mapped_thing):
            thing.name = mapped_thing.name
            thing.value = mapped_thing.value

        for attr in labels.attrs:
            self._check_thing(attr, valid_in_schema, add_to_schema,
                              get_target_iterable, assign_mapped_value)

    def _check_frames(self, labels):
        for frame in labels.iter_frames():
            self._check_frame_attrs(frame)
            self._check_objects(frame.objects)

    def _check_frame_attrs(self, frame):
        def valid_in_schema(schema, thing):
            return schema.is_valid_frame_attribute(thing)

        def add_to_schema(schema, thing):
            schema.add_frame_attribute(thing)

        def get_target_iterable():
            return self.target_schema.frames

        def assign_mapped_value(thing, mapped_thing):
            thing.name = mapped_thing.name
            thing.value = mapped_thing.value

        for attr in frame.attrs:
            self._check_thing(attr, valid_in_schema, add_to_schema,
                              get_target_iterable, assign_mapped_value)

    def _check_events(self, labels):
        for event in labels.iter_events():
            self._check_event_label(event)
            self._check_event_attrs(event)

    def _check_event_label(self, event):
        def valid_in_schema(schema, thing):
            return schema.is_valid_event_label(thing)

        def add_to_schema(schema, thing):
            schema.add_event_label(thing)

        def get_target_iterable():
            return self.target_schema.events.keys()

        def assign_mapped_value(thing, mapped_thing):
            event.label = mapped_thing

        self._check_thing(event.label, valid_in_schema, add_to_schema,
                          get_target_iterable, assign_mapped_value)

    def _check_event_attrs(self, event):
        if not self.target_schema.is_valid_event_label(event.label):
            return

        def valid_in_schema(schema, thing):
            return schema.is_valid_event_attribute(event.label, thing)

        def add_to_schema(schema, thing):
            schema.add_event_attribute(event.label, thing)

        def get_target_iterable():
            return self.target_schema.events[event.label]

        def assign_mapped_value(thing, mapped_thing):
            thing.name = mapped_thing.name
            thing.value = mapped_thing.value

        for attr in event.attrs:
            self._check_thing(attr, valid_in_schema, add_to_schema,
                              get_target_iterable, assign_mapped_value)