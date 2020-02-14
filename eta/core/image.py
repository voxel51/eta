'''
Core tools and data structures for working with images.

Notes:
    [image format] ETA stores images exclusively in RGB format. In contrast,
        OpenCV stores its images in BGR format, so all images that are read or
        produced outside of this library must be converted to RGB. This
        conversion can be done via `eta.core.image.bgr_to_rgb()`

Copyright 2017-2019, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
Jason Corso, jason@voxel51.com
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
from future.utils import iteritems
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict
import colorsys
import errno
import logging
import os
import operator
from subprocess import Popen, PIPE

import cv2
import numpy as np

import eta
from eta.core.data import AttributeContainer, AttributeContainerSchema, \
    AttributeContainerSchemaError
import eta.core.data as etad
from eta.core.objects import DetectedObjectContainer
from eta.core.serial import Serializable, Set, BigSet
from eta.core.utils import MATCH_ANY
import eta.core.utils as etau
import eta.core.web as etaw


logger = logging.getLogger(__name__)


#
# The file extensions of supported image files. Use LOWERCASE!
#
# In practice, any image that `cv2.imread` can read will be supported.
# Nonetheless, we enumerate this list here so that the ETA type system can
# verify the extension of an image provided to a pipeline at build time.
#
# This list was taken from
# https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#imread
#
SUPPORTED_IMAGE_FORMATS = {
    ".bmp", ".dib", ".jp2", ".jpe", ".jpeg", ".jpg", ".pbm", ".pgm", ".png",
    ".ppm", ".ras", ".sr", ".tif", ".tiff"
}


def is_supported_image(filepath):
    '''Determines whether the given file has a supported image type.'''
    return os.path.splitext(filepath)[1].lower() in SUPPORTED_IMAGE_FORMATS


def glob_images(dir_):
    '''Returns an iterator over all supported image files in the directory.'''
    return etau.multiglob(
        *SUPPORTED_IMAGE_FORMATS, root=os.path.join(dir_, "*"))


def make_image_sequence_patt(basedir, basename="", patt=None, ext=None):
    '''Makes an image sequence pattern of the following form:

    <basedir>/<basename>-<patt><ext>

    where the "-" is omitted

    Args:
        basedir: the base directory
        basename: an optional base filename. If omitted, the hyphen is also
            omitted
        patt: an optional image pattern to use. If omitted, the default pattern
            `eta.config.default_sequence_idx` is used
        ext: an optional image extension to use. If omitted, the default image
            extension `eta.config.default_image_ext`

    Returns:
        the image sequence pattern
    '''
    name = basename + "-" if basename else ""
    patt = patt or eta.config.default_sequence_idx
    ext = ext or eta.config.default_image_ext
    return os.path.join(basedir, name + patt + ext)


###### Image Labels ###########################################################


class ImageLabels(Serializable):
    '''Class encapsulating labels for an image.

    Attributes:
        filename: the filename of the image
        metadata: an ImageMetadata describing metadata about the image
        attrs: an AttributeContainer describing the attributes of the image
        objects: a DetectedObjectContainer describing the detected objects in
            the image
    '''

    def __init__(self, filename=None, metadata=None, attrs=None, objects=None):
        '''Constructs an ImageLabels instance.

        Args:
            filename: an optional filename of the image
            metadata: an optional ImageMetadata instance describing metadata
                about the image. By default, no metadata is stored
            attrs: an optional AttributeContainer of attributes for the image.
                By default, an empty AttributeContainer is created
            objects: an optional DetectedObjectContainer of detected objects
                for the image. By default, an empty DetectedObjectContainer is
                created
        '''
        self.filename = filename
        self.metadata = metadata
        self.attrs = attrs or AttributeContainer()
        self.objects = objects or DetectedObjectContainer()

    def iter_image_attrs(self, attr_type=MATCH_ANY, attr_name=MATCH_ANY,
                         attr_value=MATCH_ANY):
        iterator = self.attrs.iter_attrs(
            attr_type=attr_type, attr_name=attr_name, attr_value=attr_value)
        for attr in iterator:
            yield attr

    def iter_objects(self, label=MATCH_ANY):
        for obj in self.objects.iter_objects(label=label):
            yield obj

    def iter_object_attrs(self, label=MATCH_ANY, attr_type=MATCH_ANY,
                          attr_name=MATCH_ANY, attr_value=MATCH_ANY):
        iterator = self.objects.iter_object_attrs(
            label=label,
            attr_type=attr_type,
            attr_name=attr_name,
            attr_value=attr_value
        )
        for obj, attr in iterator:
            yield obj, attr

    def add_image_attribute(self, attr):
        '''Adds the attribute to the image.

        Args:
            attr: an Attribute
        '''
        self.attrs.add(attr)

    def add_image_attributes(self, attrs):
        '''Adds the attributes to the image.

        Args:
            attrs: an AttributeContainer
        '''
        self.attrs.add_container(attrs)

    def add_object(self, obj):
        '''Adds the object to the image.

        Args:
            obj: a DetectedObject
        '''
        self.objects.add(obj)

    def add_objects(self, objs):
        '''Adds the objects to the image.

        Args:
            objs: a DetectedObjectContainer
        '''
        self.objects.add_container(objs)

    def clear_frame_attributes(self):
        '''Removes all frame attributes from the instance.'''
        self.attrs = AttributeContainer()

    def clear_objects(self):
        '''Removes all objects from the instance.'''
        self.objects = DetectedObjectContainer()

    def merge_labels(self, image_labels):
        '''Merges the ImageLabels into this object.

        Args:
            image_labels: an ImageLabels instance
        '''
        self.add_image_attributes(image_labels.attrs)
        self.add_objects(image_labels.objects)

    def filter_by_schema(self, schema):
        '''Removes objects/attributes from this object that are not compliant
        with the given schema.

        Args:
            schema: an ImageLabelsSchema
        '''
        self.attrs.filter_by_schema(schema.attrs)
        self.objects.filter_by_schema(schema)

    def remove_objects_without_attrs(self, labels=None):
        '''Removes DetectedObjects from this instance that do not have
        attributes.

        Args:
            labels: an optional list of DetectedObject label strings to which
                to restrict attention when filtering. By default, all objects
                are processed
        '''
        self.objects.remove_objects_without_attrs(labels=labels)

    def check_for_duplicate_attrs(self, image_attr_multi_value_names=None,
                                  obj_attr_multi_value_names=None):
        '''Check for duplicate attributes (and raise exception)

        Args:
            image_attr_multi_value_names: list of attr name strings that the
                image attrs is allowed to have multiple DIFFERENT values for
            obj_attr_multi_value_names: list of attr name strings that any
                object attrs is allowed to have multiple DIFFERENT values for

        Raises:
            ValueError if:
                - multiple types for an attr name
                - multiple values for an attr name *not in multi_value_names*
                - duplicate values for an attr name
        '''
        self.attrs.check_for_duplicates(image_attr_multi_value_names)
        for obj in self.objects:
            obj.attrs.check_for_duplicates(obj_attr_multi_value_names)

    def has_duplicate_attrs(self, image_attr_multi_value_names=None,
                            obj_attr_multi_value_names=None):
        '''Check for duplicate attributes (and return boolean)

        Args:
            image_attr_multi_value_names: list of attr name strings that the
                image attrs is allowed to have multiple DIFFERENT values for
            obj_attr_multi_value_names: list of attr name strings that any
                object attrs is allowed to have multiple DIFFERENT values for

        Returns:
            True if any duplicate attributes exist (image or object)
        '''
        try:
            self.check_for_duplicate_attrs(
                image_attr_multi_value_names, obj_attr_multi_value_names)
            return False
        except ValueError:
            return True

    @property
    def has_attributes(self):
        '''Whether the container has at least one attribute.'''
        return bool(self.attrs)

    @property
    def has_objects(self):
        '''Whether the container has at least one object.'''
        return bool(self.objects)

    @property
    def is_empty(self):
        '''Whether the container has no labels of any kind.'''
        return not self.has_attributes and not self.has_objects

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        _attrs = []
        if self.filename:
            _attrs.append("filename")
        if self.metadata:
            _attrs.append("metadata")
        if self.attrs:
            _attrs.append("attrs")
        if self.objects:
            _attrs.append("objects")
        return _attrs

    @classmethod
    def from_video_frame_labels(
            cls, video_frame_labels, filename=None, metadata=None):
        '''Constructs an ImageLabels from a VideoFrameLabels.

        Args:
            video_frame_labels: a VideoFrameLabels instance
            filename: an optional filename for the image
            metadata: an optional ImageMetadata instance for the image

        Returns:
            an ImageLabels instance
        '''
        return cls(
            filename=filename, metadata=metadata,
            attrs=video_frame_labels.attrs, objects=video_frame_labels.objects)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ImageLabels from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an ImageLabels
        '''
        filename = d.get("filename", None)

        metadata = d.get("metadata", None)
        if metadata is not None:
            metadata = ImageMetadata.from_dict(metadata)

        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = AttributeContainer.from_dict(attrs)

        objects = d.get("objects", None)
        if objects is not None:
            objects = DetectedObjectContainer.from_dict(objects)

        return cls(
            filename=filename, metadata=metadata, attrs=attrs, objects=objects)


class ImageLabelsSchema(Serializable):
    '''A schema for ImageLabels instance(s).

    Attributes:
        attrs: an AttributeContainerSchema describing the attributes of the
            image(s)
        objects: a dictionary mapping object labels to AttributeContainerSchema
            instances describing the object attributes of each object class
    '''

    def __init__(self, attrs=None, objects=None):
        '''Creates an ImageLabelsSchema instance.

        Args:
            attrs: an AttributeContainerSchema describing the attributes of the
                image(s)
            objects: a dictionary mapping object labels to
                AttributeContainerSchema instances describing the object
                attributes of each object class
        '''
        self.attrs = attrs or AttributeContainerSchema()
        self.objects = defaultdict(AttributeContainerSchema)
        if objects is not None:
            self.objects.update(objects)

    def iter_attr_containers(self, labels):
        if not isinstance(labels, ImageLabels):
            raise ValueError("Unexpected input type %s" % type(labels))

        # image attrs
        yield self.attrs, labels.attrs

        # detected object attrs
        for obj in labels.objects:
            yield self.objects[obj.label], obj.attrs

    # HAS

    def has_image_attribute(self, image_attr_name):
        '''Whether the schema has an image attribute with the given name.

        Args:
            image_attr_name: an image attribute name

        Returns:
            True/False
        '''
        return self.attrs.has_attribute(image_attr_name)

    def has_object_label(self, label):
        '''Whether the schema has an object with the given label.

        Args:
            label: an object label

        Returns:
            True/False
        '''
        return label in self.objects

    def has_object_attribute(self, label, obj_attr_name):
        '''Whether the schema has an object with the given label with an
        attribute with the given name.

        Args:
            label: an object label
            obj_attr_name: an object attribute name

        Returns:
            True/False
        '''
        if not self.has_object_label(label):
            return False
        return self.objects[label].has_attribute(obj_attr_name)

    # GET ATTR CLASS

    def get_image_attribute_class(self, image_attr_name):
        '''Gets the Attribute class for the image attribute with the given
        name.

        Args:
            image_attr_name: an image attribute name

        Returns:
            an Attribute
        '''
        return self.attrs.get_attribute_class(image_attr_name)

    def get_object_attribute_class(self, label, obj_attr_name):
        '''Gets the Attribute class for the attribute of the given name for the
        object with the given label.

        Args:
            label: an object label
            obj_attr_name: an object attribute name

        Returns:
            an Attribute subclass
        '''
        self.validate_object_label(label)
        return self.objects[label].get_attribute_class(obj_attr_name)

    # ADD

    def add_image_attribute(self, image_attr):
        '''Adds the given image attribute to the schema.

        Args:
            image_attr: an Attribute
        '''
        self.attrs.add_attribute(image_attr)

    def add_image_attributes(self, image_attrs):
        '''Adds the given image attributes to the schema.

        Args:
            image_attrs: an AttributeContainer
        '''
        self.attrs.add_attributes(image_attrs)

    def add_object_label(self, label):
        '''Adds the given object label to the schema.

        Args:
            label: an object label
        '''
        self.objects[label]  # adds key to defaultdict

    def add_object_attribute(self, label, obj_attr):
        '''Adds the Attribute for the object with the given label to the
        schema.

        Args:
            label: an object label
            obj_attr: an Attribute
        '''
        self.objects[label].add_attribute(obj_attr)

    def add_object_attributes(self, label, obj_attrs):
        '''Adds the AttributeContainer for the object with the given label to
        the schema.

        Args:
            label: an object label
            obj_attrs: an AttributeContainer
        '''
        self.objects[label].add_attributes(obj_attrs)

    # CHECK & COUNT VALID

    def is_valid_labels(self, image_labels):
        '''Whether the given ImageLabels is compliant with the schema.

        Args:
            image_labels: an ImageLabels

        Returns:
            True/False
        '''
        try:
            self.validate_labels(image_labels)
            return True
        except (ImageLabelsSchemaError, AttributeContainerSchemaError):
            return False

    def count_invalid_labels(self, image_labels):
        '''Count the number of "things" in an ImageLabels not conforming to the
        schema.

        Args:
            image_labels: an ImageLabels

        Returns:
            a dictionary of the format:
                {
                    "image attrs": <# of invalid image attrs>
                    "objects":     <# of invalid objects>
                }
        '''
        return self._count_invalid_labels(image_labels, raise_error=False)

    def is_valid_image_attribute(self, image_attr):
        '''Whether the image attribute is compliant with the schema.

        Args:
            image_attr: an Attribute

        Returns:
            True/False
        '''
        try:
            self.validate_image_attribute(image_attr)
            return True
        except AttributeContainerSchemaError:
            return False

    def is_valid_object_label(self, label):
        '''Whether the object label is compliant with the schema.

        Args:
            label: an object label

        Returns:
            True/False
        '''
        try:
            self.validate_object_label(label)
            return True
        except ImageLabelsSchemaError:
            return False

    def is_valid_object_attribute(self, label, obj_attr):
        '''Whether the object attribute for the object with the given label is
        compliant with the schema.

        Args:
            label: an object label
            obj_attr: an Attribute

        Returns:
            True/False
        '''
        try:
            self.validate_object_attribute(label, obj_attr)
            return True
        except AttributeContainerSchemaError:
            return False

    def is_valid_object(self, obj):
        '''Whether the given DetectedObject is compliant with the schema.

        Args:
            obj: a DetectedObject

        Returns:
            True/False
        '''
        try:
            self.validate_object(obj)
            return True
        except (ImageLabelsSchemaError, AttributeContainerSchemaError):
            return False

    # VALIDATE

    def validate_labels(self, image_labels):
        '''Validates that the ImageLabels is compliant with the schema.

        Args:
            image_labels: an ImageLabels

        Raises:
            ImageLabelsSchemaError: if ImageLabels violates the schema
        '''
        self._count_invalid_labels(image_labels, raise_error=True)

    def validate_image_attribute(self, image_attr):
        '''Validates that the image attribute is compliant with the schema.

        Args:
            image_attr: an Attribute

        Raises:
            AttributeContainerSchemaError: if the attribute violates the schema
        '''
        self.attrs.validate_attribute(image_attr)

    def validate_object_label(self, label):
        '''Validates that the object label is compliant with the schema.

        Args:
            label: an object label

        Raises:
            ImageLabelsSchemaError: if the object label violates the schema
        '''
        if label not in self.objects:
            raise ImageLabelsSchemaError(
                "Object label '%s' is not allowed by the schema" % label)

    def validate_object_attribute(self, label, obj_attr):
        '''Validates that the object attribute for the given label is compliant
        with the schema.

        Args:
            label: an object label
            obj_attr: an Attribute

        Raises:
            AttributeContainerSchemaError: if the object attribute violates
                the schema
        '''
        obj_schema = self.objects[label]
        obj_schema.validate_attribute(obj_attr)

    def validate_object(self, obj):
        '''Validates that the detected object is compliant with the schema.

        Args:
            obj: a DetectedObject

        Raises:
            ImageLabelsSchemaError: if the object's label violates the schema
            AttributeContainerSchemaError: if any attributes of the
                DetectedObject violate the schema
        '''
        self.validate_object_label(obj.label)
        if obj.has_attributes:
            for obj_attr in obj.attrs:
                self.validate_object_attribute(obj.label, obj_attr)

    # OTHER

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        return ["attrs", "objects"]

    def merge_schema(self, schema):
        '''Merges the given ImageLabelsSchema into this schema.

        Args:
            schema: an ImageLabelsSchema
        '''
        self.attrs.merge_schema(schema.attrs)
        for k, v in iteritems(schema.objects):
            self.objects[k].merge_schema(v)

    @classmethod
    def build_active_schema(cls, image_labels):
        '''Builds an ImageLabelsSchema that describes the active schema of
        the given ImageLabels.

        Args:
            image_labels: an ImageLabels

        Returns:
            an ImageLabelsSchema
        '''
        schema = cls()
        schema.add_image_attributes(image_labels.attrs)
        for obj in image_labels.objects:
            if obj.has_attributes:
                schema.add_object_attributes(obj.label, obj.attrs)
            else:
                schema.add_object_label(obj.label)
        return schema

    @classmethod
    def from_video_labels_schema(cls, video_labels_schema):
        '''Create ImageLabelsSchema from VideoLabelsSchema, using frame attrs
        as image attrs

        Args:
            video_labels_schema: a VideoLabelsSchema instance

        Returns:
            an ImageLabelsSchema instance
        '''
        return cls(attrs=video_labels_schema.frames,
                   objects=video_labels_schema.objects)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ImageLabelsSchema from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an ImageLabelsSchema
        '''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = AttributeContainerSchema.from_dict(attrs)

        objects = d.get("objects", None)
        if objects is not None:
            objects = {
                k: AttributeContainerSchema.from_dict(v)
                for k, v in iteritems(objects)
            }

        return cls(attrs=attrs, objects=objects)

    # PRIVATE

    def _count_invalid_labels(self, labels, raise_error=False):
        if not isinstance(labels, ImageLabels):
            raise ValueError("Unexpected input type '%s' expected '%s'"
                             % (etau.get_class_name(labels),
                                etau.get_class_name(ImageLabels)))

        # intentionally not using defaultdict so that 0 counts are still here
        invalid_counts = {
            "image attrs": 0,
            "objects": 0
        }

        #
        # image attrs
        #

        for attr in labels.attrs:
            is_valid = self.is_valid_image_attribute(attr)
            if raise_error and not is_valid:
                raise ImageLabelsSchemaError(
                    "Invalid image attr:\n%s" % attr.to_str())
            invalid_counts["image attrs"] += int(not is_valid)

        #
        # objects
        #

        for obj in labels.objects:
            is_valid = self.is_valid_object(obj)
            if raise_error and not is_valid:
                raise ImageLabelsSchemaError(
                    "Invalid object:\n%s" % obj.to_str())
            invalid_counts["objects"] += int(not is_valid)

        return invalid_counts


class ImageLabelsSchemaError(Exception):
    '''Error raised when an ImageLabelsSchema is violated.'''
    pass


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

    _SCHEMA_CLS = ImageLabelsSchema
    _LABELS_CLS = ImageLabels
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


class ImageSetLabels(Set):
    '''Class encapsulating labels for a set of images.

    ImageSetLabels support item indexing by the `filename` of the ImageLabels
    instances in the set.

    ImageSetLabels instances behave like defaultdicts: new ImageLabels
    instances are automatically created if a non-existent filename is accessed.

    ImageLabels without filenames may be added to the set, but they cannot be
    accessed by `filename`-based lookup.

    Attributes:
        images: an OrderedDict of ImageLabels with filenames as keys
        schema: an ImageLabelsSchema describing the schema of the labels
    '''

    _ELE_ATTR = "images"
    _ELE_KEY_ATTR = "filename"
    _ELE_CLS = ImageLabels
    _ELE_CLS_FIELD = "_LABELS_CLS"

    def __init__(self, images=None, schema=None):
        '''Constructs an ImageSetLabels instance.

        Args:
            images: an optional iterable of ImageLabels. By default, an empty
                set is created
            schema: an optional ImageLabelsSchema to enforce on the object.
                By default, no schema is enforced
        '''
        self.schema = schema
        super(ImageSetLabels, self).__init__(images=images)

    def __getitem__(self, filename):
        '''Gets the ImageLabels for the given filename.

        If the filename is not found, an empty ImageLabels is created for it,
        and returned.

        Args:
            filename: the image name

        Returns:
            an ImageLabels
        '''
        if filename not in self:
            image_labels = ImageLabels(filename=filename)
            self.add(image_labels)

        return super(ImageSetLabels, self).__getitem__(filename)

    def __setitem__(self, filename, image_labels):
        '''Sets the labels for the image with the given filename.

        Any existing labels are overwritten.

        Args:
            filename: the image name
            image_labels: an ImageLabels
        '''
        if self.has_schema:
            self._validate_labels(image_labels)

        return super(ImageSetLabels, self).__setitem__(filename, image_labels)

    @property
    def has_schema(self):
        '''Whether this instance has an enforced schema.'''
        return self.schema is not None

    def empty(self):
        '''Returns an empty copy of the ImageSetLabels.

        The schema of the set is preserved, if applicable.

        Returns:
            an empty ImageSetLabels
        '''
        return self.__class__(schema=self.schema)

    def add(self, image_labels):
        '''Adds the ImageLabels to the set.

        Args:
            image_labels: an ImageLabels instance
        '''
        if self.has_schema:
            self._validate_labels(image_labels)

        super(ImageSetLabels, self).add(image_labels)

    def clear_frame_attributes(self):
        '''Removes all frame attributes from all ImageLabels in the set.'''
        for image_labels in self:
            image_labels.clear_frame_attributes()

    def clear_objects(self):
        '''Removes all objects from all ImageLabels in the set.'''
        for image_labels in self:
            image_labels.clear_objects()

    def get_filenames(self):
        '''Returns the set of filenames of ImageLabels in the set.

        Returns:
            the set of filenames
        '''
        return set(il.filename for il in self if il.filename)

    def get_schema(self):
        '''Gets the schema for the set, or None if no schema is enforced.

        Returns:
            an ImageLabelsSchema, or None
        '''
        return self.schema

    def get_active_schema(self):
        '''Gets the ImageLabelsSchema describing the active schema of the set.

        Returns:
            an ImageLabelsSchema
        '''
        schema = ImageLabelsSchema()
        for image_labels in self:
            schema.merge_schema(
                ImageLabelsSchema.build_active_schema(image_labels))
        return schema

    def set_schema(self, schema, filter_by_schema=False):
        '''Sets the schema to the given ImageLabelsSchema.

        Args:
            schema: the ImageLabelsSchema to use
            filter_by_schema: whether to filter any invalid objects/attributes
                from the set after changing the schema. By default, this is
                False

        Raises:
            ImageLabelsSchemaError: if `filter_by_schema` was False and the
                set contains attributes/objects that are not compliant with the
                schema
        '''
        self.schema = schema
        if not self.has_schema:
            return

        if filter_by_schema:
            self.filter_by_schema(self.schema)
        else:
            self._validate_schema()

    def filter_by_schema(self, schema):
        '''Removes objects/attributes from the ImageLabels in the set that are
        not compliant with the given schema.

        Args:
            schema: an ImageLabelsSchema
        '''
        for image_labels in self:
            image_labels.filter_by_schema(schema)

    def remove_objects_without_attrs(self, labels=None):
        '''Removes DetectedObjects from the ImageLabels in the set that do not
        have attributes.

        Args:
            labels: an optional list of DetectedObject label strings to which
                to restrict attention when filtering. By default, all objects
                are processed
        '''
        for image_labels in self:
            image_labels.remove_objects_without_attrs(labels=labels)

    def freeze_schema(self):
        '''Sets the schema for the set to the current active schema.'''
        self.set_schema(self.get_active_schema())

    def remove_schema(self):
        '''Removes the schema from the set.'''
        self.schema = None

    def sort_by_filename(self, reverse=False):
        '''Sorts the ImageLabels in this instance by filename.

        ImageLabels without filenames are always put at the end of the set.

        Args:
            reverse: whether to sort in reverse order. By default, this is
                False
        '''
        self.sort_by("filename", reverse=reverse)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        _attrs = super(ImageSetLabels, self).attributes()
        if self.has_schema:
            return ["schema"] + _attrs
        return _attrs

    def _validate_labels(self, image_labels):
        if self.has_schema:
            for image_attr in image_labels.attrs:
                self._validate_image_attribute(image_attr)
            for obj in image_labels.objects:
                self._validate_object(obj)

    def _validate_image_attribute(self, image_attr):
        if self.has_schema:
            self.schema.validate_image_attribute(image_attr)

    def _validate_object(self, obj):
        if self.has_schema:
            self.schema.validate_object(obj)

    def _validate_schema(self):
        if self.has_schema:
            for image_labels in self:
                self._validate_labels(image_labels)

    @classmethod
    def from_image_labels_patt(cls, image_labels_patt):
        '''Creates an instance of `cls` from a pattern of `_ELE_CLS` files.

        Args:
             image_labels_patt: a pattern with one or more numeric sequences:
                example: "/path/to/labels/%05d.json"

        Returns:
            a `cls` instance
        '''
        logger.warning("Using deprecated method `from_image_labels_patt`. Use"
                       " `from_numeric_patt` instead.")
        return cls.from_numeric_patt(image_labels_patt)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ImageSetLabels from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an ImageSetLabels
        '''
        schema = d.get("schema", None)
        if schema is not None:
            schema = ImageLabelsSchema.from_dict(schema)

        return super(ImageSetLabels, cls).from_dict(d, schema=schema)


class BigImageSetLabels(ImageSetLabels, BigSet):
    '''A BigSet of ImageLabels.

    Behaves identically to ImageSetLabels except that each ImageLabels is
    stored on disk.

    BigImageSetLabels store a `backing_dir` attribute that specifies the path
    on disk to the serialized elements. If a backing directory is explicitly
    provided, the directory will be maintained after the BigImageSetLabels
    object is deleted; if no backing directory is specified, a temporary
    backing directory is used and is deleted when the BigImageSetLabels
    instance is garbage collected.

    Attributes:
        images: an OrderedDict whose keys are filenames and whose values are
            uuids for locating ImageLabels on disk
        schema: an ImageLabelsSchema describing the schema of the labels
        backing_dir: the backing directory in which the ImageLabels
            are/will be stored
    '''

    def __init__(self, images=None, schema=None, backing_dir=None):
        '''Creates a BigImageSetLabels instance.

        Args:
            images: an optional dictionary or list of (key, uuid) tuples for
                elements in the set
            schema: an optional ImageLabelsSchema to enforce on the object.
                By default, no schema is enforced
            backing_dir: an optional backing directory in which the ImageLabels
                are/will be stored. If omitted, a temporary backing directory
                is used
        '''
        self.schema = schema
        BigSet.__init__(self, backing_dir=backing_dir, images=images)

    def empty_set(self):
        '''Returns an empty in-memory ImageSetLabels version of this
        BigImageSetLabels.

        Returns:
            an empty ImageSetLabels
        '''
        return ImageSetLabels(schema=self.schema)

    def filter_by_schema(self, schema):
        '''Removes objects/attributes from the ImageLabels in the set that are
        not compliant with the given schema.

        Args:
            schema: an ImageLabelsSchema
        '''
        for key in self.keys():
            image_labels = self[key]
            image_labels.filter_by_schema(schema)
            self[key] = image_labels

    def remove_objects_without_attrs(self, labels=None):
        '''Removes DetectedObjects from the ImageLabels in the set that do not
        have attributes.

        Args:
            labels: an optional list of DetectedObject label strings to which
                to restrict attention when filtering. By default, all objects
                are processed
        '''
        for key in self.keys():
            image_labels = self[key]
            image_labels.remove_objects_without_attrs(labels=labels)
            self[key] = image_labels


###### Image I/O ##############################################################


def decode(b, include_alpha=False, flag=None):
    '''Decodes an image from raw bytes.

    By default, images are returned as color images with no alpha channel.

    Args:
        bytes: the raw bytes of an image, e.g., from read() or from a web
            download
        include_alpha: whether to include the alpha channel of the image, if
            present, in the returned array. By default, this is False
        flag: an optional OpenCV image format flag to use. If provided, this
            flag takes precedence over `include_alpha`

    Returns:
        a uint8 numpy array containing the image
    '''
    flag = _get_opencv_imread_flag(flag, include_alpha)
    vec = np.asarray(bytearray(b), dtype=np.uint8)
    return _exchange_rb(cv2.imdecode(vec, flag))


def download(url, include_alpha=False, flag=None):
    '''Downloads an image from a URL.

    By default, images are returned as color images with no alpha channel.

    Args:
        url: the URL of the image
        include_alpha: whether to include the alpha channel of the image, if
            present, in the returned array. By default, this is False
        flag: an optional OpenCV image format flag to use. If provided, this
            flag takes precedence over `include_alpha`

    Returns:
        a uint8 numpy array containing the image
    '''
    bytes = etaw.download_file(url)
    return decode(bytes, include_alpha=include_alpha, flag=flag)


def read(path, include_alpha=False, flag=None):
    '''Reads image from path.

    By default, images are returned as color images with no alpha channel.

    Args:
        path: the path to the image on disk
        include_alpha: whether to include the alpha channel of the image, if
            present, in the returned array. By default, this is False
        flag: an optional OpenCV image format flag to use. If provided, this
            flag takes precedence over `include_alpha`

    Returns:
        a uint8 numpy array containing the image
    '''
    flag = _get_opencv_imread_flag(flag, include_alpha)
    img_bgr = cv2.imread(path, flag)
    if img_bgr is None:
        raise OSError("Image not found '%s'" % path)
    return _exchange_rb(img_bgr)


def write(img, path):
    '''Writes image to file. The output directory is created if necessary.

    Args:
        img: a numpy array
        path: the output path
    '''
    etau.ensure_basedir(path)
    cv2.imwrite(path, _exchange_rb(img))


def _get_opencv_imread_flag(flag, include_alpha):
    if flag is not None:
        return flag
    if include_alpha:
        return cv2.IMREAD_UNCHANGED
    return cv2.IMREAD_COLOR


class ImageMetadata(Serializable):
    '''Class encapsulating metadata about an image.

    Attributes:
        frame_size: the [width, height] of the image
        num_channels: the number of channels in the image
        size_bytes: the size of the image file on disk, in bytes
        mime_type: the MIME type of the image
    '''

    def __init__(
            self, frame_size=None, num_channels=None, size_bytes=None,
            mime_type=None):
        '''Constructs an ImageMetadata instance. All args are optional.

        Args:
            frame_size: the [width, height] of the image
            num_channels: the number of channels in the image
            size_bytes: the size of the image file on disk, in bytes
            mime_type: the MIME type of the image
        '''
        self.frame_size = frame_size
        self.num_channels = num_channels
        self.size_bytes = size_bytes
        self.mime_type = mime_type

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.'''
        _attrs = ["frame_size", "num_channels", "size_bytes", "mime_type"]
        # Exclude attributes that are None
        return [a for a in _attrs if getattr(self, a) is not None]

    @classmethod
    def build_for(cls, filepath):
        '''Builds an ImageMetadata object for the given image.

        Args:
            filepath: the path to the image on disk

        Returns:
            an ImageMetadata instance
        '''
        img = read(filepath, include_alpha=True)
        return cls(
            frame_size=to_frame_size(img=img),
            num_channels=img.shape[2] if len(img.shape) > 2 else 1,
            size_bytes=os.path.getsize(filepath),
            mime_type=etau.guess_mime_type(filepath),
        )

    @classmethod
    def from_dict(cls, d):
        '''Constructs an ImageMetadata from a JSON dictionary.'''
        return cls(
            frame_size=d.get("frame_size", None),
            num_channels=d.get("num_channels", None),
            size_bytes=d.get("size_bytes", None),
            mime_type=d.get("mime_type", None))


###### Image Manipulation #####################################################


def create(width, height, background=None):
    '''Creates a blank image and optionally fills it with a color.

    Args:
        width: the width of the image, in pixels
        height: the height of the image, in pixels
        background: hex RGB (e.g., "#ffffff")

    Returns:
        the image
    '''
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if background:
        img[:] = hex_to_rgb(background)

    return img


def overlay(im1, im2, x0=0, y0=0):
    '''Overlays im2 onto im1 at the specified coordinates.

    Args:
        im1: a non-transparent image
        im2: a possibly-transparent image
        (x0, y0): the top-left coordinate of im2 in im1 after overlaying, where
            (0, 0) corresponds to the top-left of im1. This coordinate may lie
            outside of im1, in which case some (even all) of im2 may be omitted

    Returns:
        a copy of im1 with im2 overlaid
    '''
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    # Active slice of im1
    y1t = np.clip(y0, 0, h1)
    y1b = np.clip(y0 + h2, 0, h1)
    x1l = np.clip(x0, 0, w1)
    x1r = np.clip(x0 + w2, 0, w1)
    y1 = slice(y1t, y1b)
    x1 = slice(x1l, x1r)

    # Active slice of im2
    y2t = np.clip(y1t - y0, 0, h2)
    y2b = y2t + y1b - y1t
    x2l = np.clip(x1l - x0, 0, w2)
    x2r = x2l + x1r - x1l
    y2 = slice(y2t, y2b)
    x2 = slice(x2l, x2r)

    if im2.shape[2] == 4:
        # Mix transparent image
        im1 = to_double(im1)
        im2 = to_double(im2)
        alpha = im2[y2, x2, 3][:, :, np.newaxis]
        im1[y1, x1, :] *= (1 - alpha)
        im1[y1, x1, :] += alpha * im2[y2, x2, :3]
        im1 = np.uint8(255 * im1)
    else:
        # Insert opaque image
        im1 = np.copy(im1)
        im1[y1, x1, :] = im2[y2, x2, :]

    return im1


def rasterize(vector_path, width, include_alpha=True, flag=None):
    '''Renders a vector image as a raster image with the given pixel width.

    By default, the image is returned with an alpha channel, if possible.

    Args:
        vector_path: the path to the vector image
        width: the desired image width
        include_alpha: whether to include the alpha channel of the image, if
            present, in the returned array. By default, this is True
        flag: an optional OpenCV image format flag to use. If provided, this
            flag takes precedence over `include_alpha`

    Returns:
        a uint8 numpy array containing the rasterized image
    '''
    with etau.TempDir() as d:
        try:
            png_path = os.path.join(d, "tmp.png")
            Convert(
                in_opts=["-density", "1200", "-trim"],
                out_opts=["-resize", str(width)],
            ).run(vector_path, png_path)
            return read(png_path, include_alpha=include_alpha, flag=flag)
        except Exception:
            # Fail gracefully
            return None

    # @todo why is it slightly blurry this way?
    # try:
    #     out = Convert(
    #         in_opts=["-density", "1200", "-trim"],
    #         out_opts=["-resize", str(width)],
    #     ).run(vector_path, "png:-")
    #     return read(out, include_alpha=include_alpha, flag=flag)
    # except Exception:
    #     # Fail gracefully
    #     return None


def resize(img, width=None, height=None, *args, **kwargs):
    '''Resizes the given image to the given width and height.

    At most one dimension can be None or negative, in which case the
    aspect-preserving value is used.

    Args:
        img: input image
        width: the desired image width
        height: the desired image height
        *args: valid positional arguments for `cv2.resize()`
        **kwargs: valid keyword arguments for `cv2.resize()`

    Returns:
        the resized image
    '''
    if height is None or height < 0:
        height = int(round(img.shape[0] * (width * 1.0 / img.shape[1])))
    if width is None or width < 0:
        width = int(round(img.shape[1] * (height * 1.0 / img.shape[0])))
    return cv2.resize(img, (width, height), *args, **kwargs)


def expand(img, width=None, height=None, *args, **kwargs):
    '''Resizes the given image, if necesary, so that its width and height are
    greater than or equal to the specified minimum values.

    The aspect ratio of the input image is preserved.

    Args:
        img: input image
        width: the minimum width
        height: the minimum height
        *args: valid positional arguments for `cv2.resize()`
        **kwargs: valid keyword arguments for `cv2.resize()`

    Returns:
        the expanded image
    '''
    iw, ih = to_frame_size(img=img)
    ow, oh = iw, ih
    if ow < width:
        oh = int(round(oh * (width / ow)))
        ow = width

    if oh < height:
        ow = int(round(ow * (height / oh)))
        oh = height

    if (ow > iw) or (oh > ih):
        img = resize(img, width=ow, height=oh)

    return img


def central_crop(img, frame_size=None, shape=None):
    '''Extracts a centered crop of the required size from the given image.

    The image is resized as necessary if the requested size is larger than the
    resolution of the input image.

    Pass *one* keyword argument to this function.

    Args:
        img: the input image
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape

    Returns:
        A cropped portion of the image of height `h` and width `w`.
    '''
    width, height = to_frame_size(frame_size=frame_size, shape=shape)

    # Expand image, if necessary
    img = expand(img, width=width, height=height)

    # Extract central crop
    bounding = (height, width)
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def render_instance_mask(
        obj, frame_size=None, shape=None, img=None, as_bool=True):
    '''Renders the instance mask for the DetectedObject for an image of the
    given dimensions.

    One of `frame_size`, `shape`, and `img` must be provided.

    Args:
        obj: a DetectedObject with an instance mask
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape
        img: the image itself
        as_bool: whether to return the mask as a boolean image (True) or a
            uint8 image (False). The default is True

    Returns:
        (obj_mask, offset), where `obj_mask` is a binary image describing
            the instance mask in its bounding box and `offset = (tlx, tly)`
            are the coordinates of the top-left corner of the mask within
            the image
    '''
    tlx, tly, width, height = obj.bounding_box.coords_in(
        frame_size=frame_size, shape=shape, img=img)
    offset = (tlx, tly)

    obj_mask = obj.mask.astype(np.uint8)
    # Can consider using `interpolation=cv2.INTER_NEAREST` here
    obj_mask = resize(obj_mask, width=width, height=height)

    if as_bool:
        obj_mask = obj_mask.astype(bool)

    return obj_mask, offset


def render_instance_image(obj, frame_size=None, shape=None, img=None):
    '''Renders a binary image of the specified size containing the instance
    mask for the given DetectedObject.

    One of `frame_size`, `shape`, and `img` must be provided.

    Args:
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape
        img: the image itself

    Returns:
        a binary instance mask of the specified size
    '''
    w, h = to_frame_size(frame_size=frame_size, shape=shape, img=img)
    obj_mask, offset = render_instance_mask(obj, frame_size=(w, h))
    x0, y0 = offset
    dh, dw = obj_mask.shape

    img_mask = np.zeros((h, w), dtype=bool)
    img_mask[y0:(y0 + dh), x0:(x0 + dw)] = obj_mask
    return img_mask


def to_double(img):
    '''Converts img to a double precision image with values in [0, 1].

    Args:
        img: input image

    Returns:
        a copy of the image in double precision format
    '''
    return img.astype(np.float) / np.iinfo(img.dtype).max


class Convert(object):
    '''Interface for the ImageMagick convert binary.'''

    def __init__(
            self,
            executable="convert",
            in_opts=None,
            out_opts=None,
        ):
        '''Constructs a convert command, minus the input/output paths.

        Args:
            executable: the system path to the convert binary
            in_opts: a list of input options for convert
            out_opts: a list of output options for convert
        '''
        self._executable = executable
        self._in_opts = in_opts or []
        self._out_opts = out_opts or []
        self._args = None
        self._p = None

    @property
    def cmd(self):
        '''The last executed convert command string, or None if run() has not
        yet been called.
        '''
        return " ".join(self._args) if self._args else None

    def run(self, inpath, outpath):
        '''Run the convert binary with the specified input/outpath paths.

        Args:
            inpath: the input path
            outpath: the output path. Use "-" or a format like "png:-" to pipe
                output to STDOUT

        Returns:
            out: STDOUT of the convert binary

        Raises:
            ExecutableNotFoundError: if the convert binary cannot be found
            ExecutableRuntimeError: if the convert binary raises an error
                during execution
        '''
        self._args = (
            [self._executable] +
            self._in_opts + [inpath] +
            self._out_opts + [outpath]
        )

        try:
            self._p = Popen(self._args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        except EnvironmentError as e:
            if e.errno == errno.ENOENT:
                raise etau.ExecutableNotFoundError(self._executable)
            raise

        out, err = self._p.communicate()
        if self._p.returncode != 0:
            raise etau.ExecutableRuntimeError(self.cmd, err)

        return out


###### Image Properties and Representations ###################################


def has_alpha(img):
    '''Checks if the image has an alpha channel.

    Args:
        img: an image

    Returns:
        True/False
    '''
    return img.ndim == 4


def is_gray(img):
    '''Checks if the image is grayscale, i.e., has exactly two channels.

    Args:
        img: an image

    Returns:
        True/False
    '''
    return img.ndim == 2


def is_color(img):
    '''Checks if the image is color, i.e., has at least three channels.

    Args:
        img: an image

    Returns:
        True/False
    '''
    return img.ndim > 2


def to_frame_size(frame_size=None, shape=None, img=None):
    '''Converts an image size representation to a (width, height) tuple.

    Pass *one* keyword argument to compute the frame size.

    Args:
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape
        img: the image itself

    Returns:
        a (width, height) frame size tuple

    Raises:
        TypeError: if none of the keyword arguments were passed
    '''
    if img is not None:
        shape = img.shape

    if shape is not None:
        return shape[1], shape[0]

    if frame_size is not None:
        return tuple(frame_size)

    raise TypeError("A valid keyword argument must be provided")


def aspect_ratio(**kwargs):
    '''Computes the aspect ratio of the image.

    Args:
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape
        img: the image itself

    Returns:
        the aspect ratio of the image
    '''
    fs = to_frame_size(**kwargs)
    return fs[0] / fs[1]


def parse_frame_size(frame_size):
    '''Parses the given frame size, ensuring that it is valid.

    Args:
        a (width, height) tuple or list, optionally with dimensions that are
            -1 to indicate "fill-in" dimensions

    Returns:
        the frame size converted to a tuple, if necessary

    Raises:
        ValueError: if the frame size was invalid
    '''
    if isinstance(frame_size, list):
        frame_size = tuple(frame_size)
    if not isinstance(frame_size, tuple):
        raise ValueError(
            "Frame size must be a tuple or list; found '%s'" % str(frame_size))
    if len(frame_size) != 2:
        raise ValueError(
            "frame_size must be a be a (width, height) tuple; found '%s'" %
            str(frame_size))
    return frame_size


def infer_missing_dims(frame_size, ref_size):
    '''Infers the missing entries (if any) of the given frame size.

    Args:
        frame_size: a (width, height) tuple. One or both dimensions can be -1,
            in which case the input aspect ratio is preserved
        ref_size: the reference (width, height)

    Returns:
        the concrete (width, height) with no negative values
    '''
    width, height = frame_size
    kappa = ref_size[0] / ref_size[1]
    if width < 0:
        if height < 0:
            return ref_size
        width = int(round(height * kappa))
    elif height < 0:
        height = int(round(width / kappa))
    return width, height


def scale_frame_size(frame_size, scale):
    '''Scales the frame size by the given factor.

    Args:
        frame_size: a (width, height) tuple
        scale: the desired scale factor

    Returns:
        the scaled (width, height)
    '''
    return tuple(int(round(scale * d)) for d in frame_size)


def clamp_frame_size(frame_size, max_size):
    '''Clamps the frame size to the given maximum size

    Args:
        frame_size: a (width, height) tuple
        max_size: a (max width, max height) tuple. One or both dimensions can
            be -1, in which case no constraint is applied that dimension

    Returns:
        the (width, height) scaled down if necessary so that width <= max width
            and height <= max height
    '''
    alpha = 1
    if max_size[0] > 0:
        alpha = min(alpha, max_size[0] / frame_size[0])
    if max_size[1] > 0:
        alpha = min(alpha, max_size[1] / frame_size[1])
    return scale_frame_size(frame_size, alpha)


class Length(object):
    '''Represents a length along a specified dimension of an image as a
    relative percentage or an absolute pixel count.
    '''

    def __init__(self, length_str, dim):
        '''Creates a Length instance.

        Args:
            length_str: a string of the form '<float>%' or '<int>px' describing
                a relative or absolute length, respectively
            dim: the dimension to measure length along
        '''
        self.dim = dim
        if length_str.endswith("%"):
            self.relunits = True
            self.rellength = 0.01 * float(length_str[:-1])
            self.length = None
        elif length_str.endswith("px"):
            self.relunits = False
            self.rellength = None
            self.length = int(length_str[:-2])
        else:
            raise TypeError(
                "Expected '<float>%%' or '<int>px', received '%s'" %
                str(length_str)
            )

    def render_for(self, frame_size=None, shape=None, img=None):
        '''Returns the length in pixels for the given frame size/shape/img.

        Pass any *one* of the keyword arguments to render the length.

        Args:
            frame_size: the (width, height) of the image
            shape: the (height, width, ...) of the image, e.g. from img.shape
            img: the image itself

        Raises:
            LengthError: if none of the keyword arguments were passed
        '''
        if img is not None:
            shape = img.shape
        elif frame_size is not None:
            shape = frame_size[::-1]
        elif shape is None:
            raise LengthError("One keyword argument must be provided")

        if self.relunits:
            return int(round(self.rellength * shape[self.dim]))
        return self.length


class LengthError(Exception):
    '''Error raised when an invalid Length is encountered.'''
    pass


class Width(Length):
    '''Represents the width of an image as a relative percentage or an absolute
    pixel count.
    '''

    def __init__(self, width_str):
        '''Creates a Width instance.

        Args:
            width_str: a string of the form '<float>%' or '<int>px' describing
                a relative or absolute width, respectively
        '''
        super(Width, self).__init__(width_str, 1)


class Height(Length):
    '''Represents the height of an image as a relative percentage or an
    absolute pixel count.
    '''

    def __init__(self, height_str):
        '''Creates a Height instance.

        Args:
            height_str: a string of the form '<float>%' or '<int>px' describing
                a relative or absolute height, respectively
        '''
        super(Height, self).__init__(height_str, 0)


class Location(object):
    '''Represents a location in an image.'''

    # Valid loc strings
    TOP_LEFT = ["top-left", "tl"]
    TOP_RIGHT = ["top-right", "tr"]
    BOTTOM_RIGHT = ["bottom-right", "br"]
    BOTTOM_LEFT = ["bottom-left", "bl"]

    def __init__(self, loc):
        '''Creates a Location instance.

        Args:
            loc: a (case-insenstive) string specifying a location
                ["top-left", "top-right", "bottom-right", "bottom-left"]
                ["tl", "tr", "br", "bl"]
        '''
        self._loc = loc.lower()

    @property
    def is_top_left(self):
        '''True if the location is top left, otherwise False.'''
        return self._loc in self.TOP_LEFT

    @property
    def is_top_right(self):
        '''True if the location is top right, otherwise False.'''
        return self._loc in self.TOP_RIGHT

    @property
    def is_bottom_right(self):
        '''True if the location is bottom right, otherwise False.'''
        return self._loc in self.BOTTOM_RIGHT

    @property
    def is_bottom_left(self):
        '''True if the location is bottom left, otherwise False.'''
        return self._loc in self.BOTTOM_LEFT


###### Image Composition ######################################################


def best_tiling_shape(n, kappa=1.777, **kwargs):
    '''Computes the (width, height) of the best tiling of n images in a grid
    such that the composite image would have roughly the specified aspect
    ratio.

    The returned tiling always satisfies width * height >= n.

    Args:
        n: the number of images to tile
        kappa: the desired aspect ratio of the composite image. By default,
            this is 1.777
        **kwargs: a valid keyword argument for to_frame_size(). By default,
            square images are assumed

    Returns:
        the (width, height) of the best tiling
    '''
    alpha = aspect_ratio(**kwargs) if kwargs else 1.0

    def _cost(w, h):
        return (alpha * w - kappa * h) ** 2 + (w * h - n) ** 2

    def _best_width_for_height(h):
        w = np.arange(int(np.ceil(n / h)), n + 1)
        return w[np.argmin(_cost(w, h))]

    # Caution: this is O(n^2)
    hh = np.arange(1, n + 1)
    ww = np.array([_best_width_for_height(h) for h in hh])
    idx = np.argmin(_cost(ww, hh))
    return  ww[idx], hh[idx]


def tile_images(imgs, width, height, fill_value=0):
    '''Tiles the images in the given array into a grid of the given width and
    height (row-wise).

    If fewer than width * height images are provided, the remaining tiles are
    filled with blank images.

    Args:
        imgs: a list (or num_images x height x width x num_channels numpy
            array) of same-size images
        width: the desired grid width
        height: the desired grid height
        fill_value: a value to fill any blank chips in the tiled image

    Returns:
        the tiled image
    '''
    # Parse images
    imgs = np.asarray(imgs)
    num_imgs = len(imgs)
    if num_imgs == 0:
        raise ValueError("Must have at least one image to tile")

    # Pad with blank images, if necessary
    num_blanks = width * height - num_imgs
    if num_blanks < 0:
        raise ValueError(
            "Cannot tile %d images in a %d x %d grid" %
            (num_imgs, width, height))
    if num_blanks > 0:
        blank = np.full_like(imgs[0], fill_value)
        blanks = np.repeat(blank[np.newaxis, ...], num_blanks, axis=0)
        imgs = np.concatenate((imgs, blanks), axis=0)

    # Tile images
    rows = [
        np.concatenate(imgs[(i * width):((i + 1) * width)], axis=1)
        for i in range(height)
    ]
    return np.concatenate(rows, axis=0)


###### Color Conversions ######################################################
#
# R, G, B: ints in [0, 255], [0, 255], [0, 255]
# B, G, R: ints in [0, 255], [0, 255], [0, 255]
# H, S, V: floats in [0, 1], [0, 1], [0, 1]
# H, L, S: floats in [0, 1], [0, 1], [0, 1]
#


def rgb_to_hsv(r, g, b):
    '''Converts (red, green, blue) to a (hue, saturation, value) tuple.

    Args:
        r, g, b: the RGB values

    Returns:
        an H, S, V tuple
    '''
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)


def rgb_to_hls(r, g, b):
    '''Converts (red, green, blue) to a (hue, lightness, saturation) tuple.

    Args:
        r, g, b: the RGB values

    Returns:
        an H, L, S tuple
    '''
    return colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)


def rgb_to_hex(r, g, b):
    '''Converts (red, green, blue) to a "#rrbbgg" string.

    Args:
        r, g, b: the RGB values

    Returns:
        a hex string
    '''
    return "#%02x%02x%02x" % (r, g, b)


def bgr_to_hsv(b, g, r):
    '''Converts (blue, green, red) to a (hue, saturation, value) tuple.

    Args:
        b, g, r: the BGR values

    Returns:
        an H, S, V tuple
    '''
    return rgb_to_hsv(r, g, b)


def bgr_to_hls(b, g, r):
    '''Converts (blue, green, red) to a (hue, lightness, saturation) tuple.

    Args:
        b, g, r: the BGR values

    Returns:
        an H, L, S tuple
    '''
    return rgb_to_hls(r, g, b)


def bgr_to_hex(b, g, r):
    '''Converts (blue, green, red) to a "#rrbbgg" string.

    Args:
        b, g, r: the BGR values

    Returns:
        a hex string
    '''
    return rgb_to_hex(r, g, b)


def hsv_to_rgb(h, s, v):
    '''Converts a (hue, saturation, value) tuple to a (red, green blue) tuple.

    Args:
        h, s, v: the HSV values

    Returns:
        an R, G, B tuple
    '''
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(255 * r), int(255 * g), int(255 * b))


def hsv_to_bgr(h, s, v):
    '''Converts a (hue, saturation, value) tuple to a (blue, green red) tuple.

    Args:
        h, s, v: the HSV values

    Returns:
        a B, G, R tuple
    '''
    return hsv_to_rgb(h, s, v)[::-1]


def hsv_to_hls(h, s, v):
    '''Converts a (hue, saturation, value) tuple to a
    (hue, lightness, saturation) tuple.

    Args:
        h, s, v: the HSV values

    Returns:
        an H, L, S tuple
    '''
    return rgb_to_hls(*hsv_to_rgb(h, s, v))


def hsv_to_hex(h, s, v):
    '''Converts a (hue, saturation, value) tuple to a "#rrbbgg" string.

    Args:
        h, s, v: the HSV values

    Returns:
        a hex string
    '''
    return rgb_to_hex(*hsv_to_rgb(h, s, v))


def hls_to_rgb(h, l, s):
    '''Converts a (hue, lightness, saturation) tuple to a (red, green blue)
    tuple.

    Args:
        h, l, s: the HLS values

    Returns:
        an R, G, B tuple
    '''
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(255 * r), int(255 * g), int(255 * b))


def hls_to_bgr(h, l, s):
    '''Converts a (hue, lightness, saturation) tuple to a (blue, green red)
    tuple.

    Args:
        h, l, s: the HLS values

    Returns:
        a B, G, R tuple
    '''
    return hls_to_rgb(h, l, s)[::-1]


def hls_to_hsv(h, l, s):
    '''Converts a (hue, lightness, saturation) tuple to a
    (hue, saturation, value) tuple.

    Args:
        h, l, s: the HLS values

    Returns:
        an H, S, V tuple
    '''
    return rgb_to_hls(*hls_to_rgb(h, l, s))


def hls_to_hex(h, l, s):
    '''Converts a (hue, lightness, saturation) tuple to a "#rrbbgg" string.

    Args:
        h, l, s: the HLS values

    Returns:
        a hex string
    '''
    return rgb_to_hex(*hls_to_rgb(h, l, s))


def hex_to_rgb(h):
    '''Converts a "#rrbbgg" string to a (red, green, blue) tuple.

    Args:
        h: a hex string

    Returns:
        an R, G, B tuple
    '''
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_bgr(h):
    '''Converts a "#rrbbgg" string to a (blue, green, red) tuple.

    Args:
        h: a hex string

    Returns:
        a B, G, R tuple
    '''
    return hex_to_rgb(h)[::-1]


def hex_to_hsv(h):
    '''Converts a "#rrbbgg" string to a (hue, saturation, value) tuple.

    Args:
        h: a hex string

    Returns:
        an H, S, V tuple
    '''
    return rgb_to_hsv(*hex_to_rgb(h))


def hex_to_hls(h):
    '''Converts a "#rrbbgg" string to a (hue, lightness, saturation) tuple.

    Args:
        h: a hex string

    Returns:
        an H, L, S tuple
    '''
    return rgb_to_hls(*hex_to_rgb(h))


def rgb_to_gray(img):
    '''Converts the input RGB image to a grayscale image.

    Args:
        img: an RGB image

    Returns:
        a grayscale image
    '''
    if is_gray(img):
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def bgr_to_gray(img):
    '''Converts the input BGR image to a grayscale image.

    Args:
        img: a BGR image

    Returns:
        a grayscale image
    '''
    if is_gray(img):
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gray_to_bgr(img):
    '''Convert a grayscale image to an BGR image.

    Args:
        img: a grayscale image

    Returns:
        a BGR image
    '''
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def gray_to_rgb(img):
    '''Convert a grayscale image to an RGB image.

    Args:
        img: a grayscale image

    Returns:
        an RGB image
    '''
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def rgb_to_bgr(img):
    '''Converts an RGB image to a BGR image (supports alpha).

    Args:
        img: an RGB image

    Returns:
        a BGR image
    '''
    return _exchange_rb(img)


def bgr_to_rgb(img):
    '''Converts a BGR image to an RGB image (supports alpha).

    Args:
        img: a BGR image

    Returns:
        an RGB image
    '''
    return _exchange_rb(img)


def _exchange_rb(img):
    '''Converts an image from BGR to/from RGB format by exchanging the red and
    blue channels.

    Handles gray (passthrough), 3-channel, and 4-channel images.

    Args:
        img: an image

    Returns:
        a copy of the input image with its first and third channels swapped
    '''
    if is_gray(img):
        return img
    return img[..., [2, 1, 0] + list(range(3, img.shape[2]))]
