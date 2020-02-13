'''
Core tools and data structures for working with frames of videos.

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
import six
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import numpy as np

import eta.core.data as etad
import eta.core.labels as etal
import eta.core.objects as etao
import eta.core.serial as etas
import eta.core.utils as etau


class FrameLabels(etal.Labels):
    '''Class encapsulating labels for a frame, i.e., an image or a specific
    frame of a video.

    Attributes:
        attrs: AttributeContainer describing attributes of the frame
        objects: DetectedObjectContainer describing detected objects in the
            frame
    '''

    def __init__(self, attrs=None, objects=None):
        '''Constructs a FrameLabels instance.

        Args:
            attrs: (optional) AttributeContainer of attributes for the frame.
                By default, an empty AttributeContainer is created
            objects: (optional) DetectedObjectContainer of detected objects for
                the frame. By default, an empty DetectedObjectContainer is
                created
        '''
        self.attrs = attrs or etad.AttributeContainer()
        self.objects = objects or etao.DetectedObjectContainer()

    @property
    def has_attributes(self):
        '''Whether the frame has at least one attribute.'''
        return bool(self.attrs)

    @property
    def has_objects(self):
        '''Whether the frame has at least one object.'''
        return bool(self.objects)

    @property
    def is_empty(self):
        '''Whether the frame has no labels of any kind.'''
        return not self.has_attributes and not self.has_objects

    def add_attribute(self, frame_attr):
        '''Adds the attribute to the frame.

        Args:
            frame_attr: an Attribute
        '''
        self.attrs.add(frame_attr)

    def add_attributes(self, frame_attrs):
        '''Adds the attributes to the frame.

        Args:
            frame_attrs: an AttributeContainer
        '''
        self.attrs.add_container(frame_attrs)

    def add_object(self, obj):
        '''Adds the object to the frame.

        Args:
            obj: a DetectedObject
        '''
        self.objects.add(obj)

    def add_objects(self, objs):
        '''Adds the objects to the frame.

        Args:
            objs: a DetectedObjectContainer
        '''
        self.objects.add_container(objs)

    def clear(self):
        '''Removes all labels from the frame.'''
        self.clear_attributes()
        self.clear_objects()

    def clear_attributes(self):
        '''Removes all attributes from the frame.'''
        self.attrs = etad.AttributeContainer()

    def clear_objects(self):
        '''Removes all objects from the frame.'''
        self.objects = etao.DetectedObjectContainer()

    def merge_labels(self, frame_labels):
        '''Merges the labels into the frame.

        Args:
            frame_labels: a FrameLabels
        '''
        self.add_attributes(frame_labels.attrs)
        self.add_objects(frame_labels.objects)

    def filter_by_schema(self, schema):
        '''Removes objects/attributes from this object that are not compliant
        with the given schema.

        Args:
            schema: a FrameLabelsSchema
        '''
        self.attrs.filter_by_schema(schema.attrs)
        self.objects.filter_by_schema(schema.objects)

    def remove_objects_without_attrs(self, labels=None):
        '''Removes objects from the frame that do not have attributes.

        Args:
            labels: an optional list of object `label` strings to which to
                restrict attention when filtering. By default, all objects are
                processed
        '''
        self.objects.remove_objects_without_attrs(labels=labels)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        _attrs = []
        if self.attrs:
            _attrs.append("attrs")
        if self.objects:
            _attrs.append("objects")
        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs a `FrameLabels` from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a FrameLabels
        '''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainer.from_dict(attrs)

        objects = d.get("objects", None)
        if objects is not None:
            objects = etao.DetectedObjectContainer.from_dict(objects)

        return cls(attrs=attrs, objects=objects)


class FrameLabelsSchema(etal.LabelsSchema):
    '''Schema for FrameLabels.

    Attributes:
        attrs: an AttributeContainerSchema describing the attributes of the
            frame(s)
        objects: an ObjectContainerSchema describing the objects of the
            frame(s)
    '''

    def __init__(self, attrs=None, objects=None):
        '''Creates a FrameLabelsSchema instance.

        Args:
            attrs: (optional) an AttributeContainerSchema describing the
                attributes of the frame(s)
            objects: (optional) an ObjectContainerSchema describing the objects
                of the frame(s)
        '''
        self.attrs = attrs or etad.AttributeContainerSchema()
        self.objects = objects or etao.ObjectContainerSchema()

    @property
    def is_empty(self):
        '''Whether this schema has no labels of any kind.'''
        return not bool(self.attrs) and not bool(self.objects)

    def has_attribute(self, attr_name):
        '''Whether the schema has an attribute with the given name.

        Args:
            attr_name: an attribute name

        Returns:
            True/False
        '''
        return self.attrs.has_attribute(attr_name)

    def get_attribute_class(self, attr_name):
        '''Gets the `Attribute` class for the attribute with the given name.

        Args:
            attr_name: an attribute name

        Returns:
            the Attribute class
        '''
        return self.attrs.get_attribute_class(attr_name)

    def has_object_label(self, label):
        '''Whether the schema has an object with the given label.

        Args:
            label: an object label

        Returns:
            True/False
        '''
        return self.objects.has_object_label(label)

    def get_object_schema(self, label):
        '''Gets the `ObjectSchema` for the object with the given label.

        Args:
            label: the object label

        Returns:
            the ObjectSchema
        '''
        return self.objects.get_object_schema(label)

    def has_object_attribute(self, label, attr_name):
        '''Whether the schema has an object with the given label with a
        frame-level attribute with the given name.

        Args:
            label: an object label
            attr_name: a frame-level object attribute name

        Returns:
            True/False
        '''
        return self.objects.has_frame_attribute(label, attr_name)

    def get_object_attribute_schema(self, label, attr_name):
        '''Gets the `AttributeSchema` for the frame-level attribute of the
        given name for the object with the given label.

        Args:
            label: the object label
            attr_name: the name of the frame-level object attribute

        Returns:
            the AttributeSchema
        '''
        return self.objects.get_frame_attribute_schema(label, attr_name)

    def get_object_attribute_class(self, label, attr_name):
        '''Gets the `Attribute` class for the frame-level attribute of the
        given name for the object with the given label.

        Args:
            label: an object label
            attr_name: a frame-level object attribute name

        Returns:
            the Attribute class
        '''
        return self.objects.get_frame_attribute_class(label, attr_name)

    def add_attribute(self, attr):
        '''Adds the given attribute to the schema.

        Args:
            attr: an Attribute
        '''
        self.attrs.add_attribute(attr)

    def add_attributes(self, attrs):
        '''Adds the given attributes to the schema.

        Args:
            attrs: an AttributeContainer
        '''
        self.attrs.add_attributes(attrs)

    def add_object_label(self, label):
        '''Adds the given object label to the schema.

        Args:
            label: an object label
        '''
        self.objects.add_object_label(label)

    def add_object_attribute(self, label, attr):
        '''Adds the frame-level attribute for the object with the given label
        to the schema.

        Args:
            label: an object label
            attr: a frame-level Attribute
        '''
        self.objects.add_frame_attribute(label, attr)

    def add_object_attributes(self, label, attrs):
        '''Adds the frame-level attributes for the object with the given label
        to the schema.

        Args:
            label: an object label
            attrs: a frame-level AttributeContainer
        '''
        self.objects.add_frame_attributes(label, attrs)

    def add_object(self, obj):
        '''Adds the object to the schema.

        Args:
            obj: a DetectedObject
        '''
        self.objects.add_object(obj)

    def add_objects(self, objects):
        '''Adds the objects to the schema.

        Args:
            objects: a DetectedObjectContainer
        '''
        self.objects.add_objects(objects)

    def add_labels(self, frame_labels):
        '''Adds the FrameLabels to the schema.

        Args:
            frame_labels: a FrameLabels
        '''
        self.add_attributes(frame_labels.attrs)
        self.add_objects(frame_labels.objects)

    def is_valid_attribute(self, attr):
        '''Whether the attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Returns:
            True/False
        '''
        return self.attrs.is_valid_attribute(attr)

    def is_valid_object_label(self, label):
        '''Whether the object label is compliant with the schema.

        Args:
            label: an object label

        Returns:
            True/False
        '''
        return self.objects.is_valid_object_label(label)

    def is_valid_object_attribute(self, label, attr):
        '''Whether the frame-level attribute for the object with the given
        label is compliant with the schema.

        Args:
            label: an object label
            attr: a frame-level Attribute

        Returns:
            True/False
        '''
        return self.objects.is_valid_frame_attribute(label, attr)

    def is_valid_object(self, obj):
        '''Whether the given object is compliant with the schema.

        Args:
            obj: a DetectedObject

        Returns:
            True/False
        '''
        return self.objects.is_valid_object(obj)

    def validate_attribute(self, attr):
        '''Validates that the attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Raises:
            AttributeContainerSchemaError: if the attribute violates the schema
        '''
        self.attrs.validate_attribute(attr)

    def validate_object_label(self, label):
        '''Validates that the object label is compliant with the schema.

        Args:
            label: an object label

        Raises:
            ObjectContainerSchemaError: if the object label violates the schema
        '''
        self.objects.validate_object_label(label)

    def validate_object_attribute(self, label, attr):
        '''Validates that the frame-level attribute for the object with the
        given label is compliant with the schema.

        Args:
            label: an object label
            attr: a frame-level Attribute

        Raises:
            ObjectContainerSchemaError: if the object label violates the schema
            AttributeContainerSchemaError: if the frame-level attribute
                violates the schema
        '''
        self.objects.validate_frame_attribute(label, attr)

    def validate_object(self, obj):
        '''Validates that the object is compliant with the schema.

        Args:
            obj: a DetectedObject

        Raises:
            ObjectContainerSchemaError: if the object label violates the schema
            AttributeContainerSchemaError: if any attributes of the object
                violate the schema
        '''
        self.objects.validate_object(obj)

    def validate(self, frame_labels):
        '''Validates that the FrameLabels are compliant with the schema.

        Args:
            frame_labels: a FrameLabels instance

        Raises:
            AttributeContainerSchemaError: if a frame/object attribute
                violates the schema
            ObjectContainerSchemaError: if an object label violates the schema
        '''
        for attr in frame_labels.attrs:
            self.validate_attribute(attr)

        for obj in frame_labels.objects:
            self.validate_object(obj)

    def merge_schema(self, schema):
        '''Merges the given FrameLabelsSchema into this schema.

        Args:
            schema: a FrameLabelsSchema
        '''
        self.attrs.merge_schema(schema.attrs)
        self.objects.merge_schema(schema.objects)

    @classmethod
    def build_active_schema(cls, frame_labels):
        '''Builds a FrameLabelsSchema that describes the active schema of the
        given FrameLabels.

        Args:
            frame_labels: a FrameLabels instance

        Returns:
            a FrameLabelsSchema
        '''
        schema = cls()
        schema.add_labels(frame_labels)
        return schema

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        _attrs = []
        if self.attrs:
            _attrs.append("attrs")
        if self.objects:
            _attrs.append("objects")

        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs a FrameLabelsSchema from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a FrameLabelsSchema
        '''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainerSchema.from_dict(attrs)

        objects = d.get("objects", None)
        if objects is not None:
            objects = etao.ObjectContainerSchema.from_dict(objects)

        return cls(attrs=attrs, objects=objects)


class FrameLabelsSchemaError(etal.LabelsSchemaError):
    '''Error raised when an `FrameLabelsSchema` is violated.'''
    pass


def frame_number_to_timestamp(frame_number, total_frame_count, duration):
    '''Converts the given frame number to a timestamp.

    Args:
        frame_number: the frame number of interest
        total_frame_count: the total number of frames in the video
        duration: the length of the video (in seconds)

    Returns:
        the timestamp (in seconds) of the given frame number in the video
    '''
    if total_frame_count == 1:
        return 0
    alpha = (frame_number - 1) / (total_frame_count - 1)
    return alpha * duration


def timestamp_to_frame_number(timestamp, duration, total_frame_count):
    '''Converts the given timestamp in a video to a frame number.

    Args:
        timestamp: the timestamp (in seconds or "HH:MM:SS.XXX" format) of
            interest
        duration: the length of the video (in seconds)
        total_frame_count: the total number of frames in the video

    Returns:
        the frame number associated with the given timestamp in the video
    '''
    if isinstance(timestamp, six.string_types):
        timestamp = timestamp_str_to_seconds(timestamp)
    alpha = timestamp / duration
    return 1 + int(round(alpha * (total_frame_count - 1)))


def timestamp_str_to_seconds(timestamp):
    '''Converts a timestamp string in "HH:MM:SS.XXX" format to seconds.

    Args:
        timestamp: a string in "HH:MM:SS.XXX" format

    Returns:
        the number of seconds
    '''
    return sum(
        float(n) * m for n, m in zip(
            reversed(timestamp.split(":")), (1, 60, 3600)))


def world_time_to_timestamp(world_time, start_time):
    '''Converts the given world time to a timestamp in a video.

    If one (but not both) of the datetimes are timezone-aware, the other
    datetime is assumed to be expressed in UTC time.

    Args:
        world_time: a datetime describing a time of interest
        start_time: a datetime indicating the start time of the video

    Returns:
        the corresponding timestamp (in seconds) in the video
    '''
    return etau.datetime_delta_seconds(start_time, world_time)


def world_time_to_frame_number(
        world_time, start_time, duration, total_frame_count):
    '''Converts the given world time to a frame number in a video.

    Args:
        world_time: a datetime describing a time of interest
        start_time: a datetime indicating the start time of the video
        duration: the length of the video (in seconds)
        total_frame_count: the total number of frames in the video

    Returns:
        the corresponding timestamp (in seconds) in the video
    '''
    timestamp = world_time_to_timestamp(world_time, start_time)
    return timestamp_to_frame_number(timestamp, duration, total_frame_count)


def parse_frame_ranges(frames):
    '''Parses the given frames quantity into a FrameRanges instance.

    Args:
        frames: one of the following quantities:
            - a string like "1-3,6,8-10"
            - a FrameRange or FrameRanges instance
            - an iterable, e.g., [1, 2, 3, 6, 8, 9, 10]. The frames do not
                need to be in sorted order

    Returns:
        a FrameRanges instance describing the frames
    '''
    if isinstance(frames, six.string_types):
        # Human-readable frames string
        frame_ranges = FrameRanges.from_human_str(frames)
    elif isinstance(frames, FrameRange):
        # FrameRange
        frame_ranges = FrameRanges.from_frame_range(frames)
    elif isinstance(frames, FrameRanges):
        # FrameRanges
        frame_ranges = frames
    elif hasattr(frames, "__iter__"):
        # Frames iterable
        frame_ranges = FrameRanges.from_iterable(frames)
    else:
        raise ValueError("Invalid frames %s" % frames)

    return frame_ranges


class FrameRanges(etas.Serializable):
    '''Class representing a monotonically increasing and disjoint series of
    frames.
    '''

    def __init__(self, ranges=None):
        '''Creates a FrameRanges instance.

        Args:
            ranges: can either be a human-readable frames string like
                "1-3,6,8-10" or an iterable of (first, last) tuples, which must
                be disjoint and monotonically increasing. By default, an empty
                instance is created
        '''
        self._ranges = []
        self._idx = 0
        self._started = False

        if ranges is not None:
            if isinstance(ranges, six.string_types):
                ranges = self._parse_frames_str(ranges)

            for new_range in ranges:
                self._ingest_range(new_range)

    def __str__(self):
        return self.to_human_str()

    def __len__(self):
        return sum(len(r) for r in self._ranges)

    def __bool__(self):
        return bool(self._ranges)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        self._started = True
        try:
            frame = next(self._ranges[self._idx])
        except StopIteration:
            self._idx += 1
            return next(self)
        except IndexError:
            raise StopIteration

        return frame

    @staticmethod
    def _parse_frames_str(frames_str):
        ranges = []
        for r in frames_str.split(","):
            if r:
                fr = FrameRange.from_human_str(r)
                ranges.append((fr.first, fr.last))

        return ranges

    def _ingest_range(self, new_range):
        first, last = new_range
        end = self.limits[1]

        if end is not None and first <= end:
            raise FrameRangesError(
                "Expected first:%d > end:%d" % (first, end))

        self._ranges.append(FrameRange(first, last))

    @property
    def limits(self):
        '''A (first, last) tuple describing the limits of the frame ranges.

        Returns (None, None) if the instance is empty.
        '''
        if not self:
            return (None, None)

        first = self._ranges[0].limits[0]
        last = self._ranges[-1].limits[1]
        return (first, last)

    @property
    def num_ranges(self):
        '''The number of `FrameRange`s in this object.'''
        return len(self._ranges)

    @property
    def frame(self):
        '''The current frame number, or -1 if no frames have been read.'''
        if self._started:
            return self._ranges[self._idx].frame

        return -1

    @property
    def ranges(self):
        '''A serialized string representation of this object.'''
        # This controls how `FrameRanges` instances are serialized
        #return self.to_range_tuples()  # can be used if strings aren't liked
        return self.to_human_str()

    @property
    def frame_range(self):
        '''The (first, last) values for the current range, or (-1, -1) if no
        frames have been read.
        '''
        if self._started:
            return self._ranges[self._idx].first, self._ranges[self._idx].last

        return (-1, -1)

    @property
    def is_new_frame_range(self):
        '''Whether the current frame is the first in a new range.'''
        if self._started:
            return self._ranges[self._idx].is_first_frame

        return False

    @property
    def is_contiguous(self):
        '''Determines whether the frame range is contiguous, i.e., whether it
        consists of a single `FrameRange`.

        If you want to ensure that this instance does not contain trivial
        adjacent `FrameRange`s, then call `simplify()` first.

        Returns:
            True/False
        '''
        return self.num_ranges == 1

    def reset(self):
        '''Resets the FrameRanges instance so that the next frame will be the
        first.
        '''
        for r in self._ranges[:(self._idx + 1)]:
            r.reset()

        self._started = False
        self._idx = 0

    def clear(self):
        '''Clears the FrameRanges instance.'''
        self._ranges = []
        self.reset()

    def add_range(self, new_range):
        '''Adds the given frame range to the instance.

        Args:
            new_range: a (first, last) tuple describing the range

        Raises:
            FrameRangesError: if the new range is not disjoint and
                monotonically increasing
        '''
        self._ingest_range(new_range)

    def simplify(self):
        '''Simplifies the frame ranges, if possible, by merging any adjacent
        `FrameRange` instances into a single range.

        This operation will `reset()` the instance.
        '''
        if not self:
            return

        did_something = False
        last_range = list(self._ranges[0].limits)
        new_ranges = [last_range]
        for old_range in self._ranges[1:]:
            ofirst, olast = old_range.limits
            if ofirst <= last_range[1] + 1:
                did_something = True
                last_range[1] = olast
            else:
                last_range = [ofirst, olast]
                new_ranges.append(last_range)

        if not did_something:
            self.reset()
            return

        self.clear()
        for new_range in new_ranges:
            self.add_range(new_range)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attributes
        '''
        return ["ranges"]

    def to_range_tuples(self):
        '''Returns the list of (first, last) tuples defining the frame ranges
        in this instance.

        Returns:
            a list of (first, last) tuples
        '''
        return [r.limits for r in self._ranges]

    def to_list(self):
        '''Returns the list of frames, in sorted order, described by this
        object.

        Returns:
            list of frames
        '''
        frames = []
        for r in self._ranges:
            frames += r.to_list()

        return frames

    def to_human_str(self):
        '''Returns a human-readable string representation of this object.

        Returns:
            a string like "1-3,6,8-10" describing the frame ranges
        '''
        return ",".join([fr.to_human_str() for fr in self._ranges])

    def to_bools(self, total_frame_count=None):
        '''Returns a boolean array indicating the frames described by this
        object.

        Note that the boolean array uses 0-based indexing. Thus, the returned
        array satisfies `bools[idx] == True` iff frame `idx + 1` is in this
        object.

        Args:
            total_frame_count: an optional total frame count. Can be less or
                greater than the maximum frame in this object, if desired. By
                default, `self.limits[1]` is used.

        Returns:
            a boolean numpy array of length `total_frame_count`
        '''
        if total_frame_count is None:
            total_frame_count = self.limits[1]

        bools = np.zeros(total_frame_count, dtype=bool)

        inds = [i - 1 for i in self.to_list() if i <= total_frame_count]
        bools[inds] = True

        return bools

    @staticmethod
    def build_simple(first, last):
        '''Builds a FrameRanges from a simple [first, last] range.

        Args:
            first: the first frame
            last: the last frame

        Returns:
            a FrameRanges instance
        '''
        return FrameRanges(ranges=[(first, last)])

    @classmethod
    def from_bools(cls, bools):
        '''Constructs a FrameRanges object from a boolean array describing the
        frames in the ranges.

        Note that the 0-based indexes in the boolean array are converted to
        1-based frame numbers. In other words, the returned FrameRanges
        contains `frame` iff `bools[frame - 1] == True`.

        Args:
            bools: a boolean array

        Returns:
            a FrameRanges instance
        '''
        return cls.from_iterable(1 + np.flatnonzero(bools))

    @classmethod
    def from_human_str(cls, frames_str):
        '''Constructs a FrameRanges object from a human-readable frames string.

        Args:
            frames_str: a human-readable frames string like "1-3,6,8-10"

        Returns:
            a FrameRanges instance

        Raises:
            FrameRangesError: if the frames string is invalid
        '''
        return cls(ranges=frames_str)

    @classmethod
    def from_iterable(cls, frames):
        '''Constructs a FrameRanges object from an iterable of frames.

        The frames do not need to be in sorted order.

        Args:
            frames: an iterable of frames, e.g., [1, 2, 3, 6, 8, 9, 10]

        Returns:
            a FrameRanges instance

        Raises:
            FrameRangesError: if the frames list is invalid
        '''
        return cls(ranges=_iterable_to_ranges(frames))

    @classmethod
    def from_frame_range(cls, frame_range):
        '''Constructs a FrameRanges instance from a FrameRange instance.

        Args:
            frame_range: a FrameRange instance

        Returns:
            a FrameRanges instance
        '''
        return cls(ranges=[(frame_range.first, frame_range.last)])

    @classmethod
    def from_dict(cls, d):
        '''Constructs a FrameRanges from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a FrameRanges instance
        '''
        ranges = d.get("ranges", None)
        return cls(ranges=ranges)


class FrameRangesError(Exception):
    '''Exception raised when an invalid FrameRanges is encountered.'''
    pass


class FrameRange(etas.Serializable):
    '''Class representing a range of frames.'''

    def __init__(self, first, last):
        '''Creates a FrameRange instance.

        Args:
            first: the first frame in the range (inclusive)
            last: the last frame in the range (inclusive)
        '''
        self.first = first
        self.last = last
        self._frame = -1

        self._validate_range(first, last)

    @staticmethod
    def _validate_range(first, last):
        if first < 1:
            raise FrameRangeError("Expected first:%d >= 1" % first)

        if last < first:
            raise FrameRangeError(
                "Expected first:%d <= last:%d" % (first, last))

    def __str__(self):
        return self.to_human_str()

    def __len__(self):
        return self.last + 1 - self.first

    def __bool__(self):
        return True

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self._frame < 0:
            self._frame = self.first
        elif self._frame < self.last:
            self._frame += 1
        else:
            raise StopIteration

        return self._frame

    def reset(self):
        '''Resets the FrameRange instance so that the next frame will be the
        first.
        '''
        self._frame = -1

    @property
    def frame(self):
        '''The current frame number, or -1 if no frames have been read.'''
        if self._frame < 0:
            return -1

        return self._frame

    @property
    def limits(self):
        '''A (first, last) tuple describing the frame range.'''
        return (self.first, self.last)

    @property
    def is_first_frame(self):
        '''Whether the current frame is first in the range.'''
        return self._frame == self.first

    def to_list(self):
        '''Returns the list of frames in the range.

        Returns:
            a list of frames
        '''
        return list(range(self.first, self.last + 1))

    def to_human_str(self):
        '''Returns a human-readable string representation of the range.

        Returns:
            a string like "1-5"
        '''
        if self.first == self.last:
            return "%d" % self.first

        return "%d-%d" % (self.first, self.last)

    @classmethod
    def from_human_str(cls, frames_str):
        '''Constructs a FrameRange object from a human-readable string.

        Args:
            frames_str: a human-readable frames string like "1-5"

        Returns:
            a FrameRange instance

        Raises:
            FrameRangeError: if the frame range string is invalid
        '''
        try:
            v = list(map(int, frames_str.split("-")))
            return cls(v[0], v[-1])
        except ValueError:
            raise FrameRangeError(
                "Invalid frame range string '%s'" % frames_str)

    @classmethod
    def from_iterable(cls, frames):
        '''Constructs a FrameRange object from an iterable of frames.

        The frames do not need to be in sorted order, but they must define a
        single interval.

        Args:
            frames: an iterable of frames, e.g., [1, 2, 3, 4, 5]

        Returns:
            a FrameRange instance

        Raises:
            FrameRangeError: if the frame range list is invalid
        '''
        ranges = list(_iterable_to_ranges(frames))
        if len(ranges) != 1:
            raise FrameRangeError("Invalid frame range list %s" % frames)

        return cls(*ranges[0])

    @classmethod
    def from_dict(cls, d):
        '''Constructs a FrameRange from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a FrameRange instance
        '''
        return cls(d["first"], d["last"])


class FrameRangeError(Exception):
    '''Exception raised when an invalid FrameRange is encountered.'''
    pass


def _iterable_to_ranges(vals):
    # This will convert numpy arrays to list, and it's important to do this
    # before checking for falseness below, since numpy arrays don't support it
    vals = sorted(vals)

    if not vals:
        return

    first = last = vals[0]
    for val in vals[1:]:
        if val == last + 1:
            last += 1
        else:
            yield (first, last)
            first = last = val

    yield (first, last)
