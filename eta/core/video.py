'''
Core tools and data structures for working with videos.

Notes:
    [frame numbers] ETA uses 1-based indexing for all frame numbers

    [image format] ETA stores images exclusively in RGB format. In contrast,
        OpenCV stores its images in BGR format, so all images that are read or
        produced outside of this library must be converted to RGB. This
        conversion can be done via `eta.core.image.bgr_to_rgb()`

Copyright 2017-2019, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
Jason Corso, jason@voxel51.com
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from future.utils import iteritems, itervalues
import six
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict, OrderedDict
import dateutil.parser
import errno
import logging
import os
from subprocess import Popen, PIPE
import threading

import cv2
import numpy as np

from eta.core.data import AttributeContainer, AttributeContainerSchema
import eta.core.gps as etag
import eta.core.image as etai
from eta.core.objects import DetectedObjectContainer
from eta.core.serial import load_json, Serializable, Set, BigSet
import eta.core.utils as etau


logger = logging.getLogger(__name__)


#
# The file extensions of supported video files. Use LOWERCASE!
#
# In practice, any video that ffmpeg can read will be supported. Nonetheless,
# we enumerate this list here so that the ETA type system can verify the
# extension of a video provided to a pipeline at build time.
#
# This list was taken from https://en.wikipedia.org/wiki/Video_file_format
#
SUPPORTED_VIDEO_FILE_FORMATS = {
    ".3g2", ".3gp", ".m2ts", ".mts", ".amv", ".avi", ".f4a", ".f4b", ".f4p",
    ".f4v", ".flv", ".m2v", ".m4p", ".m4v", ".mkv", ".mov", ".mp2", ".mp4",
    ".mpe", ".mpeg", ".mpg", ".mpv", ".nsv", ".ogg", ".ogv", ".qt", ".rm",
    ".rmvb", ".svi", ".vob", ".webm", ".wmv", ".yuv"
}


def is_supported_video(path):
    '''Determines whether the given filepath points to a supported video.

    Args:
        path: the path to a video, like `/path/to/video.mp4` or
            `/path/to/frames-%05d.jpg`

    Returns:
        True/False if the path refers to a supported video type
    '''
    return is_supported_video_file(path) or is_supported_image_sequence(path)


def is_supported_video_file(path):
    '''Determines whether the given filepath points to a supported video file
    type.

    Args:
        path: the path to a video file, like `/path/to/video.mp4`

    Returns:
        True/False if the path refers to a supported video file type
    '''
    return os.path.splitext(path)[1].lower() in SUPPORTED_VIDEO_FILE_FORMATS


def is_supported_image_sequence(path):
    '''Determines whether the given filepath points to a supported image
    sequence type.

    Args:
        path: the path to an image sequence, like `/path/to/frames-%05d.jpg`

    Returns:
        True/False if the path refers to a supported video file type
    '''
    try:
        _ = path % 1
        return etai.is_supported_image(path)
    except TypeError:
        return False


def is_same_video_file_format(path1, path2):
    '''Determines whether the video files have the same supported format.

    Args:
        path1: the path to a video
        path2: the path to a video

    Returns:
        True/False
    '''
    return (
        is_supported_video(path1) and
        os.path.splitext(path1)[1] == os.path.splitext(path2)[1]
    )


def is_valid_video_file(path):
    '''Determines if the given video file is valid, i.e., it has a supported
    type and can be read by our system.

    This method does not support videos represented as image sequences (i.e.,
    it will return False for them).

    Args:
        path: the path to a video file

    Returns:
        True/False if the video is valid
    '''
    if not is_supported_video_file(path):
        return False
    try:
        with FFmpegVideoReader(path):
            return True
    except etau.ExecutableRuntimeError:
        return False


def glob_videos(dir_):
    '''Returns an iterator over all supported video files in the directory.'''
    return etau.multiglob(
        *SUPPORTED_VIDEO_FILE_FORMATS, root=os.path.join(dir_, "*"))


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

    Args:
        world_time: a datetime describing a time of interest
        start_time: a datetime indicating the start time of the video

    Returns:
        the corresponding timestamp (in seconds) in the video
    '''
    return (world_time - start_time).total_seconds()


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


class VideoMetadata(Serializable):
    '''Class encapsulating metadata about a video.

    Attributes:
        start_time: a datetime describing the start (world) time of the video
        frame_size: the [width, height] of the video frames
        frame_rate: the frame rate of the video
        total_frame_count: the total number of frames in the video
        duration: the duration of the video, in seconds
        size_bytes: the size of the video file on disk, in bytes
        mime_type: the MIME type of the video
        encoding_str: the encoding string for the video
        gps_waypoints: a GPSWaypoints instance describing the GPS coordinates
            for the video
    '''

    def __init__(
            self, start_time=None, frame_size=None, frame_rate=None,
            total_frame_count=None, duration=None, size_bytes=None,
            mime_type=None, encoding_str=None, gps_waypoints=None):
        '''Creates a VideoMetadata instance.

        Args:
            start_time: a datetime describing
            frame_size: the [width, height] of the video frames
            frame_rate: the frame rate of the video
            total_frame_count: the total number of frames in the video
            duration: the duration of the video, in seconds
            size_bytes: the size of the video file on disk, in bytes
            mime_type: the MIME type of the video
            encoding_str: the encoding string for the video
            gps_waypoints: a GPSWaypoints instance describing the GPS
                coordinates for the video
        '''
        self.start_time = start_time
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.total_frame_count = total_frame_count
        self.duration = duration
        self.size_bytes = size_bytes
        self.mime_type = mime_type
        self.encoding_str = encoding_str
        self.gps_waypoints = gps_waypoints

    @property
    def has_gps(self):
        '''Returns True/False if this object has GPS waypoints.'''
        return self.gps_waypoints is not None

    def get_timestamp(self, frame_number=None, world_time=None):
        '''Gets the timestamp for the given point in the video.

        Exactly one keyword argument must be supplied.

        Args:
            frame_number: the frame number of interest
            world_time: a datetime describing the world time of interest

        Returns:
            the timestamp (in seconds) in the video
        '''
        if world_time is not None:
            return world_time_to_timestamp(world_time, self.start_time)

        return frame_number_to_timestamp(
            frame_number, self.total_frame_count, self.duration)

    def get_frame_number(self, timestamp=None, world_time=None):
        '''Gets the frame number for the given point in the video.

        Exactly one keyword argument must be supplied.

        Args:
            timestamp: the timestamp (in seconds or "HH:MM:SS.XXX" format) of
                interest
            world_time: a datetime describing the world time of interest

        Returns:
            the frame number in the video
        '''
        if world_time is not None:
            return world_time_to_frame_number(
                world_time, self.start_time, self.duration,
                self.total_frame_count)

        return timestamp_to_frame_number(
            timestamp, self.duration, self.total_frame_count)

    def get_gps_location(
            self, frame_number=None, timestamp=None, world_time=None):
        '''Gets the GPS location at the given point in the video.

        Exactly one keyword argument must be supplied.

        Nearest neighbors is used to interpolate between waypoints.

        Args:
            frame_number: the frame number of interest
            timestamp: the timestamp (in seconds or "HH:MM:SS.XXX" format) of
                interest
            world_time: a datetime describing the absolute (world) time of
                interest

        Returns:
            the (lat, lon) at the given frame in the video, or None if the
                video has no GPS waypoints
        '''
        if not self.has_gps:
            return None
        if world_time is not None:
            timestamp = self.get_timestamp(world_time=world_time)
        if timestamp is not None:
            frame_number = self.get_frame_number(timestamp=timestamp)
        return self.gps_waypoints.get_location(frame_number)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.'''
        _attrs = [
            "start_time", "frame_size", "frame_rate", "total_frame_count",
            "duration", "size_bytes", "mime_type", "encoding_str",
            "gps_waypoints"
        ]
        # Exclude attributes that are None
        return [a for a in _attrs if getattr(self, a) is not None]

    @classmethod
    def build_for(
            cls, filepath, start_time=None, gps_waypoints=None):
        '''Builds a VideoMetadata object for the given video.

        Args:
            filepath: the path to the video on disk
            start_time: an optional datetime specifying the start time of the
                video
            gps_waypoints: an optional GPSWaypoints instance describing the
                GPS coordinates of the video

        Returns:
            a VideoMetadata instance
        '''
        vsi = VideoStreamInfo.build_for(filepath)
        return cls(
            start_time=start_time,
            frame_size=vsi.frame_size,
            frame_rate=vsi.frame_rate,
            total_frame_count=vsi.total_frame_count,
            duration=vsi.duration,
            size_bytes=os.path.getsize(filepath),
            mime_type=etau.guess_mime_type(filepath),
            encoding_str=vsi.encoding_str,
            gps_waypoints=gps_waypoints,
        )

    @classmethod
    def from_dict(cls, d):
        '''Constructs a VideoMetadata from a JSON dictionary.'''
        start_time = d.get("start_time", None)
        if start_time is not None:
            start_time = dateutil.parser.parse(start_time)

        gps_waypoints = d.get("gps_waypoints", None)
        if isinstance(gps_waypoints, dict):
            gps_waypoints = etag.GPSWaypoints.from_dict(gps_waypoints)
        elif isinstance(gps_waypoints, list):
            # this supports a list of GPSWaypoint instances rather than a
            # serialized GPSWaypoints instance. for backwards compatability
            points = [etag.GPSWaypoint.from_dict(p) for p in gps_waypoints]
            gps_waypoints = etag.GPSWaypoints(points=points)

        return cls(
            start_time=start_time,
            frame_size=d.get("frame_size", None),
            frame_rate=d.get("frame_rate", None),
            total_frame_count=d.get("total_frame_count", None),
            duration=d.get("duration", None),
            size_bytes=d.get("size_bytes", None),
            mime_type=d.get("mime_type", None),
            encoding_str=d.get("encoding_str", None),
            gps_waypoints=gps_waypoints)


class VideoFrameLabels(Serializable):
    '''Class encapsulating labels for a frame of a video.

    Attributes:
        frame_number: the frame number
        attrs: an AttributeContainer describing the attributes of the frame
        objects: a DetectedObjectContainer describing the detected objects in
            the frame
    '''

    def __init__(self, frame_number, attrs=None, objects=None):
        '''Constructs a VideoFrameLabels instance.

        Args:
            frame_number: the frame number of the video
            attrs: an optional AttributeContainer of attributes for the frame.
                By default, an empty AttributeContainer is created
            objects: an optional DetectedObjectContainer of detected objects
                for the frame. By default, an empty DetectedObjectContainer is
                created
        '''
        self.frame_number = frame_number
        self.attrs = attrs or AttributeContainer()
        self.objects = objects or DetectedObjectContainer()

    def add_frame_attribute(self, frame_attr):
        '''Adds the attribute to the frame.

        Args:
            frame_attr: an Attribute
        '''
        self.attrs.add(frame_attr)

    def add_frame_attributes(self, frame_attrs):
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

    def clear_frame_attributes(self):
        '''Removes all frame attributes from the instance.'''
        self.attrs = AttributeContainer()

    def clear_objects(self):
        '''Removes all objects from the instance.'''
        self.objects = DetectedObjectContainer()

    def merge_frame_labels(self, frame_labels):
        '''Merges the labels into the frame.

        Args:
            frame_labels: a VideoFrameLabels
        '''
        self.add_frame_attributes(frame_labels.attrs)
        self.add_objects(frame_labels.objects)

    def filter_by_schema(self, schema):
        '''Removes objects/attributes from this object that are not compliant
        with the given schema.

        Args:
            schema: a VideoLabelsSchema
        '''
        self.attrs.filter_by_schema(schema.frames)
        self.objects.filter_by_schema(schema)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.'''
        _attrs = ["frame_number"]
        if self.attrs:
            _attrs.append("attrs")
        if self.objects:
            _attrs.append("objects")
        return _attrs

    @classmethod
    def from_image_labels(cls, image_labels, frame_number):
        '''Constructs a VideoFrameLabels from an ImageLabels.

        Args:
            image_labels: an ImageLabels instance
            frame_number: the frame number

        Returns:
            a VideoFrameLabels instance
        '''
        return cls(frame_number, image_labels.attrs, image_labels.objects)

    @classmethod
    def from_dict(cls, d):
        '''Constructs a VideoFrameLabels from a JSON dictionary.'''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = AttributeContainer.from_dict(attrs)

        objects = d.get("objects", None)
        if objects is not None:
            objects = DetectedObjectContainer.from_dict(objects)

        return cls(d["frame_number"], attrs=attrs, objects=objects)


class VideoLabels(Serializable):
    '''Class encapsulating labels for a video.

    Note that any falsey fields of this class will be omitted during
    serialization.

    Note that when VideoLabels objects are serialized, the keys of the `frames`
    dict will be converted to strings, because all JSON object keys _must_ be
    strings. The `from_dict` method of this class handles converting the keys
    back to integers when VideoLabels instances are loaded.

    Attributes:
        filename: the filename of the video
        metadata: a VideoMetadata describing metadata about the video
        attrs: an AttributeContainer containing video attributes
        frames: a dictionary mapping frame number strings to VideoFrameLabels
            instances
        schema: a VideoLabelsSchema describing the schema of the video labels
    '''

    def __init__(
            self, filename=None, metadata=None, attrs=None, frames=None,
            schema=None):
        '''Constructs a VideoLabels instance.

        Args:
            filename: an optional filename for the video. By default, no
                filename is stored
            metadata: an optional VideoMetadata instance describing metadata
                about the video. By default, no metadata is stored
            attrs: an optional AttributeContainer of video attributes. By
                default, an empty AttributeContainer is created
            frames: an optional dictionary mapping frame numbers to
                VideoFrameLabels instances. By default, an empty dictionary
                is created
            schema: an optional VideoLabelsSchema to enforce on the object.
                By default, no schema is enforced
        '''
        self.filename = filename
        self.metadata = metadata
        self.attrs = attrs or AttributeContainer()
        self.frames = frames or {}
        self.schema = schema

    def __getitem__(self, frame_number):
        return self.get_frame(frame_number)

    def __setitem__(self, frame_number, frame_labels):
        frame_labels.frame_number = frame_number
        self.add_frame(frame_labels, overwrite=True)

    def __delitem__(self, frame_number):
        self.delete_frame(frame_number)

    def __iter__(self):
        # Always iterate over the keys in sorted order
        return iter(sorted(self.frames))

    def __len__(self):
        return len(self.frames)

    def __bool__(self):
        return bool(self.frames)

    @property
    def has_frame_attributes(self):
        '''Returns True/False whether the container has at least one frame
        attribute.
        '''
        for frame_number in self:
            if self[frame_number].attrs:
                return True

        return False

    @property
    def has_objects(self):
        '''Returns True/False whether the container has at least one
        DetectedObject.
        '''
        for frame_number in self:
            if self[frame_number].objects:
                return True

        return False

    @property
    def has_schema(self):
        '''Returns True/False whether the container has an enforced schema.'''
        return self.schema is not None

    def has_frame(self, frame_number):
        '''Returns True/False whether this object contains a VideoFrameLabels
        for the given frame number.
        '''
        return frame_number in self.frames

    def get_frame(self, frame_number):
        '''Gets the VideoFrameLabels for the given frame number, or an empty if
        VideoFrameLabels if the frame has no labels.
        '''
        try:
            return self.frames[frame_number]
        except KeyError:
            return VideoFrameLabels(frame_number)

    def delete_frame(self, frame_number):
        '''Deletes the VideoFrameLabels for the given frame number.'''
        del self.frames[frame_number]

    def get_frame_numbers(self):
        '''Returns a sorted list of all frames with VideoFrameLabels.'''
        return sorted(self.frames.keys())

    def get_frame_range(self):
        '''Returns the (min, max) frame numbers with VideoFrameLabels.'''
        fns = self.get_frame_numbers()
        return (fns[0], fns[-1]) if fns else (None, None)

    def merge_video_labels(self, video_labels):
        '''Merges the given VideoLabels into this labels.'''
        self.attrs.add_container(video_labels.attrs)
        for frame_number in video_labels:
            self.add_frame(video_labels[frame_number], overwrite=False)

    def add_video_attribute(self, video_attr):
        '''Adds the given video attribute to the video.

        Args:
            video_attr: an Attribute
        '''
        if self.has_schema:
            self._validate_video_attribute(video_attr)
        self.attrs.add(video_attr)

    def add_video_attributes(self, video_attrs):
        '''Adds the given video attributes to the video.

        Args:
            video_attrs: an AttributeContainer
        '''
        if self.has_schema:
            for video_attr in video_attrs:
                self._validate_video_attribute(video_attr)
        self.attrs.add_container(video_attrs)

    def add_frame(self, frame_labels, overwrite=True):
        '''Adds the frame labels to the video.

        Args:
            frame_labels: a VideoFrameLabels instance
            overwrite: whether to overwrite any existing VideoFrameLabels
                instance for the frame or merge the new labels. By default,
                this is True
        '''
        if self.has_schema:
            self._validate_frame_labels(frame_labels)

        frame_number = frame_labels.frame_number
        if overwrite or not self.has_frame(frame_number):
            self.frames[frame_number] = frame_labels
        else:
            self.frames[frame_number].merge_frame_labels(frame_labels)

    def add_frame_attribute(self, frame_attr, frame_number):
        '''Adds the given frame attribute to the video.

        Args:
            frame_attr: an Attribute
            frame_number: the frame number
        '''
        if self.has_schema:
            self._validate_frame_attribute(frame_attr)
        self._ensure_frame(frame_number)
        self.frames[frame_number].add_frame_attribute(frame_attr)

    def add_frame_attributes(self, frame_attrs, frame_number):
        '''Adds the given frame attributes to the video.

        Args:
            frame_attrs: an AttributeContainer
            frame_number: the frame number
        '''
        if self.has_schema:
            for frame_attr in frame_attrs:
                self._validate_frame_attribute(frame_attr)
        self._ensure_frame(frame_number)
        self.frames[frame_number].add_frame_attributes(frame_attrs)

    def add_object(self, obj, frame_number):
        '''Adds the object to the video.

        Args:
            obj: a DetectedObject
            frame_number: the frame number
        '''
        if self.has_schema:
            self._validate_object(obj)
        self._ensure_frame(frame_number)
        obj.frame_number = frame_number
        self.frames[frame_number].add_object(obj)

    def add_objects(self, objs, frame_number):
        '''Adds the objects to the video.

        Args:
            objs: a DetectedObjectContainer
            frame_number: the frame number
        '''
        if self.has_schema:
            for obj in objs:
                self._validate_object(obj)
        self._ensure_frame(frame_number)
        for obj in objs:
            obj.frame_number = frame_number
            self.frames[frame_number].add_object(obj)

    def clear_frame_attributes(self):
        '''Removes all frame attributes from the instance.'''
        for frame_number in self:
            self[frame_number].clear_frame_attributes()

    def clear_objects(self):
        '''Removes all objects from the instance.'''
        for frame_number in self:
            self[frame_number].clear_objects()

    def get_schema(self):
        '''Gets the current enforced schema for the video, or None if no schema
        is enforced.
        '''
        return self.schema

    def get_active_schema(self):
        '''Returns a VideoLabelsSchema describing the active schema of the
        video.
        '''
        return VideoLabelsSchema.build_active_schema(self)

    def set_schema(self, schema, filter_by_schema=False):
        '''Sets the enforced schema to the given VideoLabelsSchema.

        Args:
            schema: a VideoLabelsSchema to assign
            filter_by_schema: whether to filter objects/attributes that are not
                compliant with the schema. By default, this is False

        Raises:
            VideoLabelsSchemaError: if `filter_by_schema` was False and this
                object contains attributes/objects that are not compliant with
                the schema
        '''
        self.schema = schema
        if not self.has_schema:
            return

        if filter_by_schema:
            self.filter_by_schema(self.schema)
        else:
            self._validate_schema()

    def filter_by_schema(self, schema):
        '''Removes objects/attributes from this object that are not compliant
        with the given schema.

        Args:
            schema: a VideoLabelsSchema
        '''
        self.attrs.filter_by_schema(schema.attrs)
        for frame_labels in itervalues(self.frames):
            frame_labels.filter_by_schema(schema)

    def freeze_schema(self):
        '''Sets the enforced schema for the video to the current active
        schema.
        '''
        self.set_schema(self.get_active_schema())

    def remove_schema(self):
        '''Removes the enforced schema from the video.'''
        self.schema = None

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.'''
        _attrs = []
        if self.filename:
            _attrs.append("filename")
        if self.metadata:
            _attrs.append("metadata")
        if self.has_schema:
            _attrs.append("schema")
        if self.attrs:
            _attrs.append("attrs")
        if self.frames:
            _attrs.append("frames")
        return _attrs

    def _validate_video_attribute(self, video_attr):
        if self.has_schema:
            self.schema.validate_video_attribute(video_attr)

    def _ensure_frame(self, frame_number):
        if not self.has_frame(frame_number):
            self.frames[frame_number] = VideoFrameLabels(frame_number)

    def _validate_frame_labels(self, frame_labels):
        if self.has_schema:
            for frame_attr in frame_labels.attrs:
                self.schema.validate_frame_attribute(frame_attr)
            for obj in frame_labels.objects:
                self.schema.validate_object(obj)

    def _validate_frame_attribute(self, frame_attr):
        if self.has_schema:
            self.schema.validate_frame_attribute(frame_attr)

    def _validate_object(self, obj):
        if self.has_schema:
            self.schema.validate_object(obj)

    def _validate_schema(self):
        if self.has_schema:
            for video_attr in self.attrs:
                self._validate_video_attribute(video_attr)
            for frame_labels in itervalues(self.frames):
                self._validate_frame_labels(frame_labels)

    @classmethod
    def from_detected_objects(cls, objects):
        '''Builds a VideoLabels instance from a DetectedObjectContainer.

        The DetectedObjects must have their `frame_number` attributes set.

        Args:
            objects: a DetectedObjectContainer

        Returns:
            a VideoLabels instance
        '''
        labels = cls()
        for obj in objects:
            labels.add_object(obj, obj.frame_number)
        return labels

    @classmethod
    def from_dict(cls, d):
        '''Constructs a VideoLabels from a JSON dictionary.'''
        filename = d.get("filename", None)

        metadata = d.get("metadata", None)
        if metadata is not None:
            metadata = VideoMetadata.from_dict(metadata)

        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = AttributeContainer.from_dict(attrs)

        frames = d.get("frames", None)
        if frames is not None:
            frames = OrderedDict(
                (int(fn), VideoFrameLabels.from_dict(vfl))
                for fn, vfl in iteritems(frames)
            )

        schema = d.get("schema", None)
        if schema is not None:
            schema = VideoLabelsSchema.from_dict(schema)

        return cls(
            filename=filename, metadata=metadata, attrs=attrs, frames=frames,
            schema=schema)


class VideoLabelsSchema(Serializable):
    '''A schema for a VideoLabels instance.

    Attributes:
        attrs: an AttributeContainerSchema describing the video attributes of
            the video
        frames: an AttributeContainerSchema describing the frame attributes
                of the video
        objects: a dictionary mapping object labels to AttributeContainerSchema
            instances describing the object attributes of each object class
    '''

    def __init__(self, attrs=None, frames=None, objects=None):
        '''Creates a VideoLabelsSchema instance.

        Args:
            attrs: an AttributeContainerSchema describing the video attributes
                of the video
            frames: an AttributeContainerSchema describing the frame attributes
                of the video
            objects: a dictionary mapping object labels to
                AttributeContainerSchema instances describing the object
                attributes of each object class
        '''
        self.attrs = attrs or AttributeContainerSchema()
        self.frames = frames or AttributeContainerSchema()
        self.objects = defaultdict(lambda: AttributeContainerSchema())
        if objects is not None:
            self.objects.update(objects)

    def has_video_attribute(self, video_attr_name):
        '''Returns True/False if the schema has a video attribute with the
        given name.
        '''
        return self.attrs.has_attribute(video_attr_name)

    def get_video_attribute_class(self, video_attr_name):
        '''Gets the Attribute class for the video attribute with the given
        name.
        '''
        return self.attrs.get_attribute_class(video_attr_name)

    def has_frame_attribute(self, frame_attr_name):
        '''Returns True/False if the schema has a frame attribute with the
        given name.
        '''
        return self.frames.has_attribute(frame_attr_name)

    def get_frame_attribute_class(self, frame_attr_name):
        '''Gets the Attribute class for the frame attribute with the given
        name.
        '''
        return self.frames.get_attribute_class(frame_attr_name)

    def has_object_label(self, label):
        '''Returns True/False if the schema has an object with the given
        label.
        '''
        return label in self.objects

    def has_object_attribute(self, label, obj_attr_name):
        '''Returns True/False if the schema has an object attribute of the
        given name for object with the given label.
        '''
        if not self.has_object_label(label):
            return False
        return self.objects[label].has_attribute(obj_attr_name)

    def get_object_attribute_class(self, label, obj_attr_name):
        '''Gets the Attribute class for the attribute of the given name for
        the object with the given label.
        '''
        self.validate_object_label(label)
        return self.objects[label].get_attribute_class(obj_attr_name)

    def add_video_attribute(self, video_attr):
        '''Incorporates the given video attribute into the schema.

        Args:
            video_attr: an Attribute
        '''
        self.attrs.add_attribute(video_attr)

    def add_video_attributes(self, video_attrs):
        '''Incorporates the given video attributes into the schema.

        Args:
            video_attrs: an AttributeContainer of video attributes
        '''
        self.attrs.add_attributes(video_attrs)

    def add_frame_attribute(self, frame_attr):
        '''Incorporates the given frame attribute into the schema.

        Args:
            frame_attr: an Attribute
        '''
        self.frames.add_attribute(frame_attr)

    def add_frame_attributes(self, frame_attrs):
        '''Incorporates the given frame attributes into the schema.

        Args:
            frame_attrs: an AttributeContainer of frame attributes
        '''
        self.frames.add_attributes(frame_attrs)

    def add_object_label(self, label):
        '''Incorporates the given object label into the schema.'''
        self.objects[label]  # adds key to defaultdict

    def add_object_attribute(self, label, obj_attr):
        '''Incorporates the Attribute for the object with the given label
        into the schema.
        '''
        self.objects[label].add_attribute(obj_attr)

    def add_object_attributes(self, label, obj_attrs):
        '''Incorporates the AttributeContainer for the object with the given
        label into the schema.
        '''
        self.objects[label].add_attributes(obj_attrs)

    def merge_schema(self, schema):
        '''Merges the given VideoLabelsSchema into this schema.'''
        self.attrs.merge_schema(schema.attrs)
        self.frames.merge_schema(schema.frames)
        for k, v in iteritems(schema.objects):
            self.objects[k].merge_schema(v)

    def is_valid_video_attribute(self, video_attr):
        '''Returns True/False if the video attribute is compliant with the
        schema.
        '''
        try:
            self.validate_video_attribute(video_attr)
            return True
        except:
            return False

    def is_valid_frame_attribute(self, frame_attr):
        '''Returns True/False if the frame attribute is compliant with the
        schema.
        '''
        try:
            self.validate_frame_attribute(frame_attr)
            return True
        except:
            return False

    def is_valid_object_label(self, label):
        '''Returns True/False if the object label is compliant with the
        schema.
        '''
        try:
            self.validate_object_label(label)
            return True
        except:
            return False

    def is_valid_object_attribute(self, label, obj_attr):
        '''Returns True/False if the object attribute for the given label is
        compliant with the schema.
        '''
        try:
            self.validate_object_attribute(label, obj_attr)
            return True
        except:
            return False

    def is_valid_object(self, obj):
        '''Returns True/False if the DetectedObject is compliant with the
        schema.
        '''
        try:
            self.validate_object(obj)
            return True
        except:
            return False

    def validate_video_attribute(self, video_attr):
        '''Validates that the video attribute is compliant with the schema.

        Args:
            video_attr: an Attribute

        Raises:
            AttributeContainerSchemaError: if the attribute violates the schema
        '''
        self.attrs.validate_attribute(video_attr)

    def validate_frame_attribute(self, frame_attr):
        '''Validates that the frame attribute is compliant with the schema.

        Args:
            frame_attr: an Attribute

        Raises:
            AttributeContainerSchemaError: if the attribute violates the schema
        '''
        self.frames.validate_attribute(frame_attr)

    def validate_object_label(self, label):
        '''Validates that the object label is compliant with the schema.

        Args:
            label: an object label

        Raises:
            VideoLabelsSchemaError: if the object label violates the schema
        '''
        if label not in self.objects:
            raise VideoLabelsSchemaError(
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
            VideoLabelsSchemaError: if the object's label violates the schema
            AttributeContainerSchemaError: if any attributes of the
                DetectedObject violate the schema
        '''
        self.validate_object_label(obj.label)
        if obj.has_attributes:
            for obj_attr in obj.attrs:
                self.validate_object_attribute(obj.label, obj_attr)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.'''
        return ["attrs", "frames", "objects"]

    @classmethod
    def build_active_schema_for_frame(cls, frame_labels):
        '''Builds a VideoLabelsSchema that describes the active schema of
        the given VideoFrameLabels.
        '''
        schema = cls()
        schema.add_frame_attributes(frame_labels.attrs)
        for obj in frame_labels.objects:
            if obj.has_attributes:
                schema.add_object_attributes(obj.label, obj.attrs)
            else:
                schema.add_object_label(obj.label)
        return schema

    @classmethod
    def build_active_schema(cls, video_labels):
        '''Builds a VideoLabelsSchema that describes the active schema of the
        given VideoLabels.
        '''
        schema = cls()
        schema.add_video_attributes(video_labels.attrs)
        for frame_labels in itervalues(video_labels.frames):
            schema.merge_schema(
                VideoLabelsSchema.build_active_schema_for_frame(frame_labels))
        return schema

    @classmethod
    def from_dict(cls, d):
        '''Constructs a VideoLabelsSchema from a JSON dictionary.'''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = AttributeContainerSchema.from_dict(attrs)

        frames = d.get("frames", None)
        if frames is not None:
            frames = AttributeContainerSchema.from_dict(frames)

        objects = d.get("objects", None)
        if objects is not None:
            objects = {
                k: AttributeContainerSchema.from_dict(v)
                for k, v in iteritems(objects)
            }

        return cls(attrs=attrs, frames=frames, objects=objects)


class VideoLabelsSchemaError(Exception):
    '''Error raised when a VideoLabelsSchema is violated.'''
    pass


class VideoSetLabels(Set):
    '''Class encapsulating labels for a set of videos.

    VideoSetLabels support item indexing by the `filename` of the VideoLabels
    instances in the set.

    VideoSetLabels instances behave like defaultdicts: new VideoLabels
    instances are automatically created if a non-existent filename is accessed.

    VideoLabels without filenames may be added to the set, but they cannot be
    accessed by `filename`-based lookup.

    Attributes:
        videos: an OrderedDict of VideoLabels with filenames as keys
        schema: a VideoLabelsSchema describing the schema of the labels
    '''

    _ELE_ATTR = "videos"
    _ELE_KEY_ATTR = "filename"
    _ELE_CLS = VideoLabels
    _ELE_CLS_FIELD = "_LABELS_CLS"

    def __init__(self, videos=None, schema=None):
        '''Constructs a VideoSetLabels instance.

        Args:
            videos: an optional iterable of VideoLabels. By default, an empty
                set is created
            schema: an optional VideoLabelsSchema to enforce on the object.
                By default, no schema is enforced
        '''
        self.schema = schema
        super(VideoSetLabels, self).__init__(videos=videos)

    def __getitem__(self, filename):
        if filename not in self:
            video_labels = VideoLabels(filename=filename)
            self.add(video_labels)

        return super(VideoSetLabels, self).__getitem__(filename)

    def __setitem__(self, filename, video_labels):
        if self.has_schema:
            self._apply_schema_to_video(video_labels)

        super(VideoSetLabels, self).__setitem__(filename, video_labels)

    @property
    def has_schema(self):
        '''Returns True/False whether the container has an enforced schema.'''
        return self.schema is not None

    def empty(self):
        '''Returns an empty copy of the VideoSetLabels.

        The schema of the set is preserved, if applicable.

        Returns:
            an empty VideoSetLabels
        '''
        return self.__class__(schema=self.schema)

    def add(self, video_labels):
        '''Adds the VideoLabels to the set.

        Args:
            video_labels: a VideoLabels instance
        '''
        if self.has_schema:
            self._apply_schema_to_video(video_labels)
        super(VideoSetLabels, self).add(video_labels)

    def clear_frame_attributes(self):
        '''Removes all frame attributes from all VideoLabels in the set.'''
        for video_labels in self:
            video_labels.clear_frame_attributes()

    def clear_objects(self):
        '''Removes all objects from all VideoLabels in the set.'''
        for video_labels in self:
            video_labels.clear_objects()

    def get_filenames(self):
        '''Returns the set of filenames of VideoLabels in the set.

        Returns:
            the set of filenames
        '''
        return set(vl.filename for vl in self if vl.filename)

    def get_schema(self):
        '''Gets the schema for the set, or None if no schema is enforced.'''
        return self.schema

    def get_active_schema(self):
        '''Returns a VideoLabelsSchema describing the active schema of the
        set.
        '''
        schema = VideoLabelsSchema()
        for video_labels in self:
            schema.merge_schema(
                VideoLabelsSchema.build_active_schema(video_labels))
        return schema

    def set_schema(self, schema, filter_by_schema=False):
        '''Sets the schema to the given VideoLabelsSchema.

        Args:
            schema: the VideoLabelsSchema to use
            filter_by_schema: whether to filter any invalid objects/attributes
                from the set after changing the schema. By default, this is
                False

        Raises:
            VideoLabelsSchemaError: if `filter_by_schema` was False and the
                set contains attributes/objects that are not compliant with the
                schema
        '''
        self.schema = schema

        if filter_by_schema and self.has_schema:
            self.filter_by_schema(schema)

        self._apply_schema()

    def filter_by_schema(self, schema):
        '''Removes objects/attributes from the VideoLabels in the set that are
        not compliant with the given schema.

        Args:
            schema: a VideoLabelsSchema
        '''
        for video_labels in self:
            video_labels.filter_by_schema(schema)

    def freeze_schema(self):
        '''Sets the schema for the set to the current active schema.'''
        self.set_schema(self.get_active_schema())

    def remove_schema(self):
        '''Removes the schema from the set.'''
        self.schema = None
        self._apply_schema()

    def sort_by_filename(self, reverse=False):
        '''Sorts the VideoLabels in this instance by filename.

        VideoLabels without filenames are always put at the end of the set.

        Args:
            reverse: whether to sort in reverse order. By default, this is
                False
        '''
        self.sort_by("filename", reverse=reverse)

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.'''
        _attrs = super(VideoSetLabels, self).attributes()
        if self.has_schema:
            return ["schema"] + _attrs
        return _attrs

    def _apply_schema_to_video(self, video_labels):
        if self.has_schema:
            video_labels.set_schema(self.get_schema())
        else:
            video_labels.remove_schema()

    def _apply_schema(self):
        for video_labels in self:
            self._apply_schema_to_video(video_labels)

    @classmethod
    def from_video_labels_patt(cls, video_labels_patt):
        '''Creates an instance of `cls` from a pattern of `_ELE_CLS` files.

        Args:
             video_labels_patt: a pattern with one or more numeric sequences:
                example: "/path/to/labels/%05d.json"

        Returns:
            a `cls` instance

        '''
        image_set_labels = cls()
        for labels_path in etau.get_pattern_matches(video_labels_patt):
            image_set_labels.add(cls._ELE_CLS.from_json(labels_path))
        return image_set_labels

    @classmethod
    def from_dict(cls, d):
        '''Constructs a VideoSetLabels from a JSON dictionary.'''
        schema = d.pop("schema", None)
        if schema is not None:
            schema = VideoLabelsSchema.from_dict(schema)

        return super(VideoSetLabels, cls).from_dict(d, schema=schema)


class BigVideoSetLabels(VideoSetLabels, BigSet):
    '''A BigSet of VideoLabels.

    Behaves identically to VideoSetLabels except that each VideoLabels is
    stored on disk.

    BigVideoSetLabels store a `backing_dir` attribute that specifies the path
    on disk to the serialized elements. If a backing directory is explicitly
    provided, the directory will be maintained after the BigVideoSetLabels
    object is deleted; if no backing directory is specified, a temporary
    backing directory is used and is deleted when the BigVideoSetLabels
    instance is garbage collected.

    Attributes:
        videos: an OrderedDict whose keys are filenames and whose values are
            uuids for locating VideoLabels on disk
        schema: a VideoLabelsSchema describing the schema of the labels
        backing_dir: the backing directory in which the VideoLabels
            are/will be stored
    '''

    def __init__(self, videos=None, schema=None, backing_dir=None):
        '''Creates a BigVideoSetLabels instance.

        Args:
            videos: an optional dictionary or list of (key, uuid) tuples for
                elements in the set
            schema: an optional VideoLabelsSchema to enforce on the object.
                By default, no schema is enforced
            backing_dir: an optional backing directory in which the VideoLabels
                are/will be stored. If omitted, a temporary backing directory
                is used
        '''
        self.schema = schema
        BigSet.__init__(self, backing_dir=backing_dir, videos=videos)

    def empty_set(self):
        '''Returns an empty in-memory VideoSetLabels version of this
        BigVideoSetLabels.

        Returns:
            an empty VideoSetLabels
        '''
        return VideoSetLabels(schema=self.schema)

    def filter_by_schema(self, schema):
        '''Removes objects/attributes from the VideoLabels in the set that are
        not compliant with the given schema.

        Args:
            schema: a VideoLabelsSchema
        '''
        for key in self.keys():
            video_labels = self[key]
            video_labels.filter_by_schema(schema)
            self[key] = video_labels

    def _apply_schema(self):
        for key in self.keys():
            video_labels = self[key]
            self._apply_schema_to_video(video_labels)
            self[key] = video_labels


class VideoStreamInfo(Serializable):
    '''Class encapsulating the stream info for a video.'''

    def __init__(self, stream_info, format_info=None):
        '''Constructs a VideoStreamInfo instance.

        Args:
            stream_info: a dictionary of video stream info
            format_info: an optional dictionary of video format info. By
                default, no format info is stored
        '''
        self.stream_info = stream_info
        self.format_info = format_info or {}

    @property
    def encoding_str(self):
        '''The video encoding string, or "" if it code not be found.'''
        _encoding_str = str(self.stream_info.get("codec_tag_string", ""))
        if _encoding_str is None:
            logger.warning("Unable to determine encoding string")
        return _encoding_str

    @property
    def frame_size(self):
        '''The (width, height) of each frame.

        Raises:
            VideoStreamInfoError if the frame size could not be determined
        '''
        try:
            return (
                int(self.stream_info["width"]),
                int(self.stream_info["height"]),
            )
        except KeyError:
            raise VideoStreamInfoError(
                "Unable to determine frame size of the video")

    @property
    def aspect_ratio(self):
        '''The aspect ratio of the video.

        Raises a VideoStreamInfoError if the frame size could not be
        determined.
        '''
        width, height = self.frame_size
        return width * 1.0 / height

    @property
    def frame_rate(self):
        '''The frame rate of the video.

        Raises:
            VideoStreamInfoError if the frame rate could not be determined
        '''
        try:
            try:
                num, denom = self.stream_info["avg_frame_rate"].split("/")
                return float(num) / float(denom)
            except ZeroDivisionError:
                num, denom = self.stream_info["r_frame_rate"].split("/")
                return float(num) / float(denom)
        except (KeyError, ValueError):
            raise VideoStreamInfoError(
                "Unable to determine frame rate of the video")

    @property
    def total_frame_count(self):
        '''The total number of frames in the video, or -1 if it could not be
        determined.
        '''
        try:
            # try `nb_frames`
            return int(self.stream_info["nb_frames"])
        except KeyError:
            pass

        try:
            # try `duration` x `frame rate`
            return int(round(self.duration * self.frame_rate))
        except VideoStreamInfoError:
            pass

        try:
            #
            # Fallback to `duration_ts` as a last resort. This will not be
            # accurate for videos with `time_base` != 1, but the assumption is
            # that one of the preceeding methods will have already worked for
            # videos. This is here as a last resort for sequences of images,
            # where `duration_ts` seems to directly contain the number of
            # frames.
            #
            return int(self.stream_info["duration_ts"])
        except KeyError:
            pass

        logger.warning("Unable to determine total frame count; returning -1")
        return -1

    @property
    def duration(self):
        '''The duration of the video, in seconds, or -1 if it could not be
        determined.
        '''
        try:
            # try `duration`
            return float(self.stream_info["duration"])
        except KeyError:
            pass

        try:
            # try `duration_ts` x `time_base`
            duration_ts = float(self.stream_info["duration_ts"])
            num, denom = self.stream_info["time_base"].split("/")
            return duration_ts * float(num) / float(denom)
        except KeyError:
            pass

        try:
            # try `duration` from format info
            return float(self.format_info["duration"])
        except KeyError:
            pass

        logger.warning("Unable to determine duration; returning -1")
        return -1

    def get_raw_value(self, key):
        '''Gets a value from the raw stream info dictionary.

        Args:
            key: the key to lookup in the stream info dictionary

        Returns:
            the value for the given key

        Raises:
            KeyError: if the key was not found in the stream info dictionary
        '''
        return self.stream_info[key]

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.'''
        return self.custom_attributes(dynamic=True)

    @classmethod
    def build_for(cls, inpath):
        '''Builds a VideoStreamInfo instance for the given video.

        Args:
            inpath: the path to the input video

        Returns:
            a VideoStreamInfo instance
        '''
        stream_info, format_info = _get_stream_info(inpath)
        return cls(stream_info, format_info=format_info)

    @classmethod
    def from_dict(cls, d):
        '''Constructs a VideoStreamInfo from a JSON dictionary.'''
        stream_info = d["stream_info"]
        format_info = d.get("format_info", None)
        return cls(stream_info, format_info=format_info)


class VideoStreamInfoError(Exception):
    '''Exception raised when an invalid video stream info dictionary is
    encountered.
    '''
    pass


def _get_stream_info(inpath):
    # Get stream info via ffprobe
    ffprobe = FFprobe(opts=[
        "-show_format",              # get format info
        "-show_streams",             # get stream info
        "-print_format", "json",     # return in JSON format
    ])
    out = ffprobe.run(inpath, decode=True)
    info = load_json(out)

    # Get format info
    format_info = info["format"]

    # Get stream info
    video_streams = [s for s in info["streams"] if s["codec_type"] == "video"]
    num_video_streams = len(video_streams)
    if num_video_streams == 1:
        stream_info = video_streams[0]
    elif num_video_streams == 0:
        logger.warning("No video stream found; defaulting to first stream")
        stream_info = info["streams"][0]
    else:
        logger.warning("Found multiple video streams; using first stream")
        stream_info = video_streams[0]

    return stream_info, format_info


def get_encoding_str(inpath):
    '''Get the encoding string of the input video.

    Args:
        inpath: video path

    Returns:
        the encoding string
    '''
    return VideoStreamInfo.build_for(inpath).encoding_str


def get_frame_rate(inpath):
    '''Get the frame rate of the input video.

    Args:
        inpath: video path

    Returns:
        the frame rate
    '''
    return VideoStreamInfo.build_for(inpath).frame_rate


def get_frame_size(inpath):
    '''Get the frame (width, height) of the input video.

    Args:
        inpath: video path

    Returns:
        the (width, height) of the video frames
    '''
    return VideoStreamInfo.build_for(inpath).frame_size


def get_frame_count(inpath):
    '''Get the number of frames in the input video.

    Args:
        inpath: video path

    Returns:
        the frame count, or -1 if it could not be determined
    '''
    return VideoStreamInfo.build_for(inpath).total_frame_count


def get_duration(inpath):
    '''Gets the duration of the video, in seconds.

    Args:
        inpath: video path

    Returns:
        the duration of the video, in seconds, or -1 if it could not be
            determined
    '''
    return VideoStreamInfo.build_for(inpath).duration


def get_raw_frame_number(raw_frame_rate, raw_frame_count, fps, sampled_frame):
    '''Get the raw frame number corresponding to the given sampled frame
    number.

    This function assumes that the sampling was performed using the command:
    ```
    FFmpeg(fps=fps).run(raw_video_path, ...)
    ```

    Args:
        raw_frame_rate: the frame rate of the raw video
        raw_frame_count: the number of frames in the raw video
        fps: the sampling rate that was used
        sampled_frame: the sampled frame number

    Returns:
        raw_frame: the raw frame number from the input video corresponding to
            the given sampled frame number
    '''
    delta = raw_frame_rate / (1.0 * fps)
    raw_frame = np.minimum(
        np.ceil(delta * (sampled_frame - 0.5)), raw_frame_count)
    return int(raw_frame)


def extract_clip(
        video_path, output_path, start_time=None, duration=None, fast=False):
    '''Extracts the specified clip from the video.

    When fast=False, the following ffmpeg command is used:
    ```
    # Slow, accurate option
    ffmpeg -ss <start_time> -i <video_path> -t <duration> <output_path>
    ```

    When fast is True, the following two-step ffmpeg process is used:
    ```
    # Faster, less accurate option
    ffmpeg -ss <start_time> -i <video_path> -t <duration> -c copy <tmp_path>
    ffmpeg -i <tmp_path> <output_path>
    ```

    Args:
        video_path: the path to a video
        output_path: the path to write the extracted video clip
        start_time: the start timestamp, which can either be a float value of
            seconds or a string in "HH:MM:SS.XXX" format. If omitted, the
            beginning of the video is used
        duration: the clip duration, which can either be a float value of
            seconds or a string in "HH:MM:SS.XXX" format. If omitted, the clip
            extends to the end of the video
        fast: whether to use a faster-but-potentially-less-accurate strategy to
            extract the clip. By default, the slow accurate strategy is used
    '''
    #
    # @todo is this accurate? should we use VideoProcessor to ensure that the
    # frames of the clip will be exactly the same as those encountered via
    # other clip-based methods in ETA?
    #
    in_opts = ["-vsync", "0"]
    if start_time is not None:
        if not isinstance(start_time, six.string_types):
            start_time = "%.3f" % start_time
        in_opts.extend(["-ss", start_time])

    out_opts = ["-vsync", "0"]
    if duration is not None:
        if not isinstance(duration, six.string_types):
            duration = "%.3f" % duration
        out_opts.extend(["-t", duration])

    if not fast:
        # Extract clip carefully and accurately by decoding every frame
        ffmpeg = FFmpeg(in_opts=in_opts, out_opts=out_opts)
        ffmpeg.run(video_path, output_path)
        return

    with etau.TempDir() as d:
        tmp_path = os.path.join(d, os.path.basename(output_path))

        # Extract clip as accurately and quickly as possible by only touching
        # key frames. May lave blank frames in the video
        out_opts.extend(["-c", "copy"])
        ffmpeg = FFmpeg(in_opts=in_opts, out_opts=out_opts)
        ffmpeg.run(video_path, tmp_path)

        # Clean up fast output by re-encoding the extracted clip
        # Note that this may not exactly correspond to the slow, accurate
        # implementation above
        ffmpeg = FFmpeg(out_opts=["-vsync", "0"])
        ffmpeg.run(tmp_path, output_path)


def _make_ffmpeg_select_arg(frames):
    ss = "+".join(["eq(n\,%d)" % (f - 1) for f in frames])
    return "select='%s'" % ss


def sample_select_frames(
        video_path, frames, output_patt=None, size=None, fast=False):
    '''Samples the specified frames of the video.

    When `fast=False`, this implementation uses `VideoProcessor`. When
    `fast=True`, this implementation uses ffmpeg's `-vf select` option.

    Args:
        video_path: the path to a video
        frames: a sorted list of frame numbers to sample
        output_patt: an optional output pattern like "/path/to/frames-%d.png"
            specifying where to write the sampled frames. If omitted, the
            frames are instead returned in an in-memory list
        size: an optional (width, height) to resize the sampled frames. By
            default, the native dimensions of the frames are used
        fast: whether to use a native ffmpeg method to perform the extraction.
            While faster, this may be inconsistent with other video processing
            methods in ETA. By default, this is False

    Returns:
        a list of the sampled frames if output_patt is None, and None otherwise
    '''
    # Parse parameters
    resize_images = size is not None

    #
    # Revert to `fast=False` if necessary
    #
    # As per https://stackoverflow.com/questions/29801975, one cannot pass an
    # argument of length > 131072 to subprocess. So, we have to make sure the
    # user isn't requesting too many frames to handle
    #
    if fast:
        select_arg_str = _make_ffmpeg_select_arg(frames)
        if len(select_arg_str) > 131072:
            logger.info(
                "Number of frames (%d) requested too large; reverting to "
                "`fast=False`", len(frames))
            fast = False

    #
    # In "slow mode", we sample the requested frames via VideoProcessor
    #

    if not fast:
        if output_patt:
            # Sample frames to disk via VideoProcessor
            p = VideoProcessor(
                video_path, frames=frames, out_images_path=output_patt)
            with p:
                for img in p:
                    if resize_images:
                        img = etai.resize(img, *size)
                    p.write(img)
            return None

        # Sample frames in memory via FFmpegVideoReader
        with FFmpegVideoReader(video_path, frames=frames) as r:
            if resize_images:
                return [etai.resize(img, *size) for img in r]

            return [img for img in r]

    #
    # In "fast mode", we use ffmpeg's native  `-vf select` option to sample
    # the requested frames
    #

    # If reading into memory, use `png` to ensure lossless-ness
    ext = os.path.splitext(output_patt)[1] if output_patt else ".png"

    with etau.TempDir() as d:
        # Sample frames to disk temporarily
        tmp_patt = os.path.join(d, "frame-%d" + ext)
        ffmpeg = FFmpeg(
            size=size, out_opts=["-vf", select_arg_str, "-vsync", "0"])
        ffmpeg.run(video_path, tmp_patt)

        if output_patt is not None:
            # Move frames into place with correct output names
            for idx, fn in enumerate(frames, 1):
                etau.move_file(tmp_patt % idx, output_patt % fn)
            return

        # Read frames into memory
        imgs = []
        for idx in range(1, len(frames) + 1):
            imgs.append(etai.read(tmp_patt % idx))

        return imgs


def sample_first_frames(imgs_or_video_path, k, stride=1, size=None):
    '''Samples the first k frames in a video.

    Args:
        imgs_or_video_path: can be either the path to the input video or an
            array of frames of size [num_frames, height, width, num_channels]
        k: number of frames to extract
        stride: number of frames to be skipped in between. By default, a
            contiguous array of frames in extracted
        size: an optional (width, height) to resize the sampled frames. By
            default, the native dimensions of the frames are used

    Returns:
        a numpy array of size [k, height, width, num_channels]
    '''
    # Read frames ...
    if isinstance(imgs_or_video_path, six.string_types):
        # ... from disk
        video_path = imgs_or_video_path
        frames = [i for i in range(1, stride * k + 1, stride)]
        with FFmpegVideoReader(video_path, frames=frames) as vr:
            imgs_out = [img for img in vr]
    else:
        # ... from tensor
        imgs = imgs_or_video_path
        imgs_out = imgs[:(k * stride):stride]

    # Duplicate last frame if necessary
    if k > len(imgs_out):
        num_repeats = k - len(imgs_out)
        imgs_out = np.asarray(imgs_out)
        imgs_out = np.concatenate((
            imgs_out, np.repeat(imgs_out[-1][np.newaxis], num_repeats, axis=0)
        ))

    # Resize frames, if necessary
    if size is not None:
        imgs_out = [etai.resize(img, *size) for img in imgs_out]

    return np.array(imgs_out)


def uniformly_sample_frames(imgs_or_video_path, k, size=None):
    '''Uniformly samples k frames from the video, always including the first
    and last frames.

    If k is larger than the number of frames in the video, duplicate frames
    will be included as necessary so that k frames are always returned.

    Args:
        imgs_or_video_path: can be either the path to the input video or an
            array of frames of size [num_frames, height, width, num_channels]
        k: the number of frames to extract
        size: an optional (width, height) to resize the sampled frames. By
            default, the native dimensions of the frames are used

    Returns:
        a numpy array of size [k, height, width, num_channels]
    '''
    is_video = isinstance(imgs_or_video_path, six.string_types)
    if is_video:
        video_path = imgs_or_video_path
    else:
        imgs = imgs_or_video_path

    # Compute 1-based frames
    num_frames = get_frame_count(video_path) if is_video else len(imgs)
    frames = [int(round(i)) for i in np.linspace(1, min(num_frames, k), k)]

    # Read frames ...
    if is_video:
        # ... from disk
        with FFmpegVideoReader(video_path, frames=frames) as vr:
            imgs_out = [img for img in vr]
    else:
        # ... from tensor
        imgs_out = [imgs[f - 1] for f in frames]

    # Resize frames, if necessary
    if size is not None:
        imgs_out = [etai.resize(img, *size) for img in imgs_out]

    return np.array(imgs_out)


def sliding_window_sample_frames(imgs_or_video_path, k, stride, size=None):
    '''Samples clips from the video using a sliding window of the given
    length and stride.

    If k is larger than the number of frames in the video, duplicate frames
    will be included as necessary so that one window of size k can be returned.

    Args:
        imgs_or_video_path: can be either the path to the input video or an
            array of frames of size [num_frames, height, width, num_channels]
        k: the size of each window
        stride: the stride for sliding window
        size: an optional (width, height) to resize the sampled frames. By
            default, the native dimensions of the frames are used

    Returns:
        a numpy array of size [XXXX, k, height, width, num_channels]
    '''
    is_video = isinstance(imgs_or_video_path, six.string_types)
    if is_video:
        video_path = imgs_or_video_path
    else:
        imgs = imgs_or_video_path

    # Determine clip indices
    num_frames = get_frame_count(video_path) if is_video else len(imgs)
    if k <= num_frames:
        delta = np.arange(1, k + 1)
        offsets = np.array(list(range(0, num_frames + 1 - k, stride)))
        clip_inds = offsets[:, np.newaxis] + delta[np.newaxis, :]
    else:
        # Duplicate last frame as necessary to fill one window of size k
        clip_inds = np.concatenate((
            np.arange(1, num_frames + 1),
            [num_frames] * (k - num_frames)))[np.newaxis]

    # Read frames ...
    imgs_dict = {}
    frames = list(np.unique(clip_inds))
    if is_video:
        # ... from disk
        with FFmpegVideoReader(video_path, frames=frames) as vr:
            for img in vr:
                imgs_dict[vr.frame_number] = img
    else:
        # ... from tensor
        for fn in frames:
            imgs_dict[fn] = imgs[fn - 1]

    # Resize frames, if necessary
    if size is not None:
        imgs_dict = {
            fn: etai.resize(img, *size) for fn, img in iteritems(imgs_dict)
        }

    # Generate clips tensor
    clips = []
    for inds in clip_inds:
        clips.append(np.array([imgs_dict[k] for k in inds]))

    return np.array(clips)


def extract_keyframes(video_path, output_patt=None):
    '''Extracts the keyframes from the video.

    Keyframes are a set of video frames that mark the start of a transition,
    and are faster to extract than an arbitrary frame.

    Args:
        video_path: the path to a video
        output_patt: an optional output pattern like "/path/to/frames-%d.png"
            specifying where to write the sampled frames. If omitted, the
            frames are instead returned in an in-memory list

    Returns:
        a list of the keyframes if output_patt is None, and None otherwise
    '''
    if output_patt:
        # Write frames to disk via VideoProcessor
        p = VideoProcessor(
            video_path, keyframes_only=True, out_images_path=output_patt)
        with p:
            for img in p:
                p.write(img)
        return

    # Load frames into memory via FFmpegVideoReader
    with FFmpegVideoReader(video_path, keyframes_only=True) as r:
        return [img for img in r]


def split_video(
        video_path, output_patt, num_clips=None, clip_duration=None,
        clip_size_bytes=None):
    '''Splits the video into (roughly) equal-sized clips of the specified size.

    Exactly one keyword argument should be provided.

    This implementation uses an `ffmpeg` command of the following form:

    ```
    ffmpeg \
        -i input.mp4 \
        -c copy -segment_time SS.XXX -f segment -reset_timestamps 1 \
        output-%03d.mp4
    ```

    Args:
        video_path: the path to a video
        output_patt: an output pattern like "/path/to/clips-%03d.mp4"
            specifying where to write the output clips
        num_clips: the number of (roughly) equal size clips to break the
            video into
        clip_duration: the (approximate) duration, in seconds, of each clip to
            generate. The last clip may be shorter
        clip_size_bytes: the (approximate) size, in bytes, of each clip to
            generate. The last clip may be smaller
    '''
    #
    # Determine segment time
    #

    metadata = VideoMetadata.build_for(video_path)
    if clip_size_bytes:
        num_clips = metadata.size_bytes / clip_size_bytes

    if num_clips:
        # Round up to nearest second to ensure that an additional small clip
        # is not created at the end
        segment_time = np.ceil(metadata.duration / num_clips)
    elif clip_duration:
        segment_time = clip_duration
    else:
        raise ValueError("One keyword argument must be provided")

    #
    # Perform clipping
    #

    in_opts = []
    out_opts = [
        "-c:v", "copy",
        "-segment_time", "%.3f" % segment_time,
        "-f", "segment",
        "-reset_timestamps", "1",
    ]
    ffmpeg = FFmpeg(in_opts=in_opts, out_opts=out_opts)
    ffmpeg.run(video_path, output_patt)


class VideoProcessor(object):
    '''Class for reading a video and writing a new video frame-by-frame.

    The typical usage is:
    ```
    with VideoProcessor(...) as p:
        for img in p:
            new_img = ... # process img
            p.write(new_img)
    ```
    '''

    def __init__(
            self,
            inpath,
            frames=None,
            keyframes_only=False,
            in_use_ffmpeg=True,
            out_use_ffmpeg=True,
            out_images_path=None,
            out_video_path=None,
            out_clips_path=None,
            out_fps=None,
            out_size=None,
            out_opts=None):
        '''Creates a VideoProcessor instance.

        Args:
            inpath: path to the input video. Passed directly to a VideoReader
            frames: an optional range(s) of frames to process. This argument
                is passed directly to VideoReader
            keyframes_only: whether to only extract keyframes when reading the
                video. Can only be set to True when `in_use_ffmpeg=True`. When
                this is True, `frames` is interpreted as keyframe numbers
            in_use_ffmpeg: whether to use FFmpegVideoReader to read input
                videos rather than OpenCVVideoReader
            out_use_ffmpeg: whether to use FFmpegVideoWriter to write output
                videos rather than OpenCVVideoWriter
            out_images_path: a path like "/path/to/frames/%05d.png" with one
                placeholder that specifies where to save frames as individual
                images when the write() method is called. When out_images_path
                is None or "", no images are written
            out_video_path: a path like "/path/to/video.mp4" that specifies
                where to save a single output video that contains all of the
                frames passed to the write() method concatenated together,
                regardless of any potential frame range gaps. When
                out_video_path is None or "", no video is written
            out_clips_path: a path like "/path/to/video/%05d-%05d.mp4" with two
                placeholders that specifies where to save output video clips
                for each frame range when the write() method is called. When
                out_clips_path is None or "", no videos are written
            out_fps: a frame rate for the output video, if any. If the input
                source is a video and fps is None, the same frame rate is used
            out_size: the frame size for the output video, if any. If out_size
                is None, the input frame size is assumed
            out_opts: a list of output video options for FFmpeg. Passed
                directly to FFmpegVideoWriter. Only applicable when
                out_use_ffmpeg = True

        Raises:
            VideoProcessorError: if insufficient options are supplied to
                construct a VideoWriter
        '''
        if in_use_ffmpeg:
            self._reader = FFmpegVideoReader(
                inpath, frames=frames, keyframes_only=keyframes_only)
        elif keyframes_only:
            raise VideoProcessorError(
                "Must have `in_use_ffmpeg=True` when `keyframes_only=True`")
        else:
            self._reader = OpenCVVideoReader(inpath, frames=frames)
        self._video_clip_writer = None
        self._video_writer = None
        self._write_images = bool(out_images_path)
        self._write_video = bool(out_video_path)
        self._write_clips = bool(out_clips_path)

        self.inpath = inpath
        self.frames = frames
        self.in_use_ffmpeg = in_use_ffmpeg
        self.out_use_ffmpeg = out_use_ffmpeg
        self.out_images_path = out_images_path
        self.out_video_path = out_video_path
        self.out_clips_path = out_clips_path
        if out_fps is not None and out_fps > 0:
            self.out_fps = out_fps
        elif self._reader.frame_rate > 0:
            self.out_fps = self._reader.frame_rate
        else:
            raise VideoProcessorError(
                "The inferred frame rate '%s' cannot be used. You must " +
                "manually specify a frame rate" % str(self._reader.frame_rate))
        self.out_size = out_size if out_size else self._reader.frame_size
        self.out_opts = out_opts

        if self._write_video:
            self._video_writer = self._new_video_writer(
                self.out_video_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.process()

    @property
    def input_frame_size(self):
        '''The (width, height) of each input frame.'''
        return self._reader.frame_size

    @property
    def output_frame_size(self):
        '''The (width, height) of each output frame.'''
        return self.out_size

    @property
    def input_frame_rate(self):
        '''The input frame rate.'''
        return self._reader.frame_rate

    @property
    def output_frame_rate(self):
        '''The output frame rate.'''
        return self.out_fps

    @property
    def frame_number(self):
        '''The current frame number, or -1 if no frames have been read.'''
        return self._reader.frame_number

    @property
    def frame_range(self):
        '''The (first, last) frames for the current range, or (-1, -1) if no
        frames have been read.
        '''
        return self._reader.frame_range

    @property
    def is_new_frame_range(self):
        '''Whether the current frame is the first in a new range.'''
        return self._reader.is_new_frame_range

    @property
    def total_frame_count(self):
        '''The total number of frames in the video.'''
        return self._reader.total_frame_count

    def process(self):
        '''Returns the next frame.

        Returns:
            img: the frame as a numpy array
        '''
        img = self._reader.read()
        if self._write_clips and self._reader.is_new_frame_range:
            self._reset_video_clip_writer()
        return img

    def write(self, img):
        '''Appends the image to the output VideoWriter(s).

        Args:
            img: an numpy array containing the image
        '''
        if self._write_images:
            etai.write(img, self.out_images_path % self._reader.frame_number)
        if self._write_video:
            self._video_writer.write(img)
        if self._write_clips:
            self._video_clip_writer.write(img)

    def close(self):
        '''Closes the video processor.'''
        self._reader.close()
        if self._video_writer is not None:
            self._video_writer.close()
        if self._video_clip_writer is not None:
            self._video_clip_writer.close()

    def _reset_video_clip_writer(self):
        if self._video_clip_writer is not None:
            self._video_clip_writer.close()

        outpath = self.out_clips_path % self._reader.frame_range
        self._video_clip_writer = self._new_video_writer(outpath)

    def _new_video_writer(self, outpath):
        if self.out_use_ffmpeg:
            return FFmpegVideoWriter(
                outpath, self.out_fps, self.out_size, out_opts=self.out_opts)

        return OpenCVVideoWriter(
            outpath, self.out_fps, self.out_size)


class VideoProcessorError(Exception):
    '''Exception raised when an error occurs within a VideoProcessor.'''
    pass


class VideoReader(object):
    '''Base class for reading videos.'''

    def __init__(self, inpath, frames):
        '''Initializes a VideoReader base instance.

        Args:
            inpath: the input video path
            frames: one of the following quantities specifying a collection of
                frames to process:
                - None (all frames)
                - "*" (all frames)
                - a string like "1-3,6,8-10"
                - a FrameRange or FrameRanges instance
                - an iterable, e.g., [1, 2, 3, 6, 8, 9, 10]. The frames do not
                    need to be in sorted order
        '''
        self.inpath = inpath
        if frames is None:
            self.frames = "1-%d" % self.total_frame_count
            self._ranges = FrameRanges.from_str(self.frames)
        elif isinstance(frames, six.string_types):
            # Frames string
            if frames == "*":
                frames = "1-%d" % self.total_frame_count
            self.frames = frames
            self._ranges = FrameRanges.from_str(frames)
        elif isinstance(frames, (FrameRange, FrameRanges)):
            # FrameRange or FrameRanges
            self._ranges = frames
            self.frames = frames.to_str()
        elif hasattr(frames, "__iter__"):
            # Frames iterable
            self._ranges = FrameRanges.from_iterable(frames)
            self.frames = self._ranges.to_str()
        else:
            raise VideoReaderError("Invalid frames %s" % frames)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.read()

    def close(self):
        '''Closes the VideoReader.'''
        pass

    @property
    def frame_number(self):
        '''The current frame number, or -1 if no frames have been read.'''
        return self._ranges.frame

    @property
    def frame_range(self):
        '''The (first, last) frames for the current range, or (-1, -1) if no
        frames have been read.
        '''
        return self._ranges.frame_range

    @property
    def is_new_frame_range(self):
        '''Whether the current frame is the first in a new range.'''
        return self._ranges.is_new_frame_range

    @property
    def encoding_str(self):
        '''The video encoding string.'''
        raise NotImplementedError("subclass must implement encoding_str")

    @property
    def frame_size(self):
        '''The (width, height) of each frame.'''
        raise NotImplementedError("subclass must implement frame_size")

    @property
    def frame_rate(self):
        '''The frame rate.'''
        raise NotImplementedError("subclass must implement frame_rate")

    @property
    def total_frame_count(self):
        '''The total number of frames in the video.'''
        raise NotImplementedError("subclass must implement total_frame_count")

    def read(self):
        '''Reads the next frame.

        Returns:
            img: the next frame
        '''
        raise NotImplementedError("subclass must implement read()")


class VideoReaderError(Exception):
    '''Exception raised when an error occured while reading a video.'''
    pass


class FFmpegVideoReader(VideoReader):
    '''Class for reading video using ffmpeg.

    The input video can be a standalone video file like "/path/to/video.mp4"
    or a directory of frames like "/path/to/frames/%05d.png". This path is
    passed directly to ffmpeg.

    A frames string like "1-5,10-15" can optionally be passed to only read
    certain frame ranges.

    This class uses 1-based indexing for all frame operations.
    '''

    def __init__(self, inpath, frames=None, keyframes_only=False):
        '''Creates an FFmpegVideoReader instance.

        Args:
            inpath: path to the input video, which can be a standalone video
                file like "/path/to/video.mp4" or a directory of frames like
                "/path/to/frames/%05d.png". This path is passed directly to
                ffmpeg
            frames: one of the following optional quantities specifying a
                collection of frames to process:
                - None (all frames - the default)
                - "*" (all frames)
                - a string like "1-3,6,8-10"
                - a FrameRange or FrameRanges instance
                - an iterable, e.g., [1, 2, 3, 6, 8, 9, 10]. The frames do not
                    need to be in sorted order
            keyframes_only: whether to only read keyframes. By default, this
                is False. When this is True, `frames` is interpreted as
                keyframe numbers
        '''
        # Parse args
        if keyframes_only:
            in_opts = ["-skip_frame", "nokey", "-vsync", "0"]
        else:
            in_opts = None

        self._stream_info = VideoStreamInfo.build_for(inpath)
        self._ffmpeg = FFmpeg(
            in_opts=in_opts,
            out_opts=[
                "-vsync", "0",              # never omit frames
                "-f", 'image2pipe',         # pipe frames to stdout
                "-vcodec", "rawvideo",      # output will be raw video
                "-pix_fmt", "rgb24",        # pixel format
            ],
        )
        self._ffmpeg.run(inpath, "-")
        self._raw_frame = None

        super(FFmpegVideoReader, self).__init__(inpath, frames)

    @property
    def encoding_str(self):
        '''The video encoding string.'''
        return self._stream_info.encoding_str

    @property
    def frame_size(self):
        '''The (width, height) of each frame.'''
        return self._stream_info.frame_size

    @property
    def frame_rate(self):
        '''The frame rate.'''
        return self._stream_info.frame_rate

    @property
    def total_frame_count(self):
        '''The total number of frames in the video, or 0 if it could not be
        determined.
        '''
        return self._stream_info.total_frame_count

    def read(self):
        '''Reads the next frame.

        If any problem is encountered while reading the frame, a warning is
        logged and a StopIteration is raised. This means that FFmpegVideoReader
        will gracefully fail when malformed videos are encountered.

        Returns:
            img: the next frame

        Raises:
            StopIteration: if there are no more frames to process or the next
                frame could not be read or parsed for any reason
        '''
        for _ in range(max(0, self.frame_number), next(self._ranges)):
            if not self._grab():
                logger.warning(
                    "Failed to grab frame %d. Raising StopIteration now",
                    self.frame_number)
                raise StopIteration
        return self._retrieve()

    def close(self):
        '''Closes the video reader.'''
        self._ffmpeg.close()

    def _grab(self):
        try:
            width, height = self.frame_size
            self._raw_frame = self._ffmpeg.read(width * height * 3)
            return True
        except Exception as e:
            logger.warning(e, exc_info=True)
            self._raw_frame = None
            return False

    def _retrieve(self):
        # Stop when ffmpeg returns empty bits. This can happen when the end of
        # the video is reached
        if not self._raw_frame:
            logger.warning(
                "Found empty frame %d. Raising StopIteration now",
                self.frame_number)
            raise StopIteration

        width, height = self.frame_size
        try:
            vec = np.fromstring(self._raw_frame, dtype="uint8")
            return vec.reshape((height, width, 3))
        except ValueError as e:
            # Possible alternative: return all zeros matrix instead
            # return np.zeros((height, width, 3), dtype="uint8")
            logger.warning(e, exc_info=True)
            logger.warning(
                "Unable to parse frame %d; Raising StopIteration now",
                self.frame_number)
            raise StopIteration


class SampledFramesVideoReader(VideoReader):
    '''Class for reading video stored as sampled frames on disk.

    A frames string like "1-5,10-15" can optionally be passed to only read
    certain frame ranges.

    This class uses 1-based indexing for all frame operations.
    '''

    def __init__(self, frames_dir, frames=None):
        '''Creates a SampledFramesVideoReader instance.

        Args:
            frames_dir: the path to a directory of frames, which must be
                parseable by `eta.core.utils.parse_dir_pattern()`
            frames: one of the following optional quantities specifying a
                collection of frames to process:
                - None (all frames - the default)
                - "*" (all frames)
                - a string like "1-3,6,8-10"
                - a FrameRange or FrameRanges instance
                - an iterable, e.g., [1, 2, 3, 6, 8, 9, 10]. The frames do not
                    need to be in sorted order
        '''
        self._frames_dir = None
        self._frames_patt = None
        self._frame_size = None
        self._total_frame_count = None

        all_frames = self._init_for_frames_dir(frames_dir)
        if frames is None or frames == "*":
            frames = all_frames

        super(SampledFramesVideoReader, self).__init__(frames_dir, frames)

    @property
    def encoding_str(self):
        '''The video encoding string.'''
        return None

    @property
    def frame_size(self):
        '''The (width, height) of each frame.'''
        return self._frame_size

    @property
    def frame_rate(self):
        '''The frame rate.'''
        return None

    @property
    def total_frame_count(self):
        '''The total number of frames in the video, or 0 if it could not be
        determined.
        '''
        return self._total_frame_count

    def read(self):
        '''Reads the next frame.

        Returns:
            img: the next frame

        Raises:
            StopIteration: if there are no more frames to process or the next
                frame could not be read or parsed for any reason
        '''
        frame_number = next(self._ranges)
        try:
            return etai.read(self._frames_patt % frame_number)
        except:
            logger.warning(
                "Failed to grab frame %d. Raising StopIteration now",
                frame_number)
            raise StopIteration

    def _init_for_frames_dir(self, frames_dir):
        frames_patt, all_frames = etau.parse_dir_pattern(frames_dir)
        if not all_frames:
            raise ValueError("Found no frames in '%s'" % frames_dir)

        img = etai.read(frames_patt % all_frames[0])

        self._frames_dir = frames_dir
        self._frames_patt = frames_patt
        self._frame_size = etai.to_frame_size(img=img)
        self._total_frame_count = all_frames[-1]

        return all_frames


class OpenCVVideoReader(VideoReader):
    '''Class for reading video using OpenCV.

    The input video can be a standalone video file like "/path/to/video.mp4"
    or a directory of frames like "/path/to/frames/%05d.png". This path is
    passed directly to cv2.VideoCapture. So, for example, if you specify a
    directory of frames, the frame numbering must start from 0-3.

    A frames string like "1-5,10-15" can optionally be passed to only read
    certain frame ranges.

    This class uses 1-based indexing for all frame operations.
    '''

    def __init__(self, inpath, frames=None):
        '''Creates an OpenCVVideoReader instance.

        Args:
            inpath: path to the input video, which can be a standalone video
                file like "/path/to/video.mp4" or a directory of frames like
                "/path/to/frames/%05d.png". This path is passed directly to
                cv2.VideoCapture
            frames: one of the following optional quantities specifying a
                collection of frames to process:
                - None (all frames - the default)
                - "*" (all frames)
                - a string like "1-3,6,8-10"
                - a FrameRange or FrameRanges instance
                - an iterable, e.g., [1, 2, 3, 6, 8, 9, 10]. The frames do not
                    need to be in sorted order

        Raises:
            VideoReaderError: if the input video could not be opened.
        '''
        self._cap = cv2.VideoCapture(inpath)
        if not self._cap.isOpened():
            raise VideoReaderError("Unable to open '%s'" % inpath)

        super(OpenCVVideoReader, self).__init__(inpath, frames)

    @property
    def encoding_str(self):
        '''Return the video encoding string.'''
        try:
            # OpenCV 3
            code = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        except AttributeError:
            # OpenCV 2
            code = int(self._cap.get(cv2.cv.CV_CAP_PROP_FOURCC))
        return FOURCC.int_to_str(code)

    @property
    def frame_size(self):
        '''The (width, height) of each frame.'''
        try:
            # OpenCV 3
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        except AttributeError:
            # OpenCV 2
            return (
                int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)),
            )

    @property
    def frame_rate(self):
        '''The frame rate.'''
        try:
            # OpenCV 3
            return float(self._cap.get(cv2.CAP_PROP_FPS))
        except AttributeError:
            # OpenCV 2
            return float(self._cap.get(cv2.cv.CV_CAP_PROP_FPS))

    @property
    def total_frame_count(self):
        '''The total number of frames in the video.'''
        try:
            # OpenCV 3
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except AttributeError:
            # OpenCV 2
            return int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    def read(self):
        '''Reads the next frame.

        If any problem is encountered while reading the frame, a warning is
        logged and a StopIteration is raised. This means that OpenCVVideoReader
        will gracefully fail when malformed videos are encountered.

        Returns:
            img: the next frame

        Raises:
            StopIteration: if there are no more frames to process or the next
                frame could not be read or parsed for any reason
        '''
        for _ in range(max(0, self.frame_number), next(self._ranges)):
            if not self._grab():
                logger.warning(
                    "Failed to grab frame %d. Raising StopIteration now",
                    self.frame_number)
                raise StopIteration
        return self._retrieve()

    def close(self):
        '''Closes the video reader.'''
        self._cap.release()

    def _grab(self):
        try:
            return self._cap.grab()
        except Exception as e:
            logger.warning(e, exc_info=True)
            return False

    def _retrieve(self):
        try:
            img_bgr = self._cap.retrieve()[1]
            return etai.bgr_to_rgb(img_bgr)
        except Exception as e:
            logger.warning(e, exc_info=True)
            logger.warning(
                "Unable to parse frame %d; Raising StopIteration now",
                self.frame_number)
            raise StopIteration


class VideoWriter(object):
    '''Base class for writing videos.'''

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, img):
        '''Appends the image to the output video.

        Args:
            img: a numpy array
        '''
        raise NotImplementedError("subclass must implement write()")

    def close(self):
        '''Closes the video writer.'''
        raise NotImplementedError("subclass must implement close()")


class VideoWriterError(Exception):
    '''Exception raised when a VideoWriter encounters an error.'''
    pass


class FFmpegVideoWriter(VideoWriter):
    '''Class for writing videos using ffmpeg.'''

    def __init__(self, outpath, fps, size, out_opts=None):
        '''Creates an FFmpegVideoWriter instance.

        Args:
            outpath: the output video path. Existing files are overwritten,
                and the directory is created if necessary
            fps: the frame rate
            size: the (width, height) of each frame
            out_opts: an optional list of output options for FFmpeg
        '''
        self.outpath = outpath
        self.fps = fps
        self.size = size

        self._ffmpeg = FFmpeg(
            in_opts=[
                "-f", "rawvideo",           # input will be raw video
                "-vcodec", "rawvideo",      # input will be raw video
                "-s", "%dx%d" % self.size,  # frame size
                "-pix_fmt", "rgb24",        # pixel format
                "-r", str(self.fps),        # frame rate
            ],
            out_opts=out_opts,
        )
        self._ffmpeg.run("-", self.outpath)

    def write(self, img):
        '''Appends the image to the output video.

        Args:
            img: a numpy array
        '''
        self._ffmpeg.stream(img.tostring())

    def close(self):
        '''Closes the video writer.'''
        self._ffmpeg.close()


class OpenCVVideoWriter(VideoWriter):
    '''Class for writing videos using cv2.VideoWriter.

    Uses the default encoding scheme for the extension of the output path.
    '''

    def __init__(self, outpath, fps, size):
        '''Creates an OpenCVVideoWriter instance.

        Args:
            outpath: the output video path. Existing files are overwritten,
                and the directory is created if necessary
            fps: the frame rate
            size: the (width, height) of each frame

        Raises:
            VideoWriterError: if the writer failed to open
        '''
        self.outpath = outpath
        self.fps = fps
        self.size = size
        self._writer = cv2.VideoWriter()

        etau.ensure_path(self.outpath)
        self._writer.open(self.outpath, -1, self.fps, self.size, True)
        if not self._writer.isOpened():
            raise VideoWriterError("Unable to open '%s'" % self.outpath)

    def write(self, img):
        '''Appends the image to the output video.

        Args:
            img: a numpy array
        '''
        self._writer.write(etai.rgb_to_bgr(img))

    def close(self):
        '''Closes the video writer.'''
        # self._writer.release()  # warns to use a separate thread
        threading.Thread(target=self._writer.release, args=()).start()


class FFprobe(object):
    '''Interface for the ffprobe binary.'''

    DEFAULT_GLOBAL_OPTS = ["-loglevel", "error"]

    def __init__(self, global_opts=None, opts=None):
        '''Creates an FFprobe instance.

        Args:
            global_opts: a list of global options for ffprobe. By default,
                self.DEFAULT_GLOBAL_OPTS is used
            opts: a list of options for ffprobe
        '''
        self._global_opts = global_opts or self.DEFAULT_GLOBAL_OPTS
        self._opts = opts or []

        self._args = None
        self._p = None

    @property
    def cmd(self):
        '''The last executed ffprobe command string, or None if run() has not
        yet been called.
        '''
        return " ".join(self._args) if self._args else None

    def run(self, inpath, decode=False):
        '''Run the ffprobe binary with the specified input path.

        Args:
            inpath: the input path

        Returns:
            out: the stdout from the ffprobe binary
            decode: whether to decode the output bytes into utf-8 strings. By
                default, the raw bytes are returned

        Raises:
            ExecutableNotFoundError: if the ffprobe binary cannot be found
            ExecutableRuntimeError: if the ffprobe binary raises an error
                during execution
        '''
        self._args = (
            ["ffprobe"] +
            self._global_opts +
            self._opts +
            ["-i", inpath]
        )

        try:
            self._p = Popen(
                self._args,
                stdout=PIPE,
                stderr=PIPE,
            )
        except EnvironmentError as e:
            if e.errno == errno.ENOENT:
                raise etau.ExecutableNotFoundError("ffprobe")
            else:
                raise

        out, err = self._p.communicate()
        if self._p.returncode != 0:
            raise etau.ExecutableRuntimeError(self.cmd, err)

        return out.decode("utf-8") if decode else out


class FFmpeg(object):
    '''Interface for the ffmpeg binary.

    Example usages:
        # Convert a video to sampled frames
        ffmpeg = = FFmpeg()
        ffmpeg.run("/path/to/video.mp4", "/path/to/frames/%05d.png")

        # Resize a video
        ffmpeg = FFmpeg(size=(512, -1))
        ffmpeg.run("/path/to/video.mp4", "/path/to/resized.mp4")

        # Change the frame rate of a video
        ffmpeg = FFmpeg(fps=10)
        ffmpeg.run("/path/to/video.mp4", "/path/to/resampled.mp4")
    '''

    DEFAULT_GLOBAL_OPTS = ["-loglevel", "error"]

    DEFAULT_IN_OPTS = ["-vsync", "0"]

    DEFAULT_VIDEO_OUT_OPTS = [
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p", "-vsync", "0", "-an"]

    DEFAULT_IMAGES_OUT_OPTS = ["-vsync", "0"]

    def __init__(
            self,
            fps=None,
            size=None,
            scale=None,
            global_opts=None,
            in_opts=None,
            out_opts=None):
        '''Creates an FFmpeg instance.

        Args:
            fps: an optional output frame rate. By default, the native frame
                rate of the input video is used
            size: an optional output (width, height) for each frame. At most
                one dimension can be -1, in which case the aspect ratio is
                preserved
            scale: an optional positive number by which to scale the input
                video (e.g., 0.5 or 2)
            global_opts: an optional list of global options for ffmpeg. By
                default, self.DEFAULT_GLOBAL_OPTS is used
            in_opts: an optional list of input options for ffmpeg, By default,
                self.DEFAULT_IN_OPTS is used
            out_opts: an optional list of output options for ffmpeg. By
                default, self.DEFAULT_VIDEO_OUT_OPTS is used when the output
                path is a video file and self.DEFAULT_IMAGES_OUT_OPTS is used
                when the output path is an image sequence
        '''
        self.is_input_streaming = False
        self.is_output_streaming = False

        self._filter_opts = self._gen_filter_opts(fps, size, scale)
        self._global_opts = global_opts or self.DEFAULT_GLOBAL_OPTS
        self._in_opts = in_opts
        self._out_opts = out_opts
        self._args = None
        self._p = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def cmd(self):
        '''The last executed ffmpeg command string, or None if run() has not
        yet been called.
        '''
        return " ".join(self._args) if self._args else None

    def run(self, inpath, outpath):
        '''Run the ffmpeg binary with the specified input/outpath paths.

        Args:
            inpath: the input path. If inpath is "-", input streaming mode is
                activated and data can be passed via the stream() method
            outpath: the output path. Existing files are overwritten, and the
                directory is created if needed. If outpath is "-", output
                streaming mode is activated and data can be read via the
                read() method

        Raises:
            ExecutableNotFoundError: if the ffmpeg binary cannot be found
            ExecutableRuntimeError: if the ffmpeg binary raises an error during
                execution
        '''
        self.is_input_streaming = (inpath == "-")
        self.is_output_streaming = (outpath == "-")

        # Input options
        if self._in_opts is None:
            in_opts = self.DEFAULT_IN_OPTS
        else:
            in_opts = self._in_opts

        # Output options
        if self._out_opts is None:
            if is_supported_video_file(outpath):
                out_opts = self.DEFAULT_VIDEO_OUT_OPTS
            else:
                out_opts = self.DEFAULT_IMAGES_OUT_OPTS
        else:
            out_opts = self._out_opts

        # Add filters to output options, if necessary
        out_opts = list(out_opts)
        if self._filter_opts:
            merged = False
            for idx, o in enumerate(out_opts):
                if o.strip() == self._filter_opts[0]:
                    # Merge with existing filter(s)
                    out_opts[idx + 1] += "," + self._filter_opts[1]
                    merged = True
                    break

            if not merged:
                # Append filters
                out_opts += self._filter_opts

        # Construct ffmpeg command
        self._args = (
            ["ffmpeg"] +
            self._global_opts +
            in_opts + ["-i", inpath] +
            out_opts + [outpath]
        )

        if not self.is_output_streaming:
            etau.ensure_path(outpath)

        try:
            logger.debug("Executing '%s'", self.cmd)
            self._p = Popen(self._args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        except EnvironmentError as e:
            if e.errno == errno.ENOENT:
                raise etau.ExecutableNotFoundError("ffmpeg")
            else:
                raise

        # Run non-streaming jobs immediately
        if not (self.is_input_streaming or self.is_output_streaming):
            err = self._p.communicate()[1]
            if self._p.returncode != 0:
                raise etau.ExecutableRuntimeError(self.cmd, err)

    def stream(self, string):
        '''Writes the string to ffmpeg's stdin stream.

        Args:
            string: the string to write

        Raises:
            FFmpegStreamingError: if input streaming mode is not active
        '''
        if not self.is_input_streaming:
            raise FFmpegStreamingError("Not currently input streaming")
        self._p.stdin.write(string)

    def read(self, num_bytes):
        '''Reads the given number of bytes from ffmpeg's stdout stream.

        Args:
            num_bytes: the number of bytes to read

        Returns:
            the bytes

        Raises:
            FFmpegStreamingError: if output streaming mode is not active
        '''
        if not self.is_output_streaming:
            raise FFmpegStreamingError("Not currently output streaming")
        return self._p.stdout.read(num_bytes)

    def close(self):
        '''Closes a streaming ffmpeg program, if necessary.'''
        if self.is_input_streaming or self.is_output_streaming:
            self._p.stdin.close()
            self._p.stdout.close()
            self._p.wait()
        self._p = None
        self.is_input_streaming = False
        self.is_output_streaming = False

    @staticmethod
    def _gen_filter_opts(fps, size, scale):
        filters = []
        if fps is not None and fps > 0:
            filters.append("fps={0}".format(fps))
        if size:
            filters.append("scale={0}:{1}".format(*size))

            #
            # If the aspect ratio is changing, we must manually set SAR/DAR
            # https://stackoverflow.com/questions/34148780/ffmpeg-setsar-value-gets-overriden
            #
            if all(p > 0 for p in size):
                # Force square pixels
                filters.append("setsar=sar=1:1")

                # Force correct display aspect ratio when playing video
                filters.append("setdar=dar={0}:{1}".format(*size))
        elif scale:
            filters.append("scale=iw*{0}:ih*{0}".format(scale))
        return ["-vf", ",".join(filters)] if filters else []


class FFmpegStreamingError(Exception):
    '''Exception raised when an error occurs while operating an FFmpeg instance
    in streaming mode.
    '''
    pass


class FOURCC(object):
    '''Class reprsesenting a FOURCC code.'''

    def __init__(self, _i=None, _s=None):
        '''Creates a FOURCC instance.

        Don't call this directly! Instead, use `from_str ` or `from_int` to
        create a FOURCC instance.

        Args:
            _i: the integer representation of the FOURCC code
            _s: the string representation of the FOURCC code
        '''
        if _i:
            self.int = _i
            self.str = FOURCC.int_to_str(_i)
        elif _s:
            self.int = FOURCC.str_to_int(_s)
            self.str = _s

    @classmethod
    def from_str(cls, s):
        '''Construct a FOURCC instance from a string.

        Args:
            s: the string representation of the FOURCC code

        Returns:
            a FOURCC instance
        '''
        return cls(_s=s)

    @classmethod
    def from_int(cls, i):
        '''Construct a FOURCC instance from an integer.

        Args:
            i: the integer representation of the FOURCC code

        Returns:
            a FOURCC instance
        '''
        return cls(_i=i)

    @staticmethod
    def str_to_int(s):
        '''Returns the integer representation of the given FOURCC string.

        Args:
            s: the string representation of the FOURCC code

        Returns:
            the integer representation of the FOURCC code
        '''
        try:
            # OpenCV 3
            return cv2.VideoWriter_fourcc(*s)
        except AttributeError:
            # OpenCV 2
            return cv2.cv.FOURCC(*s)

    @staticmethod
    def int_to_str(i):
        '''Returns the string representation of the given FOURCC integer.

        Args:
            i: the integer representation of the FOURCC code

        Returns:
            the string representation of the FOURCC code
        '''
        return chr((i & 0x000000FF) >> 0) + \
               chr((i & 0x0000FF00) >> 8) + \
               chr((i & 0x00FF0000) >> 16) + \
               chr((i & 0xFF000000) >> 24)


class FrameRanges(object):
    '''Class representing a monotonically increasing and disjoint series of
    frames.
    '''

    def __init__(self, ranges):
        '''Creates a FrameRanges instance.

        Args:
            ranges: an iterable of (first, last) tuples, which must be disjoint
                and monotonically increasing

        Raises:
            FrameRangesError: if the series is not disjoint and monotonically
                increasing
        '''
        self._idx = 0
        self._ranges = []
        self._started = False

        end = -1
        for first, last in ranges:
            if first <= end:
                raise FrameRangesError(
                    "Expected first:%d > last:%d" % (first, end))

            self._ranges.append(FrameRange(first, last))
            end = last

    def __iter__(self):
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

    @property
    def frame(self):
        '''The current frame number, or -1 if no frames have been read.'''
        if self._started:
            return self._ranges[self._idx].idx

        return -1

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

    def to_str(self):
        '''Returns a string representation of this object.

        Returns:
            a string like "1-3,6,8-10" describing the frame ranges
        '''
        return ",".join([r.to_str() for r in self._ranges])

    @classmethod
    def from_str(cls, frames_str):
        '''Constructs a FrameRanges object from a frames string.

        Args:
            frames_str: a frames string like "1-3,6,8-10"

        Returns:
            a FrameRanges instance

        Raises:
            FrameRangesError: if the frames string is invalid
        '''
        ranges = []
        for r in frames_str.split(","):
            if r:
                fr = FrameRange.from_str(r)
                ranges.append((fr.first, fr.last))

        return cls(ranges)

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
        return cls(_iterable_to_ranges(frames))


class FrameRangesError(Exception):
    '''Exception raised when an invalid FrameRanges is encountered.'''
    pass


class FrameRange(object):
    '''An iterator over a range of frames.'''

    def __init__(self, first, last):
        '''Creates a FrameRange instance.

        Args:
            first: the first frame in the range (inclusive)
            last: the last frame in the range (inclusive)

        Raises:
            FrameRangeError: if last < first
        '''
        if last < first:
            raise FrameRangeError(
                "Expected first:%d <= last:%d" % (first, last))

        self.first = first
        self.last = last
        self.idx = -1

    def __iter__(self):
        return self

    @property
    def is_first_frame(self):
        '''Whether the current frame is first in the range.'''
        return self.idx == self.first

    def __next__(self):
        if self.idx < 0:
            self.idx = self.first
        elif self.idx < self.last:
            self.idx += 1
        else:
            raise StopIteration

        return self.idx

    def to_list(self):
        '''Returns the list of frames in the range.

        Returns:
            a list of frames
        '''
        return list(range(self.first, self.last + 1))

    def to_str(self):
        '''Returns a string representation of the range.

        Returns:
            a string like "1-5"
        '''
        if self.first == self.last:
            return "%d" % self.first

        return "%d-%d" % (self.first, self.last)

    @classmethod
    def from_str(cls, frames_str):
        '''Constructs a FrameRange object from a string.

        Args:
            frames_str: a frames string like "1-5"

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


class FrameRangeError(Exception):
    '''Exception raised when an invalid FrameRange is encountered.'''
    pass


def _iterable_to_ranges(vals):
    if not vals:
        return

    vals = sorted(vals)
    first = last = vals[0]
    for val in vals[1:]:
        if val == last + 1:
            last += 1
        else:
            yield (first, last)
            first = last = val

    yield (first, last)
