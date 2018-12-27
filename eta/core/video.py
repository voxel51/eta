'''
Core tools and data structures for working with videos.

Notes:
    [frame numbers] ETA uses 1-based indexing for all frame numbers

    [image format] ETA stores images exclusively in RGB format. In contrast,
        OpenCV stores its images in BGR format, so all images that are read or
        produced outside of this library must be converted to RGB. This
        conversion can be done via `eta.core.image.bgr_to_rgb()`

Copyright 2017-2018, Voxel51, Inc.
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

from collections import defaultdict
import dateutil.parser
import errno
import json
import logging
import os
from subprocess import Popen, PIPE
import threading

import cv2
import numpy as np
import scipy.interpolate as spi

from eta.core.data import AttributeContainer, AttributeContainerSchema
import eta.core.image as etai
from eta.core.objects import DetectedObjectContainer
from eta.core.serial import Serializable
import eta.core.utils as etau


logger = logging.getLogger(__name__)


SUPPORTED_VIDEO_FILE_FORMATS = [
    ".mp4", ".mpg", ".mpeg", ".avi", ".mov", ".wmv", ".flv", ".mkv", ".m4v"
]


def is_supported_video_file(path):
    '''Determines whether the given file has a supported video type.

    This method does not support videos represented as image sequences (i.e.,
    it will return False for them).

    Args:
        path: the path to a video file

    Returns:
        True/False if the file has a supported video type
    '''
    return os.path.splitext(path)[1] in SUPPORTED_VIDEO_FILE_FORMATS


def is_same_video_file_format(path1, path2):
    '''Determines whether the video files have the same supported format.

    This method does not support videos represented as image sequences (i.e.,
    it will return False for them).

    Args:
        path1: the path to a video file
        path2: the path to a video file
    '''
    return (
        is_supported_video_file(path1) and
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
    alpha = (frame_number - 1) / (total_frame_count - 1)
    return alpha * duration


def timestamp_to_frame_number(timestamp, duration, total_frame_count):
    '''Converts the given timestamp in a video to a frame number.

    Args:
        timestamp: the timestanp (in seconds) of interest
        duration: the length of the video (in seconds)
        total_frame_count: the total number of frames in the video

    Returns:
        the frame number associated with the given timestamp in the video
    '''
    alpha = timestamp / duration
    return 1 + int(round(alpha * (total_frame_count - 1)))


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


class GPSWaypoint(Serializable):
    '''Class encapsulating a GPS waypoint in a video.

    Attributes:
        latitude: the latitude, in degrees
        longitude: the longitude, in degrees
        frame_number: the associated frame number in the video
    '''

    def __init__(self, latitude, longitude, frame_number):
        '''Constructs a GPSWaypoint instance.

        Args:
            latitude: the latitude, in degrees
            longitude: the longitude, in degrees
            frame_number: the associated frame number in the video
        '''
        self.latitude = latitude
        self.longitude = longitude
        self.frame_number = frame_number

    def attributes(self):
        return ["latitude", "longitude", "frame_number"]

    @classmethod
    def from_dict(cls, d):
        '''Constructs a GPSWaypoint from a JSON dictionary.'''
        return cls(
            latitude=d["latitude"],
            longitude=d["longitude"],
            frame_number=d["frame_number"],
        )


class VideoMetadata(Serializable):
    '''Class encapsulating metadata about a video.

    Attributes:
        uuid: a uuid string for the video
        start_time: a datetime describing
        frame_size: the [width, height] of the video frames
        frame_rate: the frame rate of the video
        total_frame_count: the total number of frames in the video
        duration: the duration of the video, in seconds
        size_bytes: the size of the video file on disk, in bytes
        encoding_str: the encoding string for the video
        filepath: the path to the video on disk
        gps_waypoints: an optional list of GPSWaypoints for the video
    '''

    def __init__(
            self, uuid=None, start_time=None, frame_size=None, frame_rate=None,
            total_frame_count=None, duration=None, size_bytes=None,
            encoding_str=None, filepath=None, gps_waypoints=None):
        '''Constructs a VideoMetadata instance.

        Args:
            uuid: a uuid string for the video
            start_time: a datetime describing
            frame_size: the [width, height] of the video frames
            frame_rate: the frame rate of the video
            total_frame_count: the total number of frames in the video
            duration: the duration of the video, in seconds
            size_bytes: the size of the video file on disk, in bytes
            encoding_str: the encoding string for the video
            filepath: the path to the video on disk
            gps_waypoints: a list of GPSWaypoints
        '''
        self.uuid = uuid
        self.start_time = start_time
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.total_frame_count = total_frame_count
        self.duration = duration
        self.size_bytes = size_bytes
        self.encoding_str = encoding_str
        self.filepath = filepath
        self.gps_waypoints = gps_waypoints

        self._flat = None
        self._flon = None
        self._init_gps()

    @property
    def has_gps(self):
        '''Returns True/False if this object has GPS waypoints.'''
        return self.gps_waypoints is not None

    def add_gps_waypoints(self, waypoints):
        '''Adds the list of GPSWaypoints to this object.'''
        if not self.has_gps:
            self.gps_waypoints = []
        self.gps_waypoints.extend(waypoints)
        self._init_gps()

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
            timestamp: the timestamp (in seconds) of interest
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

        Nearest neighbors is used to interpolate between waypoints, if
        necessary.

        Args:
            frame_number: the frame number of interest
            timestamp: the number of seconds into the video
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
        return self._flat(frame_number), self._flon(frame_number)

    def attributes(self):
        _attrs = [
            "uuid", "start_time", "frame_size", "frame_rate",
            "total_frame_count", "duration", "size_bytes", "encoding_str",
            "filepath", "gps_waypoints"
        ]
        # Exclue attributres that are None
        return [a for a in _attrs if getattr(self, a) is not None]

    @classmethod
    def build_for(
            cls, filepath, uuid=None, start_time=None, gps_waypoints=None):
        '''Builds a VideoMetadata object for the given video.

        Args:
            filepath: the path to the video on disk
            uuid: an optional uuid to assign to the video
            start_time: an optional datetime specifying the start time of the
                video
            gps_waypoints: an optional list of GPSWaypoint instances describing
                the GPS coordinates of the video

        Returns:
            a VideoMetadata instance
        '''
        vsi = VideoStreamInfo.build_for(filepath)
        return cls(
            uuid=uuid,
            start_time=start_time,
            frame_size=vsi.frame_size,
            frame_rate=vsi.frame_rate,
            total_frame_count=vsi.total_frame_count,
            duration=float(vsi.get_raw_value("duration")),
            size_bytes=os.path.getsize(filepath),
            encoding_str=vsi.encoding_str,
            filepath=filepath,
            gps_waypoints=gps_waypoints,
        )

    @classmethod
    def from_dict(cls, d):
        '''Constructs a VideoMetadata from a JSON dictionary.'''
        start_time = d.get(d["start_time"], None)
        if start_time is not None:
            start_time = dateutil.parser.parse(start_time)

        gps_waypoints = d.get("gps_waypoints", None)
        if gps_waypoints is not None:
            gps_waypoints = [GPSWaypoint.from_dict(g) for g in gps_waypoints]

        return cls(
            uuid=d.get("uuid", None),
            start_time=start_time,
            frame_size=d.get("frame_size", None),
            frame_rate=d.get("frame_rate", None),
            total_frame_count=d.get("total_frame_count", None),
            duration=d.get("duration", None),
            size_bytes=d.get("size_bytes", None),
            encoding_str=d.get("encoding_str", None),
            filepath=d.get("filepath", None),
            gps_waypoints=gps_waypoints,
        )

    def _init_gps(self):
        if not self.has_gps:
            return
        frames = [loc.frame_number for loc in self.gps_waypoints]
        lats = [loc.latitude for loc in self.gps_waypoints]
        lons = [loc.longitude for loc in self.gps_waypoints]
        self._flat = self._make_interp(frames, lats)
        self._flon = self._make_interp(frames, lons)

    def _make_interp(self, x, y):
        return spi.interp1d(
            x, y, kind="nearest", bounds_error=False, fill_value="extrapolate")


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

    @classmethod
    def from_dict(cls, d):
        '''Constructs a VideoFrameLabels from a JSON dictionary.'''
        return cls(
            d["frame_number"],
            attrs=AttributeContainer.from_dict(d["attrs"]),
            objects=DetectedObjectContainer.from_dict(d["objects"]))


class VideoLabels(Serializable):
    '''Class encapsulating labels for a video.

    Attributes:
        frames: a dictionary mapping frame number strings to VideoFrameLabels
            instances
    '''

    def __init__(self, frames=None, schema=None):
        '''Constructs a VideoLabels instance.

        Args:
            frames: an optional dictionary mapping frame numbers to
                VideoFrameLabels instances. By default, an empty dictionary
                is created
            schema: an optional VideoLabelsSchema to enforce on the object.
                By default, no schema is enforced
        '''
        self.frames = defaultdict(lambda: VideoFrameLabels())
        if frames is not None:
            for k, v in iteritems(frames):
                self.frames[str(k)] = v
        self.schema = schema

    def __getitem__(self, frame_number):
        return self.get_frame(frame_number)

    def __delitem__(self, frame_number):
        self.delete_frame(frame_number)

    def __iter__(self):
        return iter(self.frames)

    def __len__(self):
        return len(self.frames)

    def __bool__(self):
        return bool(self.frames)

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
        '''Gets the VideoFrameLabels for the given frame number, or None if
        the frame has not been labeled.
        '''
        return self.frames.get(str(frame_number), None)

    def delete_frame(self, frame_number):
        '''Deletes the VideoFrameLabels for the given frame number.'''
        del self.frames[str(frame_number)]

    def add_frame(self, frame_labels):
        '''Adds the frame labels to the video.

        Args:
            frame_labels: a VideoFrameLabels instance
        '''
        if self.has_schema:
            self._validate_frame_labels(frame_labels)
        self.frames[str(frame_labels.frame_number)] = frame_labels

    def add_frame_attribute(self, frame_attr, frame_number):
        '''Adds the given frame attribute to the video.

        Args:
            frame_attr: an Attribute
            frame_number: the frame number
        '''
        if self.has_schema:
            self._validate_frame_attribute(frame_attr)
        self.frames[str(frame_number)].add_frame_attribute(frame_attr)

    def add_frame_attributes(self, frame_attrs, frame_number):
        '''Adds the given frame attributes to the video.

        Args:
            frame_attrs: an AttributeContainer
            frame_number: the frame number
        '''
        if self.has_schema:
            for frame_attr in frame_attrs:
                self._validate_frame_attribute(frame_attr)
        self.frames[str(frame_number)].add_frame_attributes(frame_attrs)

    def add_object(self, obj):
        '''Adds the object to the video.

        Args:
            obj: a DetectedObject
        '''
        if self.has_schema:
            self._validate_object(obj)
        self.frames[str(obj.frame_number)].add_object(obj)

    def add_objects(self, objs):
        '''Adds the objects to the video.

        Args:
            objs: a DetectedObjectContainer
        '''
        if self.has_schema:
            for obj in objs:
                self._validate_object(obj)
        for obj in objs:
            self.frames[str(obj.frame_number)].add_object(obj)

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

    def set_schema(self, schema):
        '''Sets the enforced schema to the given VideoLabelsSchema.'''
        self.schema = schema
        self._validate_schema()

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
        _attrs = ["frames"]
        if self.has_schema:
            _attrs.append("schema")
        return _attrs

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
            for frame_labels in itervalues(self.frames):
                self._validate_frame_labels(frame_labels)

    @classmethod
    def from_dict(cls, d):
        '''Constructs a VideoLabels from a JSON dictionary.'''
        schema = d.get("schema", None)
        if schema is not None:
            schema = VideoLabelsSchema.from_dict(schema)

        return cls(
            frames={
                fn: VideoFrameLabels.from_dict(vfl)
                for fn, vfl in iteritems(d["frames"])
            },
            schema=schema,
        )


class VideoLabelsSchema(Serializable):
    '''A schema for a VideoLabels instance.'''

    def __init__(self, frames=None, objects=None):
        '''Creates a VideoLabelsSchema instance.

        Args:
            frames: an AttributeContainerSchema describing the frame attributes
                of the video
            objects: a dictionary mapping object labels to
                AttributeContainerSchemas describing the object attributes of
                each object class
        '''
        self.frames = frames or AttributeContainerSchema()
        self.objects = defaultdict(lambda: AttributeContainerSchema())
        if objects is not None:
            self.objects.update(objects)

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
        self.frames.merge_schema(schema.frames)
        for k, v in iteritems(schema.objects):
            self.objects[k].merge_schema(v)

    def validate_frame_attribute(self, frame_attr):
        '''Validates that the frame attribute is compliant with the schema.

        Args:
            attr: an Attribute

        Raises:
            AttributeContainerSchemaError: if the attribute violates the schema
        '''
        self.frames.validate_attribute(frame_attr)

    def validate_object(self, obj):
        '''Validates that the detected object is compliant with the schema.

        Args:
            obj: a DetectedObject

        Raises:
            VideoLabelsSchemaError: if the object's label violates the schema
            AttributeContainerSchemaError: if any attributes of the
                DetectedObject violate the schema
        '''
        if obj.label not in self.objects:
            raise VideoLabelsSchemaError(
                "Object label '%s' is not allowed by the schema" % obj.label)
        if obj.has_attributes:
            obj_schema = self.objects[obj.label]
            for obj_attr in obj.attrs:
                obj_schema.validate_attribute(obj_attr)

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
        for frame_labels in itervalues(video_labels.frames):
            schema.merge_schema(
                VideoLabelsSchema.build_active_schema_for_frame(frame_labels))
        return schema

    @classmethod
    def from_dict(cls, d):
        '''Constructs an AttributeContainerSchema from a JSON dictionary.'''
        frames = d.get("frames", None)
        if frames is not None:
            frames = AttributeContainerSchema.from_dict(frames)

        objects = d.get("objects", None)
        if objects is not None:
            objects = {
                k: AttributeContainerSchema.from_dict(v)
                for k, v in iteritems(objects)
            }

        return cls(frames=frames, objects=objects)


class VideoLabelsSchemaError(Exception):
    '''Error raised when a VideoLabelsSchema is violated.'''
    pass


class VideoStreamInfo(Serializable):
    '''Class encapsulating the stream info for a video.'''

    def __init__(self, stream_info):
        '''Constructs a VideoStreamInfo instance.

        This constructor should not normally be called directly. The proper way
        to instantiate this class is via the `build_for` factory method.

        Args:
            stream_info: a dictionary generated by `get_stream_info()`
        '''
        self.stream_info = stream_info

    @property
    def encoding_str(self):
        '''Return the video encoding string.'''
        return str(self.stream_info["codec_tag_string"])

    @property
    def frame_size(self):
        '''The (width, height) of each frame.'''
        return (
            int(self.stream_info["width"]),
            int(self.stream_info["height"]),
        )

    @property
    def aspect_ratio(self):
        '''The aspect ratio of the video.'''
        width, height = self.frame_size
        return width * 1.0 / height

    @property
    def frame_rate(self):
        '''The frame rate.'''
        try:
            try:
                num, denom = self.stream_info["avg_frame_rate"].split("/")
                return float(num) / float(denom)
            except ZeroDivisionError:
                num, denom = self.stream_info["r_frame_rate"].split("/")
                return float(num) / float(denom)
        except (KeyError, ValueError):
            raise VideoStreamInfoError(
                "Unable to determine frame rate from stream info")

    @property
    def total_frame_count(self):
        '''The total number of frames in the video, or 0 if it could not be
        determined.
        '''
        try:
            # this fails for directories of frames
            return int(self.stream_info["nb_frames"])
        except KeyError:
            # this seems to work for directories of frames
            return int(self.stream_info.get("duration_ts", 0))

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
        return self.custom_attributes(dynamic=True)

    @classmethod
    def build_for(cls, inpath):
        '''Builds a VideoStreamInfo object for the given video using
        `get_stream_info()`.

        Args:
            inpath: the path to the input video

        Returns:
            a VideoStreamInfo instance
        '''
        return cls(get_stream_info(inpath))

    @classmethod
    def from_dict(cls, d):
        '''Constructs a VideoStreamInfo from a JSON dictionary.'''
        return cls(d["stream_info"])


class VideoStreamInfoError(Exception):
    '''Exception raised when an invalid video stream info dictionary is
    encountered.
    '''
    pass


def get_stream_info(inpath):
    '''Get stream info for the video using `ffprobe -show_streams`.

    Args:
        inpath: video path

    Returns:
        stream: a dictionary of stream info

    Raises:
        FFprobeError: if no stream info was found
    '''
    try:
        ffprobe = FFprobe(opts=[
            "-show_streams",             # get stream info
            "-print_format", "json",     # return in JSON format
        ])
        out = ffprobe.run(inpath, decode=True)

        info = json.loads(out)

        for stream in info["streams"]:
            if stream["codec_type"] == "video":
                return stream

        logger.warning(
            "No stream found with codec_type = video. Returning the first "
            "stream")
        return info["streams"][0]  # default to the first stream
    except:
        raise FFprobeError("Unable to get stream info for '%s'" % inpath)


def get_encoding_str(inpath, use_ffmpeg=True):
    '''Get the encoding string of the input video.

    Args:
        inpath: video path
        use_ffmpeg: whether to use ffmpeg (True) or OpenCV (False)
    '''
    r = FFmpegVideoReader(inpath) if use_ffmpeg else OpenCVVideoReader(inpath)
    with r:
        return r.encoding_str


def get_frame_rate(inpath, use_ffmpeg=True):
    '''Get the frame rate of the input video.

    Args:
        inpath: video path
        use_ffmpeg: whether to use ffmpeg (True) or OpenCV (False)
    '''
    r = FFmpegVideoReader(inpath) if use_ffmpeg else OpenCVVideoReader(inpath)
    with r:
        return r.frame_rate


def get_frame_size(inpath, use_ffmpeg=True):
    '''Get the frame (width, height) of the input video.

    Args:
        inpath: video path
        use_ffmpeg: whether to use ffmpeg (True) or OpenCV (False)
    '''
    r = FFmpegVideoReader(inpath) if use_ffmpeg else OpenCVVideoReader(inpath)
    with r:
        return r.frame_size


def get_frame_count(inpath, use_ffmpeg=True):
    '''Get the number of frames in the input video.

    Args:
        inpath: video path
        use_ffmpeg: whether to use ffmpeg (True) or OpenCV (False)
    '''
    r = FFmpegVideoReader(inpath) if use_ffmpeg else OpenCVVideoReader(inpath)
    with r:
        return r.total_frame_count


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


def sample_first_frames(arg, k, size=None):
    '''Samples the first k frames in a video.

    Args:
        arg: can be either the path to the input video or an array of frames
            of size [num_frames, height, width, num_channels]
        k: number of frames to extract
        size: an optional [width, height] to resize the sampled frames. By
            default, the native dimensions of the frames are used

    Returns:
        A numpy array of size [k, height, width, num_channels]
    '''
    # Read frames ...
    if isinstance(arg, six.string_types):
        # ... from disk
        with FFmpegVideoReader(arg, frames="1-%d" % k) as vr:
            imgs = [img for img in vr]
    else:
        # ... from tensor
        imgs = arg[:k]

    # Resize frames, if necessary
    if size:
        imgs = [etai.resize(img, *size) for img in imgs]

    return np.array(imgs)


def uniformly_sample_frames(arg, k, size=None):
    '''Uniformly samples k frames from the video, always including the first
    and last frames.

    Args:
        arg: can be either the path to the input video or an array of frames
            of size [num_frames, height, width, num_channels]
        k: the number of frames to extract
        size: an optional [width, height] to resize the sampled frames. By
            default, the native dimensions of the frames are used

    Returns:
        A numpy array of size [k, height, width, num_channels]
    '''
    is_video_file = isinstance(arg, six.string_types)

    # Compute 1-based frames
    num_frames = get_frame_count(arg) if is_video_file else len(arg)
    frames = [int(round(i)) for i in np.linspace(1, min(num_frames, k), k)]

    # Read frames ...
    if is_video_file:
        # ... from disk
        with FFmpegVideoReader(arg, frames=frames) as vr:
            imgs = [img for img in vr]
    else:
        # ... from tensor
        imgs = [arg[f - 1] for f in frames]

    # Resize frames, if necessary
    if size:
        imgs = [etai.resize(img, *size) for img in imgs]

    return np.array(imgs)


def sliding_window_sample_frames(arg, k, stride, size=None):
    '''Samples clips from the video using a sliding window of the given
    length and stride.

    Args:
        arg: can be either the path to the input video or an array of frames
            of size [num_frames, height, width, num_channels]
        k: the size of each window
        stride: the stride for sliding window
        size: an optional [width, height] to resize the sampled frames. By
            default, the native dimensions of the frames are used

    Returns:
        A numpy array of size [XXXX, k, height, width, num_channels]
    '''
    is_video_file = isinstance(arg, six.string_types)

    # Determine clip indices
    num_frames = get_frame_count(arg) if is_video_file else len(arg)
    delta = np.arange(1, k + 1)
    offsets = np.array(list(range(0, num_frames + 1 - k, stride)))
    clip_inds = offsets[:, np.newaxis] + delta[np.newaxis, :]
    frames = list(np.unique(clip_inds))

    # Read frames ...
    imgs = {}
    if is_video_file:
        # ... from disk
        with FFmpegVideoReader(arg, frames=frames) as vr:
            for img in vr:
                imgs[vr.frame_number] = img
    else:
        # ... from tensor
        for fn in frames:
            imgs[fn] = arg[fn - 1]

    # Resize frames, if necessary
    if size:
        imgs = {fn: etai.resize(img, *size) for fn, img in iteritems(imgs)}

    # Generate clips tensor
    clips = []
    for inds in clip_inds:
        clips.append(np.array([imgs[k] for k in inds]))

    return np.array(clips)


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
            in_use_ffmpeg=True,
            out_use_ffmpeg=True,
            out_images_path=None,
            out_video_path=None,
            out_clips_path=None,
            out_fps=None,
            out_size=None,
            out_opts=None):
        '''Constructs a new VideoProcessor instance.

        Args:
            inpath: path to the input video. Passed directly to a VideoReader
            frames: an optional string specifying the range(s) of frames to
                process. Passed directly to a VideoReader
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
            self._reader = FFmpegVideoReader(inpath, frames=frames)
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
        '''Returns the next frame.'''
        img = self._reader.read()
        if self._write_clips and self._reader.is_new_frame_range:
            self._reset_video_clip_writer()
        return img

    def write(self, img):
        '''Writes the given image to the output writer(s).'''
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
    pass


class VideoReader(object):
    '''Base class for reading videos.'''

    def __init__(self, inpath, frames):
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
        elif isinstance(frames, list):
            # Frames list
            self._ranges = FrameRanges.from_list(frames)
            self.frames = self._ranges.to_str()
        elif isinstance(frames, (FrameRange, FrameRanges)):
            # FrameRange or FrameRanges
            self._ranges = frames
            self.frames = frames.to_str()
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
        raise NotImplementedError("subclass must implement encoding_str")

    @property
    def frame_size(self):
        raise NotImplementedError("subclass must implement frame_size")

    @property
    def frame_rate(self):
        raise NotImplementedError("subclass must implement frame_rate")

    @property
    def total_frame_count(self):
        raise NotImplementedError("subclass must implement total_frame_count")

    def read(self):
        raise NotImplementedError("subclass must implement read()")

    def close(self):
        raise NotImplementedError("subclass must implement close()")


class VideoReaderError(Exception):
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

    def __init__(self, inpath, frames=None):
        '''Constructs a new VideoReader with ffmpeg backend.

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
                    - a list like [1, 2, 3, 6, 8, 9, 10]
                    - a FrameRange or FrameRanges instance
        '''
        self._stream_info = VideoStreamInfo.build_for(inpath)
        self._ffmpeg = FFmpeg(
            out_opts=[
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
        '''Return the video encoding string.'''
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

        Returns:
            img: the next frame

        Raises:
            StopIteration: if there are no more frames to process
            VideoReaderError: if unable to load the next frame from file
        '''
        for _ in range(max(0, self.frame_number), next(self._ranges)):
            if not self._grab():
                raise VideoReaderError(
                    "Failed to grab frame %d" % self.frame_number)
        return self._retrieve()

    def close(self):
        '''Closes the video reader.'''
        self._ffmpeg.close()

    def _grab(self):
        try:
            width, height = self.frame_size
            self._raw_frame = self._ffmpeg.read(width * height * 3)
            return True
        except Exception:
            return False

    def _retrieve(self):
        try:
            vec = np.fromstring(self._raw_frame, dtype="uint8")
            width, height = self.frame_size
            return vec.reshape((height, width, 3))
        except ValueError:
            raise VideoReaderError(
                "Unable to parse frame %d" % self.frame_number)


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
        '''Constructs a new VideoReader with OpenCV backend.

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
                    - a list like [1, 2, 3, 6, 8, 9, 10]
                    - a FrameRange or FrameRanges instance

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

        Returns:
            img: the next frame

        Raises:
            StopIteration: if there are no more frames to process
            VideoReaderError: if unable to load the next frame from file
        '''
        for idx in range(max(0, self.frame_number), next(self._ranges)):
            if not self._cap.grab():
                raise VideoReaderError(
                    "Failed to grab frame %d" % (idx + 1))
        return etai.bgr_to_rgb(self._cap.retrieve()[1])

    def close(self):
        '''Closes the video reader.'''
        self._cap.release()


class VideoWriter(object):
    '''Base class for writing videos.'''

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, img):
        raise NotImplementedError("subclass must implement write()")

    def close(self):
        raise NotImplementedError("subclass must implement close()")


class VideoWriterError(Exception):
    pass


class FFmpegVideoWriter(VideoWriter):
    '''Class for writing videos using ffmpeg.'''

    def __init__(self, outpath, fps, size, out_opts=None):
        '''Constructs a VideoWriter with ffmpeg backend.

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
            img: an image in ETA format (RGB)
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
        '''Constructs a VideoWriter with OpenCV backend.

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
            img: an image in ETA format
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
        '''Constructs an ffprobe command, minus the input path.

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

        return out.decode() if decode else out


class FFprobeError(Exception):
    '''Exception raised when FFprobe was unable to analyze a video.'''
    pass


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

    DEFAULT_VIDEO_OUT_OPTS = [
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p", "-an"]

    def __init__(
            self,
            fps=None,
            size=None,
            scale=None,
            global_opts=None,
            in_opts=None,
            out_opts=None):
        '''Constructs an ffmpeg command, minus the input/output paths.

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
            in_opts: an optional list of input options for ffmpeg
            out_opts: an optional list of output options for ffmpeg. By
                default, self.DEFAULT_VIDEO_OUT_OPTS is used when the output
                path is a video file
        '''
        self.is_input_streaming = False
        self.is_output_streaming = False

        self._filter_opts = self._gen_filter_opts(fps, size, scale)
        self._global_opts = global_opts or self.DEFAULT_GLOBAL_OPTS
        self._in_opts = in_opts or []
        self._out_opts = out_opts
        self._args = None
        self._p = None

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

        if self._out_opts is None and is_supported_video_file(outpath):
            out_opts = self.DEFAULT_VIDEO_OUT_OPTS
        else:
            out_opts = self._out_opts or []

        self._args = (
            ["ffmpeg"] +
            self._global_opts +
            self._in_opts + ["-i", inpath] +
            self._filter_opts + out_opts + [outpath]
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

        Raises:
            FFmpegStreamingError: if input streaming mode is not active
        '''
        if not self.is_input_streaming:
            raise FFmpegStreamingError("Not currently input streaming")
        self._p.stdin.write(string)

    def read(self, num_bytes):
        '''Reads the given number of bytes from ffmpeg's stdout stream.

        Raises:
            FFmpegStreamingError: if output streaming mode is not active
        '''
        if not self.is_output_streaming:
            raise FFmpegStreamingError("Not currently output streaming")
        return self._p.stdout.read(num_bytes)

    def close(self):
        '''Closes a streaming ffmpeg program.

        Raises:
            FFmpegStreamingError: if a streaming mode is not active
        '''
        if not (self.is_input_streaming or self.is_output_streaming):
            raise FFmpegStreamingError("Not currently streaming")
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
        elif scale:
            filters.append("scale=iw*{0}:ih*{0}".format(scale))
        return ["-vf", ",".join(filters)] if filters else []


class FFmpegStreamingError(Exception):
    pass


class FOURCC(object):
    '''Class reprsesenting a FOURCC code.'''

    def __init__(self, _i=None, _s=None):
        '''Don't call this directly!'''
        if _i:
            self.int = _i
            self.str = FOURCC.int_to_str(_i)
        elif _s:
            self.int = FOURCC.str_to_int(_s)
            self.str = _s

    @classmethod
    def from_str(cls, s):
        '''Construct a FOURCC instance from a string.'''
        return cls(_s=s)

    @classmethod
    def from_int(cls, i):
        '''Construct a FOURCC instance from an integer.'''
        return cls(_i=i)

    @staticmethod
    def str_to_int(s):
        '''Convert the FOURCC string to an int.'''
        try:
            # OpenCV 3
            return cv2.VideoWriter_fourcc(*s)
        except AttributeError:
            # OpenCV 2
            return cv2.cv.FOURCC(*s)

    @staticmethod
    def int_to_str(i):
        '''Convert the FOURCC int to a string.'''
        return chr((i & 0x000000FF) >> 0) + \
               chr((i & 0x0000FF00) >> 8) + \
               chr((i & 0x00FF0000) >> 16) + \
               chr((i & 0xFF000000) >> 24)


class FrameRanges(object):
    '''A monotonically increasing and disjoint series of frames.'''

    def __init__(self, ranges):
        '''Constructs a frame range series from a list of (first, last) tuples,
        which must be disjoint and monotonically increasing.

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
        '''Returns the next frame number.

        Raises:
            StopIteration: if there are no more frames to process
        '''
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
        '''Return a list of frames in the frame ranges.'''
        frames = []
        for r in self._ranges:
            frames += r.to_list()

        return frames

    def to_str(self):
        '''Return a string representation of the frame ranges.'''
        return ",".join([r.to_str() for r in self._ranges])

    @classmethod
    def from_str(cls, frames_str):
        '''Constructs a FrameRanges object from a frames string.

        Args:
            frames_str: a string like "1-3,6,8-10"

        Raises:
            FrameRangesError: if the frames string is invalid
        '''
        ranges = []
        for r in frames_str.split(','):
            if r:
                fr = FrameRange.from_str(r)
                ranges.append((fr.first, fr.last))

        return cls(ranges)

    @classmethod
    def from_list(cls, frames_list):
        '''Constructs a FrameRanges object from a frames list.

        Args:
            frames_list: a list like [1, 2, 3, 6, 8, 9, 10]

        Raises:
            FrameRangesError: if the frames list is invalid
        '''
        return cls(_list_to_ranges(frames_list))


class FrameRangesError(Exception):
    '''Exception raised when an invalid FrameRanges is encountered.'''
    pass


class FrameRange(object):
    '''An iterator over a range of frames.'''

    def __init__(self, first, last):
        '''Constructs a frame range with the given first and last values,
        inclusive.

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
        '''Returns the next frame number.

        Raises:
            StopIteration: if there are no more frames in the range
        '''
        if self.idx < 0:
            self.idx = self.first
        elif self.idx < self.last:
            self.idx += 1
        else:
            raise StopIteration

        return self.idx

    def to_list(self):
        '''Return a list of frames in the range.'''
        return list(range(self.first, self.last + 1))

    def to_str(self):
        '''Return a string representation of the range.'''
        if self.first == self.last:
            return "%d" % self.first

        return "%d-%d" % (self.first, self.last)

    @classmethod
    def from_str(cls, frames_str):
        '''Constructs a FrameRange object from a string.

        Args:
            frames_str: a string like "1-5"

        Raises:
            FrameRangeError: if the frame range string is invalid
        '''
        try:
            v = list(map(int, frames_str.split('-')))
            return cls(v[0], v[-1])
        except ValueError:
            raise FrameRangeError(
                "Invalid frame range string '%s'" % frames_str)

    @classmethod
    def from_list(cls, frames_list):
        '''Constructs a FrameRange object from a frames list.

        Args:
            frames_list: a consecutive list like [1, 2, 3, 4, 5]

        Raises:
            FrameRangeError: if the frame range list is invalid
        '''
        ranges = list(_list_to_ranges(frames_list))
        if len(ranges) != 1:
            raise FrameRangeError("Invalid frame range list %s" % frames_list)

        return cls(*ranges[0])


class FrameRangeError(Exception):
    '''Exception raised when an invalid FrameRange is encountered.'''
    pass


def _list_to_ranges(vals):
    if not vals:
        raise StopIteration

    vals = sorted(vals)
    first = last = vals[0]
    for val in vals[1:]:
        if val == last + 1:
            last += 1
        else:
            yield (first, last)
            first = last = val

    yield (first, last)
