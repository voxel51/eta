"""
Core tools and data structures for working with videos.

Notes::

    [frame numbers] ETA uses 1-based indexing for all frame numbers

    [image format] ETA stores images exclusively in RGB format. In contrast,
        OpenCV stores its images in BGR format, so all images that are read or
        produced outside of this library must be converted to RGB. This
        conversion can be done via `eta.core.image.bgr_to_rgb()`

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
from future.utils import iteritems, itervalues

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from copy import deepcopy
import errno
import logging
import os
from subprocess import Popen, PIPE
import threading

import cv2
import dateutil.parser
import numpy as np
from sortedcontainers import SortedDict

import eta.core.data as etad
import eta.core.events as etae
from eta.core.frames import FrameLabels, FrameLabelsSchema
import eta.core.frameutils as etaf
import eta.core.gps as etag
import eta.core.image as etai
import eta.core.labels as etal
import eta.core.objects as etao
import eta.core.serial as etas
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
# with additions from other places on an ad-hoc basis
#
SUPPORTED_VIDEO_FILE_FORMATS = {
    ".3g2",
    ".3gp",
    ".m2ts",
    ".mts",
    ".amv",
    ".avi",
    ".f4a",
    ".f4b",
    ".f4p",
    ".f4v",
    ".flv",
    ".m2v",
    ".m4p",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp2",
    ".mp4",
    ".mpe",
    ".mpeg",
    ".mpg",
    ".mpv",
    ".m2ts",
    ".mts",
    ".nsv",
    ".ogg",
    ".ogv",
    ".qt",
    ".rm",
    ".rmvb",
    ".svi",
    ".ts",
    ".tsv",
    ".tsa",
    ".vob",
    ".webm",
    ".wmv",
    ".yuv",
}


def is_video_mime_type(filepath):
    """Determines whether the given file has a `video` MIME type.

    Args:
        filepath: the path to the file

    Returns:
        True/False
    """
    return etau.guess_mime_type(filepath).startswith("video")


def is_supported_video(path):
    """Determines whether the given filepath points to a supported video.

    Args:
        path: the path to a video, like `/path/to/video.mp4` or
            `/path/to/frames-%05d.jpg`

    Returns:
        True/False if the path refers to a supported video type
    """
    return is_supported_video_file(path) or is_supported_image_sequence(path)


def is_supported_video_file(path):
    """Determines whether the given filepath points to a supported video file
    type.

    Args:
        path: the path to a video file, like `/path/to/video.mp4`

    Returns:
        True/False if the path refers to a supported video file type
    """
    return os.path.splitext(path)[1].lower() in SUPPORTED_VIDEO_FILE_FORMATS


def is_supported_image_sequence(path):
    """Determines whether the given filepath points to a supported image
    sequence type.

    Args:
        path: the path to an image sequence, like `/path/to/frames-%05d.jpg`

    Returns:
        True/False if the path refers to a supported video file type
    """
    try:
        _ = path % 1
        return etai.is_supported_image(path)
    except TypeError:
        return False


def is_same_video_file_format(path1, path2):
    """Determines whether the video files have the same supported format.

    Args:
        path1: the path to a video
        path2: the path to a video

    Returns:
        True/False
    """
    return (
        is_supported_video(path1)
        and os.path.splitext(path1)[1] == os.path.splitext(path2)[1]
    )


def is_valid_video_file(path):
    """Determines if the given video file is valid, i.e., it has a supported
    type and can be read by our system.

    This method does not support videos represented as image sequences (i.e.,
    it will return False for them).

    Args:
        path: the path to a video file

    Returns:
        True/False if the video is valid
    """
    if not is_supported_video_file(path):
        return False
    try:
        with FFmpegVideoReader(path):
            return True
    except etau.ExecutableRuntimeError:
        return False


def glob_videos(dir_):
    """Returns an iterator over all supported video files in the directory."""
    return etau.multiglob(
        *SUPPORTED_VIDEO_FILE_FORMATS, root=os.path.join(dir_, "*")
    )


class VideoMetadata(etas.Serializable):
    """Class encapsulating metadata about a video.

    Attributes:
        start_time: (optional) a datetime describing the start (world) time of
            the video
        frame_size: the [width, height] of the video frames
        frame_rate: the frame rate of the video
        total_frame_count: the total number of frames in the video
        duration: the duration of the video, in seconds
        size_bytes: the size of the video file on disk, in bytes
        mime_type: the MIME type of the video
        encoding_str: the encoding string for the video
        gps_waypoints: (optional) a GPSWaypoints instance describing the GPS
            coordinates for the video
    """

    def __init__(
        self,
        start_time=None,
        frame_size=None,
        frame_rate=None,
        total_frame_count=None,
        duration=None,
        size_bytes=None,
        mime_type=None,
        encoding_str=None,
        gps_waypoints=None,
    ):
        """Creates a VideoMetadata instance.

        Args:
            start_time: (optional) a datetime describing the start (world) time
                of the video
            frame_size: the [width, height] of the video frames
            frame_rate: the frame rate of the video
            total_frame_count: the total number of frames in the video
            duration: the duration of the video, in seconds
            size_bytes: the size of the video file on disk, in bytes
            mime_type: the MIME type of the video
            encoding_str: the encoding string for the video
            gps_waypoints: (optional) a GPSWaypoints instance describing the
                GPS coordinates for the video
        """
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
    def aspect_ratio(self):
        """The aspect ratio of the video."""
        width, height = self.frame_size
        return width * 1.0 / height

    @property
    def has_gps(self):
        """Whether this object has GPS waypoints."""
        return self.gps_waypoints is not None

    def get_timestamp(self, frame_number=None, world_time=None):
        """Gets the timestamp for the given point in the video.

        Exactly one keyword argument must be supplied.

        Args:
            frame_number: the frame number of interest
            world_time: a datetime describing the world time of interest

        Returns:
            the timestamp (in seconds) in the video
        """
        if world_time is not None:
            timestamp = etaf.world_time_to_timestamp(
                world_time, self.start_time
            )
            return etaf.timestamp_to_frame_number(
                timestamp, self.duration, self.total_frame_count
            )

        return etaf.frame_number_to_timestamp(
            frame_number, self.total_frame_count, self.duration
        )

    def get_frame_number(self, timestamp=None, world_time=None):
        """Gets the frame number for the given point in the video.

        Exactly one keyword argument must be supplied.

        Args:
            timestamp: the timestamp (in seconds or "HH:MM:SS.XXX" format) of
                interest
            world_time: a datetime describing the world time of interest

        Returns:
            the frame number in the video
        """
        if world_time is not None:
            return etaf.world_time_to_frame_number(
                world_time,
                self.start_time,
                self.duration,
                self.total_frame_count,
            )

        return etaf.timestamp_to_frame_number(
            timestamp, self.duration, self.total_frame_count
        )

    def get_gps_location(
        self, frame_number=None, timestamp=None, world_time=None
    ):
        """Gets the GPS location at the given point in the video.

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
        """
        if not self.has_gps:
            return None
        if world_time is not None:
            timestamp = self.get_timestamp(world_time=world_time)
        if timestamp is not None:
            frame_number = self.get_frame_number(timestamp=timestamp)
        return self.gps_waypoints.get_location(frame_number)

    def attributes(self):
        """Returns the list of class attributes that will be serialized."""
        _attrs = [
            "start_time",
            "frame_size",
            "frame_rate",
            "total_frame_count",
            "duration",
            "size_bytes",
            "mime_type",
            "encoding_str",
            "gps_waypoints",
        ]
        return [a for a in _attrs if getattr(self, a) is not None]

    @classmethod
    def build_for(cls, video_path, verbose=False):
        """Builds a VideoMetadata instance for the given video.

        Args:
            video_path: the path to the video
            verbose: whether to generously log the process of extracting the
                video metadata. By default, this is False

        Returns:
            a VideoMetadata instance
        """
        vsi = VideoStreamInfo.build_for(video_path, verbose=verbose)
        metadata = cls.from_stream_info(vsi)
        if verbose:
            logger.info("Extracted video metadata: %s", str(metadata))

        return metadata

    @classmethod
    def from_stream_info(cls, stream_info):
        """Builds a VideoMetadata from a VideoStreamInfo.

        Args:
            stream_info: a VideoStreamInfo

        Returns:
            a VideoMetadata
        """
        return cls(
            frame_size=stream_info.frame_size,
            frame_rate=stream_info.frame_rate,
            total_frame_count=stream_info.total_frame_count,
            duration=stream_info.duration,
            size_bytes=stream_info.size_bytes,
            mime_type=stream_info.mime_type,
            encoding_str=stream_info.encoding_str,
        )

    @classmethod
    def from_dict(cls, d):
        """Constructs a VideoMetadata from a JSON dictionary."""
        start_time = d.get("start_time", None)
        if start_time is not None:
            start_time = dateutil.parser.parse(start_time)

        gps_waypoints = d.get("gps_waypoints", None)
        if isinstance(gps_waypoints, dict):
            gps_waypoints = etag.GPSWaypoints.from_dict(gps_waypoints)
        elif isinstance(gps_waypoints, list):
            #
            # This supports a list of GPSWaypoint instances rather than a
            # serialized GPSWaypoints instance. for backwards compatability
            #
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
            gps_waypoints=gps_waypoints,
        )


class VideoFrameLabels(FrameLabels):
    """FrameLabels for a specific frame of a video.

    VideoFrameLabels are spatial concepts that describe a collection of
    information about a specific frame in a video. VideoFrameLabels can have
    frame-level attributes, object detections, event detections, and
    segmentation masks.

    Attributes:
        frame_number: the frame number
        mask: (optional) a segmentation mask for the frame
        mask_index: (optional) a MaskIndex describing the semantics of the
            segmentation mask
        attrs: an AttributeContainer of attributes of the frame
        objects: a DetectedObjectContainer of objects in the frame
        keypoints: a KeypointsContainer of keypoints in the frame
        polylines: a PolylineContainer of polylines in the frame
        events: a DetectedEventContainer of events in the frame
    """

    @classmethod
    def from_image_labels(cls, image_labels, frame_number):
        """Constructs a VideoFrameLabels from an ImageLabels.

        Args:
            image_labels: an ImageLabels
            frame_number: the frame number

        Returns:
            a VideoFrameLabels
        """
        return cls(
            frame_number=frame_number,
            mask=image_labels.mask,
            mask_index=image_labels.mask_index,
            attrs=image_labels.attrs,
            objects=image_labels.objects,
            keypoints=image_labels.keypoints,
            polylines=image_labels.polylines,
            events=image_labels.events,
        )

    @classmethod
    def from_frame_labels(cls, frame_labels):
        """Constructs a VideoFrameLabels from a FrameLabels.

        Args:
            frame_labels: a FrameLabels

        Returns:
            a VideoFrameLabels
        """
        return cls(
            frame_number=frame_labels.frame_number,
            mask=frame_labels.mask,
            mask_index=frame_labels.mask_index,
            attrs=frame_labels.attrs,
            objects=frame_labels.objects,
            keypoints=frame_labels.keypoints,
            polylines=frame_labels.polylines,
            events=frame_labels.events,
        )


class VideoLabels(
    etal.Labels,
    etal.HasLabelsSchema,
    etal.HasLabelsSupport,
    etal.HasFramewiseView,
    etal.HasSpatiotemporalView,
):
    """Class encapsulating labels for a video.

    VideoLabels are spatiotemporal concepts that describe the content of a
    video. VideoLabels can have video-level attributes that apply to the entire
    video, frame-level attributes, frame-level object detections, frame-level
    keypoints/polylines/polygons/masks, frame-level event detections,
    spatiotemporal objects, and spatiotemporal events.

    Note that the VideoLabels class implements the `HasFramewiseView` and
    `HasSpatiotemporalView` mixins. This means that all VideoLabels instances
    can be rendered in both *framewise* and *spatiotemporal* format. Converting
    between these formats is guaranteed to be lossless and idempotent.

    In framewise format, VideoLabels store all information at the frame-level
    in VideoFrameLabels. In particular, the following invariants will hold:

        - The `attrs` field will be empty. All video-level attributes will be
          stored as frame-level `Attribute`s in `frames` with
          `constant == True`

        - The `objects` field will be empty. All video objects will be stored
          stored as frame-level `DetectedObject`s in the VideoFrameLabels
          corresponding to the frames in which they are observed

        - The `events` field will be empty. All video events will be stored
          stored as frame-level `DetectedEvent`s in the VideoFrameLabels
          corresponding to the frames in which they are observed

    In spatiotemporal format, VideoLabels store all possible information in
    the highest-available video construct. In particular, the following
    invariants will hold:

        - The `attrs` fields of all VideoFrameLabels will contain only
          non-constant `Attribute`s. All constant attributes will be upgraded
          to video-level attributes in the top-level `attrs` field

        - The `objects` fields of all VideoFrameLabels will be empty. All
          objects will be stored as `VideoObject`s in the top-level `objects`
          field. Detections for objects with `index`es will be collected in a
          single VideoObject, and each DetectedObject without an index will be
          given its own VideoObject. Additionally, all constant attributes of
          `DetectedObject`s will be upgraded to object-level attributes in
          their parent VideoObject

        - The `events` fields of all VideoFrameLabels will be empty. All
          events will be stored as `VideoEvent`s in the top-level `events`
          field. Detections for events with `index`es will be collected in a
          single VideoEvent, and each DetectedEvent without an index will be
          given its own VideoEvent. Additionally, all objects (and their
          constant attributes) within events will be upgraded to `VideoObject`s
          using the strategy described in the previous bullet point

    Attributes:
        filename: (optional) the filename of the video
        metadata: (optional) a VideoMetadata of metadata about the video
        support: a FrameRanges instance describing the support of the labels
        mask_index: (optional) a MaskIndex describing the semantics of all
            segmentation masks in the video
        attrs: an AttributeContainer of video-level attributes
        frames: a SortedDict mapping frame numbers to VideoFrameLabels
        objects: a VideoObjectContainer of objects
        events: a VideoEventContainer of events
        schema: (optional) a VideoLabelsSchema describing the video's schema
    """

    def __init__(
        self,
        filename=None,
        metadata=None,
        support=None,
        mask_index=None,
        attrs=None,
        frames=None,
        objects=None,
        events=None,
        schema=None,
    ):
        """Creates a VideoLabels instance.

        Args:
            filename: (optional) the filename of the video
            metadata: (optional) a VideoMetadata of metadata about the video
            support: (optional) a FrameRanges instance describing the frozen
                support of the labels
            mask_index: (optional) a MaskIndex describing the semantics of all
                segmentation masks in the video
            attrs: (optional) an AttributeContainer of video-level attributes
            frames: (optional) a dictionary mapping frame numbers to
                VideoFrameLabels
            objects: (optional) a VideoObjectContainer of objects
            events: (optional) a VideoEventContainer of events
            schema: (optional) a VideoLabelsSchema to enforce on the video
        """
        self.filename = filename
        self.metadata = metadata
        self.mask_index = mask_index
        self.attrs = attrs or etad.AttributeContainer()
        self.frames = SortedDict(frames or {})
        self.objects = objects or etao.VideoObjectContainer()
        self.events = events or etae.VideoEventContainer()
        etal.HasLabelsSchema.__init__(self, schema=schema)
        etal.HasLabelsSupport.__init__(self, support=support)

    def __getitem__(self, frame_number):
        """Gets the VideoFrameLabels for the given frame number, or an empty
        if no VideoFrameLabels exists.

        Args:
            frame_number: the frame number

        Returns:
            a VideoFrameLabels
        """
        return self.get_frame(frame_number)

    def __setitem__(self, frame_number, frame_labels):
        """Sets the VideoFrameLabels for the given frame number.

        If a VideoFrameLabels already exists for the frame, it is overwritten.

        Args:
            frame_number: the frame number
            frame_labels: a VideoFrameLabels
        """
        frame_labels.frame_number = frame_number
        self.add_frame(frame_labels, overwrite=True)

    def __delitem__(self, frame_number):
        """Deletes the VideoFrameLabels for the given frame number.

        Args:
            frame_number: the frame number
        """
        self.delete_frame(frame_number)

    def __iter__(self):
        """Returns an iterator over the frames with VideoFrameLabels.

        The frames are traversed in sorted order.

        Returns:
            an iterator over frame numbers
        """
        return iter(self.frames)

    def iter_attributes(self):
        """Returns an iterator over the video-level attributes in the video.

        Returns:
            an iterator over `Attribute`s
        """
        return iter(self.attrs)

    def iter_video_objects(self):
        """Returns an iterator over the `VideoObject`s in the video.

        Returns:
            an iterator over `VideoObject`s
        """
        return iter(self.objects)

    def iter_video_events(self):
        """Returns an iterator over the `VideoEvent`s in the video.

        Returns:
            an iterator over `VideoEvent`s
        """
        return iter(self.events)

    def iter_frames(self):
        """Returns an iterator over the VideoFrameLabels in the video.

        The frames are traversed in sorted order.

        Returns:
            an iterator over VideoFrameLabels
        """
        return itervalues(self.frames)

    @property
    def framewise_renderer_cls(self):
        """The LabelsFrameRenderer used by this class."""
        return VideoLabelsFrameRenderer

    @property
    def spatiotemporal_renderer_cls(self):
        """The LabelsSpatiotemporalRenderer used by this class."""
        return VideoLabelsSpatiotemporalRenderer

    @property
    def has_filename(self):
        """Whether the video has a filename."""
        return self.filename is not None

    @property
    def has_metadata(self):
        """Whether the video has metadata."""
        return self.metadata is not None

    @property
    def has_mask_index(self):
        """Whether the video has a video-wide frame segmentation mask index."""
        return self.mask_index is not None

    @property
    def has_video_attributes(self):
        """Whether the video has at least one video-level attribute."""
        return bool(self.attrs)

    @property
    def has_frame_attributes(self):
        """Whether the video has at least one frame-level attribute."""
        for frame_number in self:
            if self[frame_number].has_frame_attributes:
                return True

        return False

    @property
    def has_attributes(self):
        """Whether the video has video- or frame-level attributes."""
        return self.has_video_attributes or self.has_frame_attributes

    @property
    def has_video_objects(self):
        """Whether the video has at least one VideoObject."""
        return bool(self.objects)

    @property
    def has_detected_objects(self):
        """Whether the video has at least one frame-level DetectedObject."""
        for frame_labels in self.iter_frames():
            if frame_labels.has_objects:
                return True

        return False

    @property
    def has_objects(self):
        """Whether the video has at least one VideoObject or DetectedObject."""
        return self.has_video_objects or self.has_detected_objects

    @property
    def has_video_events(self):
        """Whether the video has at least one VideoEvent."""
        return bool(self.events)

    @property
    def has_detected_events(self):
        """Whether the video has at least one frame-level DetectedEvent."""
        for frame_labels in self.iter_frames():
            if frame_labels.has_events:
                return True

        return False

    @property
    def has_events(self):
        """Whether the video has at least one VideoEvent or DetectedEvent."""
        return self.has_video_events or self.has_detected_events

    @property
    def has_keypoints(self):
        """Whether the video has at least one frame with keypoints."""
        for frame_labels in self.iter_frames():
            if frame_labels.has_keypoints:
                return True

        return False

    @property
    def has_polylines(self):
        """Whether the video has at least one frame with polylines."""
        for frame_labels in self.iter_frames():
            if frame_labels.has_polylines:
                return True

        return False

    @property
    def has_frame_labels(self):
        """Whether the video has at least one VideoFrameLabels."""
        return bool(self.frames)

    @property
    def is_empty(self):
        """Whether the video has no labels of any kind."""
        return not (
            self.has_video_attributes
            or self.has_frame_attributes
            or self.has_video_objects
            or self.has_video_events
            or self.has_frame_labels
        )

    @property
    def num_frames(self):
        """The number of frames with VideoFrameLabels."""
        return len(self.frames)

    def has_frame(self, frame_number):
        """Determines whether this object contains a VideoFrameLabels for the
        given frame number.

        Args:
            frame_number: the frame number

        Returns:
            True/False
        """
        return frame_number in self.frames

    def get_frame(self, frame_number):
        """Gets the VideoFrameLabels for the given frame number, or an empty
        VideoFrameLabels if one does not yet exist.

        Args:
            frame_number: the frame number

        Returns:
            a VideoFrameLabels
        """
        try:
            return self.frames[frame_number]
        except KeyError:
            return VideoFrameLabels(frame_number=frame_number)

    def delete_frame(self, frame_number):
        """Deletes the VideoFrameLabels for the given frame number.

        Args:
            frame_number: the frame number
        """
        del self.frames[frame_number]

    def get_frame_numbers_with_labels(self):
        """Returns a sorted list of all frames with VideoFrameLabels.

        Returns:
            a list of frame numbers
        """
        return list(self.frames.keys())

    def get_frame_numbers_with_masks(self):
        """Returns a sorted list of frames with frame-level masks.

        Returns:
            a list of frame numbers
        """
        return [fn for fn in self if self[fn].has_mask]

    def get_frame_numbers_with_attributes(self):
        """Returns a sorted list of frames with one or more frame-level
        attributes.

        Returns:
            a list of frame numbers
        """
        return [fn for fn in self if self[fn].has_frame_attributes]

    def get_frame_numbers_with_objects(self):
        """Returns a sorted list of frames with one or more `DetectedObject`s.

        Returns:
            a list of frame numbers
        """
        return [fn for fn in self if self[fn].has_objects]

    def get_frame_numbers_with_keypoints(self):
        """Returns a sorted list of frames with frame-level `Keypoints`s.

        Returns:
            a list of frame numbers
        """
        return [fn for fn in self if self[fn].has_keypoints]

    def get_frame_numbers_with_polylines(self):
        """Returns a sorted list of frames with frame-level `Polyline`s.

        Returns:
            a list of frame numbers
        """
        return [fn for fn in self if self[fn].has_polylines]

    def get_frame_numbers_with_events(self):
        """Returns a sorted list of frames with one or more `DetectedEvent`s.

        Returns:
            a list of frame numbers
        """
        return [fn for fn in self if self[fn].has_events]

    def get_object_indexes(self):
        """Returns the set of `index`es of all objects in the video.

        `None` indexes are omitted.

        Returns:
            a set of indexes
        """
        obj_indexes = self.objects.get_indexes()
        obj_indexes.update(self.events.get_object_indexes())
        for frame_labels in self.iter_frames():
            obj_indexes.update(frame_labels.get_object_indexes())

        return obj_indexes

    def offset_object_indexes(self, offset):
        """Adds the given offset to all objects with `index`es in the video.

        Args:
            offset: the integer offset
        """
        self.objects.offset_indexes(offset)
        self.events.offset_object_indexes(offset)
        for frame_labels in self.iter_frames():
            frame_labels.offset_object_indexes(offset)

    def clear_object_indexes(self):
        """Clears the `index` of all objects in the video."""
        self.objects.clear_indexes()
        self.events.clear_object_indexes()
        for frame_labels in self.iter_frames():
            frame_labels.clear_object_indexes()

    def get_keypoints_indexes(self):
        """Returns the set of `index`es of all keypoints in the video.

        `None` indexes are omitted.

        Returns:
            a set of indexes
        """
        keypoints_indexes = set()
        for frame_labels in self.iter_frames():
            keypoints_indexes.update(frame_labels.get_keypoints_indexes())

        return keypoints_indexes

    def offset_keypoints_indexes(self, offset):
        """Adds the given offset to all keypoints with `index`es in the video.

        Args:
            offset: the integer offset
        """
        for frame_labels in self.iter_frames():
            frame_labels.offset_keypoints_indexes(offset)

    def clear_keypoints_indexes(self):
        """Clears the `index` of all keypoints in the video."""
        for frame_labels in self.iter_frames():
            frame_labels.clear_keypoints_indexes()

    def get_polyline_indexes(self):
        """Returns the set of `index`es of all polylines in the video.

        `None` indexes are omitted.

        Returns:
            a set of indexes
        """
        polyline_indexes = set()
        for frame_labels in self.iter_frames():
            polyline_indexes.update(frame_labels.get_polyline_indexes())

        return polyline_indexes

    def offset_polyline_indexes(self, offset):
        """Adds the given offset to all polylines with `index`es in the video.

        Args:
            offset: the integer offset
        """
        for frame_labels in self.iter_frames():
            frame_labels.offset_polyline_indexes(offset)

    def clear_polyline_indexes(self):
        """Clears the `index` of all polylines in the video."""
        for frame_labels in self.iter_frames():
            frame_labels.clear_polyline_indexes()

    def get_event_indexes(self):
        """Returns the set of `index`es of all events in the video.

        `None` indexes are omitted.

        Returns:
            a set of indexes
        """
        event_indexes = self.events.get_indexes()
        for frame_labels in self.iter_frames():
            event_indexes.update(frame_labels.get_event_indexes())

        return event_indexes

    def offset_event_indexes(self, offset):
        """Adds the given offset to all events with `index`es in the video.

        Args:
            offset: the integer offset
        """
        self.events.offset_indexes(offset)
        for frame_labels in self.iter_frames():
            frame_labels.offset_event_indexes(offset)

    def clear_event_indexes(self):
        """Clears the `index` of all events in the video."""
        self.events.clear_indexes()
        for frame_labels in self.iter_frames():
            frame_labels.clear_event_indexes()

    def add_video_attribute(self, attr):
        """Adds the given video-level attribute to the video.

        Args:
            attr: an Attribute
        """
        self.attrs.add(attr)

    def add_video_attributes(self, attrs):
        """Adds the given video-level attributes to the video.

        Args:
            attrs: an AttributeContainer
        """
        self.attrs.add_container(attrs)

    def add_frame(self, frame_labels, frame_number=None, overwrite=True):
        """Adds the frame labels to the video.

        Args:
            frame_labels: a FrameLabels instance
            frame_number: an optional frame number. If not specified, the
                FrameLabels must have its `frame_number` set
            overwrite: whether to overwrite any existing VideoFrameLabels
                instance for the frame or merge the new labels. By default,
                this is True
        """
        self._add_frame_labels(frame_labels, frame_number, overwrite)

    def add_frame_attribute(self, attr, frame_number):
        """Adds the given frame-level attribute to the video.

        Args:
            attr: an Attribute
            frame_number: the frame number
        """
        self._ensure_frame(frame_number)
        self.frames[frame_number].add_attribute(attr)

    def add_frame_attributes(self, attrs, frame_number):
        """Adds the given frame-level attributes to the video.

        Args:
            attrs: an AttributeContainer
            frame_number: the frame number
        """
        self._ensure_frame(frame_number)
        self.frames[frame_number].add_attributes(attrs)

    def add_object(self, obj, frame_number=None):
        """Adds the object to the video.

        Args:
            obj: a VideoObject or DetectedObject
            frame_number: (DetectedObject only) the frame number. If omitted,
                the DetectedObject must have its `frame_number` set
        """
        if isinstance(obj, etao.DetectedObject):
            self._add_detected_object(obj, frame_number)
        else:
            self.objects.add(obj)

    def add_objects(self, objects, frame_number=None):
        """Adds the objects to the video.

        Args:
            objects: a VideoObjectContainer or DetectedObjectContainer
            frame_number: (DetectedObjectContainer only) the frame number. If
                omitted, the DetectedObjects must have their `frame_number` set
        """
        if isinstance(objects, etao.DetectedObjectContainer):
            self._add_detected_objects(objects, frame_number)
        else:
            self.objects.add_container(objects)

    def add_event(self, event, frame_number=None):
        """Adds the event to the video.

        Args:
            event: a VideoEvent or DetectedEvent
            frame_number: (DetectedEvent only) the frame number. If omitted,
                the DetectedEvent must have its `frame_number` set
        """
        if isinstance(event, etae.DetectedEvent):
            self._add_detected_event(event, frame_number)
        else:
            self.events.add(event)

    def add_events(self, events, frame_number=None):
        """Adds the events to the video.

        Args:
            events: a VideoEventContainer or DetectedEventContainer
            frame_number: (DetectedEventContainer only) the frame number. If
                omitted, the `DetectedEvent`s must have their `frame_number`s
                set
        """
        if isinstance(events, etae.DetectedEventContainer):
            self._add_detected_events(events, frame_number)
        else:
            self.events.add_container(events)

    def clear_video_attributes(self):
        """Removes all video-level attributes from the video."""
        self.attrs = etad.AttributeContainer()

    def clear_frame_attributes(self):
        """Removes all frame-level attributes from the video."""
        for frame_labels in self.iter_frames():
            frame_labels.clear_frame_attributes()

    def clear_video_objects(self):
        """Removes all `VideoObject`s from the video."""
        self.objects = etao.VideoObjectContainer()

    def clear_detected_objects(self):
        """Removes all `DetectedObject`s from the video."""
        for frame_labels in self.iter_frames():
            frame_labels.clear_objects()

    def clear_objects(self):
        """Removes all `VideoObject`s and `DetectedObject`s from the video."""
        self.clear_video_objects()
        self.clear_detected_objects()

    def clear_video_events(self):
        """Removes all `VideoEvent`s from the video."""
        self.events = etae.VideoEventContainer()

    def clear_detected_events(self):
        """Removes all `DetectedEvent`s from the video."""
        for frame_labels in self.iter_frames():
            frame_labels.clear_events()

    def clear_events(self):
        """Removes all `VideoEvent`s and `DetectedEvent`s from the video."""
        self.clear_video_events()
        self.clear_detected_events()

    def clear_frames(self):
        """Removes all VideoFrameLabels from the video."""
        self.frames.clear()

    def remove_empty_frames(self):
        """Removes all empty VideoFrameLabels from the video."""
        self.frames = SortedDict(
            {fn: vfl for fn, vfl in iteritems(self.frames) if not vfl.is_empty}
        )

    def merge_labels(self, video_labels, reindex=False):
        """Merges the given VideoLabels into this labels.

        Args:
            video_labels: a VideoLabels
            reindex: whether to offset the `index` fields of objects,
                polylines, keypoints, and events in `video_labels` before
                merging so that all indexes are unique. The default is False
        """
        if reindex:
            self._reindex_objects(video_labels)
            self._reindex_keypoints(video_labels)
            self._reindex_polylines(video_labels)
            self._reindex_events(video_labels)

        # Merge metadata
        if video_labels.has_filename and not self.has_filename:
            self.filename = video_labels.filename
        if video_labels.has_metadata and not self.has_metadata:
            self.metadata = video_labels.metadata
        if self.is_support_frozen or video_labels.is_support_frozen:
            self.merge_support(video_labels.support)
        if video_labels.has_mask_index and not self.has_mask_index:
            self.mask_index = video_labels.mask_index
        if video_labels.has_schema:
            if self.has_schema:
                self.schema.merge_schema(video_labels.schema)
            else:
                self.schema = video_labels.schema

        # Merge labels
        self.add_video_attributes(video_labels.attrs)
        self.add_objects(video_labels.objects)
        self.add_events(video_labels.events)
        for frame_labels in video_labels.iter_frames():
            self.add_frame(frame_labels, overwrite=False)

    def filter_by_schema(self, schema):
        """Filters the labels by the given schema.

        Args:
            schema: a VideoLabelsSchema
        """
        self.attrs.filter_by_schema(schema.attrs)
        self.objects.filter_by_schema(schema.objects)
        self.events.filter_by_schema(schema.events)
        for frame_labels in self.iter_frames():
            frame_labels.filter_by_schema(schema)

    def remove_objects_without_attrs(self, labels=None):
        """Removes objects from the VideoLabels that do not have attributes.

        Args:
            labels: an optional list of object label strings to which to
                restrict attention when filtering. By default, all objects are
                processed
        """
        self.objects.remove_objects_without_attrs(labels=labels)
        self.events.remove_objects_without_attrs(labels=labels)
        for frame_labels in self.iter_frames():
            frame_labels.remove_objects_without_attrs(labels=labels)

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Returns:
            a list of attributes
        """
        _attrs = []
        if self.filename:
            _attrs.append("filename")
        if self.metadata:
            _attrs.append("metadata")
        if self.has_schema:
            _attrs.append("schema")
        if self.is_support_frozen:
            _attrs.append("support")
        if self.has_mask_index:
            _attrs.append("mask_index")
        if self.attrs:
            _attrs.append("attrs")
        if self.frames:
            _attrs.append("frames")
        if self.objects:
            _attrs.append("objects")
        if self.events:
            _attrs.append("events")
        return _attrs

    @classmethod
    def from_objects(cls, objects):
        """Builds a VideoLabels instance from a container of objects.

        If a DetectedObjectContainer is provided, the `DetectedObject`s must
        have their `frame_number`s set.

        Args:
            objects: a VideoObjectContainer or DetectedObjectContainer

        Returns:
            a VideoLabels
        """
        labels = cls()
        labels.add_objects(objects)
        return labels

    @classmethod
    def from_events(cls, events):
        """Builds a VideoLabels instance from an event container.

        If a DetectedEventContainer is provided, the `DetectedEvent`s must
        have their `frame_number`s set.

        Args:
            events: a VideoEventContainer or DetectedEventContainer

        Returns:
            a VideoLabels
        """
        labels = cls()
        labels.add_events(events)
        return labels

    @classmethod
    def from_dict(cls, d):
        """Constructs a VideoLabels from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a VideoLabels
        """
        filename = d.get("filename", None)

        metadata = d.get("metadata", None)
        if metadata is not None:
            metadata = VideoMetadata.from_dict(metadata)

        support = d.get("support", None)
        if support is not None:
            support = etaf.FrameRanges.from_dict(support)

        mask_index = d.get("mask_index", None)
        if mask_index is not None:
            mask_index = etad.MaskIndex.from_dict(mask_index)

        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = etad.AttributeContainer.from_dict(attrs)

        frames = d.get("frames", None)
        if frames is not None:
            frames = {
                int(fn): VideoFrameLabels.from_dict(vfl)
                for fn, vfl in iteritems(frames)
            }

        objects = d.get("objects", None)
        if objects is not None:
            objects = etao.VideoObjectContainer.from_dict(objects)

        events = d.get("events", None)
        if events is not None:
            events = etae.VideoEventContainer.from_dict(events)

        schema = d.get("schema", None)
        if schema is not None:
            schema = VideoLabelsSchema.from_dict(schema)

        return cls(
            filename=filename,
            metadata=metadata,
            support=support,
            mask_index=mask_index,
            attrs=attrs,
            frames=frames,
            objects=objects,
            events=events,
            schema=schema,
        )

    def _ensure_frame(self, frame_number):
        if not self.has_frame(frame_number):
            self.frames[frame_number] = VideoFrameLabels(
                frame_number=frame_number
            )

    def _add_detected_object(self, obj, frame_number):
        if frame_number is None:
            if not obj.has_frame_number:
                raise ValueError(
                    "Either `frame_number` must be provided or the "
                    "DetectedObject must have its `frame_number` set"
                )

            frame_number = obj.frame_number

        obj.frame_number = frame_number
        self._ensure_frame(frame_number)
        self.frames[frame_number].add_object(obj)

    def _add_detected_objects(self, objects, frame_number):
        for obj in objects:
            self._add_detected_object(obj, frame_number)

    def _add_detected_event(self, event, frame_number):
        if frame_number is None:
            if not event.has_frame_number:
                raise ValueError(
                    "Either `frame_number` must be provided or the "
                    "DetectedEvent must have its `frame_number` set"
                )

            frame_number = event.frame_number

        event.frame_number = frame_number
        self._ensure_frame(frame_number)
        self.frames[frame_number].add_event(event)

    def _add_detected_events(self, events, frame_number):
        for event in events:
            self._add_detected_event(event, frame_number)

    def _add_frame_labels(self, frame_labels, frame_number, overwrite):
        if frame_number is None:
            if not frame_labels.has_frame_number:
                raise ValueError(
                    "Either `frame_number` must be provided or the "
                    "FrameLabels must have its `frame_number` set"
                )

            frame_number = frame_labels.frame_number

        if overwrite or not self.has_frame(frame_number):
            if not isinstance(frame_labels, VideoFrameLabels):
                frame_labels = VideoFrameLabels.from_frame_labels(frame_labels)

            frame_labels.frame_number = frame_number
            self.frames[frame_number] = frame_labels
        else:
            self.frames[frame_number].merge_labels(frame_labels)

    def _compute_support(self):
        frame_ranges = etaf.FrameRanges.from_iterable(self.frames.keys())
        frame_ranges.merge(*[obj.support for obj in self.objects])
        frame_ranges.merge(*[event.support for event in self.events])
        return frame_ranges

    def _reindex_objects(self, video_labels):
        self_indexes = self.get_object_indexes()
        if not self_indexes:
            return

        new_indexes = video_labels.get_object_indexes()
        if not new_indexes:
            return

        offset = max(self_indexes) + 1 - min(new_indexes)
        video_labels.offset_object_indexes(offset)

    def _reindex_keypoints(self, video_labels):
        self_indexes = self.get_keypoints_indexes()
        if not self_indexes:
            return

        new_indexes = video_labels.get_keypoints_indexes()
        if not new_indexes:
            return

        offset = max(self_indexes) + 1 - min(new_indexes)
        video_labels.offset_keypoints_indexes(offset)

    def _reindex_polylines(self, video_labels):
        self_indexes = self.get_polyline_indexes()
        if not self_indexes:
            return

        new_indexes = video_labels.get_polyline_indexes()
        if not new_indexes:
            return

        offset = max(self_indexes) + 1 - min(new_indexes)
        video_labels.offset_polyline_indexes(offset)

    def _reindex_events(self, video_labels):
        self_indexes = self.get_event_indexes()
        if not self_indexes:
            return

        new_indexes = video_labels.get_event_indexes()
        if not new_indexes:
            return

        offset = max(self_indexes) + 1 - min(new_indexes)
        video_labels.offset_event_indexes(offset)


class VideoLabelsSchema(FrameLabelsSchema):
    """Schema describing the content of one or more VideoLabels.

    VideoLabelsSchema introduces the term "video-level attribute", which is
    merely an alias for the term "constant frame attribute" as used in
    `FrameLabelsSchema`s.

    Attributes:
        attrs: an AttributeContainerSchema describing the video-level
            attributes of the video(s)
        frames: an AttributeContainerSchema describing the frame-level
            attributes of the video(s)
        objects: an ObjectContainerSchema describing the objects of the
            video(s)
        keypoints: a KeypointsContainerSchema describing the keypoints of the
            video(s)
        polylines: a PolylineContainerSchema describing the polylines of the
            video(s)
        events: an EventContainerSchema describing the events of the video(s)
    """

    @property
    def has_video_attributes(self):
        """Whether the schema has at least one video-level AttributeSchema.

        This property is an alias for `has_constant_attributes`.
        """
        return self.has_constant_attributes

    def has_video_attribute(self, attr_name):
        """Whether the schema has a video-level attribute with the given name.

        This method is an alias for `has_constant_attribute()`.

        Args:
            attr_name: the video-level attribute name

        Returns:
            True/False
        """
        return self.has_constant_attribute(attr_name)

    def get_video_attribute_schema(self, attr_name):
        """Gets the AttributeSchema for the video-level attribute with the
        given name.

        This method is an alias for `get_video_attribute_schema()`.

        Args:
            attr_name: the video-level attribute name

        Returns:
            the AttributeSchema
        """
        return self.get_constant_attribute_schema(attr_name)

    def get_video_attribute_class(self, attr_name):
        """Gets the Attribute class for the video-level attribute with the
        given name.

        This method is an alias for `get_constant_attribute_class()`.

        Args:
            attr_name: the video-level attribute name

        Returns:
            the Attribute class
        """
        return self.get_constant_attribute_class(attr_name)

    def add_video_attribute(self, attr):
        """Adds the given video-level attribute to the schema.

        This method is an alias for `add_constant_attribute()`.

        Args:
            attr: an Attribute
        """
        self.add_constant_attribute(attr)

    def add_video_attributes(self, attrs):
        """Adds the given video-level attributes to the schema.

        This method is an alias for `add_constant_attributes()`.

        Args:
            attrs: an AttributeContainer
        """
        self.add_constant_attributes(attrs)

    def is_valid_video_attribute(self, attr):
        """Whether the video-level attribute is compliant with the schema.

        This method is an alias for `is_valid_constant_attribute()`.

        Args:
            attr: an Attribute

        Returns:
            True/False
        """
        return self.is_valid_constant_attribute(attr)

    def is_valid_video_attributes(self, attrs):
        """Whether the video-level attributes are compliant with the schema.

        This method is an alias for `is_valid_constant_attributes()`.

        Args:
            attrs: an AttributeContainer

        Returns:
            True/False
        """
        return self.is_valid_constant_attributes(attrs)

    def validate_video_attribute(self, attr):
        """Validates that the video-level attribute is compliant with the
        schema.

        This method is an alias for `validate_constant_attribute()`.

        Args:
            attr: an Attribute

        Raises:
            LabelsSchemaError: if the attribute violates the schema
        """
        self.validate_constant_attribute(attr)

    def validate_video_attributes(self, attrs):
        """Validates that the video-level attributes are compliant with the
        schema.

        This method is an alias for `validate_constant_attributes()`.

        Args:
            attrs: an AttributeContainer

        Raises:
            LabelsSchemaError: if the attributes violate the schema
        """
        self.validate_constant_attributes(attrs)

    def validate(self, video_labels):
        """Validates that the video labels are compliant with the schema.

        Args:
            video_labels: a VideoLabels

        Raises:
            LabelsSchemaError: if the labels violate the schema
        """
        self.validate_video_labels(video_labels)

    @classmethod
    def build_active_schema(cls, video_labels):
        """Builds a VideoLabelsSchema that describes the active schema of the
        given labels.

        Args:
            video_labels: a VideoLabels

        Returns:
            a VideoLabelsSchema
        """
        return cls.build_active_schema_for_video_labels(video_labels)


class VideoLabelsFrameRenderer(etal.LabelsFrameRenderer):
    """Class for rendering VideoLabels at the frame-level.

    See the VideoLabels class docstring for the framewise format spec.
    """

    _LABELS_CLS = VideoLabels
    _FRAME_LABELS_CLS = VideoFrameLabels

    def __init__(self, video_labels):
        """Creates a VideoLabelsFrameRenderer instance.

        Args:
            video_labels: a VideoLabels
        """
        self._video_labels = video_labels

    def render(self, in_place=False):
        """Renders the VideoLabels in framewise format.

        Args:
            in_place: whether to perform the rendering in-place. By default,
                this is False

        Returns:
            a VideoLabels
        """
        labels = self._video_labels
        frames = self.render_all_frames(in_place=in_place)

        if in_place:
            # Render in-place
            labels.clear_video_attributes()
            labels.clear_video_objects()
            labels.clear_video_events()
            labels.clear_frames()
            labels.frames = frames
            return labels

        # Render new copy of labels
        filename = deepcopy(labels.filename)
        metadata = deepcopy(labels.metadata)
        if labels.is_support_frozen:
            support = deepcopy(labels.support)
        else:
            support = None

        mask_index = deepcopy(labels.mask_index)
        schema = deepcopy(labels.schema)

        return VideoLabels(
            filename=filename,
            metadata=metadata,
            support=support,
            mask_index=mask_index,
            frames=frames,
            schema=schema,
        )

    def render_frame(self, frame_number, in_place=False):
        """Renders the VideoLabels for the given frame.

        Args:
            frame_number: the frame number
            in_place: whether to perform the rendering in-place (i.e., without
                deep copying objects). By default, this is False

        Returns:
            a VideoFrameLabels, or None if no labels exist for the given frame
        """
        if frame_number not in self._video_labels.support:
            return None

        video_attrs = self._get_video_attrs()
        dobjs = self._render_object_frame(frame_number, in_place)
        devents = self._render_event_frame(frame_number, in_place)
        return self._render_frame(
            frame_number, video_attrs, dobjs, devents, in_place
        )

    def render_all_frames(self, in_place=False):
        """Renders the VideoLabels for all possible frames.

        Args:
            in_place: whether to perform the rendering in-place (i.e., without
                deep copying objects). By default, this is False

        Returns:
            a dictionary mapping frame numbers to VideoFrameLabels instances
        """
        video_attrs = self._get_video_attrs()
        dobjs_map = self._render_all_object_frames(in_place)
        devents_map = self._render_all_event_frames(in_place)

        frame_labels_map = {}
        for frame_number in self._video_labels.support:
            dobjs = dobjs_map.get(frame_number, None)
            devents = devents_map.get(frame_number, None)
            frame_labels_map[frame_number] = self._render_frame(
                frame_number, video_attrs, dobjs, devents, in_place
            )

        return frame_labels_map

    def _render_frame(
        self, frame_number, video_attrs, dobjs, devents, in_place
    ):
        labels = self._video_labels

        # Base VideoFrameLabels
        if labels.has_frame(frame_number):
            frame_labels = labels.get_frame(frame_number)
            if not in_place:
                frame_labels = deepcopy(frame_labels)
        else:
            frame_labels = VideoFrameLabels(frame_number=frame_number)

        # Render video-level attributes
        if video_attrs is not None:
            #
            # Prepend video-level attributes
            #
            # We cannot avoid `deepcopy` here because video-level attributes
            # must be embedded in each frame
            #
            frame_labels.attrs.prepend_container(deepcopy(video_attrs))

        # Render objects
        if dobjs is not None:
            frame_labels.add_objects(dobjs)

        # Render events
        if devents is not None:
            frame_labels.add_events(devents)

        return frame_labels

    def _render_object_frame(self, frame_number, in_place):
        labels = self._video_labels

        if not labels.has_video_objects:
            return None

        r = etao.VideoObjectContainerFrameRenderer(labels.objects)
        return r.render_frame(frame_number, in_place=in_place)

    def _render_all_object_frames(self, in_place):
        labels = self._video_labels

        if not labels.has_video_objects:
            return {}

        r = etao.VideoObjectContainerFrameRenderer(labels.objects)
        return r.render_all_frames(in_place=in_place)

    def _render_event_frame(self, frame_number, in_place):
        labels = self._video_labels

        if not labels.has_video_events:
            return None

        r = etae.VideoEventContainerFrameRenderer(labels.events)
        return r.render_frame(frame_number, in_place=in_place)

    def _render_all_event_frames(self, in_place):
        labels = self._video_labels

        if not labels.has_video_events:
            return {}

        r = etae.VideoEventContainerFrameRenderer(labels.events)
        return r.render_all_frames(in_place=in_place)

    def _get_video_attrs(self):
        labels = self._video_labels

        if not labels.has_video_attributes:
            return None

        # There's no need to avoid `deepcopy` here when `in_place == True`
        # because copies of video-level attributes must be made for each frame
        video_attrs = deepcopy(labels.attrs)
        for attr in video_attrs:
            attr.constant = True

        return video_attrs


class VideoLabelsSpatiotemporalRenderer(etal.LabelsSpatiotemporalRenderer):
    """Class for rendering VideoLabels in spatiotemporal format.

    See the VideoLabels class docstring for the spatiotemporal format spec.
    """

    _LABELS_CLS = VideoLabels

    def __init__(self, video_labels):
        """Creates a VideoLabelsSpatiotemporalRenderer instance.

        Args:
            video_labels: a VideoLabels
        """
        self._video_labels = video_labels

    def render(self, in_place=False):
        """Renders the VideoLabels in spatiotemporal format.

        Args:
            in_place: whether to perform the rendering in-place. By default,
                this is False

        Returns:
            a VideoLabels
        """
        labels = self._video_labels
        if not in_place:
            labels = deepcopy(labels)

        # Ensure that existing `VideoObject`s are in spatiotemporal format
        obj_renderer = etao.VideoObjectContainerSpatiotemporalRenderer(
            labels.objects
        )
        obj_renderer.render(in_place=True)

        # Ensure that existing `VideoEvent`s are in spatiotemporal format
        event_renderer = etae.VideoEventContainerSpatiotemporalRenderer(
            labels.events
        )
        event_renderer.render(in_place=True)

        # Upgrade spatiotemporal elements from frames
        attrs, objects, events = strip_spatiotemporal_content_from_frames(
            labels
        )
        labels.attrs.add_container(attrs)
        labels.add_objects(objects)
        labels.add_events(events)
        labels.remove_empty_frames()

        return labels


def strip_spatiotemporal_content_from_frames(video_labels):
    """Strips the spatiotemporal content from the frames of the given
    VideoLabels.

    The input labels are modified in-place.

    Args:
        video_labels: a VideoLabels

    Returns:
        attrs: an AttributeContainer of constant frame attributes. By
            convention, these attributes are no longer marked as constant, as
            this is assumed to be implicit
        objects: a VideoObjectContainer containing the objects that were
            stripped from the frames
        events: a VideoEventContainer containing the events that were stripped
            from the frames
    """
    # Extract spatiotemporal content from frames
    attrs_map = {}
    dobjs = etao.DetectedObjectContainer()
    devents = etae.DetectedEventContainer()
    for frame_labels in video_labels.iter_frames():
        # Extract objects
        dobjs.add_container(frame_labels.pop_objects())

        # Extract events
        devents.add_container(frame_labels.pop_events())

        # Extract constant attributes
        for const_attr in frame_labels.attrs.pop_constant_attrs():
            # @todo could verify here that duplicate constant attributes are
            # exactly equal, as they should be
            attrs_map[const_attr.name] = const_attr

    # Store video-level attributes in a container with `constant == False`
    attrs = etad.AttributeContainer()
    for attr in itervalues(attrs_map):
        attr.constant = False
        attrs.add(attr)

    # Build VideoObjects
    objects = etao.VideoObjectContainer.from_detections(dobjs)

    # Build VideoEvents
    events = etae.VideoEventContainer.from_detections(devents)

    return attrs, objects, events


class VideoSetLabels(etal.LabelsSet):
    """Class encapsulating labels for a set of videos.

    VideoSetLabels support item indexing by the `filename` of the VideoLabels
    instances in the set.

    VideoSetLabels instances behave like defaultdicts: new VideoLabels
    instances are automatically created if a non-existent filename is accessed.

    VideoLabels without filenames may be added to the set, but they cannot be
    accessed by `filename`-based lookup.

    Attributes:
        videos: an OrderedDict of VideoLabels with filenames as keys
        schema: a VideoLabelsSchema describing the schema of the labels
    """

    _ELE_ATTR = "videos"
    _ELE_KEY_ATTR = "filename"
    _ELE_CLS = VideoLabels
    _ELE_CLS_FIELD = "_LABELS_CLS"

    def sort_by_filename(self, reverse=False):
        """Sorts the VideoLabels in this instance by filename.

        VideoLabels without filenames are always put at the end of the set.

        Args:
            reverse: whether to sort in reverse order. By default, this is
                False
        """
        self.sort_by("filename", reverse=reverse)

    def clear_video_attributes(self):
        """Removes all video-level attributes from all VideoLabels in the set.
        """
        for video_labels in self:
            video_labels.clear_video_attributes()

    def clear_frame_attributes(self):
        """Removes all frame-level attributes from all VideoLabels in the set.
        """
        for video_labels in self:
            video_labels.clear_frame_attributes()

    def clear_video_objects(self):
        """Removes all `VideoObject`s from all VideoLabels in the set."""
        for video_labels in self:
            video_labels.clear_video_objects()

    def clear_detected_objects(self):
        """Removes all `DetectedObject`s from all VideoLabels in the set."""
        for video_labels in self:
            video_labels.clear_detected_objects()

    def clear_objects(self):
        """Removes all `VideoObject`s and `DetectedObject`s from all
        VideoLabels in the set.
        """
        for video_labels in self:
            video_labels.clear_objects()

    def clear_video_events(self):
        """Removes all `VideoEvent`s from all VideoLabels in the set."""
        for video_labels in self:
            video_labels.clear_video_events()

    def clear_detected_events(self):
        """Removes all `DetectedEvent`s from all VideoLabels in the set."""
        for video_labels in self:
            video_labels.clear_detected_events()

    def clear_events(self):
        """Removes all `VideoEvent`s and `DetectedEvent`s from all VideoLabels
        in the set.
        """
        for video_labels in self:
            video_labels.clear_events()

    def get_filenames(self):
        """Returns the set of filenames of VideoLabels in the set.

        Returns:
            the set of filenames
        """
        return set(vl.filename for vl in self if vl.filename)

    def remove_objects_without_attrs(self, labels=None):
        """Removes objects from the VideoLabels in the set that do not have
        attributes.

        Args:
            labels: an optional list of object label strings to which to
                restrict attention when filtering. By default, all objects are
                processed
        """
        for video_labels in self:
            video_labels.remove_objects_without_attrs(labels=labels)

    @classmethod
    def from_video_labels_patt(cls, video_labels_patt):
        """Creates a VideoSetLabels from a pattern of VideoLabels files.

        Args:
             video_labels_patt: a pattern with one or more numeric sequences
                for VideoLabels files on disk

        Returns:
            a VideoSetLabels instance
        """
        return cls.from_labels_patt(video_labels_patt)


class BigVideoSetLabels(VideoSetLabels, etas.BigSet):
    """A BigSet of VideoLabels.

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
    """

    def __init__(self, videos=None, schema=None, backing_dir=None):
        """Creates a BigVideoSetLabels instance.

        Args:
            videos: an optional dictionary or list of (key, uuid) tuples for
                elements in the set
            schema: an optional VideoLabelsSchema to enforce on the object.
                By default, no schema is enforced
            backing_dir: an optional backing directory in which the VideoLabels
                are/will be stored. If omitted, a temporary backing directory
                is used
        """
        etas.BigSet.__init__(self, backing_dir=backing_dir, videos=videos)
        etal.HasLabelsSchema.__init__(self, schema=schema)

    def empty_set(self):
        """Returns an empty in-memory VideoSetLabels version of this
        BigVideoSetLabels.

        Returns:
            an empty VideoSetLabels
        """
        return VideoSetLabels(schema=self.schema)

    def filter_by_schema(self, schema):
        """Removes objects/attributes from the VideoLabels in the set that
        are not compliant with the given schema.

        Args:
            schema: a VideoLabelsSchema
        """
        for key in self.keys():
            video_labels = self[key]
            video_labels.filter_by_schema(schema)
            self[key] = video_labels

    def set_schema(self, schema, filter_by_schema=False, validate=False):
        """Sets the enforced schema to the given VideoLabelsSchema.

        Args:
            schema: a VideoLabelsSchema to assign
            filter_by_schema: whether to filter labels that are not compliant
                with the schema. By default, this is False
            validate: whether to validate that the labels (after filtering, if
                applicable) are compliant with the new schema. By default, this
                is False

        Raises:
            LabelsSchemaError: if `validate` was `True` and this set contains
                labels that are not compliant with the schema
        """
        self.schema = schema
        for key in self.keys():
            video_labels = self[key]
            video_labels.set_schema(
                schema, filter_by_schema=filter_by_schema, validate=validate
            )
            self[key] = video_labels

    def remove_objects_without_attrs(self, labels=None):
        """Removes all objects from the BigVideoSetLabels that do not have
        attributes.

        Args:
            labels: an optional list of object label strings to which to
                restrict attention when filtering. By default, all objects are
                processed
        """
        for key in self.keys():
            video_labels = self[key]
            video_labels.remove_objects_without_attrs(labels=labels)
            self[key] = video_labels


class VideoStreamInfo(etas.Serializable):
    """Class encapsulating the stream info for a video."""

    def __init__(self, stream_info, format_info, mime_type=None):
        """Creates a VideoStreamInfo instance.

        Args:
            stream_info: a dictionary of video stream info
            format_info: a dictionary of video format info
            mime_type: (optional) the MIME type of the video
        """
        self.stream_info = stream_info
        self.format_info = format_info
        self._mime_type = mime_type

    @property
    def frame_size(self):
        """The (width, height) of each frame.

        Raises:
            VideoStreamInfoError: if the frame size could not be determined
        """
        try:
            # Must check if the video is rotated!
            rotation = int(self.stream_info["tags"]["rotate"])
        except (KeyError, ValueError):
            rotation = 0

        try:
            width = int(self.stream_info["width"])
            height = int(self.stream_info["height"])

            if (rotation // 90) % 2:
                logger.debug(
                    "Found video with rotation %d; swapping width and height",
                    rotation,
                )
                width, height = height, width

            return width, height
        except (KeyError, ValueError):
            raise VideoStreamInfoError(
                "Unable to determine frame size of the video"
            )

    @property
    def aspect_ratio(self):
        """The aspect ratio of the video.

        Raises:
            VideoStreamInfoError: if the frame size could not be determined
        """
        width, height = self.frame_size
        return width * 1.0 / height

    @property
    def frame_rate(self):
        """The frame rate of the video.

        Raises:
            VideoStreamInfoError: if the frame rate could not be determined
        """
        try:
            try:
                num, denom = self.stream_info["avg_frame_rate"].split("/")
                return float(num) / float(denom)
            except ZeroDivisionError:
                num, denom = self.stream_info["r_frame_rate"].split("/")
                return float(num) / float(denom)
        except (KeyError, ValueError):
            raise VideoStreamInfoError(
                "Unable to determine frame rate of the video"
            )

    @property
    def total_frame_count(self):
        """The total number of frames in the video, or -1 if it could not be
        determined.
        """
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
        """The duration of the video, in seconds, or -1 if it could not be
        determined.
        """
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

    @property
    def size_bytes(self):
        """The size of the video on disk, in bytes, or -1 if it could not be
        determined.
        """
        try:
            return int(self.format_info["size"])
        except KeyError:
            pass

        logger.warning("Unable to determine video size; returning -1")
        return -1

    @property
    def mime_type(self):
        """The MIME type of the video, or None if it is not available."""
        return self._mime_type

    @property
    def encoding_str(self):
        """The video encoding string, or "" if it code not be found."""
        _encoding_str = str(self.stream_info.get("codec_tag_string", ""))
        if _encoding_str is None:
            logger.warning("Unable to determine encoding string")

        return _encoding_str

    def attributes(self):
        """Returns the list of class attributes that will be serialized."""
        return self.custom_attributes(dynamic=True)

    @classmethod
    def build_for(cls, video_path, mime_type=None, verbose=False):
        """Builds a VideoStreamInfo instance for the given video.

        Args:
            video_path: the path to the video
            mime_type: the MIME type of the video, if already known
            verbose: whether to generously log the process of extracting the
                stream info. By default, this is False

        Returns:
            a VideoStreamInfo instance
        """
        if verbose:
            logger.info("Getting stream info for '%s'", video_path)

        stream_info, format_info = _get_stream_info(
            video_path, verbose=verbose
        )

        if verbose:
            logger.info("Found format info: %s", etas.json_to_str(format_info))
            logger.info(
                "Found video stream: %s", etas.json_to_str(stream_info)
            )

        if mime_type is None:
            mime_type = etau.guess_mime_type(video_path)

        return cls(stream_info, format_info, mime_type=mime_type)

    @classmethod
    def from_dict(cls, d):
        """Constructs a VideoStreamInfo from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a VideoStreamInfo
        """
        stream_info = d["stream_info"]
        format_info = d["format_info"]
        mime_type = d.get("mime_type", None)
        return cls(stream_info, format_info, mime_type=mime_type)


class VideoStreamInfoError(Exception):
    """Exception raised when a problem with a VideoStreamInfo occurs."""

    pass


def _get_stream_info(inpath, verbose=False):
    # Get stream info via ffprobe
    ffprobe = FFprobe(
        opts=[
            "-show_format",  # get format info
            "-show_streams",  # get stream info
            "-print_format",
            "json",  # return in JSON format
        ]
    )
    out = ffprobe.run(inpath, decode=True, verbose=verbose)
    info = etas.load_json(out)

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


def get_frame_rate(inpath):
    """Get the frame rate of the input video.

    Args:
        inpath: video path

    Returns:
        the frame rate
    """
    return VideoStreamInfo.build_for(inpath).frame_rate


def get_frame_size(inpath):
    """Get the frame (width, height) of the input video.

    Args:
        inpath: video path

    Returns:
        the (width, height) of the video frames
    """
    return VideoStreamInfo.build_for(inpath).frame_size


def get_frame_count(inpath):
    """Get the number of frames in the input video.

    Args:
        inpath: video path

    Returns:
        the frame count, or -1 if it could not be determined
    """
    return VideoStreamInfo.build_for(inpath).total_frame_count


def get_duration(inpath):
    """Gets the duration of the video, in seconds.

    Args:
        inpath: video path

    Returns:
        the duration of the video, in seconds, or -1 if it could not be
            determined
    """
    return VideoStreamInfo.build_for(inpath).duration


def get_encoding_str(inpath):
    """Get the encoding string of the input video.

    Args:
        inpath: video path

    Returns:
        the encoding string
    """
    return VideoStreamInfo.build_for(inpath).encoding_str


def get_raw_frame_number(raw_frame_rate, raw_frame_count, fps, sampled_frame):
    """Get the raw frame number corresponding to the given sampled frame
    number.

    This function assumes that the sampling was performed using the command::

        FFmpeg(fps=fps).run(raw_video_path, ...)

    Args:
        raw_frame_rate: the frame rate of the raw video
        raw_frame_count: the number of frames in the raw video
        fps: the sampling rate that was used
        sampled_frame: the sampled frame number

    Returns:
        raw_frame: the raw frame number from the input video corresponding to
            the given sampled frame number
    """
    delta = raw_frame_rate / (1.0 * fps)
    raw_frame = np.minimum(
        np.ceil(delta * (sampled_frame - 0.5)), raw_frame_count
    )
    return int(raw_frame)


def read_video_as_array(video_path):
    """Reads the video from the given path into an in-memory array.

    CAUTION: in-memory videos are huge; use this at your own risk!

    Args:
        video_path: the path to the video to load

    Returns:
        a numpy array of size (num_frames, height, width, num_channels)
    """
    with FFmpegVideoReader(video_path) as vr:
        return np.asarray([img for img in vr])


def get_frame_range_for_clip(video_path, start_time=None, duration=None):
    """Gets a FrameRange instance describing the specified range of frames in
    the given video.

    Args:
        video_path: the path to the video
        start_time: the start timestamp, which can either be a float value of
            seconds or a string in "HH:MM:SS.XXX" format. If omitted, the
            beginning of the video is used
        duration: the clip duration, which can either be a float value of
            seconds or a string in "HH:MM:SS.XXX" format. If omitted, the clip
            extends to the end of the video

    Returns:
        a FrameRange
    """
    metadata = VideoMetadata.build_for(video_path)

    if start_time is not None:
        start_time = etaf.timestamp_to_seconds(start_time)
    else:
        start_time = 0

    start_frame = metadata.get_frame_number(timestamp=start_time)

    if duration is not None:
        stop_time = start_time + duration
        stop_frame = metadata.get_frame_number(timestamp=stop_time)
    else:
        stop_frame = metadata.total_frame_count

    return etaf.FrameRanges.build_simple(start_frame, stop_frame)


def extract_clip(
    video_path, output_path, start_time=None, duration=None, fast=False
):
    """Extracts the specified clip from the video.

    When fast=False, the following ffmpeg command is used::

        # Slower, more accurate option
        ffmpeg -ss <start_time> -i <video_path> -t <duration> <output_path>


    When fast is True, the following two-step ffmpeg process is used::

        # Faster, less accurate option
        ffmpeg -ss <start_time> -i <video_path> -t <duration> -c copy <tmp_path>
        ffmpeg -i <tmp_path> <output_path>

    Args:
        video_path: the path to the video
        output_path: the path to write the extracted video clip
        start_time: the start timestamp, which can either be a float value of
            seconds or a string in "HH:MM:SS.XXX" format. If omitted, the
            beginning of the video is used
        duration: the clip duration, which can either be a float value of
            seconds or a string in "HH:MM:SS.XXX" format. If omitted, the clip
            extends to the end of the video
        fast: whether to use a faster-but-potentially-less-accurate strategy to
            extract the clip. By default, the slow accurate strategy is used
    """
    #
    # @todo using FFmpeg directly here may not yield exact alignment with frame
    # numbers generated by FFmpegVideoReader, VideoProcessor, etc. Should we
    # use VideoProcessor here to ensure alignment?
    #
    in_opts = ["-vsync", "0"]
    if start_time is not None:
        if not etau.is_str(start_time):
            start_time = "%.3f" % start_time

        in_opts.extend(["-ss", start_time])

    out_opts = ["-vsync", "0"]
    if duration is not None:
        if not etau.is_str(duration):
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


def extract_clip_frames(
    video_path,
    output_patt,
    start_time=None,
    duration=None,
    keep_source_frame_numbers=False,
):
    """Extracts the frames of the specified clip from the video.

    Args:
        video_path: the path to the video
        output_patt: the pattern to which to write the extracted frames, like
            "/path/to/frames/%06d.png"
        start_time: the start timestamp, which can either be a float value of
            seconds or a string in "HH:MM:SS.XXX" format. If omitted, the
            beginning of the video is used
        duration: the clip duration, which can either be a float value of
            seconds or a string in "HH:MM:SS.XXX" format. If omitted, the clip
            extends to the end of the video
        keep_source_frame_numbers: whether to write the output frames with
            their frame numbers from the source video. If False (the default),
            the output frames are written with frame numbers starting from 1
    """
    frames = get_frame_range_for_clip(
        video_path, start_time=start_time, duration=duration
    )

    with FFmpegVideoReader(video_path, frames=frames) as vr:
        for idx, img in enumerate(vr, 1):
            if keep_source_frame_numbers:
                outpath = output_patt % vr.frame_number
            else:
                outpath = output_patt % idx

            etai.write(img, outpath)


def extract_frame(video_path, output_path, start_time=None):
    """Extracts a single frame from the local video or from the live stream and
    saves it to an image.

    This extraction function is the simplest of the extract command in this
    video code and strictly extracts a single frame either from the beginning
    of a video or stream or at some point in the video or stream.

    Uses the following ffmpeg command::

        ffmpeg -y -i <video_path> -ss <start_time> -vframes 1 ${output_path} \
            >/dev/null 2>&1

    Args:
        video_path: the path or m3u8 stream to a video
        output_path: the path to the image to write the frame
        start_time: a string in the ffmpeg time duration format, as follows,
            [-][HH:]MM:SS[.m...]
            https://ffmpeg.org/ffmpeg-utils.html#time-duration-syntax
    """
    in_opts = ["-vsync", "0"]
    if start_time is not None:
        if not etau.is_str(start_time):
            start_time = "%.3f" % start_time

        in_opts.extend(["-ss", start_time])

    ffmpeg = FFmpeg(in_opts=in_opts, out_opts=["-vsync", "0", "-vframes", "1"])
    ffmpeg.run(video_path, output_path)


def _make_ffmpeg_select_arg(frames):
    ss = "+".join(["eq(n\\,%d)" % (f - 1) for f in frames])
    return "select='%s'" % ss


def sample_select_frames(
    video_path, frames, output_patt=None, size=None, fast=False
):
    """Samples the specified frames of the video.

    This method is *intentionally* designed to be extremely graceful. It will
    sample whatever frames it can from the video you provide and will
    gracefully exit rather than raising an error if `ffmpeg` cannot understand
    some frames of the video you provide.

    When `fast=False`, this implementation uses VideoProcessor.

    When `fast=True`, this implementation uses ffmpeg's `-vf select` option.
    In this case, it may resort to `fast=False` internally if one of the
    following conditions occur:

        (a) more than 131072 frames are requested. This is a limitation of
            `subprocess` (cf. https://stackoverflow.com/q/29801975)

        (b) the fast implementation failed to generate at least 90%% of the
            target frames. This can happen if `ffmpeg -vf select` is confused
            by the the video it encounters. We have empirically found that
            VideoProcessor may be able to extract more frames such cases

    Args:
        video_path: the path to the video
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
        If `output_patt != None`, this function returns None.
        If `output_patt == None`, this method returns an (imgs, frames) tuple
        where `imgs` is the list of sampled frames, and `frames` is the list
        of frames that were succesfully sampled. If no errors were encountered,
        the output `frames` will match the input `frames`
    """
    if fast:
        try:
            return _sample_select_frames_fast(
                video_path, frames, output_patt, size
            )
        except SampleSelectFramesError as e:
            logger.warning("Select frames fast mode failed: '%s'", e)
            logger.info("Reverting to `fast=False`")

    return _sample_select_frames_slow(video_path, frames, output_patt, size)


class SampleSelectFramesError(Exception):
    """Exception raised when the `sample_select_frames` method encounters an
    error.
    """

    pass


def _sample_select_frames_fast(video_path, frames, output_patt, size):
    #
    # As per https://stackoverflow.com/q/29801975, one cannot pass an
    # argument of length > 131072 to subprocess. So, we have to make sure the
    # user isn't requesting too many frames to handle
    #
    select_arg_str = _make_ffmpeg_select_arg(frames)
    if len(select_arg_str) > 131072:
        raise SampleSelectFramesError(
            "Number of frames (%d) requested too large" % len(frames)
        )

    # If reading into memory, use `png` to ensure lossless-ness
    ext = os.path.splitext(output_patt)[1] if output_patt else ".png"

    #
    # Analogous to FFmpegVideoReader, our approach here is to gracefully
    # fail and just give the user however many frames we can...
    #

    with etau.TempDir() as d:
        # Sample frames to disk temporarily
        tmp_patt = os.path.join(d, "frame-%06d" + ext)
        ffmpeg = FFmpeg(
            size=size, out_opts=["-vf", select_arg_str, "-vsync", "0"]
        )

        try:
            ffmpeg.run(video_path, tmp_patt)
        except etau.ExecutableRuntimeError as e:
            # Graceful failure if frames couldn't be sampled
            logger.warning(etau.summarize_long_str(str(e), 500))
            logger.warning(
                "A sampling error occured; attempting to gracefully continue"
            )

        sampled_frames = etau.parse_pattern(tmp_patt)
        out_frames = [frames[i - 1] for i in sampled_frames]
        num_frames = len(sampled_frames)
        num_target_frames = len(frames)

        # Warn user if not all frames were sampled
        if num_frames < num_target_frames:
            logger.warning(
                "Only %d/%d expected frames were sampled",
                num_frames,
                num_target_frames,
            )

        #
        # If an insufficient number of frames were succesfully sampled, revert
        # to slow mode
        #
        target_percent_complete = 0.9  # warning: magic number
        percent_complete = num_frames / num_target_frames
        if percent_complete < target_percent_complete:
            raise SampleSelectFramesError(
                "We only managed to sample %.1f%% of the frames; this is "
                "below our target of %.1f%%, so let's try slow mode"
                % (100 * percent_complete, 100 * target_percent_complete)
            )

        # Move frames into place with correct output names
        if output_patt is not None:
            for sample_idx, frame_number in zip(sampled_frames, out_frames):
                tmp_path = tmp_patt % sample_idx
                outpath = output_patt % frame_number
                etau.move_file(tmp_path, outpath)

            return None

        # Return frames into memory
        imgs = []
        for sample_idx in sampled_frames:
            imgs.append(etai.read(tmp_patt % sample_idx))

        return imgs, out_frames


def _sample_select_frames_slow(video_path, frames, output_patt, size):
    # Parse parameters
    resize_images = size is not None

    # Sample frames to disk via VideoProcessor
    if output_patt:
        p = VideoProcessor(
            video_path, frames=frames, out_images_path=output_patt
        )
        with p:
            for img in p:
                if resize_images:
                    img = etai.resize(img, *size)

                p.write(img)

        return None

    # Sample frames in memory via FFmpegVideoReader
    imgs = []
    out_frames = []
    with FFmpegVideoReader(video_path, frames=frames) as r:
        for img in r:
            if resize_images:
                img = etai.resize(img, *size)

            imgs.append(img)
            out_frames.append(r.frame_number)

    return imgs, out_frames


def sample_first_frames(imgs_or_video_path, k, stride=1, size=None):
    """Samples the first k frames in a video.

    Args:
        imgs_or_video_path: can be either the path to the input video or an
            array of frames of size (num_frames, height, width, num_channels)
        k: number of frames to extract
        stride: number of frames to be skipped in between. By default, a
            contiguous array of frames in extracted
        size: an optional (width, height) to resize the sampled frames. By
            default, the native dimensions of the frames are used

    Returns:
        a numpy array of size [k, height, width, num_channels]
    """
    # Read frames ...
    if etau.is_str(imgs_or_video_path):
        # ... from disk
        video_path = imgs_or_video_path
        frames = [i for i in range(1, stride * k + 1, stride)]
        with FFmpegVideoReader(video_path, frames=frames) as vr:
            imgs_out = [img for img in vr]
    else:
        # ... from tensor
        imgs = imgs_or_video_path
        imgs_out = imgs[: (k * stride) : stride]

    # Duplicate last frame if necessary
    if k > len(imgs_out):
        num_repeats = k - len(imgs_out)
        imgs_out = np.asarray(imgs_out)
        imgs_out = np.concatenate(
            (
                imgs_out,
                np.repeat(imgs_out[-1][np.newaxis], num_repeats, axis=0),
            )
        )

    # Resize frames, if necessary
    if size is not None:
        imgs_out = [etai.resize(img, *size) for img in imgs_out]

    return np.array(imgs_out)


def uniformly_sample_frames(imgs_or_video_path, k, size=None):
    """Uniformly samples k frames from the video, always including the first
    and last frames.

    If k is larger than the number of frames in the video, duplicate frames
    will be included as necessary so that k frames are always returned.

    Args:
        imgs_or_video_path: can be either the path to the input video or an
            array of frames of size (num_frames, height, width, num_channels)
        k: the number of frames to extract
        size: an optional (width, height) to resize the sampled frames. By
            default, the native dimensions of the frames are used

    Returns:
        a numpy array of size [k, height, width, num_channels]
    """
    is_video = etau.is_str(imgs_or_video_path)
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
    """Samples clips from the video using a sliding window of the given
    length and stride.

    If k is larger than the number of frames in the video, duplicate frames
    will be included as necessary so that one window of size k can be returned.

    Args:
        imgs_or_video_path: can be either the path to the input video or an
            array of frames of size (num_frames, height, width, num_channels)
        k: the size of each window
        stride: the stride for sliding window
        size: an optional (width, height) to resize the sampled frames. By
            default, the native dimensions of the frames are used

    Returns:
        a numpy array of size [XXXX, k, height, width, num_channels]
    """
    is_video = etau.is_str(imgs_or_video_path)
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
        clip_inds = np.concatenate(
            (np.arange(1, num_frames + 1), [num_frames] * (k - num_frames))
        )[np.newaxis]

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
    """Extracts the keyframes from the video.

    Keyframes are a set of video frames that mark the start of a transition,
    and are faster to extract than an arbitrary frame.

    Args:
        video_path: the path to the video
        output_patt: an optional output pattern like "/path/to/frames-%d.png"
            specifying where to write the sampled frames. If omitted, the
            frames are instead returned in an in-memory list

    Returns:
        a list of the keyframes if output_patt is None, and None otherwise
    """
    if output_patt:
        # Write frames to disk via VideoProcessor
        p = VideoProcessor(
            video_path, keyframes_only=True, out_images_path=output_patt
        )
        with p:
            for img in p:
                p.write(img)

        return None

    # Load frames into memory via FFmpegVideoReader
    with FFmpegVideoReader(video_path, keyframes_only=True) as r:
        return [img for img in r]


def split_video(
    video_path,
    output_patt,
    num_clips=None,
    clip_duration=None,
    clip_size_bytes=None,
):
    """Splits the video into (roughly) equal-sized clips of the specified size.

    Exactly one keyword argument should be provided.

    This implementation uses an `ffmpeg` command of the following form::

        ffmpeg \
            -i input.mp4 \
            -c copy -segment_time SS.XXX -f segment -reset_timestamps 1 \
            output-%03d.mp4

    Args:
        video_path: the path to the video
        output_patt: an output pattern like "/path/to/clips-%03d.mp4"
            specifying where to write the output clips
        num_clips: the number of (roughly) equal size clips to break the
            video into
        clip_duration: the (approximate) duration, in seconds, of each clip to
            generate. The last clip may be shorter
        clip_size_bytes: the (approximate) size, in bytes, of each clip to
            generate. The last clip may be smaller
    """
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
        "-c:v",
        "copy",
        "-segment_time",
        "%.3f" % segment_time,
        "-f",
        "segment",
        "-reset_timestamps",
        "1",
    ]
    ffmpeg = FFmpeg(in_opts=in_opts, out_opts=out_opts)
    ffmpeg.run(video_path, output_patt)


class VideoProcessor(object):
    """Class for reading a video and writing a new video frame-by-frame.

    The typical usage is::

        with VideoProcessor(...) as p:
            for img in p:
                new_img = ... # process img
                p.write(new_img)
    """

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
        out_opts=None,
    ):
        """Creates a VideoProcessor instance.

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
        """
        if in_use_ffmpeg:
            self._reader = FFmpegVideoReader(
                inpath, frames=frames, keyframes_only=keyframes_only
            )
        elif keyframes_only:
            raise VideoProcessorError(
                "Must have `in_use_ffmpeg=True` when `keyframes_only=True`"
            )
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
                "The inferred frame rate '%s' cannot be used. You must "
                + "manually specify a frame rate"
                % str(self._reader.frame_rate)
            )
        self.out_size = out_size if out_size else self._reader.frame_size
        self.out_opts = out_opts

        if self._write_video:
            self._video_writer = self._new_video_writer(self.out_video_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        return len(self._reader)

    def __iter__(self):
        return self

    def __next__(self):
        return self.process()

    @property
    def input_frame_size(self):
        """The (width, height) of each input frame."""
        return self._reader.frame_size

    @property
    def output_frame_size(self):
        """The (width, height) of each output frame."""
        return self.out_size

    @property
    def input_frame_rate(self):
        """The input frame rate."""
        return self._reader.frame_rate

    @property
    def output_frame_rate(self):
        """The output frame rate."""
        return self.out_fps

    @property
    def frame_number(self):
        """The current frame number, or -1 if no frames have been read."""
        return self._reader.frame_number

    @property
    def frame_range(self):
        """The (first, last) frames for the current range, or (-1, -1) if no
        frames have been read.
        """
        return self._reader.frame_range

    @property
    def is_new_frame_range(self):
        """Whether the current frame is the first in a new range."""
        return self._reader.is_new_frame_range

    @property
    def total_frame_count(self):
        """The total number of frames in the video."""
        return self._reader.total_frame_count

    def process(self):
        """Returns the next frame.

        Returns:
            img: the frame as a numpy array
        """
        img = self._reader.read()
        if self._write_clips and self._reader.is_new_frame_range:
            self._reset_video_clip_writer()
        return img

    def write(self, img):
        """Appends the image to the output VideoWriter(s).

        Args:
            img: an numpy array containing the image
        """
        if self._write_images:
            etai.write(img, self.out_images_path % self._reader.frame_number)
        if self._write_video:
            self._video_writer.write(img)
        if self._write_clips:
            self._video_clip_writer.write(img)

    def close(self):
        """Closes the video processor."""
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
                outpath, self.out_fps, self.out_size, out_opts=self.out_opts
            )

        return OpenCVVideoWriter(outpath, self.out_fps, self.out_size)


class VideoProcessorError(Exception):
    """Exception raised when a problem with a VideoProcessor is encountered."""

    pass


class VideoReader(object):
    """Base class for reading videos.

    This class declares the following conventions:

        (a) `VideoReader`s implement the context manager interface. This means
            that models can optionally use context to perform any necessary
            setup and teardown, and so any code that uses a VideoReader
            should use the `with` syntax

        (b) `VideoReader`s support a `reset()` method that allows them to be
            reset back to their first frame, on demand
    """

    def __init__(self, inpath, frames):
        """Initializes a VideoReader base instance.

        Args:
            inpath: the input video path
            frames: one of the following quantities specifying a collection of
                frames to process:
                - None (all frames)
                - "*" (all frames)
                - a string like "1-3,6,8-10"
                - an `eta.core.frameutils.FrameRange` instance
                - an `eta.core.frameutils.FrameRanges` instance
                - an iterable, e.g., [1, 2, 3, 6, 8, 9, 10]. The frames do not
                    need to be in sorted order
        """
        self.inpath = inpath

        # Parse frames
        if frames is None or frames == "*":
            frames = "1-%d" % self.total_frame_count

        self._ranges = etaf.parse_frame_ranges(frames)
        self.frames = self._ranges.to_human_str()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        return len(self._ranges)

    def __iter__(self):
        return self

    def __next__(self):
        return self.read()

    def close(self):
        """Closes the VideoReader.

        Subclasses can override this method if necessary.
        """
        pass

    def reset(self):
        """Resets the VideoReader so that the next call to `read()` will return
        the first frame.
        """
        raise NotImplementedError("subclass must implement reset()")

    def _reset(self):
        """Base VideoReader implementation of `reset()`. Subclasses must call
        this method internally within `reset()`.
        """
        self._ranges.reset()

    @property
    def frame_number(self):
        """The current frame number, or -1 if no frames have been read."""
        return self._ranges.frame

    @property
    def frame_range(self):
        """The (first, last) frames for the current range, or (-1, -1) if no
        frames have been read.
        """
        return self._ranges.frame_range

    @property
    def is_new_frame_range(self):
        """Whether the current frame is the first in a new range."""
        return self._ranges.is_new_frame_range

    @property
    def encoding_str(self):
        """The video encoding string."""
        raise NotImplementedError("subclass must implement encoding_str")

    @property
    def frame_size(self):
        """The (width, height) of each frame."""
        raise NotImplementedError("subclass must implement frame_size")

    @property
    def frame_rate(self):
        """The frame rate."""
        raise NotImplementedError("subclass must implement frame_rate")

    @property
    def total_frame_count(self):
        """The total number of frames in the video."""
        raise NotImplementedError("subclass must implement total_frame_count")

    def read(self):
        """Reads the next frame.

        Returns:
            img: the next frame
        """
        raise NotImplementedError("subclass must implement read()")


class VideoReaderError(Exception):
    """Exception raised when a problem with a VideoReader is encountered."""

    pass


class FFmpegVideoReader(VideoReader):
    """Class for reading video using ffmpeg.

    The input video can be a standalone video file like "/path/to/video.mp4"
    or a directory of frames like "/path/to/frames/%05d.png". This path is
    passed directly to ffmpeg.

    A frames string like "1-5,10-15" can optionally be passed to only read
    certain frame ranges.

    This class uses 1-based indexing for all frame operations.
    """

    def __init__(self, inpath, frames=None, keyframes_only=False):
        """Creates an FFmpegVideoReader instance.

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
                - an `eta.core.frameutils.FrameRange` instance
                - an `eta.core.frameutils.FrameRanges` instance
                - an iterable, e.g., [1, 2, 3, 6, 8, 9, 10]. The frames do not
                    need to be in sorted order
            keyframes_only: whether to only read keyframes. By default, this
                is False. When this is True, `frames` is interpreted as
                keyframe numbers
        """
        # Parse args
        if keyframes_only:
            in_opts = ["-skip_frame", "nokey", "-vsync", "0"]
        else:
            in_opts = None

        self._stream_info = VideoStreamInfo.build_for(inpath)
        self._ffmpeg = FFmpeg(
            in_opts=in_opts,
            out_opts=[
                "-vsync",
                "0",  # never omit frames
                "-f",
                "image2pipe",  # pipe frames to stdout
                "-vcodec",
                "rawvideo",  # output will be raw video
                "-pix_fmt",
                "rgb24",  # pixel format
            ],
        )
        self._raw_frame = None

        self._open_stream(inpath)
        super(FFmpegVideoReader, self).__init__(inpath, frames)

    def close(self):
        """Closes the FFmpegVideoReader."""
        self._ffmpeg.close()

    def reset(self):
        """Resets the FFmpegVideoReader."""
        self.close()
        self._reset()
        self._open_stream(self.inpath)

    @property
    def encoding_str(self):
        """The video encoding string."""
        return self._stream_info.encoding_str

    @property
    def frame_size(self):
        """The (width, height) of each frame."""
        return self._stream_info.frame_size

    @property
    def frame_rate(self):
        """The frame rate."""
        return self._stream_info.frame_rate

    @property
    def total_frame_count(self):
        """The total number of frames in the video, or 0 if it could not be
        determined.
        """
        return self._stream_info.total_frame_count

    def read(self):
        """Reads the next frame.

        If any problem is encountered while reading the frame, a warning is
        logged and a StopIteration is raised. This means that FFmpegVideoReader
        will gracefully fail when malformed videos are encountered.

        Returns:
            img: the next frame

        Raises:
            StopIteration: if there are no more frames to process or the next
                frame could not be read or parsed for any reason
        """
        for _ in range(max(0, self.frame_number), next(self._ranges)):
            if not self._grab():
                logger.warning(
                    "Failed to grab frame %d. Raising StopIteration now",
                    self.frame_number,
                )
                raise StopIteration
        return self._retrieve()

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
                self.frame_number,
            )
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
                self.frame_number,
            )
            raise StopIteration

    def _open_stream(self, inpath):
        self._ffmpeg.run(inpath, "-")
        self._raw_frame = None


class SampledFramesVideoReader(VideoReader):
    """Class for reading video stored as sampled frames on disk.

    This class uses 1-based indexing for all frame operations.
    """

    def __init__(self, frames_dir, frames=None):
        """Creates a SampledFramesVideoReader instance.

        Args:
            frames_dir: the path to a directory of frames, which must be
                parseable by `eta.core.utils.parse_dir_pattern()`
            frames: one of the following optional quantities specifying a
                collection of frames to process:
                - None (all frames - the default)
                - "*" (all frames)
                - a string like "1-3,6,8-10"
                - an `eta.core.frameutils.FrameRange` instance
                - an `eta.core.frameutils.FrameRanges` instance
                - an iterable, e.g., [1, 2, 3, 6, 8, 9, 10]. The frames do not
                    need to be in sorted order
        """
        self._frames_dir = None
        self._frames_patt = None
        self._frame_size = None
        self._total_frame_count = None

        all_frames = self._init_for_frames_dir(frames_dir)
        if frames is None or frames == "*":
            frames = all_frames

        super(SampledFramesVideoReader, self).__init__(frames_dir, frames)

    def reset(self):
        """Resets the SampledFramesVideoReader."""
        self.close()
        self._reset()

    @property
    def encoding_str(self):
        """The video encoding string."""
        return None

    @property
    def frame_size(self):
        """The (width, height) of each frame."""
        return self._frame_size

    @property
    def frame_rate(self):
        """The frame rate."""
        return None

    @property
    def total_frame_count(self):
        """The total number of frames in the video, or 0 if it could not be
        determined.
        """
        return self._total_frame_count

    def read(self):
        """Reads the next frame.

        Returns:
            img: the next frame

        Raises:
            StopIteration: if there are no more frames to process or the next
                frame could not be read or parsed for any reason
        """
        frame_number = next(self._ranges)
        try:
            return etai.read(self._frames_patt % frame_number)
        except:
            logger.warning(
                "Failed to grab frame %d. Raising StopIteration now",
                frame_number,
            )
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
    """Class for reading video using OpenCV.

    The input video can be a standalone video file like "/path/to/video.mp4"
    or a directory of frames like "/path/to/frames/%05d.png". This path is
    passed directly to cv2.VideoCapture. So, for example, if you specify a
    directory of frames, the frame numbering must start from 0-3.

    A frames string like "1-5,10-15" can optionally be passed to only read
    certain frame ranges.

    This class uses 1-based indexing for all frame operations.
    """

    def __init__(self, inpath, frames=None):
        """Creates an OpenCVVideoReader instance.

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
                - an `eta.core.frameutils.FrameRange` instance
                - an `eta.core.frameutils.FrameRanges` instance
                - an iterable, e.g., [1, 2, 3, 6, 8, 9, 10]. The frames do not
                    need to be in sorted order

        Raises:
            OpenCVVideoReaderError: if the input video could not be opened
        """
        self._cap = None

        self._open_stream(inpath)
        super(OpenCVVideoReader, self).__init__(inpath, frames)

    def close(self):
        """Closes the OpenCVVideoReader."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def reset(self):
        """Resets the OpenCVVideoReader."""
        self.close()
        self._reset()
        self._open_stream(self.inpath)

    @property
    def encoding_str(self):
        """Return the video encoding string."""
        try:
            # OpenCV 3
            code = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        except AttributeError:
            # OpenCV 2
            # pylint: disable=no-member
            code = int(self._cap.get(cv2.cv.CV_CAP_PROP_FOURCC))
        return FOURCC.int_to_str(code)

    @property
    def frame_size(self):
        """The (width, height) of each frame."""
        try:
            # OpenCV 3
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        except AttributeError:
            # OpenCV 2
            return (
                # pylint: disable=no-member
                int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                # pylint: disable=no-member
                int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)),
            )

    @property
    def frame_rate(self):
        """The frame rate."""
        try:
            # OpenCV 3
            return float(self._cap.get(cv2.CAP_PROP_FPS))
        except AttributeError:
            # OpenCV 2
            # pylint: disable=no-member
            return float(self._cap.get(cv2.cv.CV_CAP_PROP_FPS))

    @property
    def total_frame_count(self):
        """The total number of frames in the video."""
        try:
            # OpenCV 3
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except AttributeError:
            # OpenCV 2
            # pylint: disable=no-member
            return int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    def read(self):
        """Reads the next frame.

        If any problem is encountered while reading the frame, a warning is
        logged and a StopIteration is raised. This means that OpenCVVideoReader
        will gracefully fail when malformed videos are encountered.

        Returns:
            img: the next frame

        Raises:
            StopIteration: if there are no more frames to process or the next
                frame could not be read or parsed for any reason
        """
        for _ in range(max(0, self.frame_number), next(self._ranges)):
            if not self._grab():
                logger.warning(
                    "Failed to grab frame %d. Raising StopIteration now",
                    self.frame_number,
                )
                raise StopIteration
        return self._retrieve()

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
                self.frame_number,
            )
            raise StopIteration

    def _open_stream(self, inpath):
        self._cap = cv2.VideoCapture(inpath)
        if not self._cap.isOpened():
            raise OpenCVVideoReaderError("Unable to open '%s'" % inpath)


class OpenCVVideoReaderError(VideoReaderError):
    """Error raised when a problem with an OpenCVVideoReader is encountered."""

    pass


class VideoWriter(object):
    """Base class for writing videos.

    This class declares the following conventions:

        (a) `VideoWriter`s implement the context manager interface. This means
            that subclasses can optionally use context to perform any necessary
            setup and teardown, and so any code that uses a VideoWriter
            should use the `with` syntax
    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, img):
        """Appends the image to the output video.

        Args:
            img: a numpy array
        """
        raise NotImplementedError("subclass must implement write()")

    def close(self):
        """Closes the VideoWriter."""
        pass


class VideoWriterError(Exception):
    """Exception raised when a problem with a VideoWriter is encountered."""

    pass


class FFmpegVideoWriter(VideoWriter):
    """Class for writing videos using ffmpeg."""

    def __init__(self, outpath, fps, size, out_opts=None):
        """Creates an FFmpegVideoWriter instance.

        Args:
            outpath: the output video path. Existing files are overwritten,
                and the directory is created if necessary
            fps: the frame rate
            size: the (width, height) of each frame
            out_opts: an optional list of output options for FFmpeg
        """
        self.outpath = outpath
        self.fps = fps
        self.size = size

        self._ffmpeg = FFmpeg(
            in_opts=[
                "-f",
                "rawvideo",  # input will be raw video
                "-vcodec",
                "rawvideo",  # input will be raw video
                "-s",
                "%dx%d" % self.size,  # frame size
                "-pix_fmt",
                "rgb24",  # pixel format
                "-r",
                str(self.fps),  # frame rate
            ],
            out_opts=out_opts,
        )
        self._ffmpeg.run("-", self.outpath)

    def write(self, img):
        """Appends the image to the output video.

        Args:
            img: a numpy array
        """
        self._ffmpeg.stream(img.tobytes())

    def close(self):
        """Closes the FFmpegVideoWriter."""
        self._ffmpeg.close()


class OpenCVVideoWriter(VideoWriter):
    """Class for writing videos using cv2.VideoWriter.

    Uses the default encoding scheme for the extension of the output path.
    """

    def __init__(self, outpath, fps, size):
        """Creates an OpenCVVideoWriter instance.

        Args:
            outpath: the output video path. Existing files are overwritten,
                and the directory is created if necessary
            fps: the frame rate
            size: the (width, height) of each frame

        Raises:
            OpenCVVideoWriterError: if the writer failed to open
        """
        self.outpath = outpath
        self.fps = fps
        self.size = size
        self._writer = cv2.VideoWriter()

        etau.ensure_path(self.outpath)
        self._writer.open(self.outpath, -1, self.fps, self.size, True)
        if not self._writer.isOpened():
            raise OpenCVVideoWriterError("Unable to open '%s'" % self.outpath)

    def write(self, img):
        """Appends the image to the output video.

        Args:
            img: a numpy array
        """
        self._writer.write(etai.rgb_to_bgr(img))

    def close(self):
        """Closes the video writer."""
        # self._writer.release()  # warns to use a separate thread
        threading.Thread(target=self._writer.release, args=()).start()


class OpenCVVideoWriterError(VideoWriterError):
    """Exception raised when a problem with an OpenCVVideoWriter is
    encountered.
    """

    pass


class FFprobe(object):
    """Interface for the ffprobe binary."""

    DEFAULT_GLOBAL_OPTS = ["-loglevel", "error"]

    def __init__(self, global_opts=None, opts=None):
        """Creates an FFprobe instance.

        Args:
            global_opts: a list of global options for ffprobe. By default,
                self.DEFAULT_GLOBAL_OPTS is used
            opts: a list of options for ffprobe
        """
        self._global_opts = global_opts or self.DEFAULT_GLOBAL_OPTS
        self._opts = opts or []

        self._args = None
        self._p = None

    @property
    def cmd(self):
        """The last executed ffprobe command string, or None if run() has not
        yet been called.
        """
        return " ".join(self._args) if self._args else None

    def run(self, inpath, decode=False, verbose=False):
        """Run the ffprobe binary with the specified input path.

        Args:
            inpath: the input path

        Returns:
            out: the stdout from the ffprobe binary
            decode: whether to decode the output bytes into utf-8 strings. By
                default, the raw bytes are returned
            verbose: whether to log the ffprobe command used at INFO level. By
                default, this is False

        Raises:
            ExecutableNotFoundError: if the ffprobe binary cannot be found
            ExecutableRuntimeError: if the ffprobe binary raises an error
                during execution
        """
        self._args = (
            ["ffprobe"] + self._global_opts + self._opts + ["-i", inpath]
        )

        if verbose:
            logger.info("Executing '%s'", self.cmd)
        else:
            logger.debug("Executing '%s'", self.cmd)

        try:
            self._p = Popen(self._args, stdout=PIPE, stderr=PIPE)
        except EnvironmentError as e:
            if e.errno == errno.ENOENT:
                raise etau.ExecutableNotFoundError(exe="ffprobe")

            raise

        out, err = self._p.communicate()
        if self._p.returncode != 0:
            raise etau.ExecutableRuntimeError(self.cmd, err)

        return out.decode("utf-8") if decode else out


class FFmpeg(object):
    """Interface for the ffmpeg binary.

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
    """

    DEFAULT_GLOBAL_OPTS = ["-loglevel", "error"]

    DEFAULT_IN_OPTS = ["-vsync", "0"]

    DEFAULT_VIDEO_OUT_OPTS = [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "0",
        "-an",
    ]

    DEFAULT_IMAGES_OUT_OPTS = ["-vsync", "0"]

    def __init__(
        self,
        fps=None,
        size=None,
        scale=None,
        global_opts=None,
        in_opts=None,
        out_opts=None,
    ):
        """Creates an FFmpeg instance.

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
        """
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
        """The last executed ffmpeg command string, or None if run() has not
        yet been called.
        """
        return " ".join(self._args) if self._args else None

    def run(self, inpath, outpath, verbose=False):
        """Run the ffmpeg binary with the specified input/outpath paths.

        Args:
            inpath: the input path. If inpath is "-", input streaming mode is
                activated and data can be passed via the stream() method
            outpath: the output path. Existing files are overwritten, and the
                directory is created if needed. If outpath is "-", output
                streaming mode is activated and data can be read via the
                read() method
            verbose: whether to log the ffmpeg command used at INFO level. By
                default, this is False

        Raises:
            ExecutableNotFoundError: if the ffmpeg binary cannot be found
            ExecutableRuntimeError: if the ffmpeg binary raises an error during
                execution
        """
        self.is_input_streaming = inpath == "-"
        self.is_output_streaming = outpath == "-"

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
            ["ffmpeg"]
            + self._global_opts
            + in_opts
            + ["-i", inpath]
            + out_opts
            + [outpath]
        )

        if not self.is_output_streaming:
            etau.ensure_path(outpath)

        if verbose:
            logger.info("Executing '%s'", self.cmd)
        else:
            logger.debug("Executing '%s'", self.cmd)

        try:
            self._p = Popen(self._args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        except EnvironmentError as e:
            if e.errno == errno.ENOENT:
                raise etau.ExecutableNotFoundError(exe="ffmpeg")

            raise

        # Run non-streaming jobs immediately
        if not (self.is_input_streaming or self.is_output_streaming):
            err = self._p.communicate()[1]
            if self._p.returncode != 0:
                raise etau.ExecutableRuntimeError(self.cmd, err)

    def stream(self, string):
        """Writes the string to ffmpeg's stdin stream.

        Args:
            string: the string to write

        Raises:
            FFmpegStreamingError: if input streaming mode is not active
        """
        if not self.is_input_streaming:
            raise FFmpegStreamingError("Not currently input streaming")
        self._p.stdin.write(string)

    def read(self, num_bytes):
        """Reads the given number of bytes from ffmpeg's stdout stream.

        Args:
            num_bytes: the number of bytes to read

        Returns:
            the bytes

        Raises:
            FFmpegStreamingError: if output streaming mode is not active
        """
        if not self.is_output_streaming:
            raise FFmpegStreamingError("Not currently output streaming")
        return self._p.stdout.read(num_bytes)

    def close(self):
        """Closes a streaming ffmpeg program, if necessary."""
        if self.is_input_streaming or self.is_output_streaming:
            self._p.stdin.close()
            self._p.stdout.close()
            self._p.stderr.close()
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
                filters.append("setdar=dar={0}/{1}".format(*size))

        elif scale:
            filters.append("scale=iw*{0}:ih*{0}".format(scale))
        return ["-vf", ",".join(filters)] if filters else []


class FFmpegStreamingError(Exception):
    """Exception raised when an error occurs while operating an FFmpeg instance
    in streaming mode.
    """

    pass


class FOURCC(object):
    """Class reprsesenting a FOURCC code."""

    def __init__(self, _i=None, _s=None):
        """Creates a FOURCC instance.

        Don't call this directly! Instead, use `from_str ` or `from_int` to
        create a FOURCC instance.

        Args:
            _i: the integer representation of the FOURCC code
            _s: the string representation of the FOURCC code
        """
        if _i:
            self.int = _i
            self.str = FOURCC.int_to_str(_i)
        elif _s:
            self.int = FOURCC.str_to_int(_s)
            self.str = _s

    @classmethod
    def from_str(cls, s):
        """Construct a FOURCC instance from a string.

        Args:
            s: the string representation of the FOURCC code

        Returns:
            a FOURCC instance
        """
        return cls(_s=s)

    @classmethod
    def from_int(cls, i):
        """Construct a FOURCC instance from an integer.

        Args:
            i: the integer representation of the FOURCC code

        Returns:
            a FOURCC instance
        """
        return cls(_i=i)

    @staticmethod
    def str_to_int(s):
        """Returns the integer representation of the given FOURCC string.

        Args:
            s: the string representation of the FOURCC code

        Returns:
            the integer representation of the FOURCC code
        """
        try:
            # OpenCV 3
            return cv2.VideoWriter_fourcc(*s)
        except AttributeError:
            # OpenCV 2
            # pylint: disable=no-member
            return cv2.cv.FOURCC(*s)

    @staticmethod
    def int_to_str(i):
        """Returns the string representation of the given FOURCC integer.

        Args:
            i: the integer representation of the FOURCC code

        Returns:
            the string representation of the FOURCC code
        """
        return (
            chr((i & 0x000000FF) >> 0)
            + chr((i & 0x0000FF00) >> 8)
            + chr((i & 0x00FF0000) >> 16)
            + chr((i & 0xFF000000) >> 24)
        )
