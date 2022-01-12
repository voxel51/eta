"""
Utilities for working with frames of videos.

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

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import numpy as np

import eta.core.serial as etas
import eta.core.utils as etau


def frame_number_to_timestamp(frame_number, total_frame_count, duration):
    """Converts the given frame number to a timestamp.

    Args:
        frame_number: the frame number of interest
        total_frame_count: the total number of frames in the video
        duration: the length of the video (in seconds)

    Returns:
        the timestamp (in seconds) of the given frame number in the video
    """
    if total_frame_count == 1:
        return 0

    alpha = (frame_number - 1) / (total_frame_count - 1)
    return alpha * duration


def timestamp_to_frame_number(timestamp, duration, total_frame_count):
    """Converts the given timestamp in a video to a frame number.

    Args:
        timestamp: the timestamp (in seconds or "HH:MM:SS.XXX" format) of
            interest
        duration: the length of the video (in seconds)
        total_frame_count: the total number of frames in the video

    Returns:
        the frame number associated with the given timestamp in the video
    """
    timestamp = timestamp_to_seconds(timestamp)
    alpha = timestamp / duration
    return 1 + int(round(alpha * (total_frame_count - 1)))


def timestamp_to_seconds(timestamp):
    """Converts a timestamp that is in either seconds or "HH:MM:SS.XXX" format
    to seconds.

    Args:
        timestamp: a timestamp in seconds or "HH:MM:SS.XXX" format

    Returns:
        a timestamp in seconds
    """
    if etau.is_str(timestamp):
        return timestamp_str_to_seconds(timestamp)

    return timestamp


def timestamp_str_to_seconds(timestamp):
    """Converts a timestamp string in "HH:MM:SS.XXX" format to seconds.

    Args:
        timestamp: a string in "HH:MM:SS.XXX" format

    Returns:
        the number of seconds
    """
    return sum(
        float(n) * m
        for n, m in zip(reversed(timestamp.split(":")), (1, 60, 3600))
    )


def world_time_to_timestamp(world_time, start_time):
    """Converts the given world time to a timestamp in a video.

    If one (but not both) of the datetimes are timezone-aware, the other
    datetime is assumed to be expressed in UTC time.

    Args:
        world_time: a datetime describing a time of interest
        start_time: a datetime indicating the start time of the video

    Returns:
        the corresponding timestamp (in seconds) in the video
    """
    return etau.datetime_delta_seconds(start_time, world_time)


def world_time_to_frame_number(
    world_time, start_time, duration, total_frame_count
):
    """Converts the given world time to a frame number in a video.

    Args:
        world_time: a datetime describing a time of interest
        start_time: a datetime indicating the start time of the video
        duration: the length of the video (in seconds)
        total_frame_count: the total number of frames in the video

    Returns:
        the corresponding timestamp (in seconds) in the video
    """
    timestamp = world_time_to_timestamp(world_time, start_time)
    return timestamp_to_frame_number(timestamp, duration, total_frame_count)


def parse_frame_ranges(frames):
    """Parses the given frames quantity into a FrameRanges instance.

    Args:
        frames: one of the following quantities:
            - a string like "1-3,6,8-10"
            - a FrameRange or FrameRanges instance
            - an iterable, e.g., [1, 2, 3, 6, 8, 9, 10]. The frames do not
                need to be in sorted order

    Returns:
        a FrameRanges instance describing the frames
    """
    if etau.is_str(frames):
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
    """Class representing a monotonically increasing and disjoint series of
    frames.
    """

    def __init__(self, ranges=None):
        """Creates a FrameRanges instance.

        Args:
            ranges: can either be a human-readable frames string like
                "1-3,6,8-10" or an iterable of (first, last) tuples, which must
                be disjoint and monotonically increasing. By default, an empty
                instance is created
        """
        self._ranges = []
        self._idx = 0
        self._started = False

        if ranges is not None:
            if etau.is_str(ranges):
                ranges = self._parse_frames_str(ranges)

            self._set_ranges(ranges)

    def __str__(self):
        return self.to_human_str()

    def __len__(self):
        return sum(len(r) for r in self._ranges)

    def __bool__(self):
        return bool(self._ranges)

    def __iter__(self):
        self.reset()
        return self

    def __contains__(self, frame):
        for r in self._ranges:
            if frame in r:
                return True

        return False

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

    def _set_ranges(self, ranges):
        self.clear()
        for new_range in ranges:
            self._ingest_range(new_range)

    def _ingest_range(self, new_range):
        first, last = new_range
        end = self.limits[1]

        if end is not None and first <= end:
            raise FrameRangesError("Expected first:%d > end:%d" % (first, end))

        self._ranges.append(FrameRange(first, last))

    @property
    def limits(self):
        """A (first, last) tuple describing the limits of the frame ranges.

        Returns (None, None) if the instance is empty.
        """
        if not self:
            return (None, None)

        first = self._ranges[0].limits[0]
        last = self._ranges[-1].limits[1]
        return (first, last)

    @property
    def num_ranges(self):
        """The number of `FrameRange`s in this object."""
        return len(self._ranges)

    @property
    def frame(self):
        """The current frame number, or -1 if no frames have been read."""
        if self._started:
            return self._ranges[self._idx].frame

        return -1

    @property
    def ranges(self):
        """A serialized string representation of this object."""
        return self.to_human_str()

    @property
    def frame_range(self):
        """The (first, last) values for the current range, or (-1, -1) if no
        frames have been read.
        """
        if self._started:
            return self._ranges[self._idx].first, self._ranges[self._idx].last

        return (-1, -1)

    @property
    def is_new_frame_range(self):
        """Whether the current frame is the first in a new range."""
        if self._started:
            return self._ranges[self._idx].is_first_frame

        return False

    @property
    def is_contiguous(self):
        """Determines whether the frame range is contiguous, i.e., whether it
        consists of a single `FrameRange`.

        If you want to ensure that this instance does not contain trivial
        adjacent `FrameRange`s, then call `simplify()` first.

        Returns:
            True/False
        """
        return self.num_ranges == 1

    def reset(self):
        """Resets the FrameRanges instance so that the next frame will be the
        first.
        """
        for r in self._ranges[: (self._idx + 1)]:
            r.reset()

        self._started = False
        self._idx = 0

    def clear(self):
        """Clears and resets the FrameRanges instance."""
        self._ranges = []
        self.reset()

    def add_range(self, new_range):
        """Adds the given frame range to the instance.

        Args:
            new_range: a (first, last) tuple describing the range

        Raises:
            FrameRangesError: if the new range is not disjoint and
                monotonically increasing w.r.t. the existing ranges
        """
        self._ingest_range(new_range)

    def merge(self, *args):
        """Merges the given FrameRange and/or FrameRanges instances into this
        range.

        Merges are successful regardless of overlap between ranges.

        This operation will reset the instance.

        Args:
            *args: one or more FrameRange or FrameRanges instances
        """
        new_frames = set(self.to_list())
        for r in args:
            new_frames.update(r.to_list())

        new_ranges = _iterable_to_ranges(new_frames)
        self._set_ranges(new_ranges)

    def simplify(self):
        """Simplifies the frame ranges, if possible, by merging any adjacent
        `FrameRange` instances into a single range.

        This operation will reset the instance if any simplification was
        performed.

        Returns:
            True/False whether any simplification was performed
        """
        if not self:
            return False

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
            return False

        self._set_ranges(new_ranges)
        return True

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Returns:
            a list of attributes
        """
        return ["ranges"]

    def to_range_tuples(self):
        """Returns the list of (first, last) tuples defining the frame ranges
        in this instance.

        Returns:
            a list of (first, last) tuples
        """
        return [r.limits for r in self._ranges]

    def to_list(self):
        """Returns the list of frames, in sorted order, described by this
        object.

        Returns:
            list of frames
        """
        frames = []
        for r in self._ranges:
            frames += r.to_list()

        return frames

    def to_human_str(self):
        """Returns a human-readable string representation of this object.

        Returns:
            a string like "1-3,6,8-10" describing the frame ranges
        """
        return ",".join([fr.to_human_str() for fr in self._ranges])

    def to_bools(self, total_frame_count=None):
        """Returns a boolean array indicating the frames described by this
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
        """
        if total_frame_count is None:
            total_frame_count = self.limits[1]

        bools = np.zeros(total_frame_count, dtype=bool)

        inds = [i - 1 for i in self.to_list() if i <= total_frame_count]
        bools[inds] = True

        return bools

    @staticmethod
    def build_simple(first, last):
        """Builds a FrameRanges from a simple [first, last] range.

        Args:
            first: the first frame
            last: the last frame

        Returns:
            a FrameRanges instance
        """
        return FrameRanges(ranges=[(first, last)])

    @classmethod
    def from_bools(cls, bools):
        """Constructs a FrameRanges object from a boolean array describing the
        frames in the ranges.

        Note that the 0-based indexes in the boolean array are converted to
        1-based frame numbers. In other words, the returned FrameRanges
        contains `frame` iff `bools[frame - 1] == True`.

        Args:
            bools: a boolean array

        Returns:
            a FrameRanges instance
        """
        return cls.from_iterable(1 + np.flatnonzero(bools))

    @classmethod
    def from_human_str(cls, frames_str):
        """Constructs a FrameRanges object from a human-readable frames string.

        Args:
            frames_str: a human-readable frames string like "1-3,6,8-10"

        Returns:
            a FrameRanges instance

        Raises:
            FrameRangesError: if the frames string is invalid
        """
        return cls(ranges=frames_str)

    @classmethod
    def from_iterable(cls, frames):
        """Constructs a FrameRanges object from an iterable of frames.

        The frames do not need to be in sorted order, and they may contain
        duplicates.

        Args:
            frames: an iterable of frames, e.g., [1, 2, 3, 6, 8, 9, 10]

        Returns:
            a FrameRanges instance

        Raises:
            FrameRangesError: if the frames list is invalid
        """
        return cls(ranges=_iterable_to_ranges(frames))

    @classmethod
    def from_frame_range(cls, frame_range):
        """Constructs a FrameRanges instance from a FrameRange instance.

        Args:
            frame_range: a FrameRange instance

        Returns:
            a FrameRanges instance
        """
        return cls(ranges=[(frame_range.first, frame_range.last)])

    @classmethod
    def from_dict(cls, d):
        """Constructs a FrameRanges from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a FrameRanges instance
        """
        ranges = d.get("ranges", None)
        return cls(ranges=ranges)


class FrameRangesError(Exception):
    """Exception raised when an invalid FrameRanges is encountered."""

    pass


class FrameRange(etas.Serializable):
    """Class representing a range of frames."""

    def __init__(self, first, last):
        """Creates a FrameRange instance.

        Args:
            first: the first frame in the range (inclusive)
            last: the last frame in the range (inclusive)
        """
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
                "Expected first:%d <= last:%d" % (first, last)
            )

    def __str__(self):
        return self.to_human_str()

    def __len__(self):
        return self.last + 1 - self.first

    def __bool__(self):
        return True

    def __iter__(self):
        self.reset()
        return self

    def __contains__(self, frame):
        return self.first <= frame <= self.last

    def __next__(self):
        if self._frame < 0:
            self._frame = self.first
        elif self._frame < self.last:
            self._frame += 1
        else:
            raise StopIteration

        return self._frame

    def reset(self):
        """Resets the FrameRange instance so that the next frame will be the
        first.
        """
        self._frame = -1

    @property
    def frame(self):
        """The current frame number, or -1 if no frames have been read."""
        if self._frame < 0:
            return -1

        return self._frame

    @property
    def limits(self):
        """A (first, last) tuple describing the frame range."""
        return (self.first, self.last)

    @property
    def is_first_frame(self):
        """Whether the current frame is first in the range."""
        return self._frame == self.first

    def to_list(self):
        """Returns the list of frames in the range.

        Returns:
            a list of frames
        """
        return list(range(self.first, self.last + 1))

    def to_human_str(self):
        """Returns a human-readable string representation of the range.

        Returns:
            a string like "1-5"
        """
        if self.first == self.last:
            return "%d" % self.first

        return "%d-%d" % (self.first, self.last)

    @classmethod
    def from_human_str(cls, frames_str):
        """Constructs a FrameRange object from a human-readable string.

        Args:
            frames_str: a human-readable frames string like "1-5"

        Returns:
            a FrameRange instance

        Raises:
            FrameRangeError: if the frame range string is invalid
        """
        try:
            v = list(map(int, frames_str.split("-")))
            return cls(v[0], v[-1])
        except ValueError:
            raise FrameRangeError(
                "Invalid frame range string '%s'" % frames_str
            )

    @classmethod
    def from_iterable(cls, frames):
        """Constructs a FrameRange object from an iterable of frames.

        The frames do not need to be in sorted order, and there may be
        duplicates. However, the frames must define a single interval.

        Args:
            frames: an iterable of frames, e.g., [1, 2, 3, 4, 5]

        Returns:
            a FrameRange instance

        Raises:
            FrameRangeError: if the frame range list is invalid
        """
        ranges = list(_iterable_to_ranges(frames))
        if len(ranges) != 1:
            raise FrameRangeError("Invalid frame range list %s" % frames)

        return cls(*ranges[0])

    @classmethod
    def from_dict(cls, d):
        """Constructs a FrameRange from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a FrameRange instance
        """
        return cls(d["first"], d["last"])


class FrameRangeError(Exception):
    """Exception raised when an invalid FrameRange is encountered."""

    pass


def _iterable_to_ranges(vals):
    # This will convert numpy arrays to list, and it's important to do this
    # before checking for falseness below, since numpy arrays don't support it
    vals = sorted(set(vals))

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
