'''
ETA core type system.

@todo document the semantics of each type more fully.

Copyright 2018, Voxel51, LLC
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

import numbers
import os

import eta.core.utils as etau
import eta.core.video as etav


###### Utilities ###############################################################


def parse_type(type_str):
    '''Parses the type string and returns the associated type.

    Args:
        type_str: a string like "eta.core.types.Video

    Returns:
        the Type class referred to by the given type string

    Raises:
        TypeError: is the type string was not a recognized type
    '''
    try:
        return etau.get_class(type_str)
    except ImportError:
        raise TypeError("Unknown type '%s'" % type_str)


def is_pipeline(type_):
    '''Returns True/False if the given type is a subtype of Pipeline.'''
    return issubclass(type_, Pipeline)


def is_module(type_):
    '''Returns True/False if the given type is a subtype of Module.'''
    return issubclass(type_, Module)


def is_builtin(type_):
    '''Returns True/False if the given type is a subtype of Builtin.'''
    return issubclass(type_, Builtin)


def is_data(type_):
    '''Returns True/False if the given type is a subtype of Data.'''
    return issubclass(type_, Data)


###### Base type ###############################################################


class Type(object):
    '''The base type for all types.'''


###### Pipeline types ##########################################################


class Pipeline(Type):
    '''The base type for pipelines.'''
    pass


###### Module types ############################################################


class Module(Type):
    '''The base type for modules.'''
    pass


###### Builtin types ###########################################################


class Builtin(Type):
    '''The base type for builtins, which are types whose values are consumed
    directly.
    '''

    @staticmethod
    def is_valid_value(self, val):
        '''Returns True/False if `val` is a valid value for this type.'''
        raise NotImplementedError("subclass must implement is_valid_value()")


class Null(Builtin):
    '''A JSON null value. None in Python.'''

    @staticmethod
    def is_valid_value(self, val):
        return val is None


class Boolean(Builtin):
    '''A JSON boolean value. A bool in Python.'''

    @staticmethod
    def is_valid_value(self, val):
        return isinstance(val, bool)


class String(Builtin):
    '''A JSON string. A str in Python.'''

    @staticmethod
    def is_valid_value(self, val):
        return isinstance(val, six.string_types)


class Number(Builtin):
    '''A numeric value.'''

    @staticmethod
    def is_valid_value(self, val):
        return isinstance(val, numbers.Number)


class Array(Builtin):
    '''A JSON array. A list in Python.'''

    @staticmethod
    def is_valid_value(self, val):
        return isinstance(val, list)


class StringArray(Array):
    '''An array of strings in JSON. A list of strings in Python.'''

    @staticmethod
    def is_valid_value(self, val):
        return (
            Array.is_valid_value(val) and
            all(String.is_valid_value(s) for s in val)
        )


class Object(Builtin):
    '''An object in JSON. A dict in Python.'''

    @staticmethod
    def is_valid_value(self, val):
        return isinstance(val, dict)


###### Data types ##############################################################


class Data(Type):
    '''The base type for data, which are types that are stored on disk.'''

    @staticmethod
    def is_valid_path(self, val):
        '''Returns True/False if `path` is a valid filepath for this type.'''
        raise NotImplementedError("subclass must implement is_valid_path()")


class Directory(Data):
    '''The base type for directories that contain data.

    Examples:
        /path/to/dir
    '''

    @staticmethod
    def is_valid_path(self, val):
        return String.is_valid_value(val)


class File(Data):
    '''The base type for file.

    Examples:
        /path/to/data.txt
    '''

    @staticmethod
    def is_valid_path(self, val):
        return String.is_valid_value(val)


class FileSequence(Data):
    '''The base type for a collection of files indexed by one numeric
    parameter.

    Examples:
        /path/to/data-%05d.txt
    '''

    @staticmethod
    def is_valid_path(self, val):
        if not String.is_valid_value(val):
            return False
        try:
            val % 1
            return True
        except TypeError:
            return False


class JSONFile(File):
    '''The base type for JSON files.

    Examples:
        /path/to/data.json
    '''

    @staticmethod
    def is_valid_path(self, val):
        return File.is_valid_path(val) and etau.has_extenstion(val, ".json")


class JSONFileSequence(FileSequence):
    '''The base type for a collection of JSON files indexed by one numeric
    parameter.

    Examples:
        /path/to/data-%05d.json
    '''

    @staticmethod
    def is_valid_path(self, val):
        return (
            FileSequence.is_valid_path(val) and
            etau.has_extenstion(val, ".json")
        )


class DualFileSequence(Data):
    '''The base type for a collection of files indexed by two numeric
    parameters.

    Examples:
        /path/to/data-%05d-%05d.json
    '''

    @staticmethod
    def is_valid_path(self, val):
        if not String.is_valid_value(val):
            return False
        try:
            val % (1, 2)
            return True
        except TypeError:
            return False


class Weights(File):
    '''The base type for model weights files.

    Examples:
        /path/to/weights.npz
    '''
    pass


class Image(File):
    '''An image.

    Examples:
        /path/to/image.png
    '''

    @staticmethod
    def is_valid_path(self, val):
        return File.is_valid_path(val) and etau.is_supported_image_type(val)


class Video(Data):
    '''The base type for a single video.

    Examples:
        /path/to/video.mp4
        /path/to/frame-%05d.png
    '''

    @staticmethod
    def is_valid_path(self, val):
        return (
            VideoFile.is_valid_path(val) or
            ImageSequence.is_valid_path(val)
        )


class VideoFile(Video, File):
    '''A video represented as a single (encoded) video file.

    Examples:
        /path/to/video.mp4
    '''

    @staticmethod
    def is_valid_path(self, val):
        return File.is_valid_path(val) and etau.is_supported_video_type(val)


class ImageSequence(Video, FileSequence):
    '''A video represented as a sequence of images with one numeric parameter.

    Examples:
        /path/to/frame-%05d.png
    '''

    @staticmethod
    def is_valid_path(self, val):
        return (
            FileSequence.is_valid_path(val) and
            etau.is_supported_image_type(val)
        )


class VideoSequece(FileSequence):
    '''A sequence of video files with one numeric parameter.

    Examples:
        /path/to/video-%05d.mp4
    '''

    @staticmethod
    def is_valid_path(self, val):
        return (
            FileSequence.is_valid_path(val) and
            etau.is_supported_video_type(val)
        )


class VideoClips(DualFileSequence):
    '''A sequence of video files with two numeric parameters.

    Examples:
        /path/to/video-%05d-%05d.mp4
    '''

    @staticmethod
    def is_valid_path(self, val):
        return (
            DualFileSequence.is_valid_path(val) and
            etau.is_supported_video_type(val)
        )


class EventDetection(JSONFile):
    '''Per-frame binary detections of an event in a video.

    Examples:
        /path/to/event_detection.json
    '''
    pass


class EventSeries(JSONFile):
    '''A series of events in a video.

    Examples:
        /path/to/event_series.json
    '''
    pass


class Frame(JSONFile):
    '''Detected objects in a frame.

    Examples:
        /path/to/frame.json
    '''
    pass


class FrameSequence(JSONFileSequence):
    '''Detected objects in a video represented as a collection of Frame files
    indexed by one numeric parameter.

    Examples:
        /path/to/frame-%05d.json
    '''
    pass
