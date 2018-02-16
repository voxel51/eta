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
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import eta.core.utils as etau


###### Utilities ###############################################################


def parse_type(type_str):
    '''Parses the type string and returns the associated type.'''
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


def is_data(type_):
    '''Returns True/False if the given type is a subtype of Data.'''
    return issubclass(type_, Data)


###### Base type ###############################################################


class Type(object):
    '''The base type for all types.'''
    pass


###### Pipeline types ##########################################################


class Pipeline(Type):
    '''The base type for pipelines.'''
    pass


###### Module types ############################################################


class Module(Type):
    '''The base type for modules.'''
    pass


###### Data types ##############################################################


class Data(Type):
    '''The base type for data.'''
    pass


class Builtin(Data):
    '''Builtin data types that can be directly read/written from JSON.'''
    pass


class Null(Builtin):
    '''A JSON null value. None in Python.'''
    pass


class Boolean(Builtin):
    '''A JSON boolean value. A bool in Python.'''
    pass


class String(Builtin):
    '''A JSON string. A str in Python.'''
    pass


class FramesString(Builtin):
    '''A string like `1,5-10,20` or `*` specifying a set of frame ranges
    in a video.
    '''
    pass


class Number(Builtin):
    '''A JSON numeric value. A float in Python.'''
    pass


class Array(Builtin):
    '''A JSON array. A list in Python.'''
    pass


class Object(Builtin):
    '''A JSON object. A dict in Python.'''
    pass


class Point(Object):
    '''An point with `x` and `y` coordinates.'''
    pass


class Rectangle(Object):
    '''A rectangle specified by `top_left` and bottom_right` points.'''
    pass


class Directory(Data):
    '''The base type for directories that contain data.'''
    pass


class File(Data):
    '''The base type for a data file.'''
    pass


class FileSequence(Data):
    '''The base type for a sequence/collection of files.'''
    pass


class Weights(File):
    '''The base type for model weights files.'''
    pass


class Image(File):
    '''An image.'''
    pass


class Video(Data):
    '''The base type for videos.'''
    pass


class VideoFile(Video, File):
    '''A video represented as a single (encoded) video file.

    Example:
        video.mp4
    '''
    pass


class VideoFiles(Video, FileSequence):
    '''A one-parameter sequence of video files.

    Example:
        video-%05d.mp4
    '''
    pass


class VideoClips(Video, FileSequence):
    '''A two-parameter sequence of video files.

    Example:
        video-%05d-%05d.mp4
    '''
    pass


class ImageSequence(Video, FileSequence):
    '''A video represented as a sequence of images in a directory.

    Example:
        image-%05d.png
    '''
    pass


class EventDetection(File):
    '''Per-frame binary detections of an event in a video.'''
    pass


class EventSeries(File):
    '''A series of events in a video.'''
    pass


class Frame(File):
    '''Detected objects in a frame.'''
    pass


class FrameSequence(FileSequence):
    '''Detected objects in a video represented as a sequence of Frame JSON
    files.

    Example:
        frame-%05d.json
    '''
    pass
