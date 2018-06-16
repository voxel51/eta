'''
ETA core type system.

More types may be defined in other modules, but they must inherit from the
base type `eta.core.types.Type` defined here.

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

import eta
import eta.core.utils as etau
import eta.core.weights as etaw


###### Utilities ###############################################################


def parse_type(type_str):
    '''Parses the type string and returns the associated Type.

    Raises:
        TypeError: is the type string was not a recognized type
    '''
    try:
        type_cls = etau.get_class(type_str)
    except ImportError:
        raise TypeError("Unknown type '%s'" % type_str)

    if not issubclass(type_cls, Type):
        raise TypeError("Type '%s' must be a subclass of Type" % type_cls)

    return type_cls


def resolve_value(val, type_):
    '''Resolves the given value of the given type.'''
    if isinstance(type_, Weights):
        val = etaw.find_weights(val)

    return val


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


def is_concrete_data(type_):
    '''Returns True/False if the given type is a subtype of ConcreteData.'''
    return issubclass(type_, ConcreteData)


def is_abstract_data(type_):
    '''Returns True/False if the given type is a subtype of AbstractData.'''
    return issubclass(type_, AbstractData) and not is_concrete_data(type_)


class ConcreteDataParams(object):
    '''Class encapsulating the string formatting parameters for generating
    paths for ConcreteData types.
    '''

    def __init__(self):
        self._params = {
            "idx": eta.config.default_sequence_idx,
            "image_ext": eta.config.default_image_ext,
            "video_ext": eta.config.default_video_ext,
        }

    def render_for(self, name):
        '''Render the type parameters for use with field `name`.

        Returns:
            a params dict
        '''
        params = self._params.copy()
        params["name"] = name
        return params


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

    Builtin types must know how to validate whether a given value is valid for
    their type.
    '''

    @staticmethod
    def is_valid_value(val):
        '''Returns True/False if `val` is a valid value for this type.'''
        raise NotImplementedError("subclass must implement is_valid_value()")


class Null(Builtin):
    '''A JSON null value. None in Python.'''

    @staticmethod
    def is_valid_value(val):
        return val is None


class Boolean(Builtin):
    '''A JSON boolean value. A bool in Python.'''

    @staticmethod
    def is_valid_value(val):
        return isinstance(val, bool)


class String(Builtin):
    '''A JSON string. A str in Python.'''

    @staticmethod
    def is_valid_value(val):
        return isinstance(val, six.string_types)


class Number(Builtin):
    '''A numeric value.'''

    @staticmethod
    def is_valid_value(val):
        return isinstance(val, numbers.Number)


class Array(Builtin):
    '''A JSON array. A list in Python.'''

    @staticmethod
    def is_valid_value(val):
        return isinstance(val, list)


class StringArray(Array):
    '''An array of strings in JSON. A list of strings in Python.'''

    @staticmethod
    def is_valid_value(val):
        return (
            Array.is_valid_value(val) and
            all(String.is_valid_value(s) for s in val)
        )


class Object(Builtin):
    '''An object in JSON. A dict in Python.'''

    @staticmethod
    def is_valid_value(val):
        return isinstance(val, dict)


class Point(Object):
    '''An (x, y) coordinate point defined by "x" and "y" coordinates, which
    must be nonnegative.

    Typically, Points represent coordinates of pixels in images.

    Example:
        ```json
        {
            "x": 0,
            "y": 128
        }
        ```
    '''

    @staticmethod
    def is_valid_value(val):
        return (
            Object.is_valid_value(val) and
            "x" in val and
            "y" in val and
            val["x"] >= 0 and
            val["y"] >= 0
        )


class RelativePoint(Object):
    '''An (x, y) coordinate point defined by "x" and "y" coordinates, which
    must take values in [0, 1].

    Typically, RelativePoints describe the coordiantes of pixels in images
    relative to the image dimensions. This is useful so that the coordinates
    can be automatically rendered for images of different resolution.

    Example:
        ```json
        {
            "x": 0.5,
            "y": 0.5
        }
        ```
    '''

    @staticmethod
    def is_valid_value(val):
        return (
            Object.is_valid_value(val) and
            "x" in val and
            "y" in val and
            0 <= val["x"] <= 1 and
            0 <= val["y"] <= 1
        )


class Rectangle(Object):
    '''A rectangle specified by its top-left and bottom-right Points.

    Example:
        ```json
        {
            "top_left": {
                "x": 32,
                "y": 64
            },
            "bottom_right": {
                "x": 64,
                "y": 128
            }
        }
        ```
    '''

    @staticmethod
    def is_valid_value(val):
        return (
            Object.is_valid_value(val) and
            "top_left" in val and
            "bottom_right" in val and
            Point.is_valid_value(val["top_left"]) and
            Point.is_valid_value(val["bottom_right"])
        )


class RelativeRectangle(Object):
    '''A rectangle specified by its top-left and bottom-right RelativePoints.

    Example:
        ```json
        {
            "top_left": {
                "x": 0.25,
                "y": 0.5
            },
            "bottom_right": {
                "x": 0.5,
                "y": 0.75
            }
        }
        ```
    '''

    @staticmethod
    def is_valid_value(val):
        return (
            Object.is_valid_value(val) and
            "top_left" in val and
            "bottom_right" in val and
            RelativePoint.is_valid_value(val["top_left"]) and
            RelativePoint.is_valid_value(val["bottom_right"])
        )


###### Data types ##############################################################


class Data(Type):
    '''The base type for data, which are types that are stored on disk.

    Data types must know how to validate whether a given path is valid path
    for their type.
    '''

    @staticmethod
    def is_valid_path(path):
        '''Returns True/False if `path` is a valid filepath for this type.'''
        raise NotImplementedError("subclass must implement is_valid_path()")


class ConcreteData(Data):
    '''The base type for concrete data types, which represent well-defined data
    types that can be written to disk.

    Concrete data types must know how to generate their own output paths
    explicitly.
    '''

    @staticmethod
    def gen_path(basedir, params):
        '''Generates the output path for the given data

        Args:
            basedir: the base output directory
            params: a dictionary of string formatting parameters generated by
                ConcreteDataParams.render_for()

        Returns:
            the appropriate path for the data
        '''
        raise NotImplementedError("subclass must implement gen_path()")


class AbstractData(Data):
    '''The base type for abstract data types, which define base data types that
    encapsulate one or more `ConcreteData` types.

    Abstract data types allow modules to declare that their inputs or
    parameters can accept one of many equivalent representations of a given
    type.

    However, abstract data types cannot be used for module outputs, which must
    return a concrete data type.
    '''

    @staticmethod
    def gen_path(*args, **kwargs):
        '''Raises an error clarifying that AbstractData types cannot generate
        output paths.
        '''
        raise ValueError("AbstractData types cannot generate output paths")


class File(AbstractData):
    '''The abstract data type describing a file.'''

    @staticmethod
    def is_valid_path(path):
        return String.is_valid_value(path)


class FileSequence(AbstractData):
    '''The abstract data type describing a collection of files indexed by one
    numeric parameter.
    '''

    @staticmethod
    def is_valid_path(path):
        if not String.is_valid_value(path):
            return False
        try:
            path % 1
            return True
        except TypeError:
            return False


class DualFileSequence(AbstractData):
    '''The abstract data type describing a collection of files indexed by two
    numeric parameters.
    '''

    @staticmethod
    def is_valid_path(path):
        if not String.is_valid_value(path):
            return False
        try:
            path % (1, 2)
            return True
        except TypeError:
            return False


class Directory(ConcreteData):
    '''The base type for directories that contain data.

    Examples:
        /path/to/dir
    '''

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}").format(**params)

    @staticmethod
    def is_valid_path(path):
        return String.is_valid_value(path)


class Weights(File, ConcreteData):
    '''A model weights file of any type.

    Examples:
        /path/to/weights.npz
    '''

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}").format(**params)


class Image(AbstractData):
    '''The abstract data type representing an image.

    Examples:
        /path/to/image.png
        /path/to/image.jpg
    '''

    @staticmethod
    def is_valid_path(path):
        return ImageFile.is_valid_path(path)


class ImageFile(Image, File, ConcreteData):
    '''An image file.

    ETA uses OpenCV to read images, so any image type understood by OpenCV is
    valid.

    Examples:
        /path/to/image.png
    '''

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}{image_ext}").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etau.is_supported_image_type(path)


class Video(AbstractData):
    '''The abstract data type representing a single video.

    Examples:
        /path/to/video.mp4
        /path/to/video/%05d.png
    '''

    @staticmethod
    def is_valid_path(path):
        return (
            VideoFile.is_valid_path(path) or
            ImageSequence.is_valid_path(path)
        )


class VideoFile(Video, File, ConcreteData):
    '''A video represented as a single encoded video file.

    ETA uses ffmpeg (default) or OpenCV (if specified) to load video files,
    so any video encoding types understood by these tools are valid.

    Examples:
        /path/to/video.mp4
    '''

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}{video_ext}").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etau.is_supported_video_type(path)


class ImageSequence(Video, FileSequence, ConcreteData):
    '''A video represented as a sequence of images with one numeric parameter.

    ETA uses ffmpeg (default) or OpenCV (if specified) to load videos stored
    as sequences of images, so any image types understood by these tools are
    valid.

    Examples:
        /path/to/video/%05d.png
    '''

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(
            basedir, "{name}", "{idx}{image_ext}").format(**params)

    @staticmethod
    def is_valid_path(path):
        return (
            FileSequence.is_valid_path(path) and
            etau.is_supported_image_type(path)
        )


class VideoSequece(FileSequence, ConcreteData):
    '''A sequence of encoded video files with one numeric parameter.

    Examples:
        /path/to/videos/%05d.mp4
    '''

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(
            basedir, "{name}", "{idx}{video_ext}").format(**params)

    @staticmethod
    def is_valid_path(path):
        return (
            FileSequence.is_valid_path(path) and
            etau.is_supported_video_type(path)
        )


class VideoClips(DualFileSequence, ConcreteData):
    '''A sequence of video files with two numeric parameters.

    Examples:
        /path/to/videos/%05d-%05d.mp4
    '''

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(
            basedir, "{name}", "{idx}-{idx}{video_ext}").format(**params)

    @staticmethod
    def is_valid_path(path):
        return (
            DualFileSequence.is_valid_path(path) and
            etau.is_supported_video_type(path)
        )


class JSONFile(File, ConcreteData):
    '''The base type for JSON files.

    Examples:
        /path/to/data.json
    '''

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}.json").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etau.has_extension(path, ".json")


class JSONFileSequence(FileSequence, ConcreteData):
    '''The base type for a collection of JSON files indexed by one numeric
    parameter.

    Examples:
        /path/to/jsons/%05d.json
    '''

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(
            basedir, "{name}", "{idx}.json").format(**params)

    @staticmethod
    def is_valid_path(path):
        return (
            FileSequence.is_valid_path(path) and
            etau.has_extension(path, ".json")
        )


class EventDetection(JSONFile):
    '''A per-frame binary event detection.

    This type is implemented in ETA by the `eta.core.events.EventDetection`
    class.

    Examples:
        /path/to/event_detection.json
    '''

    pass


class EventSeries(JSONFile):
    '''A series of events in a video.

    This type is implemented in ETA by the `eta.core.events.EventSeries` class.

    Examples:
        /path/to/event_series.json
    '''

    pass


class Frame(JSONFile):
    '''Detected objects in a frame.

    This type is implemented in ETA by the `eta.core.objects.Frame` class.

    Examples:
        /path/to/frame.json
    '''

    pass


class FrameSequence(JSONFileSequence):
    '''Detected objects in a video represented as a collection of Frame files
    indexed by one numeric parameter.

    Examples:
        /path/to/frames/%05d.json
    '''

    pass
