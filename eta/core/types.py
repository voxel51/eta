"""
Core type system for ETA modules and pipelines.

More types may be defined in other modules, but they must inherit from the
base type `eta.core.types.Type` defined here.

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
import six

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging
import os

import eta
import eta.core.features as etaf
import eta.core.image as etai
import eta.core.learning as etal
import eta.core.utils as etau
import eta.core.video as etav

etat = etau.lazy_import("eta.core.tfutils")


logger = logging.getLogger(__name__)


###### Utilities ##############################################################


def parse_type(type_str):
    """Parses the type string and returns the associated Type.

    Raises:
        TypeError: is the type string was not a recognized type
    """
    try:
        type_cls = etau.get_class(type_str)
    except ImportError:
        raise TypeError("Unknown type '%s'" % type_str)

    if not issubclass(type_cls, Type):
        raise TypeError("Type '%s' must be a subclass of Type" % type_cls)

    return type_cls


def is_pipeline(type_):
    """Returns True/False if the given type is a subtype of Pipeline."""
    return issubclass(type_, Pipeline)


def is_module(type_):
    """Returns True/False if the given type is a subtype of Module."""
    return issubclass(type_, Module)


def is_builtin(type_):
    """Returns True/False if the given type is a subtype of Builtin."""
    return issubclass(type_, Builtin)


def is_data(type_):
    """Returns True/False if the given type is a subtype of Data."""
    return issubclass(type_, Data)


def is_concrete_data(type_):
    """Returns True/False if the given type is a subtype of ConcreteData."""
    return issubclass(type_, ConcreteData)


def is_abstract_data(type_):
    """Returns True/False if the given type is a subtype of AbstractData."""
    return issubclass(type_, AbstractData) and not is_concrete_data(type_)


class ConcreteDataParams(object):
    """Class encapsulating the string formatting parameters for generating
    paths for ConcreteData types.
    """

    def __init__(self):
        """Creates a ConcreteDataParams instance."""
        self._params = {
            "name": None,
            "idx": eta.config.default_sequence_idx,
            "image_ext": eta.config.default_image_ext,
            "video_ext": eta.config.default_video_ext,
        }

    @property
    def default(self):
        """The default parameters dictionary."""
        return self._params

    def render_for(self, name, hint=None):
        """Render the type parameters for use with field `name`.

        Args:
            name: the field name
            hint: an optional path hint from which to infer custom parameter
                values (e.g. sequence indices or image/video extensions)

        Returns:
            a params dict
        """
        params = self._params.copy()
        params["name"] = name
        if hint:
            hint_idx = etau.parse_sequence_idx_from_pattern(hint)
            if hint_idx:
                params["idx"] = hint_idx
            if etai.is_supported_image(hint):
                params["image_ext"] = os.path.splitext(hint)[1]
            if etav.is_supported_video_file(hint):
                params["video_ext"] = os.path.splitext(hint)[1]
        return params


###### Base type ##############################################################


class Type(object):
    """The base type for all types."""


###### Pipeline types #########################################################


class Pipeline(Type):
    """The base type for pipelines."""

    pass


###### Module types ###########################################################


class Module(Type):
    """The base type for modules."""

    pass


###### Builtin types ##########################################################


class Builtin(Type):
    """The base type for builtins, which are types whose values are consumed
    directly.

    Builtin types must know how to validate whether a given value is valid for
    their type.
    """

    @staticmethod
    def is_valid_value(val):
        """Returns True/False if `val` is a valid value for this type."""
        raise NotImplementedError("subclass must implement is_valid_value()")


class Null(Builtin):
    """A JSON null value. None in Python."""

    @staticmethod
    def is_valid_value(val):
        return val is None


class Boolean(Builtin):
    """A JSON boolean value. A bool in Python."""

    @staticmethod
    def is_valid_value(val):
        return isinstance(val, bool)


class String(Builtin):
    """A JSON string. A str in Python."""

    @staticmethod
    def is_valid_value(val):
        return etau.is_str(val)


class Number(Builtin):
    """A numeric value."""

    @staticmethod
    def is_valid_value(val):
        return etau.is_numeric(val)


class Object(Builtin):
    """An object in JSON. A dict in Python."""

    @staticmethod
    def is_valid_value(val):
        return isinstance(val, dict)


class Array(Builtin):
    """A JSON array. A list in Python."""

    @staticmethod
    def is_valid_value(val):
        return isinstance(val, list)


class StringArray(Array):
    """An array of strings in JSON. A list of strings in Python."""

    @staticmethod
    def is_valid_value(val):
        return Array.is_valid_value(val) and all(
            String.is_valid_value(s) for s in val
        )


class ObjectArray(Array):
    """An array of objects in JSON. A list of dicts in Python."""

    @staticmethod
    def is_valid_value(val):
        return Array.is_valid_value(val) and all(
            Object.is_valid_value(o) for o in val
        )


class Config(Object):
    """Base class for objects that are serialized instances of
    `eta.core.config.Config` classes.
    """

    pass


class Featurizer(Config):
    """Configuration of an `eta.core.features.Featurizer`.

    This types is implemented in ETA by the
    `eta.core.features.FeaturizerConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etaf.FeaturizerConfig(val)
            return True
        except Exception as e:
            logger.error("An error occured while parsing the Featurizer value")
            logger.error(e, exc_info=True)
            return False


class ImageFeaturizer(Featurizer):
    """Configuration of an `eta.core.features.ImageFeaturizer`.

    This types is implemented in ETA by the
    `eta.core.features.ImageFeaturizerConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etaf.ImageFeaturizerConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occured while parsing the ImageFeaturizer value"
            )
            logger.error(e, exc_info=True)
            return False


class VideoFramesFeaturizer(Featurizer):
    """Configuration of an `eta.core.features.VideoFramesFeaturizer`.

    This types is implemented in ETA by the
    `eta.core.features.VideoFramesFeaturizerConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etaf.VideoFramesFeaturizerConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occured while parsing the VideoFramesFeaturizer "
                "value"
            )
            logger.error(e, exc_info=True)
            return False


class VideoFeaturizer(Featurizer):
    """Configuration of an `eta.core.features.VideoFeaturizer`.

    This types is implemented in ETA by the
    `eta.core.features.VideoFeaturizerConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etaf.VideoFeaturizerConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occured while parsing the VideoFeaturizer value"
            )
            logger.error(e, exc_info=True)
            return False


class Model(Config):
    """Configuration for an `eta.core.learning.Model`.

    This type is implemented in ETA by the `eta.core.learning.ModelConfig`
    class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.ModelConfig(val)
            return True
        except Exception as e:
            logger.error("An error occured while parsing the Model value")
            logger.error(e, exc_info=True)
            return False


class ImageModel(Model):
    """Configuration for an `eta.core.learning.ImageModel`.

    This type is implemented in ETA by the `eta.core.learning.ImageModelConfig`
    class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.ImageModelConfig(val)
            return True
        except Exception as e:
            logger.error("An error occured while parsing the ImageModel value")
            logger.error(e, exc_info=True)
            return False


class VideoModel(Model):
    """Configuration for an `eta.core.learning.VideoModel`.

    This type is implemented in ETA by the `eta.core.learning.VideoModelConfig`
    class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.VideoModelConfig(val)
            return True
        except Exception as e:
            logger.error("An error occured while parsing the VideoModel value")
            logger.error(e, exc_info=True)
            return False


class Classifier(Model):
    """Configuration for an `eta.core.learning.Classifier`.

    This type is implemented in ETA by the `eta.core.learning.ClassifierConfig`
    class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.ClassifierConfig(val)
            return True
        except Exception as e:
            logger.error("An error occured while parsing the Classifier value")
            logger.error(e, exc_info=True)
            return False


class ImageClassifier(Classifier):
    """Configuration for an `eta.core.learning.ImageClassifier`.

    This type is implemented in ETA by the
    `eta.core.learning.ImageClassifierConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.ImageClassifierConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occured while parsing the ImageClassifier value"
            )
            logger.error(e, exc_info=True)
            return False


class VideoFramesClassifier(Classifier):
    """Configuration for an `eta.core.learning.VideoFramesClassifier`.

    This type is implemented in ETA by the
    `eta.core.learning.VideoFramesClassifierConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.VideoFramesClassifierConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occured while parsing the VideoFramesClassifier "
                "value"
            )
            logger.error(e, exc_info=True)
            return False


class VideoClassifier(Classifier):
    """Configuration for an `eta.core.learning.VideoClassifier`.

    This type is implemented in ETA by the
    `eta.core.learning.VideoClassifierConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.VideoClassifierConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occured while parsing the VideoClassifier value"
            )
            logger.error(e, exc_info=True)
            return False


class Detector(Model):
    """Configuration for an `eta.core.learning.Detector`.

    This type is implemented in ETA by the `eta.core.learning.DetectorConfig`
    class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.DetectorConfig(val)
            return True
        except Exception as e:
            logger.error("An error occured while parsing the Detector value")
            logger.error(e, exc_info=True)
            return False


class ObjectDetector(Detector):
    """Configuration for an `eta.core.learning.ObjectDetector`.

    This type is implemented in ETA by the
    `eta.core.learning.ObjectDetectorConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.ObjectDetectorConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occured while parsing the ObjectDetector value"
            )
            logger.error(e, exc_info=True)
            return False


class VideoFramesObjectDetector(Detector):
    """Configuration for an `eta.core.learning.VideoFramesObjectDetector`.

    This type is implemented in ETA by the
    `eta.core.learning.VideoFramesObjectDetectorConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.VideoFramesObjectDetectorConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occured while parsing the VideoFramesObjectDetector "
                "value"
            )
            logger.error(e, exc_info=True)
            return False


class VideoObjectDetector(Detector):
    """Configuration for an `eta.core.learning.VideoObjectDetector`.

    This type is implemented in ETA by the
    `eta.core.learning.VideoObjectDetectorConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.VideoObjectDetectorConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occured while parsing the VideoObjectDetector value"
            )
            logger.error(e, exc_info=True)
            return False


class VideoEventDetector(Detector):
    """Configuration for an `eta.core.learning.VideoEventDetector`.

    This type interface is implemented in ETA by the
    `eta.core.learning.VideoEventDetectorConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.VideoEventDetectorConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occurred while parsing the VideoEventDetector value"
            )
            logger.error(e, exc_info=True)
            return False


class SemanticSegmenter(Model):
    """Configuration for an `eta.core.learning.SemanticSegmenter`.

    This type is implemented in ETA by the
    `eta.core.learning.SemanticSegmenterConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.SemanticSegmenterConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occured while parsing the SemanticSegmenter value"
            )
            logger.error(e, exc_info=True)
            return False


class ImageSemanticSegmenter(SemanticSegmenter):
    """Configuration for an `eta.core.learning.ImageSemanticSegmenter`.

    This type is implemented in ETA by the
    `eta.core.learning.ImageSemanticSegmenterConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.ImageSemanticSegmenterConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occured while parsing the ImageSemanticSegmenter "
                "value"
            )
            logger.error(e, exc_info=True)
            return False


class VideoSemanticSegmenter(SemanticSegmenter):
    """Configuration for an `eta.core.learning.VideoSemanticSegmenter`.

    This type is implemented in ETA by the
    `eta.core.learning.VideoSemanticSegmenterConfig` class.
    """

    @staticmethod
    def is_valid_value(val):
        try:
            etal.VideoSemanticSegmenterConfig(val)
            return True
        except Exception as e:
            logger.error(
                "An error occured while parsing the VideoSemanticSegmenter "
                "value"
            )
            logger.error(e, exc_info=True)
            return False


###### Data types #############################################################


class Data(Type):
    """The base type for data, which are types that are stored on disk.

    Data types must know how to:
        (a) validate whether a given path is valid path for their type
        (b) return metadata about an instance of data on disk
    """

    @staticmethod
    def is_valid_path(path):
        """Returns True/False if `path` is a valid filepath for this type."""
        raise NotImplementedError("subclass must implement is_valid_path()")

    @staticmethod
    def get_metadata(path):
        """Returns a dictionary containing metadata about the data at the
        given path.
        """
        raise NotImplementedError("subclass must implement get_metadata()")


class ConcreteData(Data):
    """The base type for concrete data types, which represent well-defined data
    types that can be written to disk.

    Concrete data types must know how to generate their own output paths
    explicitly.
    """

    @staticmethod
    def gen_path(basedir, params):
        """Generates the output path for the given data

        Args:
            basedir: the base output directory
            params: a dictionary of string formatting parameters generated by
                ConcreteDataParams.render_for()

        Returns:
            the appropriate path for the data
        """
        raise NotImplementedError("subclass must implement gen_path()")


class AbstractData(Data):
    """The base type for abstract data types, which define base data types that
    encapsulate one or more `ConcreteData` types.

    Abstract data types allow modules to declare that their inputs or
    parameters can accept one of many equivalent representations of a given
    type.

    However, abstract data types cannot be used for module outputs, which must
    return a concrete data type.
    """

    @staticmethod
    def gen_path(*args, **kwargs):
        """Raises an error clarifying that AbstractData types cannot generate
        output paths.
        """
        raise ValueError("AbstractData types cannot generate output paths")


class File(AbstractData):
    """The abstract data type describing a file."""

    @staticmethod
    def is_valid_path(path):
        return String.is_valid_value(path)


class FileSequence(AbstractData):
    """The abstract data type describing a collection of files indexed by one
    numeric parameter.
    """

    @staticmethod
    def is_valid_path(path):
        if not String.is_valid_value(path):
            return False
        try:
            _ = path % 1
            return True
        except TypeError:
            return False


class DualFileSequence(AbstractData):
    """The abstract data type describing a collection of files indexed by two
    numeric parameters.
    """

    @staticmethod
    def is_valid_path(path):
        if not String.is_valid_value(path):
            return False
        try:
            _ = path % (1, 2)
            return True
        except TypeError:
            return False


class FileSet(AbstractData):
    """The abstract data type describing a collection of files indexed by one
    string parameter.
    """

    @staticmethod
    def is_valid_path(path):
        if not String.is_valid_value(path):
            return False
        try:
            _ = path % "a"
            return True
        except TypeError:
            return False


class DualFileSet(AbstractData):
    """The abstract data type describing a collection of files indexed by two
    string parameters.
    """

    @staticmethod
    def is_valid_path(path):
        if not String.is_valid_value(path):
            return False
        try:
            _ = path % ("a", "b")
            return True
        except TypeError:
            return False


class FileSetSequence(AbstractData):
    """The abstract data type describing a collection of files indexed by one
    string parameter and one numeric parameter.
    """

    @staticmethod
    def is_valid_path(path):
        if not String.is_valid_value(path):
            return False
        try:
            _ = path % ("a", 1)
            return True
        except TypeError:
            return False


class FileSequenceSet(AbstractData):
    """The abstract data type describing a collection of files indexed by one
    numeric parameter and one string parameter.
    """

    @staticmethod
    def is_valid_path(path):
        if not String.is_valid_value(path):
            return False
        try:
            _ = path % (1, "a")
            return True
        except TypeError:
            return False


class Directory(ConcreteData):
    """The base type for directories that contain data.

    Examples:
        /path/to/dir
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}").format(**params)

    @staticmethod
    def is_valid_path(path):
        return String.is_valid_value(path)


class Image(AbstractData):
    """The abstract data type representing an image.

    Examples:
        /path/to/image.png
        /path/to/image.jpg
    """

    @staticmethod
    def is_valid_path(path):
        return ImageFile.is_valid_path(path)


class ImageFile(Image, File, ConcreteData):
    """An image file.

    ETA uses OpenCV to read images, so any image type understood by OpenCV is
    valid.

    Examples:
        /path/to/image.png
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}{image_ext}").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path)


class ImageFileDirectory(Directory):
    """A directory containing one or more images.

    Examples:
        /path/to/images
    """

    pass


class Video(AbstractData):
    """The abstract data type representing a single video.

    Examples:
        /path/to/video.mp4
        /path/to/video/%05d.png
    """

    @staticmethod
    def is_valid_path(path):
        return VideoFile.is_valid_path(path) or ImageSequence.is_valid_path(
            path
        )

    @staticmethod
    def get_metadata(path):
        if VideoFile.is_valid_path(path):
            return VideoFile.get_metadata(path)
        if ImageSequence.is_valid_path(path):
            return ImageSequence.get_metadata(path)
        raise TypeError("Unable to get metadata for '%s'" % path)


class VideoFile(Video, File, ConcreteData):
    """A video represented as a single encoded video file.

    ETA uses ffmpeg (default) or OpenCV (if specified) to load video files,
    so any video encoding types understood by these tools are valid.

    Examples:
        /path/to/video.mp4
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}{video_ext}").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path)

    @staticmethod
    def get_metadata(path):
        return etav.VideoMetadata.build_for(path)


class ImageSequence(Video, FileSequence, ConcreteData):
    """A video represented as a sequence of images with one numeric parameter.

    ETA uses ffmpeg (default) or OpenCV (if specified) to load videos stored
    as sequences of images, so any image types understood by these tools are
    valid.

    Examples:
        /path/to/video/%05d.png
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}", "{idx}{image_ext}").format(
            **params
        )

    @staticmethod
    def is_valid_path(path):
        return FileSequence.is_valid_path(path)


class DualImageSequence(DualFileSequence, ConcreteData):
    """A sequence of images indexed by two numeric parameters.

    Examples:
        /path/to/images/%05d-%05d.png
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(
            basedir, "{name}", "{idx}-{idx}{image_ext}"
        ).format(**params)

    @staticmethod
    def is_valid_path(path):
        return DualFileSequence.is_valid_path(path)


class VideoFileSequence(FileSequence, ConcreteData):
    """A sequence of encoded video files with one numeric parameter.

    Examples:
        /path/to/videos/%05d.mp4
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}", "{idx}{video_ext}").format(
            **params
        )

    @staticmethod
    def is_valid_path(path):
        return FileSequence.is_valid_path(path)


class VideoClips(DualFileSequence, ConcreteData):
    """A sequence of video files with two numeric parameters.

    Examples:
        /path/to/videos/%05d-%05d.mp4
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(
            basedir, "{name}", "{idx}-{idx}{video_ext}"
        ).format(**params)

    @staticmethod
    def is_valid_path(path):
        return DualFileSequence.is_valid_path(path)


class VideoFileDirectory(Directory):
    """A directory containing one or more video files.

    Examples:
        /path/to/videos
    """

    pass


class NpzFile(File, ConcreteData):
    """An .npz file.

    Examples:
        /path/to/data.npz
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}.npz").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etau.has_extension(path, ".npz")


class NpzFileDirectory(Directory):
    """A directory containing one or more .npz files.

    Examples:
        /path/to/npz_files
    """

    pass


class JSONFile(File, ConcreteData):
    """The base type for JSON files.

    Examples:
        /path/to/data.json
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}.json").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etau.has_extension(path, ".json")


class JSONFileSequence(FileSequence, ConcreteData):
    """The base type for a collection of JSON files indexed by one numeric
    parameter.

    Examples:
        /path/to/jsons/%05d.json
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}", "{idx}.json").format(**params)

    @staticmethod
    def is_valid_path(path):
        return FileSequence.is_valid_path(path) and etau.has_extension(
            path, ".json"
        )


class CSVFile(File, ConcreteData):
    """The base type for csv files.

    Examples:
        /path/to/data.csv
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}.csv").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etau.has_extension(path, ".csv")


class ExcelFile(File, ConcreteData):
    """The base type for Excel spreadsheets (.xls or .xlsx).

    Examples:
        /path/to/data.json
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}.xlsx").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etau.has_extension(
            path, ".xls", ".xlsx"
        )


class DataRecords(JSONFile):
    """A container of BaseDataRecords instane each having a certain set of
    fields.

    This type is implemented in ETA by the `eta.core.data.DataRecords` class.

    Examples:
        /path/to/data_records.json
    """

    pass


class VideoMetadata(JSONFile):
    """Metadata about a video.

    This type is implemented in ETA by the `eta.core.video.VideoMetadata`
    class.

    Examples:
        /path/to/video_metadata.json
    """

    pass


class VideoStreamInfo(JSONFile):
    """Stream info about a video.

    This type is implemented in ETA by the `eta.core.video.VideoStreamInfo`
    class.

    Examples:
        /path/to/video_stream_info.json
    """

    pass


class FrameRanges(JSONFile):
    """A monotonically increasing and disjoint series of frame ranges.

    This type is implemented in ETA by the `eta.core.frameutils.FrameRanges`
    class.

    Examples:
        /path/to/frame_ranges.json
    """

    pass


class MaskIndex(JSONFile):
    """An index of sementics for the values in a mask.

    This type is implemented in ETA by the `eta.core.data.MaskIndex` class.

    Examples:
        /path/to/mask_index.json
    """

    pass


class Labels(JSONFile):
    """Base type for labels representing attributes, objects, frames, events,
    images, videos, etc.

    This type is implemented in ETA by the `eta.core.labels.Labels` class.

    Examples:
        /path/to/labels.json
    """

    pass


class LabelsSchema(JSONFile):
    """Base type for labels schemas.

    This type is implemented in ETA by the `eta.core.labels.LabelsSchema`
    class.

    Examples:
        /path/to/labels_schema.json
    """

    pass


class Attribute(Labels):
    """Base class for attributes of entities in images or video.

    This type is implemented in ETA by the `eta.core.data.Attribute` class.

    Examples:
        /path/to/attribute.json
    """

    pass


class AttributeSchema(LabelsSchema):
    """Base class for classes that describe the values that a particular
    attribute can take.

    This type is implemented in ETA by the `eta.core.data.AttributeSchema`
    class.

    Examples:
        /path/to/attribute_schema.json
    """

    pass


class CategoricalAttribute(Attribute):
    """A categorical attribute of an entity in an image or video.

    This type is implemented in ETA by the `eta.core.data.CategoricalAttribute`
    class.

    Examples:
        /path/to/categorical_attribute.json
    """

    pass


class CategoricalAttributeSchema(AttributeSchema):
    """A schema that defines the set of possible values that a particular
    `CategoricalAttribute` can take.

    This type is implemented in ETA by the
    `eta.core.data.CategoricalAttributeSchema` class.

    Examples:
        /path/to/categorical_attribute_schema.json
    """

    pass


class NumericAttribute(Attribute):
    """A numeric attribute of an entity in an image or video.

    This type is implemented in ETA by the `eta.core.data.NumericAttribute`
    class.

    Examples:
        /path/to/numeric_attribute.json
    """

    pass


class NumericAttributeSchema(AttributeSchema):
    """A schema that defines the range of possible values that a particular
    `NumericAttribute` can take.

    This type is implemented in ETA by the
    `eta.core.data.NumericAttributeSchema` class.

    Examples:
        /path/to/numeric_attribute_schema.json
    """

    pass


class BooleanAttribute(Attribute):
    """A boolean attribute of an entity in an image or video.

    This type is implemented in ETA by the `eta.core.data.BooleanAttribute`
    class.

    Examples:
        /path/to/boolean_attribute.json
    """

    pass


class BooleanAttributeSchema(AttributeSchema):
    """A schema that declares that a given attribute is a `BooleanAttribute`
    and thus must take the values `True` and `False`.

    This type is implemented in ETA by the
    `eta.core.data.BooleanAttributeSchema` class.

    Examples:
        /path/to/boolean_attribute_schema.json
    """

    pass


class Attributes(Labels):
    """A list of `Attribute`s of an entity in an image or video. The list can
    contain attributes with any subtype of `Attribute`.

    This type is implemented in ETA by the `eta.core.data.AttributeContainer`
    class.

    Examples:
        /path/to/attribute_container.json
    """

    pass


class AttributesSchema(LabelsSchema):
    """A dictionary of `AttributesSchema`s that define the schemas of a
    collection of `Attribute`s of any type.

    This type is implemented in ETA by the
    `eta.core.data.AttributeContainerSchema` class.

    Examples:
        /path/to/attributes_schema.json
    """

    pass


class BoundingBox(JSONFile):
    """A bounding box of an object in a frame.

    This type is implemented in ETA by the `eta.core.geometry.BoundingBox`
    class.

    Examples:
        /path/to/bounding_box.json
    """

    pass


class VideoObject(Labels):
    """A spatiotemporal object in a video.

    This type is implemented in ETA by the `eta.core.objects.VideoObject`
    class.

    Examples:
        /path/to/video_object.json
    """

    pass


class VideoObjects(Labels):
    """A list of spatiotemporal objects in a video.

    This type is implemented in ETA by the
    `eta.core.objects.VideoObjectContainer` class.

    Examples:
        /path/to/video_objects.json
    """

    pass


class DetectedObject(Labels):
    """A detected object in an image or video frame.

    This type is implemented in ETA by the `eta.core.objects.DetectedObject`
    class.

    Examples:
        /path/to/detected_object.json
    """

    pass


class DetectedObjects(Labels):
    """A list of detected objects in image(s) or video frame(s).

    This type is implemented in ETA by the
    `eta.core.objects.DetectedObjectContainer` class.

    Examples:
        /path/to/detected_objects.json
    """

    pass


class DetectedObjectsSequence(JSONFileSequence):
    """Detected objects in a video represented as a collection of
    DetectedObjects files indexed by one numeric parameter.

    Examples:
        /path/to/detected_objects/%05d.json
    """

    pass


class VideoEvent(Labels):
    """A spatiotemporal event in a video.

    This type is implemented in ETA by the `eta.core.events.VideoEvent` class.

    Examples:
        /path/to/video_event.json
    """

    pass


class VideoEvents(Labels):
    """A list of spatiotemporal events in a video.

    This type is implemented in ETA by the
    `eta.core.events.VideoEventContainer` class.

    Examples:
        /path/to/video_events.json
    """

    pass


class DetectedEvent(Labels):
    """A detected event in an image or video frame.

    This type is implemented in ETA by the `eta.core.events.DetectedEvent`
    class.

    Examples:
        /path/to/detected_event.json
    """

    pass


class DetectedEvents(Labels):
    """A list of detected events in image(s) or video frame(s).

    This type is implemented in ETA by the
    `eta.core.objects.DetectedEventContainer` class.

    Examples:
        /path/to/detected_events.json
    """

    pass


class FrameLabels(Labels):
    """A description of the labeled contents of a frame.

    This type is implemented in ETA by the `eta.core.frames.FrameLabels` class.

    Examples:
        /path/to/frame_labels.json
    """

    pass


class FrameLabelsSchema(LabelsSchema):
    """A description of the schema of possible labels that can be generated for
    one or more frames.

    This type is implemented in ETA by the `eta.core.frames.FrameLabelsSchema`
    class.

    Examples:
        /path/to/frame_labels_schema.json
    """

    pass


class ImageLabels(FrameLabels):
    """A description of the labeled contents of an image.

    This type is implemented in ETA by the `eta.core.image.ImageLabels`
    class.

    Examples:
        /path/to/image_labels.json
    """

    pass


class ImageLabelsSchema(FrameLabelsSchema):
    """A description of the schema of possible labels that can be generated for
    images.

    This type is implemented in ETA by the `eta.core.image.ImageLabelsSchema`
    class.

    Examples:
        /path/to/image_labels_schema.json
    """

    pass


class ImageSetLabels(Labels):
    """A description of the labeled contents of a set of images.

    This type is implemented in ETA by the `eta.core.image.ImageSetLabels`
    class.

    Examples:
        /path/to/image_set_labels.json
    """

    pass


class VideoLabels(Labels):
    """A description of the labeled contents of a video.

    This type is implemented in ETA by the `eta.core.video.VideoLabels`
    class.

    Examples:
        /path/to/video_labels.json
    """

    pass


class VideoLabelsSchema(FrameLabelsSchema):
    """A description of the schema of possible labels that can be generated for
    a video.

    This type is implemented in ETA by the `eta.core.video.VideoLabelsSchema`
    class.

    Examples:
        /path/to/video_labels_schema.json
    """

    pass


class VideoSetLabels(Labels):
    """A description of the labeled contents of a set of videos.

    This type is implemented in ETA by the `eta.core.video.VideoSetLabels`
    class.

    Examples:
        /path/to/video_set_labels.json
    """

    pass


class ImageFeature(File, ConcreteData):
    """A feature vector for an image.

    Examples:
        /path/to/feature.npy
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}.npy").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path)


class ImageObjectsFeatures(FileSequence, ConcreteData):
    """A sequence of features for the objects in an image indexed by one
    numeric parameter.

    Examples:
        /path/to/features/%05d.npy
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}", "{idx}.npy").format(**params)

    @staticmethod
    def is_valid_path(path):
        return FileSequence.is_valid_path(path)


class ImageSetFeatures(FileSet, ConcreteData):
    """A sequence of features for a set of images indexed by one string
    parameter.

    Examples:
        /path/to/features/%s.npy
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}", "%s.npy").format(**params)

    @staticmethod
    def is_valid_path(path):
        return FileSet.is_valid_path(path)


class ImageSetObjectsFeatures(FileSetSequence, ConcreteData):
    """A collection of features for the objects in a set of images indexed by
    one string parameter and one index parameter.

    Examples:
        /path/to/features/%s-%05d.npy
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}", "%s-{idx}.npy").format(**params)

    @staticmethod
    def is_valid_path(path):
        return FileSetSequence.is_valid_path(path)


class VideoFramesFeatures(FileSequence, ConcreteData):
    """A sequence of features for the frames of a video indexed by one numeric
    parameter.

    Examples:
        /path/to/features/%05d.npy
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}", "{idx}.npy").format(**params)

    @staticmethod
    def is_valid_path(path):
        return FileSequence.is_valid_path(path)


class VideoObjectsFeatures(DualFileSequence, ConcreteData):
    """A sequence of features of objects-in-frames indexed by two numeric
    parameters.

    Examples:
        /path/to/features/%05d-%05d.npy
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}", "{idx}-{idx}.npy").format(
            **params
        )

    @staticmethod
    def is_valid_path(path):
        return DualFileSequence.is_valid_path(path)


class VideoDirectory(Directory):
    """A directory containing encoded video files.

    Examples:
        /path/to/videos
    """

    pass


class FileSequenceDirectory(Directory):
    """A directory containing a sequence of files indexed by one numeric
    parameter.

    Examples:
        /path/to/dir
    """

    pass


class DualFileSequenceDirectory(Directory):
    """A directory containing a sequence of files indexed by two numeric
    parameters.

    Examples:
        /path/to/dir
    """

    pass


class FileSetDirectory(Directory):
    """A directory containing a set of files indexed by one string parameter.

    Examples:
        /path/to/dir
    """

    pass


class DualFileSetDirectory(Directory):
    """A directory containing a set of files indexed by two string parameters.

    Examples:
        /path/to/dir
    """

    pass


class FileSetSequenceDirectory(Directory):
    """A directory containing a set of sequences of files indexed by one string
    parameter and one numeric parameter.

    Examples:
        /path/to/dir
    """

    pass


class ImageSequenceDirectory(FileSequenceDirectory):
    """A directory containing a sequence of images indexed by one numeric
    parameter.

    Examples:
        /path/to/images
    """

    pass


class DualImageSequenceDirectory(DualFileSequenceDirectory):
    """A directory containing a sequence of images indexed by two numeric
    parameters.

    Examples:
        /path/to/dual-images
    """

    pass


class JSONDirectory(Directory):
    """A directory of JSON files.

    Examples:
        /path/to/jsons
    """

    pass


class DataRecordsDirectory(JSONDirectory):
    """A directory of DataRecords files.

    Examples:
        /path/to/data_records
    """

    pass


class JSONSequenceDirectory(FileSequenceDirectory, JSONDirectory):
    """A directory containing a sequence of JSON files indexed by one numeric
    parameter.

    Examples:
        /path/to/jsons
    """

    pass


class DetectedObjectsSequenceDirectory(JSONSequenceDirectory):
    """A directory containing a sequence of DetectedObjects JSON files indexed
    by one numeric parameter.

    Examples:
        /path/to/detected_objects
    """

    pass


class ImageObjectsFeaturesDirectory(FileSequenceDirectory):
    """A directory containing features for the objects in an image indexed by
    one numeric parameter.

    Examples:
        /path/to/features
    """

    pass


class ImageSetFeaturesDirectory(FileSetDirectory):
    """A directory containing features for a set of images indexed by one
    string parameter.

    Examples:
        /path/to/features
    """

    pass


class ImageSetObjectsFeaturesDirectory(FileSetSequenceDirectory):
    """A directory containing features for the objects in a set of images
    indexed by one string parameter and one numeric parameter.

    Examples:
        /path/to/features
    """

    pass


class VideoFramesFeaturesDirectory(FileSequenceDirectory):
    """A directory containing a sequence of features for the frames of a video
    indexed by one numeric parameter.

    Examples:
        /path/to/features
    """

    pass


class VideoObjectsFeaturesDirectory(DualFileSequenceDirectory):
    """A directory containing a sequence of features of objects-in-frames
    indexed by two numeric parameters.

    Examples:
        /path/to/features
    """

    pass


class ZipFile(File, ConcreteData):
    """A zip file.

    Examples:
        /path/to/file.zip
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}.zip").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etau.has_extension(path, ".zip")


class ZippedDirectory(ZipFile):
    """A zip file containing a directory of the same name.

    Examples:
        /path/to/dir.zip
    """

    pass


class ZippedVideoFileDirectory(ZippedDirectory):
    """A zipped directory containing encoded video files.

    Examples:
        /path/to/videos.zip
    """

    pass


class ZippedImageSequenceDirectory(ZippedDirectory):
    """A zipped directory containing a sequence of images.

    Examples:
        /path/to/images.zip
    """

    pass


class ZippedDualImageSequenceDirectory(ZippedDirectory):
    """A zipped directory containing a collection of dual image sequence
    directories.

    Examples:
        /path/to/dual-images.zip
    """

    pass


class ZippedJSONDirectory(ZippedDirectory):
    """A zipped directory of JSON files.

    Examples:
        /path/to/jsons.zip
    """

    pass


class ZippedDetectedObjectsSequenceDirectory(ZippedDirectory):
    """A zipped directory containing a collection of DetectedObjectsSequence
    directories.

    Examples:
        /path/to/detected_objects.zip
    """

    pass


class ZippedVideoObjectsFeaturesDirectory(ZippedDirectory):
    """A zipped directory containing a collection of VideoObjectsFeatures
    directories.

    Examples:
        /path/to/video-object-features.zip
    """

    pass


class TFRecord(File, ConcreteData):
    """A tf.Record file, which may be sharded.

    Examples:
        /path/to/data.record
        /path/to/data.record-?????-of-XXXXX
        /path/to/data-?????-of-XXXXX.tfrecord
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}.record").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etat.is_valid_tf_record_path(path)


class TFRecordsDirectory(Directory):
    """A directory containing a sequence of tf.Records.

    Examples:
        /path/to/tf_records
    """

    pass


class PickleFile(File, ConcreteData):
    """A .pkl file.

    Examples:
        /path/to/data.pkl
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}.pkl").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etau.has_extension(path, ".pkl")


class PickleFileSequence(FileSequence, ConcreteData):
    """A collection of .pkl files indexed by one numeric parameter.

    Examples:
        /path/to/data/%05d.pkl
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}", "{idx}.pkl").format(**params)

    @staticmethod
    def is_valid_path(path):
        return FileSequence.is_valid_path(path) and etau.has_extension(
            path, ".pkl"
        )


class PickleFileDirectory(Directory):
    """A directory containing one or more .pkl files.

    Examples:
        /path/to/pkl_files
    """

    pass


class TextFile(File, ConcreteData):
    """A .txt file.

    Examples:
        /path/to/data.txt
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}.txt").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etau.has_extension(path, ".txt")


class HTMLFile(File, ConcreteData):
    """A .html file.

    Examples:
        /path/to/data.html
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}.html").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etau.has_extension(path, ".html")


class CheckpointFile(File, ConcreteData):
    """A .ckpt file.

    Examples:
        /path/to/model.ckpt
    """

    @staticmethod
    def gen_path(basedir, params):
        return os.path.join(basedir, "{name}.ckpt").format(**params)

    @staticmethod
    def is_valid_path(path):
        return File.is_valid_path(path) and etau.has_extension(path, ".ckpt")


class LabeledVideoDatasetDirectory(Directory):
    """A `eta.core.datasets.LabeledVideoDataset` directory.

    Examples:
        /path/to/labeled_video_dataset
    """

    pass


class LabeledImageDatasetDirectory(Directory):
    """A `eta.core.datasets.LabeledImageDataset` directory.

    Examples:
        /path/to/labeled_image_dataset
    """

    pass


class LabeledDatasetIndex(JSONFile):
    """An encapsulation of the manifest of a `LabeledDataset`.

    This type is implemented in ETA by the
    `eta.core.datasets.LabeledDatasetIndex` class.

    Examples:
        /path/to/labeled_dataset/manifest.json
    """

    pass
