"""
Core infrastructure for deploying ML models.

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

import logging

import numpy as np

from eta.core.config import Config, ConfigError, Configurable
import eta.core.models as etam
import eta.core.utils as etau


logger = logging.getLogger(__name__)


def load_labels_map(labels_map_path):
    """Loads the labels map from the given path.

    The labels mmap must be in the following plain text format::

        1:label1
        2:label2
        3:label3
        ...

    The indexes are irrelevant to this function, they can be in any order and
    can start from zero, one, or another number.

    Args:
        labels_map_path: the path to a labels map file

    Returns:
        a dictionary mapping indexes to label strings
    """
    labels_map = {}
    with open(labels_map_path, "r") as f:
        for line in f:
            idx, label = line.split(":")
            labels_map[int(idx)] = label.strip()

    return labels_map


def write_labels_map(labels_map, outpath):
    """Writes the labels map to disk.

    Labels maps are written to disk in the following plain text format::

        1:label1
        2:label2
        3:label3
        ...

    The indexes are irrelevant to this function, they can be in any order and
    can start from zero, one, or another number. They are, however, written
    to disk in sorted (increasing) order.

    Args:
        labels_map: the labels map dictionary
        outpath: the output path
    """
    with open(outpath, "w") as f:
        for idx in sorted(labels_map):
            f.write("%s:%s\n" % (idx, labels_map[idx]))


def get_class_labels(labels_map):
    """Returns the list of class labels from the given labels map.

    The class labels are returned for indexes sequentially from
    `min(1, min(labels_map))` to `max(labels_map)`. Any missing indices are
    given the label "class <index>".

    Args:
        a dictionary mapping indexes to label strings

    Returns:
        a list of class labels
    """
    mini = min(1, min(labels_map))
    maxi = max(labels_map)
    return [labels_map.get(i, "class %d" % i) for i in range(mini, maxi + 1)]


def has_default_deployment_model(model_name):
    """Determines whether the model with the given name has a default
    deployment.

    The model must be findable via `eta.core.models.get_model(model_name)`.

    Args:
        model_name: the name of the model, which can have "@<ver>" appended to
            refer to a specific version of the model. If no version is
            specified, the latest version of the model is assumed

    Returns:
        True/False whether the model has a default deployment
    """
    model = etam.get_model(model_name)
    return model.default_deployment_config_dict is not None


def load_default_deployment_model(
    model_name, install_requirements=False, error_level=0
):
    """Loads the default deployment for the model with the given name.

    The model must be findable via `eta.core.models.get_model(model_name)`.

    By default, any requirement(s) specified by the model are validated prior
    to loading it. Use `error_level` to configure this behavior, if desired.

    Args:
        model_name: the name of the model, which can have "@<ver>" appended to
            refer to a specific version of the model. If no version is
            specified, the latest version of the model is assumed
        install_requirements: whether to install any requirements before
            loading the model. By default, this is False
        error_level: the error level to use, defined as:

            0: raise error if a requirement is not satisfied
            1: log warning if a requirement is not satisifed
            2: ignore unsatisifed requirements

    Returns:
        the loaded `Model` instance described by the default deployment for the
            specified model
    """
    model = etam.get_model(model_name)
    if install_requirements:
        model.install_requirements(error_level=error_level)
    else:
        model.ensure_requirements(error_level=error_level)

    config = ModelConfig.from_dict(model.default_deployment_config_dict)
    return config.build()


def install_requirements(model_name, error_level=0):
    """Installs any package requirements for the model with the given name.

    Args:
        model_name: the name of the model, which can have "@<ver>" appended to
            refer to a specific version of the model. If no version is
            specified, the latest version of the model is assumed
        error_level: the error level to use, defined as:

            0: raise error if a requirement install fails
            1: log warning if a requirement install fails
            2: ignore install fails
    """
    model = etam.get_model(model_name)
    model.install_requirements(error_level=error_level)


def ensure_requirements(model_name, error_level=0):
    """Ensures that the package requirements for the model with the given name
    are satisfied.

    Args:
        model_name: the name of the model, which can have "@<ver>" appended to
            refer to a specific version of the model. If no version is
            specified, the latest version of the model is assumed
        error_level: the error level to use, defined as:

            0: raise error if a requirement is not satisfied
            1: log warning if a requirement is not satisifed
            2: ignore unsatisifed requirements
    """
    model = etam.get_model(model_name)
    model.ensure_requirements(error_level=error_level)


class HasPublishedModel(object):
    """Mixin class for `eta.core.learning.ModelConfig`s whose models are
    published via the `eta.core.models` infrastructure.

    This class provides the following functionality:

    -   The model to load can be specified either by:

        (a) providing a `model_name`, which specifies the published model to
            load. The model will be downloaded, if necessary

        (b) providing a `model_path`, which directly specifies the path to the
            model to load

    -   `ModelConfig` definitions that use published models with default
        deployments will have default values for any unspecified parameters
        loaded and applied at runtime

    Attributes:
        model_name: the name of the published model to load. If this value is
            provided, `model_path` does not need to be
        model_path: the path to a model to load. If this value is provided,
            `model_name` does not need to be
    """

    def init(self, d):
        """Initializes the published model config.

        This method should be called by `ModelConfig.__init__()`, and it
        performs the following tasks:

        -   Parses the `model_name` and `model_path` parameters
        -   Populates any default parameters in the provided ModelConfig dict

        Args:
            d: a ModelConfig dict

        Returns:
            a ModelConfig dict with any default parameters populated
        """
        # pylint: disable=no-member
        self.model_name = self.parse_string(d, "model_name", default=None)
        self.model_path = self.parse_string(d, "model_path", default=None)

        if self.model_name:
            d = self._load_default_deployment_params(d, self.model_name)

        return d

    def download_model_if_necessary(self):
        """Downloads the published model specified by the config, if necessary.

        After this method is called, the `model_path` attribute will always
        contain the path to the model on disk.
        """
        if not self.model_name and not self.model_path:
            raise ConfigError(
                "Either `model_name` or `model_path` must be provided"
            )

        if self.model_path is None:
            self.model_path = etam.download_model(self.model_name)

    @classmethod
    def _load_default_deployment_params(cls, d, model_name):
        model = cls._get_model(model_name)

        deploy_config_dict = model.default_deployment_config_dict
        if deploy_config_dict is None:
            return d

        dd = deploy_config_dict["config"]
        dd.update(d)
        return dd

    @classmethod
    def _get_model(cls, model_name):
        return etam.get_model(model_name)


class ModelConfig(Config):
    """Base configuration class that encapsulates the name of a `Model`
    subclass and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `Model` subclass
        config: an instance of the Config class associated with the specified
            `Model` subclass
    """

    def __init__(self, d):
        self.type = self.parse_string(d, "type")
        self._model_cls, self._config_cls = Configurable.parse(self.type)
        self.config = self.parse_object(
            d, "config", self._config_cls, default=None
        )
        if not self.config:
            self.config = self._load_default_config()

    def build(self):
        """Factory method that builds the Model instance from the config
        specified by this class.

        Returns:
            a Model instance
        """
        return self._model_cls(self.config)

    def _load_default_config(self):
        try:
            # Try to load the default config from disk
            return self._config_cls.load_default()
        except NotImplementedError:
            # Try default() instead
            return self._config_cls.default()

    def _validate_type(self, base_cls):
        if not issubclass(self._model_cls, base_cls):
            raise ConfigError(
                "Expected type '%s' to be a subclass of '%s'"
                % (self.type, etau.get_class_name(base_cls))
            )


class Model(Configurable):
    """Abstract base class for all models.

    This class declares the following two conventions:

        (a) `Model`s are `Configurable`. This means that their constructors
            must take a single `config` argument that is an instance of
            `<ModelClass>Config`

        (b) Models implement the context manager interface. This means that
            models can optionally use context to perform any necessary setup
            and teardown, and so any code that builds a model should use the
            `with` syntax
    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class ImageModelConfig(ModelConfig):
    """Base configuration class that encapsulates the name of an `ImageModel`
    subclass and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `ImageModel` subclass
        config: an instance of the Config class associated with the specified
            `ImageModel` subclass
    """

    def __init__(self, d):
        super(ImageModelConfig, self).__init__(d)
        self._validate_type(ImageModel)


class ImageModel(Model):
    """Interface for generic models that process images and perform arbitrary
    predictions and detections.

    Subclasses of `ImageModel` must implement the `process()` method.

    `ImageModel` is useful when implementing a highly customized model that
    does not fit any of the concrete classifier/detector interfaces.
    """

    def process(self, img):
        """Generates labels for the given image.

        Args:
            img: an image

        Returns:
            an `eta.core.image.ImageLabels` instance containing the labels
                generated for the given image
        """
        raise NotImplementedError("subclasses must implement process()")


class VideoModelConfig(ModelConfig):
    """Base configuration class that encapsulates the name of an `VideoModel`
    subclass and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `VideoModel` subclass
        config: an instance of the Config class associated with the specified
            `VideoModel` subclass
    """

    def __init__(self, d):
        super(VideoModelConfig, self).__init__(d)
        self._validate_type(VideoModel)


class VideoModel(Model):
    """Interface for generic models that process entire videos and perform
    arbitrary predictions and detections.

    Subclasses of `VideoModel` must implement the `process()` method.

    `VideoModel` is useful when implementing a highly customized model that
    does not fit any of the concrete classifier/detector interfaces.
    """

    def process(self, video_reader):
        """Generates labels for the given video.

        Args:
            video_reader: an `eta.core.video.VideoReader` that can be used to
                read the video

        Returns:
            an `eta.core.video.VideoLabels` instance containing the labels
                generated for the given video
        """
        raise NotImplementedError("subclasses must implement process()")


class ClassifierConfig(ModelConfig):
    """Configuration class that encapsulates the name of a `Classifier` and
    an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `Classifier`
        config: an instance of the Config class associated with the specified
            `Classifier`
    """

    def __init__(self, d):
        super(ClassifierConfig, self).__init__(d)
        self._validate_type(Classifier)


class Classifier(Model):
    """Interface for classifiers.

    Subclasses of `Classifier` must implement the `predict()` method, and they
    should declare via the `is_multilabel` property whether they output single
    or multiple labels per prediction.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown.
    """

    @property
    def is_multilabel(self):
        """Whether the classifier generates single labels (False) or multiple
        labels (True) per prediction.
        """
        raise NotImplementedError("subclasses must implement is_multilabel")

    def predict(self, arg):
        """Peforms prediction on the given argument.

        Args:
            arg: the data to process

        Returns:
            an `eta.core.data.AttributeContainer` describing the predictions
        """
        raise NotImplementedError("subclasses must implement predict()")


class ImageClassifierConfig(ClassifierConfig):
    """Configuration class that encapsulates the name of an `ImageClassifier`
    and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `ImageClassifier`
        config: an instance of the Config class associated with the specified
            `ImageClassifier`
    """

    def __init__(self, d):
        super(ImageClassifierConfig, self).__init__(d)
        self._validate_type(ImageClassifier)


class ImageClassifier(Classifier):
    """Base class for classifiers that operate on single images.

    Subclasses of `ImageClassifier` must implement the `predict()` method, and
    they can optionally provide a custom (efficient) implementation of the
    `predict_all()` method.

    `ImageClassifier`s may output single or multiple labels per image, and
    should declare such via the `is_multilabel` property.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown, e.g., operating a `Featurizer`
    that featurizes the input images.
    """

    @property
    def ragged_batches(self):
        """True/False whether :meth:`transforms` may return images of different
        sizes and therefore passing ragged lists of images to
        :meth:`predict_all` is not allowed.
        """
        raise NotImplementedError("subclasses must implement ragged_batches")

    @property
    def transforms(self):
        """The preprocessing transformation that will be applied to each image
        before prediction, or ``None`` if no preprocessing is performed.
        """
        raise NotImplementedError("subclasses must implement transforms")

    @property
    def preprocess(self):
        """Whether to apply :meth:`transforms` during inference (True) or to
        assume that they have already been applied (False).
        """
        raise NotImplementedError("subclasses must implement preprocess")

    @preprocess.setter
    def preprocess(self, value):
        raise NotImplementedError("subclasses must implement preprocess")

    def predict(self, img):
        """Peforms prediction on the given image.

        Args:
            img: an image

        Returns:
            an `eta.core.data.AttributeContainer` instance containing the
                predictions
        """
        raise NotImplementedError("subclasses must implement predict()")

    def predict_all(self, imgs):
        """Performs prediction on the given tensor of images.

        Subclasses can override this method to increase efficiency, but, by
        default, this method simply iterates over the images and predicts each.

        Args:
            imgs: a list (or n x h x w x 3 tensor) of images

        Returns:
            a list of `eta.core.data.AttributeContainer` instances describing
                the predictions for each image
        """
        return [self.predict(img) for img in imgs]


class VideoFramesClassifierConfig(ClassifierConfig):
    """Configuration class that encapsulates the name of a
    `VideoFramesClassifier` and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `VideoFramesClassifier`
        config: an instance of the Config class associated with the specified
            `VideoFramesClassifier`
    """

    def __init__(self, d):
        super(VideoFramesClassifierConfig, self).__init__(d)
        self._validate_type(VideoFramesClassifier)


class VideoFramesClassifier(Classifier):
    """Base class for classifiers that operate directly on videos represented
    as tensors of images.

    `VideoFramesClassifier`s may output single or multiple labels per video
    clip, and should declare such via the `is_multilabel` property.

    Subclasses of `VideoFramesClassifier` must implement the `predict()`
    method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown, e.g., operating a `Featurizer`
    that featurizes the input frames.
    """

    def predict(self, imgs):
        """Peforms prediction on the given video represented as a tensor of
        images.

        Args:
            imgs: a list (or n x h x w x 3 tensor) of images

        Returns:
            an `eta.core.data.AttributeContainer` instance describing
                the predictions for the input
        """
        raise NotImplementedError("subclasses must implement predict()")


class VideoClassifierConfig(ClassifierConfig):
    """Configuration class that encapsulates the name of a `VideoClassifier`
    and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `VideoClassifier`
        config: an instance of the Config class associated with the specified
            `VideoClassifier`
    """

    def __init__(self, d):
        super(VideoClassifierConfig, self).__init__(d)
        self._validate_type(VideoClassifier)


class VideoClassifier(Classifier):
    """Base class for classifiers that operate on entire videos.

    `VideoClassifier`s may output single or multiple (video-level) labels per
    video, and should declare such via the `is_multilabel` property.

    Subclasses of `VideoClassifier` must implement the `predict()` method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown, e.g., operating a `Featurizer`
    that featurizes the frames of the input video.
    """

    def predict(self, video_reader):
        """Peforms prediction on the given video.

        Args:
            video_reader: an `eta.core.video.VideoReader` that can be used to
                read the video

        Returns:
            an `eta.core.data.AttributeContainer` instance containing the
                predictions
        """
        raise NotImplementedError("subclasses must implement predict()")


class DetectorConfig(ModelConfig):
    """Configuration class that encapsulates the name of a `Detector` and
    an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `Detector`
        config: an instance of the Config class associated with the specified
            `Detector`
    """

    def __init__(self, d):
        super(DetectorConfig, self).__init__(d)
        self._validate_type(Detector)


class Detector(Model):
    """Interface for detectors.

    Subclasses of `Detector` must implement the `detect()` method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown.
    """

    def detect(self, arg):
        """Peforms detection on the given argument.

        Args:
            arg: the data to process

        Returns:
            an instance of a subclass of `eta.core.serial.Container` describing
            the detections, with the specific sub-class depending on the type
            of detection (e.g., objects or events)
        """
        raise NotImplementedError("subclasses must implement detect()")


class ObjectDetectorConfig(DetectorConfig):
    """Configuration class that encapsulates the name of a `ObjectDetector` and
    an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `ObjectDetector`
        config: an instance of the Config class associated with the specified
            `ObjectDetector`
    """

    def __init__(self, d):
        super(ObjectDetectorConfig, self).__init__(d)
        self._validate_type(ObjectDetector)


class ObjectDetector(Detector):
    """Base class for object detectors that operate on single images.

    `ObjectDetector`s may output single or multiple object detections per
    image.

    Subclasses of `ObjectDetector` must implement the `detect()` method, and
    they can optionally provide a custom (efficient) implementation of the
    `detect_all()` method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown, e.g., operating a `Featurizer`
    that featurizes the input images.
    """

    @property
    def ragged_batches(self):
        """True/False whether :meth:`transforms` may return images of different
        sizes and therefore passing ragged lists of images to
        :meth:`detect_all` is not allowed.
        """
        raise NotImplementedError("subclasses must implement ragged_batches")

    @property
    def transforms(self):
        """The preprocessing transformation that will be applied to each image
        before detection, or ``None`` if no preprocessing is performed.
        """
        raise NotImplementedError("subclasses must implement transforms")

    @property
    def preprocess(self):
        """Whether to apply :meth:`transforms` during inference (True) or to
        assume that they have already been applied (False).
        """
        raise NotImplementedError("subclasses must implement preprocess")

    @preprocess.setter
    def preprocess(self, value):
        raise NotImplementedError("subclasses must implement preprocess")

    def detect(self, img):
        """Detects objects in the given image.

        Args:
            img: an image

        Returns:
            an `eta.core.objects.DetectedObjectContainer` instance describing
                the detections
        """
        raise NotImplementedError("subclass must implement detect()")

    def detect_all(self, imgs):
        """Performs detection on the given tensor of images.

        Subclasses can override this method to increase efficiency, but, by
        default, this method simply iterates over the images and detects each.

        Args:
            imgs: a list (or n x h x w x 3 tensor) of images

        Returns:
            a list of `eta.core.objects.DetectedObjectContainer` instances
                describing the detections for each image
        """
        return [self.detect(img) for img in imgs]


class VideoFramesObjectDetectorConfig(DetectorConfig):
    """Configuration class that encapsulates the name of a
    `VideoFramesObjectDetector` and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `VideoFramesObjectDetector`
        config: an instance of the Config class associated with the specified
            `VideoFramesObjectDetector`
    """

    def __init__(self, d):
        super(VideoFramesObjectDetectorConfig, self).__init__(d)
        self._validate_type(VideoFramesObjectDetector)


class VideoFramesObjectDetector(Detector):
    """Base class for detectors that operate directly on videos
    represented as tensors of images.

    `VideoFramesObjectDetector`s may output single or multiple detections per
    video clip.

    Subclasses of `VideoFramesObjectDetector` must implement the `detect()`
    method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown, e.g., operating a `Featurizer`
    that featurizes the input frames.
    """

    def detect(self, imgs):
        """Peforms detection on the given video represented as a tensor of
        images.

        Args:
            imgs: a list (or n x h x w x 3 tensor) of images

        Returns:
            an `eta.core.objects.DetectedObjectContainer` instance describing
                the detections for the clip
        """
        raise NotImplementedError("subclasses must implement detect()")


class VideoObjectDetectorConfig(DetectorConfig):
    """Configuration class that encapsulates the name of a
    `VideoObjectDetector` and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `VideoObjectDetector`
        config: an instance of the Config class associated with the specified
            `VideoObjectDetector`
    """

    def __init__(self, d):
        super(VideoObjectDetectorConfig, self).__init__(d)
        self._validate_type(VideoObjectDetector)


class VideoObjectDetector(Detector):
    """Base class for detectors that operate on entire videos.

    `VideoObjectDetector`s may output one or more detections per video.

    Subclasses of `VideoObjectDetector` must implement the `detect()` method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown, e.g., operating a `Featurizer`
    that featurizes the frames of the input video.
    """

    def detect(self, video_reader):
        """Peforms detection on the given video.

        Args:
            video_reader: an `eta.core.video.VideoReader` that can be used to
                read the video

        Returns:
            an `eta.core.objects.DetectedObjectContainer` instance describing
                the detections for the video
        """
        raise NotImplementedError("subclasses must implement detect()")


class VideoEventDetectorConfig(DetectorConfig):
    """Configuration class that encapsulates the name of a `VideoEventDetector`
    and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `VideoEventDetector`
        config: an instance of the Config class associated with the specified
            `VideoEventDetector`
    """

    def __init__(self, d):
        super(VideoEventDetectorConfig, self).__init__(d)
        self._validate_type(VideoEventDetector)


class VideoEventDetector(Detector):
    """Base class for event detectors that operate on individual videos.

    `VideoEventDetector`s may output single or multiple detections per video;
    these detections may cover the entire video or may not.

    Subclasses of `VideoEventDetector` must implement the `detect()` method.
    """

    def detect(self, video_reader):
        """Detects events in the given video.

        Args:
            video_reader: an `eta.core.video.VideoReader` that can be used to
                read the video

        Returns:
            an `eta.core.events.VideoEventContainer` instance describing the
                events for the video
        """
        raise NotImplementedError("subclass must implement detect()")


class SemanticSegmenterConfig(ModelConfig):
    """Configuration class that encapsulates the name of a `SemanticSegmenter`
    and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `SemanticSegmenter`
        config: an instance of the Config class associated with the specified
            `SemanticSegmenter`
    """

    def __init__(self, d):
        super(SemanticSegmenterConfig, self).__init__(d)
        self._validate_type(SemanticSegmenter)


class SemanticSegmenter(Model):
    """Interface for sementic segmentation models.

    Subclasses of `SemanticSegmenter` must implement the `segment()` method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown.
    """

    def segment(self, arg):
        """Peforms segmentation on the given argument.

        Args:
            arg: the data to process

        Returns:
            an `eta.core.labels.Labels` describing the segmentations
        """
        raise NotImplementedError("subclasses must implement segment()")


class ImageSemanticSegmenterConfig(SemanticSegmenterConfig):
    """Configuration class that encapsulates the name of an
    `ImageSemanticSegmenter` and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `ImageSemanticSegmenter`
        config: an instance of the Config class associated with the specified
            `ImageSemanticSegmenter`
    """

    def __init__(self, d):
        super(ImageSemanticSegmenterConfig, self).__init__(d)
        self._validate_type(ImageSemanticSegmenter)


class ImageSemanticSegmenter(SemanticSegmenter):
    """Base class for sementic segmentation models that operate on single
    images.

    Subclasses of `ImageSemanticSegmenter` must implement the `segment()`
    method, and they can optionally provide a custom (efficient) implementation
    of the `segment_all()` method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown, e.g., operating a `Featurizer`
    that featurizes the input images.
    """

    @property
    def ragged_batches(self):
        """True/False whether :meth:`transforms` may return images of different
        sizes and therefore passing ragged lists of images to
        :meth:`segment_all` is not allowed.
        """
        raise NotImplementedError("subclasses must implement ragged_batches")

    @property
    def transforms(self):
        """The preprocessing transformation that will be applied to each image
        before segmentation, or ``None`` if no preprocessing is performed.
        """
        raise NotImplementedError("subclasses must implement transforms")

    @property
    def preprocess(self):
        """Whether to apply :meth:`transforms` during inference (True) or to
        assume that they have already been applied (False).
        """
        raise NotImplementedError("subclasses must implement preprocess")

    @preprocess.setter
    def preprocess(self, value):
        raise NotImplementedError("subclasses must implement preprocess")

    def segment(self, img):
        """Peforms segmentation on the given image.

        Args:
            img: an image

        Returns:
            an `eta.core.image.ImageLabels` instance containing the
                segmentation
        """
        raise NotImplementedError("subclasses must implement segment()")

    def segment_all(self, imgs):
        """Performs segmentation on the given tensor of images.

        Subclasses can override this method to increase efficiency, but, by
        default, this method simply iterates over the images and segments each.

        Args:
            imgs: a list (or n x h x w x 3 tensor) of images

        Returns:
            a list of `eta.core.image.ImageLabels` instances describing the
                segmentations for each image
        """
        return [self.segment(img) for img in imgs]


class VideoSemanticSegmenterConfig(SemanticSegmenterConfig):
    """Configuration class that encapsulates the name of a
    `VideoSemanticSegmenter` and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `VideoSemanticSegmenter`
        config: an instance of the Config class associated with the specified
            `VideoSemanticSegmenter`
    """

    def __init__(self, d):
        super(VideoSemanticSegmenterConfig, self).__init__(d)
        self._validate_type(VideoSemanticSegmenter)


class VideoSemanticSegmenter(SemanticSegmenter):
    """Base class for semantic segmentation models that operate on entire
    videos.

    Subclasses of `VideoSemanticSegmenter` must implement the `segment()`
    method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown, e.g., operating a `Featurizer`
    that featurizes the frames of the input video.
    """

    def segment(self, video_reader):
        """Peforms segmentation on the given video.

        Args:
            video_reader: an `eta.core.video.VideoReader` that can be used to
                read the video

        Returns:
            an `eta.core.video.VideoLabels` instance containing the
                segmentations
        """
        raise NotImplementedError("subclasses must implement segment()")


class ExposesMaskIndex(object):
    """Mixin for `SemanticSegmenter` subclasses that expose
    `eta.core.data.MaskIndex`s that assign semantic labels to their
    segmentations.
    """

    @property
    def exposes_mask_index(self):
        """Whether this segmenter exposes an `eta.core.data.MaskIndex`.

        This property allows for the possibility that some, but not all
        instances of a `SemanticSegmenter` expose their semantic labels.
        """
        raise NotImplementedError(
            "subclasses must implement exposes_mask_index"
        )

    def get_mask_index(self):
        """Returns the MaskIndex for the segmenter.

        Returns:
            A MaskIndex, or None if the segmenter does not expose its mask
                index
        """
        raise NotImplementedError("subclasses must implement get_mask_index()")

    @staticmethod
    def ensure_exposes_mask_index(segmenter):
        """Ensures that the given segmenter exposes its mask index.

        Args:
            segmenter: a SemanticSegmenter

        Raises:
            ValueError: if the segmenter does not expose features
        """
        if not isinstance(segmenter, ExposesMaskIndex):
            raise ValueError(
                "Expected %s to implement the %s mixin, but it does not"
                % (type(segmenter), ExposesMaskIndex)
            )

        if not segmenter.exposes_mask_index:
            raise ValueError(
                "Expected %s to expose its mask index, but it does not"
                % type(segmenter)
            )


class ExposesFeatures(object):
    """Mixin for `Model` subclasses that expose features for their predictions.

    By convention, features should be returned in an array whose shape follows
    the pattern below:

    Inference type                    Features array shape
    --------------------------------  -----------------------------------
    ImageClassifier.predict           1 x features_dim
    ImageClassifier.predict_all       num_images x features_dim
    VideoFramesClassifier.predict     1 x features_dim
    VideoClassifier.predict           1 x features_dim
    ObjectDetector.detect             1 x num_objects x features_dim
    ObjectDetector.detect_all         num_images x num_objects x features_dim
    VideoFramesObjectDetector.detect  1 x num_objects x features_dim
    VideoObjectDetector.detect        1 x num_objects x features_dim
    """

    @property
    def exposes_features(self):
        """Whether this model exposes features for its predictions.

        This property allows for the possibility that some, but not all
        instances of a `Model` are capable of exposing features.
        """
        raise NotImplementedError("subclasses must implement exposes_features")

    @property
    def features_dim(self):
        """The dimension of the features generated by this model, or None
        if it does not expose features.
        """
        raise NotImplementedError("subclasses must implement features_dim")

    def get_features(self):
        """Gets the features generated by the model from its last prediction.

        Returns:
            the features array, or None if the model has not (or does not)
                generated features
        """
        raise NotImplementedError("subclasses must implement get_features()")

    @staticmethod
    def ensure_exposes_features(model):
        """Ensures that the given model exposes features.

        Args:
            model: a Model

        Raises:
            ValueError: if the model does not expose features
        """
        if not isinstance(model, ExposesFeatures):
            raise ValueError(
                "Expected %s to implement the %s mixin, but it does not"
                % (type(model), ExposesFeatures)
            )

        if not model.exposes_features:
            raise ValueError(
                "Expected %s to expose features, but it does not" % type(model)
            )


class ExposesProbabilities(object):
    """Mixin for `Model` subclasses that expose probabilities for their
    predictions.

    By convention, class probabilities should be returned in an array whose
    shape follows the pattern below:

    Inference type                    Probabilities array shape
    --------------------------------  -----------------------------------
    ImageClassifier.predict           1 x num_preds x num_classes
    ImageClassifier.predict_all       num_images x num_preds x num_classes
    VideoFramesClassifier.predict     1 x num_preds x num_classes
    VideoClassifier.predict           1 x num_preds x num_classes
    ObjectDetector.detect             1 x num_objects x num_classes
    ObjectDetector.detect_all         num_images x num_objects x num_classes
    VideoFramesObjectDetector.detect  1 x num_objects x num_classes
    VideoObjectDetector.detect        1 x num_objects x num_classes
    """

    @property
    def exposes_probabilities(self):
        """Whether this model exposes probabilities for its predictions.

        This property allows for the possibility that some, but not all
        instances of a `Model` are capable of exposing probabilities.
        """
        raise NotImplementedError(
            "subclasses must implement exposes_probabilities"
        )

    @property
    def num_classes(self):
        """The number of classes for the model."""
        raise NotImplementedError("subclasses must implement num_classes")

    @property
    def class_labels(self):
        """The list of class labels for the model."""
        raise NotImplementedError("subclasses must implement class_labels")

    def get_probabilities(self):
        """Gets the class probabilities generated by the model from its last
        prediction.

        Returns:
            the class probabilities, or None if the model has not (or does not)
                generated probabilities
        """
        raise NotImplementedError(
            "subclasses must implement get_probabilities()"
        )

    def get_top_k_classes(self, top_k):
        """Gets the probabilities for the top-k classes generated by the model
        from its last prediction.

        Subclasses can override this method, but, by default, this information
        is extracted via `get_probabilities()` and `class_labels`.

        Args:
            top_k: the number of top classes

        Returns:
            a `num_images x num_preds/objects` array of dictionaries mapping
                class labels to probabilities, or None if the model has not
                (or does not) expose probabilities
        """
        if not self.exposes_probabilities:
            return None

        probs = self.get_probabilities()
        if probs is None:
            return None

        probs = np.asarray(probs)
        labels = np.asarray(self.class_labels)

        inds = np.argsort(probs, axis=2)
        num_images = inds.shape[0]
        num_preds = inds.shape[1]

        top_k_probs = np.empty((num_images, num_preds), dtype=dict)
        for i in range(num_images):
            for j in range(num_preds):
                probsij = probs[i, j, :]
                indsij = inds[i, j, :][-top_k:]
                top_k_probs[i, j] = dict(zip(labels[indsij], probsij[indsij]))

        return top_k_probs

    @staticmethod
    def ensure_exposes_probabilities(model):
        """Ensures that the given model exposes probabilities.

        Args:
            model: a Model

        Raises:
            ValueError: if the model does not expose probabilities
        """
        if not isinstance(model, ExposesProbabilities):
            raise ValueError(
                "Expected %s to implement the %s mixin, but it does not"
                % (type(model), ExposesProbabilities)
            )

        if not model.exposes_probabilities:
            raise ValueError(
                "Expected %s to expose probabilities, but it does not"
                % type(model)
            )
