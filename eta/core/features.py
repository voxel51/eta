'''
Core interfaces, data structures, and methods for feature extraction.

Copyright 2017-2019, Voxel51, Inc.
voxel51.com

Jason Corso, jason@voxel51.com
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

from glob import glob
import logging
import os
import tempfile

import cv2
import numpy as np

from eta.core.config import Config, Configurable, ConfigError
import eta.core.image as etai
import eta.core.utils as etau
import eta.core.video as etav


logger = logging.getLogger(__name__)


class FeaturizerConfig(Config):
    '''Configuration class that encapsulates the name of a `Featurizer` and an
    instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the Featurizer, e.g.,
            `eta.core.vgg16.VGG16Featurizer`
        config: an instance of the Config class associated with the specified
            Featurizer, e.g., `eta.core.vgg16.VGG16FeaturizerConfig`
    '''

    def __init__(self, d):
        self.type = self.parse_string(d, "type")
        self._featurizer_cls, self._config_cls = Configurable.parse(self.type)
        self.config = self.parse_object(
            d, "config", self._config_cls, default=None)
        if not self.config:
            self.config = self._load_default_config()

    def build(self):
        '''Factory method that builds the Featurizer instance from the config
        specified by this class.

        Returns:
            a Featurizer instance
        '''
        return self._featurizer_cls(self.config)

    def _load_default_config(self):
        try:
            # Try to load the default config from disk
            return self._config_cls.load_default()
        except NotImplementedError:
            # Try default() instead
            return self._config_cls.default()

    def _validate_type(self, base_cls):
        if not issubclass(self._featurizer_cls, base_cls):
            raise ConfigError(
                "Expected type '%s' to be a subclass of '%s'" % (
                    self.type, etau.get_class_name(base_cls)))


class Featurizer(Configurable):
    '''Base class for all featurizers.

    Subclasses of Featurizer must implement the `dim()` and `_featurize()`
    methods.

    If setup/teardown is required, subclasses should also implement the
    `_start()` and `_stop()` methods.

    Subclasses must call the superclass constructor defined by this base class.

    Note that Featurizer implements the context manager interface, so
    Featurizer subclasses can be used with the following convenient `with`
    syntax to automatically handle `start()` and `stop()` calls:

    ```
    with <My>Featurizer(...) as f:
        v = f.featurize(data)
    ```
    '''

    def __init__(self):
        '''Initializes the base Featurizer instance.'''
        self._is_started = False
        self._keep_alive = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def dim(self):
        '''Returns the dimension of the features extracted by this
        Featurizer.
        '''
        raise NotImplementedError("subclass must implement dim()")

    def start(self, warn_on_restart=True, keep_alive=True):
        '''Start method that handles any necessary setup to prepare the
        Featurizer for use.

        This method can be explicitly called by users. If it is not called
        manually, it will be called each time `featurize()` is called.

        Args:
            warn_on_restart: whether to generate a warning if `start()` is
                called when the Featurizer is already started. The default
                value is True
            keep_alive: whether to keep the Featurizer alive (i.e. not to call
                `stop()`) after a each `featurize()` call
        '''
        if warn_on_restart and self._is_started:
            logger.warning("Featurizer.start() called when already started")

        if self._is_started:
            return

        self._is_started = True
        self._keep_alive = keep_alive
        self._start()

    def _start(self):
        '''The backend implementation that is called when the public `start()`
        method is called.

        Subclasses that require startup configuration should implement this
        method.
        '''
        pass

    def stop(self):
        '''Stop method that handles any necessary cleanup after featurization
        is complete.

        This method can be explicitly called by users, and, in fact, it must
        be called by users who called `start()` themselves. If `start()` was
        not called manually or `keep_alive` was set to False, then this method
        will be invoked at the end of each call to `featurize()`.
        '''
        if not self._is_started:
            return

        self._stop()
        self._is_started = False
        self._keep_alive = False

    def _stop(self):
        '''The backend implementation that is called when the public `stop()`
        method is called.

        Subclasses that require cleanup after featurization should implement
        this method.
        '''
        pass

    def featurize(self, data):
        '''Featurizes the input data.

        Args:
            data: the data to featurize

        Returns:
            the feature vector
        '''
        self.start(warn_on_restart=False, keep_alive=False)
        v = self._featurize(data)
        if self._keep_alive is False:
            self.stop()

        return v

    def _featurize(self, data):
        '''The backend implementation of the feature extraction routine.
        Subclasses must implement this method.

        Args:
            data: the data to featurize

        Returns:
            the feature vector
        '''
        raise NotImplementedError("subclass must implement _featurize()")


class ImageFeaturizerConfig(FeaturizerConfig):
    '''Base configuration class that encapsulates the name of an
    `ImageFeaturizer` subclass and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `ImageFeaturizer` subclass
        config: an instance of the Config class associated with the specified
            `ImageFeaturizer` subclass
    '''

    def __init__(self, d):
        super(ImageFeaturizerConfig, self).__init__(d)
        self._validate_type(ImageFeaturizer)


class ImageFeaturizer(Featurizer):
    '''Interface for featurizers that operate on images.'''

    def _featurize(self, img):
        '''Featurizes the given image.

        Args:
            img: the image to featurize

        Returns:
            the feature vector
        '''
        raise NotImplementedError("subclass must implement _featurize()")


class VideoFramesFeaturizerConfig(FeaturizerConfig):
    '''Base configuration class that encapsulates the name of an
    `VideoFramesFeaturizer` subclass and an instance of its associated Config
    class.

    Attributes:
        type: the fully-qualified class name of the `VideoFramesFeaturizer`
            subclass
        config: an instance of the Config class associated with the specified
            `VideoFramesFeaturizer` subclass
    '''

    def __init__(self, d):
        super(VideoFramesFeaturizerConfig, self).__init__(d)
        self._validate_type(VideoFramesFeaturizer)


class VideoFramesFeaturizer(Featurizer):
    '''Interface for featurizers that operate on videos represented as
    tensors of images.
    '''

    def _featurize(self, imgs):
        '''Featurizes the given video represented as a tensor of images.

        Args:
            imgs: a list (or d x ny x nx x 3 tensor) of images defining the
                video to featurize

        Returns:
            the feature vector
        '''
        raise NotImplementedError("subclass must implement _featurize()")


class VideoFeaturizerConfig(FeaturizerConfig):
    '''Base configuration class that encapsulates the name of an
    `VideoFeaturizer` subclass and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `VideoFeaturizer` subclass
        config: an instance of the Config class associated with the specified
            `VideoFeaturizer` subclass
    '''

    def __init__(self, d):
        super(VideoFeaturizerConfig, self).__init__(d)
        self._validate_type(VideoFeaturizer)


class VideoFeaturizer(Featurizer):
    '''Base class for featurizers that operate on entire videos.'''

    def _featurize(self, video_path):
        '''Featurizes the given video.

        Args:
            video_path: the path to the video

        Returns:
            the feature vector
        '''
        raise NotImplementedError("subclass must implement _featurize()")


class ORBFeaturizerConfig(Config):
    '''Configuration settings for an ORBFeaturizer.'''

    def __init__(self, d):
        self.num_keypoints = self.parse_number(d, "num_keypoints", default=128)


class ORBFeaturizer(ImageFeaturizer):
    '''ORB (Oriented FAST and rotated BRIEF features) Featurizer.

    Reference:
        http://www.willowgarage.com/sites/default/files/orb_final.pdf
    '''

    def __init__(self, config=None):
        '''Creates a new ORBFeaturizer instance.

        Args:
            config: an optional ORBFeaturizerConfig instance. If omitted, the
                default ORBFeaturizerConfig is used
        '''
        if config is None:
            config = ORBFeaturizerConfig.default()
        self.num_keypoints = config.num_keypoints
        super(ORBFeaturizer, self).__init__()

        try:
            # OpenCV 3
            self._orb = cv2.ORB_create(nfeatures=self.num_keypoints)
        except AttributeError:
            # OpenCV 2
            self._orb = cv2.ORB(nfeatures=self.num_keypoints)

    def dim(self):
        '''Returns the dimension of the features.'''
        return 32 * self.num_keypoints

    def _featurize(self, img):
        gray = etai.rgb_to_gray(img)
        return self._orb.detectAndCompute(gray, None)[1].flatten()


class RandFeaturizerConfig(Config):
    '''Configuration settings for a RandFeaturizer.'''

    def __init__(self, d):
        self.dim = self.parse_number(d, "dim", default=1024)


class RandFeaturizer(ImageFeaturizer, VideoFramesFeaturizer, VideoFeaturizer):
    '''Featurizer that returns a feature vector with uniformly random entries
    regardless of the input data.
    '''

    def __init__(self, config=None):
        '''Creates a new RandFeaturizer instance.

        Args:
            config: an optional RandFeaturizerConfig instance. If omitted, the
                default RandFeaturizerConfig is used
        '''
        if config is None:
            config = RandFeaturizerConfig.default()
        self._dim = config.dim
        super(RandFeaturizer, self).__init__()

    def dim(self):
        '''Returns the dimension of the features.'''
        return self._dim

    def _featurize(self, _):
        return np.random.rand(self._dim)


class FeaturesHandler(object):
    '''Base class for handling the reading and writing features to disk.

    The features are stored on disk in `features_dir` in .npy format with the
    following pattern:

        `<features_dir>/<features_patt>`

    where `features_patt` is a pattern like `%08d.npy`, `%s.npy`,
    `%08d-%08d.npy`, `%s-%08d.npy`, etc. that defines how to construct the
    filenames for the features.

    FeaturesHandler also supports reading/writing sequences of features with
    **1-based** indices.

    Attributes:
        features_dir: the backing directory for the features
    '''

    def __init__(self, features_dir, features_patt, glob_sequence_patt=None):
        '''Initializes the base FeaturesHandler.

        Args:
            features_dir: the backing directory in which to read/write features
            features_patt: the pattern to generate filenames for features
            glob_sequence_patt: an optional glob pattern for matching a
                sequence of features. If omitted, sequences of features cannot
                be read or written
        '''
        self.features_dir = features_dir
        self._features_patt = features_patt
        self._glob_sequence_patt = glob_sequence_patt
        etau.ensure_dir(self.features_dir)

    def _load_feature(self, *args):
        path = self._get_feature_path(*args)
        try:
            return self._load_feature_from_path(path)
        except IOError:
            raise FeatureNotFoundError(path)

    def _load_features_sequence(self, *args):
        paths = self._get_feature_sequence_paths(*args)
        if not paths:
            return None

        v = self._load_feature_from_path(paths[0])
        features = np.empty((len(paths), v.size))
        features[0] = v
        for idx, path in enumerate(paths[1:], 1):
            features[idx] = self._load_feature_from_path(path)

        return features

    def _write_feature(self, v, *args):
        path = self._get_feature_path(*args)
        self._write_feature_to_path(v, path)

    def _write_features_sequence(self, features, *args):
        for idx, v in enumerate(features, 1):
            self._write_feature(v, *(args + (idx,)))

    def _get_feature_path(self, *args):
        filename = self._features_patt % tuple(args)
        return os.path.join(self.features_dir, filename)

    def _get_feature_sequence_paths(self, *args):
        if self._glob_sequence_patt is None:
            raise ValueError(
                "Cannot read a sequence of features when `glob_sequence_patt "
                "== None`")

        filename_patt = self._glob_sequence_patt % tuple(args)
        sequence_patt = os.path.join(self.features_dir, filename_patt)
        return sorted(glob(sequence_patt))

    @staticmethod
    def _load_feature_from_path(path):
        return np.load(path)

    @staticmethod
    def _write_feature_to_path(v, path):
        np.save(path, v)


class FeatureNotFoundError(IOError):
    '''Exception raised when a feature is not found on disk.'''

    def __init__(self, path):
        super(FeatureNotFoundError, self).__init__(
            "Feature not found at '%s'" % path)


class ImageFeaturesHandler(FeaturesHandler):
    '''Class that handles reading and writing features to disk corresponding
    to an image.

    Unlike other feature handlers, `ImageFeaturesHandler` simply stores the
    features on disk (in .npy format) via a directly specified path.
    '''

    def __init__(self):
        '''Creates a ImageFeaturesHandler instance.'''
        pass

    def load_feature(self, feature_path):
        '''Load the feature from the given path.

        Args:
            feature_path: the feature path

        Returns:
            the feature vector

        Raises:
            FeatureNotFoundError: if the feature is not found on disk
        '''
        return self._load_feature_from_path(feature_path)

    def write_feature(self, v, feature_path):
        '''Writes the feature vector to disk.

        Args:
            v: the feature vector
            feature_path: the feature path
        '''
        etau.ensure_basedir(feature_path)
        return self._write_feature_to_path(v, feature_path)


class ImageObjectsFeaturesHandler(FeaturesHandler):
    '''Class that handles reading and writing features to disk corresponding to
    objects in an image.

    The features are stored on disk in `features_dir` in .npy format with the
    following pattern:

        `<features_dir>/%08d.npy`

    where the numeric parameter holds the object number.

    By convention, object numbers passed to this handler should be **1-based**
    indices corresponding to the ordering of the objects in the corresponding
    `eta.core.objects.DetectedObjectContainer` instance.

    Attributes:
        features_dir: the backing directory for the features
    '''

    FEATURES_PATT = "%08d.npy"
    IMAGE_OBJECTS_GLOB_PATT = "*.npy"

    def __init__(self, features_dir):
        '''Creates a ImageObjectsFeaturesHandler instance.

        Args:
            features_dir: the backing directory in which to read/write features
        '''
        super(ImageObjectsFeaturesHandler, self).__init__(
            features_dir, self.FEATURES_PATT,
            glob_sequence_patt=self.IMAGE_OBJECTS_GLOB_PATT)

    def load_feature(self, object_number):
        '''Load the feature for the given object.

        Args:
            object_number: the object number

        Returns:
            the feature vector

        Raises:
            FeatureNotFoundError: if the feature is not found on disk
        '''
        return self._load_feature(object_number)

    def load_features(self):
        '''Loads the features for all objects in the image.

        Returns:
            an `num_objects x dim` array of features

        Raises:
            FeatureNotFoundError: if a feature is not found on disk
        '''
        return self._load_features_sequence()

    def write_feature(self, v, object_number):
        '''Writes the feature vector to disk.

        Args:
            v: the feature vector
            object_number: the object number
        '''
        return self._write_feature(v, object_number)

    def write_features(self, features):
        '''Writes the feature vectors for the objects to disk.

        Args:
            features: a `num_objects x dim` array of feature vectors
        '''
        return self._write_features_sequence(features)


class ImageSetFeaturesHandler(FeaturesHandler):
    '''Class that handles reading and writing features to disk corresponding to
    images in a set.

    The features are stored on disk in `features_dir` in .npy format with the
    following pattern:

        `<features_dir>/%s.npy`

    where the string parameter holds the name of the image.

    By convention, image name strings passed to this handler should correspond
    to the keys of the images in the corresponding
    `eta.core.image.ImageSetLabels` instance.

    Attributes:
        features_dir: the backing directory for the features
    '''

    FEATURES_PATT = "%s.npy"

    def __init__(self, features_dir):
        '''Creates a ImageSetFeaturesHandler instance.

        Args:
            features_dir: the backing directory in which to read/write features
        '''
        super(ImageSetFeaturesHandler, self).__init__(
            features_dir, self.FEATURES_PATT)

    def load_feature(self, image_name):
        '''Load the feature for the given image.

        Args:
            image_name: the image name

        Returns:
            the feature vector

        Raises:
            FeatureNotFoundError: if the feature is not found on disk
        '''
        return self._load_feature(image_name)

    def write_feature(self, v, image_name):
        '''Writes the feature vector to disk.

        Args:
            v: the feature vector
            image_name: the image name
        '''
        return self._write_feature(v, image_name)


class ImageSetObjectsFeaturesHandler(FeaturesHandler):
    '''Class that handles reading and writing features to disk corresponding to
    detected objects in a set of images.

    The features are stored on disk in `features_dir` in .npy format with the
    following pattern:

        `<features_dir>/%s-%08d.npy`

    where the string parameter holds the name of the image, and the numeric
    parameter holds the object number.

    By convention, image name strings passed to this handler should correspond
    to the keys of the images in the corresponding
    `eta.core.image.ImageSetLabels` instance, and object numbers passed to this
    handler should be **1-based** indices corresponding to the ordering of the
    objects in the corresponding `eta.core.objects.DetectedObjectContainer`
    instance.

    Attributes:
        features_dir: the backing directory for the features
    '''

    FEATURES_PATT = "%s-%08d.npy"
    IMAGE_OBJECTS_GLOB_PATT = "%s-*.npy"

    def __init__(self, features_dir):
        '''Creates a ImageSetObjectsFeaturesHandler instance.

        Args:
            features_dir: the backing directory in which to read/write features
        '''
        super(ImageSetObjectsFeaturesHandler, self).__init__(
            features_dir, self.FEATURES_PATT,
            glob_sequence_patt=self.IMAGE_OBJECTS_GLOB_PATT)

    def load_feature(self, image_name, object_number):
        '''Load the feature for the given object from the given image.

        Args:
            image_name: the image name
            object_number: the object number

        Returns:
            the feature vector

        Raises:
            FeatureNotFoundError: if the feature is not found on disk
        '''
        return self._load_feature(image_name, object_number)

    def load_features(self, image_name):
        '''Loads the features for all objects in the given image.

        Args:
            image_name: the image name

        Returns:
            an `num_objects x dim` array of features

        Raises:
            FeatureNotFoundError: if a feature is not found on disk
        '''
        return self._load_features_sequence(image_name)

    def write_feature(self, v, image_name, object_number):
        '''Writes the feature vector to disk.

        Args:
            v: the feature vector
            image_name: the image name
            object_number: the object number
        '''
        return self._write_feature(v, image_name, object_number)

    def write_features(self, features, image_name):
        '''Writes the feature vectors for the objects to disk.

        Args:
            features: a `num_objects x dim` array of feature vectors
            image_name: the image name
        '''
        return self._write_features_sequence(features, image_name)


class VideoFramesFeaturesHandler(FeaturesHandler):
    '''Class that handles reading and writing features to disk corresponding to
    the frames of a video.

    The features are stored on disk in `features_dir` in .npy format with the
    following pattern:

        `<features_dir>/%08d.npy`

    where the numeric parameter holds the frame number.

    Attributes:
        features_dir: the backing directory for the features
    '''

    FEATURES_PATT = "%08d.npy"

    def __init__(self, features_dir):
        '''Creates a VideoFramesFeaturesHandler instance.

        Args:
            features_dir: the backing directory in which to read/write features
        '''
        super(VideoFramesFeaturesHandler, self).__init__(
            features_dir, self.FEATURES_PATT)

    def load_feature(self, frame_number):
        '''Load the feature for the given frame.

        Args:
            frame_number: the frame number

        Returns:
            the feature vector

        Raises:
            FeatureNotFoundError: if the feature is not found on disk
        '''
        return self._load_feature(frame_number)

    def write_feature(self, v, frame_number):
        '''Writes the feature vector to disk.

        Args:
            v: the feature vector
            frame_number: the frame number
        '''
        return self._write_feature(v, frame_number)


class VideoObjectsFeaturesHandler(FeaturesHandler):
    '''Class that handles reading and writing features to disk corresponding to
    detected objects in the frames of a video.

    The features are stored on disk in `features_dir` in .npy format with the
    following pattern:

        `<features_dir>/%08d-%08d.npy`

    where the first numeric parameter holds the frame number, and the second
    numeric parameter holds the object number.

    By convention, object numbers passed to this handler should be **1-based**
    indices corresponding to the ordering of the objects in the corresponding
    `eta.core.objects.DetectedObjectContainer` instance.

    Attributes:
        features_dir: the backing directory for the features
    '''

    FEATURES_PATT = "%08d-%08d.npy"
    FRAME_OBJECTS_GLOB_PATT = "%08d-*.npy"

    def __init__(self, features_dir):
        '''Creates a VideoObjectsFeaturesHandler instance.

        Args:
            features_dir: the backing directory in which to read/write features
        '''
        super(VideoObjectsFeaturesHandler, self).__init__(
            features_dir, self.FEATURES_PATT,
            glob_sequence_patt=self.FRAME_OBJECTS_GLOB_PATT)

    def load_feature(self, frame_number, object_number):
        '''Load the feature for the given object from the given frame.

        Args:
            frame_number: the frame number
            object_number: the object number

        Returns:
            the feature vector

        Raises:
            FeatureNotFoundError: if the feature is not found on disk
        '''
        return self._load_feature(frame_number, object_number)

    def load_features(self, frame_number):
        '''Loads the features for all objects in the given frame.

        Args:
            frame_number: the frame number

        Returns:
            an `num_objects x dim` array of features

        Raises:
            FeatureNotFoundError: if a feature is not found on disk
        '''
        return self._load_features_sequence(frame_number)

    def write_feature(self, v, frame_number, object_number):
        '''Writes the feature vector to disk.

        Args:
            v: the feature vector
            frame_number: the frame number
            object_number: the object number
        '''
        return self._write_feature(v, frame_number, object_number)

    def write_features(self, features, frame_number):
        '''Writes the feature vectors for the objects to disk.

        Args:
            features: a `num_objects x dim` array of feature vectors
            frame_number: the frame number
        '''
        return self._write_features_sequence(features, frame_number)


class BackingManagerConfig(Config):
    '''Base configuration class that encapsulates the name of a
    `BackingManager` subclass and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `BackingManager` subclass
        config: an instance of the Config class associated with the specified
            `BackingManager` subclass
    '''

    def __init__(self, d):
        self.type = self.parse_string(d, "type")
        self._backing_manager_cls, self._config_cls = Configurable.parse(
            self.type)
        self.config = self.parse_object(
            d, "config", self._config_cls, default=None)
        if not self.config:
            self.config = self._load_default_config()

    @classmethod
    def default(cls):
        '''Loads the default BackingManager.

        Returns:
            a BackingManager instance
        '''
        return cls({"type": "eta.core.features.RandomBackingManager"})

    def build(self):
        '''Factory method that builds the BackingManager instance from the
        config specified by this class.

        Returns:
            a BackingManager instance
        '''
        return self._backing_manager_cls(self.config)

    def _load_default_config(self):
        try:
            # Try to load the default config from disk
            return self._config_cls.load_default()
        except NotImplementedError:
            # Try default() instead
            return self._config_cls.default()


class BackingManager(Configurable):
    '''Abstract base class for all backing managers.'''

    @property
    def backing_dir(self):
        '''Returns the current backing directory.'''
        raise NotImplementedError("subclasses must implement `backing_dir`")

    def set_video_path(self, video_path):
        '''Sets the video path and ensures that the backing directory for the
        video exists.

        Args:
            video_path: the video path
        '''
        self._set_video_path(video_path)
        logger.info("Using backing directory '%s'", self.backing_dir)
        etau.ensure_dir(self.backing_dir)

    def _set_video_path(self, video_path):
        '''Internal implementation of setting the video path.

        Subclasses can implement this method if necessary

        Args:
            video_path: the video path
        '''
        pass

    def flush(self):
        '''Deletes the backing directory.'''
        logger.info("Deleting backing directory '%s'", self.backing_dir)
        etau.delete_dir(self.backing_dir)


class ManualBackingManagerConfig(Config):
    '''Configuration settings for a ManualBackingManager.

    Attributes:
        backing_dir: the backing directory in which to store features
    '''

    def __init__(self, d):
        self.backing_dir = self.parse_string(
            d, "backing_dir", default="/tmp/eta.backing")


class ManualBackingManager(BackingManager):
    '''Backing manager that stores features directly in the base directory.'''

    def __init__(self, config):
        '''Creates the ManualBackingManager instance.

        Args:
            config: a ManualBackingManager instance.
        '''
        self.validate(config)
        self.config = config

    @property
    def backing_dir(self):
        return self.config.backing_dir


class RandomBackingManagerConfig(Config):
    '''Configuration settings for a RandomBackingManager.

    Attributes:
        basedir: the base directory in which to store features
    '''

    def __init__(self, d):
        self.basedir = self.parse_string(d, "basedir", default="/tmp")


class RandomBackingManager(BackingManager):
    '''Backing manager that stores features in random subdirectories of the
    given base directory.
    '''

    def __init__(self, config):
        '''Creates the RandomBackingManager instance.

        Args:
            config: a RandomBackingManagerConfig instance.
        '''
        self.validate(config)
        self.config = config
        self._backing_dir = None

    @property
    def backing_dir(self):
        return self._backing_dir

    def _set_video_path(self, video_path):
        etau.ensure_dir(self.config.basedir)
        self._backing_dir = tempfile.mkdtemp(
            dir=self.config.basedir, prefix="eta.backing.")


class PatternBackingManagerConfig(Config):
    '''Configuration settings for a PatternBackingManager.

    Attributes:
        path_replacers: an array of (find, replace) strings to apply to the
            video path
    '''

    def __init__(self, d):
        self.path_replacers = self.parse_array(d, "path_replacers")


class PatternBackingManager(BackingManager):
    '''Backing manager that uses a list of (find, replace) strings to generate
    a (hopefully unique) backing directory for each video path.
    '''

    def __init__(self, config):
        '''Creates the PatternBackingManager instance.

        Args:
            config: a PatternBackingManagerConfig instance.
        '''
        self.validate(config)
        self.config = config
        self._backing_dir = None

    @property
    def backing_dir(self):
        return self._backing_dir

    def _set_video_path(self, video_path):
        self._backing_dir = etau.replace_strings(
            video_path, self.config.path_replacers)


class CachingVideoFeaturizerConfig(Config):
    '''Configuration settings for a CachingVideoFeaturizer.

    Attributes:
        frame_featurizer: an ImageFeaturizerConfig specifying the Featurizer
            to use to embed the video frames
        backing_manager: a BackingManagerConfig specifying the backing manager
            to use
        frames: a string specifying the specific frames to featurize
        delete_backing_directory: whether to delete the backing directory
            when the featurizer is stopped
    '''

    def __init__(self, d):
        self.frame_featurizer = self.parse_object(
            d, "frame_featurizer", ImageFeaturizerConfig)
        self.backing_manager = self.parse_object(
            d, "backing_manager", BackingManagerConfig, default=None)
        if self.backing_manager is None:
            self.backing_manager = BackingManagerConfig.default()
        self.frames = self.parse_string(d, "frames", default=None)
        self.delete_backing_directory = self.parse_bool(
            d, "delete_backing_directory", default=True)


class CachingVideoFeaturizer(Featurizer):
    '''Meta-featurizer that uses an ImageFeaturizer to maintain a cache of the
    feature vectors for the frames of a video.

    Featurized frames are stored on disk as .npz files indexed by frame number.
    The location of the files on disk is controlled by the `backing_manager`.
    By default, a `RandomBackingManager` is used that writes features to a
    randomly generated subdirectory of `/tmp`. Alternatively, you can manually
    set the backing directory with `set_manual_backing_dir()`.

    This class provides a `frame_preprocessor` property that allows a
    preprocessing function to be applied to each frame before featurizing it.
    By default, no preprocessing is performed.

    This class implements the iterator interface, which allows you to iterate
    over the featurized frames of a video using the following syntax:

    ```
    with CachingVideoFeaturizer(...) as f:
        f.featurize(video_path)
        for v in f:
            # use feature vector, v
    ```
    '''

    def __init__(self, config):
        '''Creates a CachingVideoFeaturizer instance.

        Args:
            config: a CachingVideoFeaturizerConfig instance
        '''
        self.validate(config)
        self.config = config
        super(CachingVideoFeaturizer, self).__init__()

        self._frame_featurizer = self.config.frame_featurizer.build()
        logger.info("Loaded featurizer %s", type(self._frame_featurizer))

        self._backing_manager = self.config.backing_manager.build()
        logger.info("Loaded backing manager %s", type(self._backing_manager))
        self._manual_backing_dir = None

        self._frame_preprocessor = None
        self._frame_string = "%08d.npz"

        self._iter_frame = None
        self._iter_paths = None

    def __iter__(self):
        self._iter_frame = -1
        self._iter_paths = self.get_feature_paths()
        return self

    def __next__(self):
        self._iter_frame += 1
        try:
            return self._read_feature(self._iter_paths[self._iter_frame])
        except IndexError:
            self._iter_frame = None
            self._iter_paths = None
            raise StopIteration

    def dim(self):
        '''Returns the dimension of the underlying frame Featurizer.'''
        return self._frame_featurizer.dim()

    @property
    def backing_dir(self):
        '''The current backing directory.

        If a manual backing directory was set via `set_manual_backing_dir`, it
        will be returned here.
        '''
        if self._manual_backing_dir is not None:
            return self._manual_backing_dir
        return self._backing_manager.backing_dir

    def set_manual_backing_dir(self, manual_backing_dir):
        '''Manually sets the backing directory.

        If a manual backing directory is set, it will take precedence over the
        backing manager's directory. To remove this manual setting, call
        `clear_manual_backing_dir()`.

        Args:
            manual_backing_dir: the manual backing directory to use
        '''
        self._manual_backing_dir = manual_backing_dir

    def clear_manual_backing_dir(self):
        '''Clears the manual backing directory, if necessary.'''
        self._manual_backing_dir = None

    @property
    def frame_preprocessor(self):
        '''The frame processor applied to each frame before featurizing.'''
        return self._frame_preprocessor

    @frame_preprocessor.setter
    def frame_preprocessor(self, fcn):
        self._frame_preprocessor = fcn

    @frame_preprocessor.deleter
    def frame_preprocessor(self):
        self._frame_preprocessor = None

    def _start(self):
        self._frame_featurizer.start()

    def _stop(self):
        self._frame_featurizer.stop()
        if self.config.delete_backing_directory:
            self._backing_manager.flush()

    def featurize(self, video_path, frames=None):
        '''Featurizes the frames of the input video.

        Attributes:
            video_path: the input video path
            frames: an optional frames string to specify the frames of the
                video to featurize. By default, all frames are featurized
        '''
        self.start(warn_on_restart=False, keep_alive=False)
        self._featurize(video_path, frames)
        if self._keep_alive is False:
            self.stop()

    def _featurize(self, video_path, frames):
        if frames is None:
            frames = self.config.frames

        self._backing_manager.set_video_path(video_path)
        with etav.FFmpegVideoReader(video_path, frames=frames) as vr:
            for img in vr:
                if self._frame_preprocessor is not None:
                    _img = self._frame_preprocessor(img)
                else:
                    _img = img

                v = self._frame_featurizer.featurize(_img)
                path = self.get_featurized_frame_path(vr.frame_number)
                self._write_feature(v, path)

    def is_featurized(self, frame_number):
        '''Returns True/False if the given frame has been featurized.

        Args:
            frame_number: the frame number

        Returns:
            True/False
        '''
        path = self.get_featurized_frame_path(frame_number)
        return os.path.isfile(path)

    def get_feature_paths(self):
        '''Returns a list of absolute paths to the features on disk.

        Returns:
            a list of absolute paths to .npz files
        '''
        return [
            p for p in etau.list_files(self.backing_dir, abs_paths=True)
            if p.endswith(".npz")]

    def get_featurized_frame_path(self, frame_number):
        '''Returns the feature path on disk for the given frame number.

        The actual file may or may not exist.

        Args:
            frame_number: the frame number

        Returns:
            the path to the feature vector on disk
        '''
        return os.path.join(
            self.backing_dir, self._frame_string % frame_number)

    def load_feature_for_frame(self, frame_number):
        '''Loads the feature for the given frame.

        Args:
            frame_number: the frame number

        Returns:
            the feature vector

        Raises:
            FeaturizedFrameNotFoundError: if the feature vector was not found
            on disk
        '''
        path = self.get_featurized_frame_path(frame_number)
        return self._read_feature(path)

    def load_features_for_frames(self, frame_range):
        '''Loads the features for the given range of frames.

        Args:
            frame_range: a (start, stop) tuple definining the range of frames
                to load (inclusive)

        Returns:
            an n x d array of features, where n = stop - start + 1

        Raises:
            FeaturizedFrameNotFoundError: if any of the feature vectors were
            not found on disk
        '''
        X = []
        for frame_number in range(frame_range[0], frame_range[1] + 1):
            X.append(self.load_feature_for_frame(frame_number))
        return np.array(X)

    def load_all_features(self):
        '''Loads all features for the last featurized video.

        Use this method with caution; the return matrix may have many rows!

        Returns:
            an n x d array of features, where n = # of frames that were
                featurized

        Raises:
            FeaturizedFrameNotFoundError: if any of the feature vectors were
            not found on disk
        '''
        X = []
        for path in self.get_feature_paths():
            X.append(self._read_feature(path))
        return np.array(X)

    @staticmethod
    def _write_feature(v, path):
        np.savez_compressed(path, v=v)

    @staticmethod
    def _read_feature(path):
        try:
            return np.load(path)["v"]
        except (IOError, OSError):
            raise FeaturizedFrameNotFoundError(
                "Feature vector not found at '%s'" % path)


class FeaturizedFrameNotFoundError(Exception):
    '''Exception raised when a featurized frame is not found on disk.'''
    pass
