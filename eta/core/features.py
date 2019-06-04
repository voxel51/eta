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
import six
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

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


class CanFeaturize(object):
    '''Mixin class that exposes the ability to featurize data just-in-time via
    a provided Featurizer instance.

    This class allows you to decorate methods that should featurize data if
    necessary with one of the following strategies:
        - `@CanFeaturize.featurize_if_needed`: the first function argument will
            be featurized
        - `@CanFeaturize.featurize_if_needed(arg_name="foo")`: the arg with the
            specified name will be featurized
        - `@CanFeaturize.featurize_if_needed("foo")`: the arg with the
            specified name will be featurized
        - `@CanFeaturize.featurize_if_needed(index)`: the arg in the specified
            position will be featurized

    Arguments are featurized if they are strings that point to a valid file(s)
    on disk (including videos represented as sequences of frames).

    Alternatively, when `force_featurize=True`, all arguments to the function
    are automatically featurized.
    '''

    def __init__(self, featurizer=None, force_featurize=False):
        '''Initializes a CanFeaturize instance.

        Args:
            featurizer: the actual featurizer to use when needed
            force_featurize: whether to force any input to the
                `featurize_if_needed` decorator to be featurized. By default,
                this is False
        '''
        self.featurizer = featurizer
        self.force_featurize = force_featurize

    @property
    def has_featurizer(self):
        '''Determines whether this instance has a Featurizer.'''
        return bool(self.featurizer)

    def get_featurizer(self):
        '''Gets the Featurizer used by this instance, if any.'''
        return self.featurizer

    def set_featurizer(self, featurizer):
        '''Sets the Featurizer for this instance.'''
        self.featurizer = featurizer

    def remove_featurizer(self):
        '''Removes the Featurizer from this instance, if any.'''
        self.featurizer = None

    @staticmethod
    def featurize_if_needed(*args, **kwargs):
        '''This decorator function will check a specified argument of the
        decorated function needs to be featurized, and if so, featurizes it
        using the `featurizer` attribute of the class instances (already
        defined).

        The argument to featurize can be specified as either the numeric index
        of the argument to featurize or a named argument. The code tries to
        reconcile one of them, ultimately failing if it cannot find one.

        Args:
            arg_name ("X"): a string specifying the name of the argument
                passed to the original function that you want to featurize
                if necessary

            arg_index (0): an int specifying the index of the argument passed
                to the original function that you want to featurize if
                necessary

        Raises:
            CanFeaturizeError: if featurization failed or was not allowed
        '''
        # The default argument name to featurize.
        arg_name = "X"

        # The default positional index to featurize. This is 1, not 0, because
        # we assume the annotated method is a class method, not a standalone
        # function.
        arg_index = 1

        # Parse input arguments.
        arg = args[0] if args else None
        if not callable(arg):
            n = len(args)
            if n >= 1:
                arg_name = args[0]
            elif "arg_name" in kwargs:
                arg_name = kwargs["arg_name"]

            if n >= 2:
                arg_index = args[1]
            elif "arg_index" in kwargs:
                arg_index = kwargs["arg_index"]

        # At this point, we have processed all possible invocations of the
        # annotation (the decorator) and we have the arguments to use.

        def decorated_(caller):
            '''Just the outside decorator that will pop the caller off the
            argument list.
            '''
            def decorated(*args, **kwargs):
                '''The main decorator function that handles featurization.'''
                # args[0] is the the calling object.
                cfobject = args[0]
                if not isinstance(cfobject, CanFeaturize):
                    raise CanFeaturizeError(
                        "featurize_if_needed can only decorate CanFeaturize "
                        "subclass methods; found %s" % cfobject.__class__)

                if not cfobject.featurizer:
                    #
                    # We cannot featurize if there is no featurizer.
                    #
                    # Note that this is allowed for flexibility: if you do not
                    # want to featurize, do not set the featurizer and then
                    # we'll early exit here.
                    #
                    return caller(*args, **kwargs)

                # Determine what calling syntax was used.
                data = None
                used_name = False
                used_index = False
                if arg_name in kwargs:
                    # Argument specified by name.
                    data = kwargs[arg_name]
                    used_name = True
                elif arg_index < len(args):
                    # Argument specified by positional index.
                    data = args[arg_index]
                    used_index = True
                else:
                    logger.warning("Unknown argument; skipping featurization")
                    return caller(*args, **kwargs)

                # Determine whether we need to featurize the input data.
                should_featurize = cfobject.force_featurize
                if not should_featurize and (used_name or used_index):
                    if isinstance(data, six.string_types):
                        if os.path.exists(data):
                            should_featurize = True
                        else:
                            #
                            # The data is a string but not a single file, but
                            # it might be a file sequence.
                            #
                            # Currently the only such sequence we support is
                            # a video, so we check if data is a valid video
                            # path.
                            #
                            should_featurize = etav.is_supported_video(data)

                # Perform the actual featurization, if necessary.
                if should_featurize:
                    data = cfobject.featurizer.featurize(data)

                    # Replace the data with its features.
                    if used_name:
                        kwargs[arg_name] = data
                    if used_index:
                        targs = list(args)
                        targs[arg_index] = data
                        args = tuple(targs)

                return caller(*args, **kwargs)

            return decorated

        # If arg is callable then we called it just with @featurize_if_needed.
        # Otherwise, we gave it parameters or even just parentheses.
        return decorated_(arg) if callable(arg) else decorated_


class CanFeaturizeError(Exception):
    '''Exception raised when an invalid usage of CanFeaturize is found.'''
    pass


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
    '''Configuration settings for a RandomBackingManager.

    Attributes:
        basedir: the base directory in which to store features
    '''

    def __init__(self, d):
        self.basedir = self.parse_string(
            d, "basedir", default="/tmp/eta.backing")


class ManualBackingManager(BackingManager):
    '''Backing manager that stores features directly in the base directory.'''

    def __init__(self, config):
        '''Creates the RandomBackingManager instance.

        Args:
            config: a RandomBackingManagerConfig instance.
        '''
        self.validate(config)
        self.config = config

    @property
    def backing_dir(self):
        return self.config.basedir


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
        self.delete_backing_directory = self.parse_bool(
            d, "delete_backing_directory", default=True)


class CachingVideoFeaturizer(Featurizer):
    '''Meta-featurizer that uses an ImageFeaturizer to maintain a cache of the
    feature vectors for the frames of a video.

    Featurized frames are stored on disk as compressed pickle files indexed by
    frame number. The location of the files on disk is controlled by the
    `backing_manager`. By default, a `RandomBackingManager` is used that writes
    features to a randomly generated subdirectory of `/tmp`.

    This class also allows a `frame_preprocessor` function to be set that
    preprocesses each input frame before featurizing it. By default, no
    preprocessing is performed.
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

        self._frame_preprocessor = None
        self._frame_string = "%08d.npz"

    def dim(self):
        '''Returns the dimension of the underlying frame Featurizer.'''
        return self._frame_featurizer.dim()

    @property
    def backing_dir(self):
        '''Returns the current backing directory.'''
        return self._backing_manager.backing_dir

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
        filenames = etau.list_files(self.backing_dir)
        for path in [os.path.join(self.backing_dir, f) for f in filenames]:
            X.append(self._read_feature(path))
        return np.array(X)

    @staticmethod
    def _write_feature(v, path):
        np.savez_compressed(path, v=v)

    @staticmethod
    def _read_feature(path):
        try:
            return np.load(path)["v"]
        except OSError:
            raise FeaturizedFrameNotFoundError(
                "Feature vector not found at '%s'" % path)


class FeaturizedFrameNotFoundError(Exception):
    '''Exception raised when a featurized frame is not found on disk.'''
    pass
