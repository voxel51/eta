'''
Core interfaces, data structures, and methods for feature extraction in images.

Copyright 2017-2018, Voxel51, LLC
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

import errno
import logging
import os
import shutil
import tempfile

import cv2
import numpy as np

from eta.core.config import Config, Configurable
from eta.core.numutils import GrowableArray
import eta.core.utils as etau
import eta.core.types as etat
import eta.core.video as etav


logger = logging.getLogger(__name__)


class FeaturizerConfig(Config):
    '''Configuration class that encapsulates the name of a Featurizer and an
    instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the Featurizer, e.g.,
            "eta.core.features.VideoFramesFeaturizer"
        config: an instance of the Config class associated with the specified
            Featurizer (e.g. an instance of
            eta.core.features.VideoFramesFeaturizerConfig)
    '''

    def __init__(self, d):
        self.type = self.parse_string(d, "type")
        self._featurizer_cls, config_cls = Configurable.parse(self.type)
        self.config = self.parse_object(d, "config", config_cls, default=None)
        if not self.config:
            # Try to load the default config for the featurizer
            self.config = config_cls.default()

    def build(self):
        '''Factory method that builds the Featurizer instance from the config
        specified by this class.
        '''
        return self._featurizer_cls(self.config)


class Featurizer(Configurable):
    '''Base class for all featurizers.

    Subclasses of Featurizer must implement the `dim()` and `_featurize()`
    methods, and if necessary, should also implement the `_start()` and
    `_stop()` methods.

    Subclasses must call the superclass constructor defined by this base class.

    Note that Featurizer implements the context manager interface, so
    Featurizer subclasses can be used with the following convenient `with`
    syntax to automatically handle `start()` and `stop()` calls:

    ```
    with <My>Featurizer(...) as f:
        f.featurize(data)
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
        raise NotImplementedError("subclass must implement dim().")

    def start(self, warn_on_restart=True, keep_alive=True):
        '''Start method that handles any necessary setup to prepare the
        Featurizer for use.

        This method can be explicitly called by users. If it is not called
        manually, it will be called each time `featurize` is called.

        Args:
            warn_on_restart: whether to generate a warning if `start` is
                called when the Featurizer is already started. The default
                value is True
            keep_alive: whether to keep the Featurizer alive (i.e. not to call
                `stop`) after a each `featurize` call
        '''
        if warn_on_restart and self._is_started:
            logger.warning("Featurizer.start() called when already started.")

        if self._is_started:
            return

        self._is_started = True
        self._keep_alive = keep_alive
        self._start()

    def _start(self):
        '''The backend implementation that is called when the public `start`
        method is called. Subclasses that require startup configuration should
        override this method.
        '''
        pass

    def stop(self):
        '''Stop method that handles any necessary cleanup after featurization
        is complete.

        This method can be explicitly called by users, and, in fact, it must
        be called by users who called `start` themselves. If `start` was not
        called manually or `keep_alive` was set to False, then this method will
        be invoked at the end of each call to `featurize`.
        '''
        if not self._is_started:
            return

        self._stop()
        self._is_started = False
        self._keep_alive = False

    def _stop(self):
        '''The backend implementation that is called when the public `stop`
        method is called. Subclasses that require cleanup after featurization
        should override this method.
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
        fv = self._featurize(data)
        if self._keep_alive is False:
            self.stop()

        return fv

    def _featurize(self, data):
        '''The backend implementation of the feature extraction routine.
        Subclasses must implement this method.

        Args:
            data: the data to featurize

        Returns:
            the feature vector
        '''
        raise NotImplementedError("subclass must implement _featurize()")


class CanFeaturize(object):
    '''Mixin class that exposes the ability to featurize data just-in-time via
    a provided Featurizer instance.
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
        of the argument to featurize in *args or a named argument.  The code
        tries to reconcile one of them, ultimately failing if it cannot find
        one.

        The method we use to tell if the argument needs to be featurized is by
        checking if it is a string that points to a file on the disk, and if
        that fails, if it is a string that points to a valid video.

        You can decorate with either just `@CanFeaturize.featurize_if_needed`,
        in which case
        or by specifying specific names of arguments to operate on either as
        `@CanFeaturize.featurize_if_needed(arg_name="foo")`
        or just
        `@CanFeaturize.featurize_if_needed("foo")` and this will be a name not
        an index.

        Args:
            arg_name ("X"): a string specifying the name of the argument
                passed to the original function that you want to featurize

            arg_index (0): an int specifying the index of the argument passed
                to the original function that you want to featurize. If
                `arg_name` is provided, it takes precedence over `arg_index`

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
                            should_featurize = etat.Video.is_valid_path(data)

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


class FeaturizedFrameNotFoundError(OSError):
    '''Exception raised when a featurized frame is not found on disk.'''
    pass


class VideoFramesFeaturizerConfig(Config):
    '''Specifies the configuration settings for the VideoFeaturizer class.'''

    def __init__(self, d):
        self.backing_path = self.parse_string(
            d, "backing_path", default="/tmp")
        self.backing_manager = self.parse_string(
            d, "backing_manager", default="random")
        self.backing_manager_remove_random = self.parse_bool(
            d, "backing_manager_remove_random", default=True)
        self.backing_manager_path_replace = self.parse_array(
            d, "backing_manager_path_replace", default=[])
        self.frame_featurizer = self.parse_object(
            d, "frame_featurizer", FeaturizerConfig)
        self.frames = self.parse_string(d, "frames", default="*")


class VideoFramesFeaturizer(Featurizer):
    '''Class that encapsulates featurizing the frames of a video.

    A VideoFramesFeaturizer is a meta-Featurizer that uses the Featurizer
    specified by `frame_featurizer` internally to featurize the frames.

    Featurized frames are stored on disk as compressed pickle files indexed by
    frame number. The location of the files on disk is controlled by the
    `backing_path` attribute. By default, the backing path is `/tmp`.

    This class also allows a `frame_preprocessor` function to be installed
    that preprocesses each input frame before featurizing it. By default, no
    preprocessing is performed.

    **WARNING** if you use the same backing path for multiple videos your
    features will be invalid (features on disk are not overwritten, they are
    simply skipped).

    To automate the management of the the backing path, this class supports an
    optional `backing_manager` field that has the following options:

        "random" (default)
            a new random subdirectory of `backing_path` is generated each time
            `featurize` is called. This guarantees that features will not
            collide with existing features on disk. Note that the randomly
            created backing directory is removed after each featurization,
            unless config field `backing_manager_remove_random` is set to False

        "replace"
            the field `backing_manager_path_replace` is used to replace
            substrings in the filename of each video featurized, which will
            *hopefully* yield a new, unique output path

        "manual"
            the provided `backing_path` is used verbatim

    @todo Refactor the backing managers into standalone Configurable classes

    @todo: Generalize to allow non npz-able features
    '''

    def __init__(self, config):
        '''Creates a new VideoFramesFeaturizer and initializes the backing
        storage.

        Args:
            config: a VideoFramesFeaturizerConfig instance
        '''
        self.validate(config)
        self.config = config
        self.most_recent_frame = -1

        super(VideoFramesFeaturizer, self).__init__()

        self._frame_string = "%08d.npz"
        self._frame_preprocessor = None
        self._frame_featurizer = None
        self._backing_path = None

        backing_managers = {
            "random": self._backing_manager_random,
            "replace": self._backing_manager_replace,
            "manual": self._backing_manager_manual,
        }
        self._backing_manager = backing_managers[self.config.backing_manager]
        self.update_backing_path(self.config.backing_path)
        self._backing_manager_random_last_tempdir = None

    @property
    def frame_preprocessor(self):
        '''The frame processor applied to each frame before featurizing.'''
        return self._frame_preprocessor

    @frame_preprocessor.setter
    def frame_preprocessor(self, fp):
        self._frame_preprocessor = fp

    @frame_preprocessor.deleter
    def frame_preprocessor(self):
        self._frame_preprocessor = None

    def _backing_manager_random(self, video_path, is_featurize_start=True):
        '''Backing manager that generates a new unique subdirectory of
        `backing_path` for each video processed.
        '''
        if is_featurize_start:
            td = tempfile.mkdtemp(
                dir=self.config.backing_path, prefix="eta.backing.")
            self.update_backing_path(td)
            self._backing_manager_random_last_tempdir = td
            return

        if self.config.backing_manager_remove_random:
            shutil.rmtree(self._backing_manager_random_last_tempdir)
        self.update_backing_path(self.config.backing_path)

    def _backing_manager_replace(self, video_path, is_featurize_start=True):
        '''Backing manager that generates a (hopefully) unique backing
        directory for each video processed by peforming a find-and-replace
        string operation on the video path.
        '''
        if is_featurize_start:
            rp = etau.replace_strings(
                video_path, self.config.backing_manager_path_replace)
            self._backing_manager_random_last_tempdir = rp
            self.update_backing_path(rp)
            return

        self.update_backing_path(self.config.backing_path)

    def _backing_manager_manual(self, video_path, is_featurize_start=True):
        '''Backing manager that simply uses the provided `backing_path`.'''
        pass

    def dim(self):
        '''Returns the dimension of the underlying frame Featurizer.'''
        if not self._frame_featurizer:
            self._frame_featurizer = self.config.frame_featurizer.build()
            d = self._frame_featurizer.dim()
            self._frame_featurizer = None
        else:
            d = self._frame_featurizer.dim()

        return d

    def is_featurized(self, frame_number):
        '''Checks the backing store to determine whether or not the frame
        number is already featurized and stored to disk.
        '''
        return os.path.isfile(self.featurized_frame_path(frame_number))

    def retrieve_featurized_frame(self, frame_number):
        '''The frame_number here is rendered into a string for the filepath.
        So, if it is not available as part of the video, then it will be an
        error.

        No checking is explicitly done here. Careful about starting from
        0 or 1.
        '''
        p = self.featurized_frame_path(frame_number)
        if not os.path.isfile(p):
            raise FeaturizedFrameNotFoundError("Feature %d not found", p)

        return np.load(p)["v"]

    def featurize(self, video_path, frames=None, returnX=True):
        '''Featurizes the frames of the input video.

        Attributes:
            video_path: the input video path
            frames: an optional frames string to specify the frames of the
                video to featurize. By default, the value provided in the
                VideoFramesFeaturizerConfig is used
            returnX: whether to return the frames matrix

        Returns:
            If returnX is True, a (# frames) x (# dims) array is returned
                whose rows contain the computed features
        '''
        if not frames:
            frames = self.config.frames

        self._backing_manager(video_path)
        self.start(warn_on_restart=False, keep_alive=False)
        v = self._featurize(video_path, frames, returnX)
        if self._keep_alive is False:
            self.stop()

        self._backing_manager(video_path, False)

        return v

    def _featurize(self, video_path, frames=None, returnX=True):
        frames = frames or self.config.frames
        logger.debug("Featurizing frames %s" % frames)

        if returnX:
            X = None

        with etav.FFmpegVideoReader(video_path, frames=frames) as vr:
            for img in vr:
                self.most_recent_frame = vr.frame_number
                path = self.featurized_frame_path(vr.frame_number)

                try:
                    # Try to load the existing feature
                    v = self.retrieve_featurized_frame(vr.frame_number)
                except FeaturizedFrameNotFoundError:
                    # Build the per-frame Featurizer, if necessary
                    if not self._frame_featurizer:
                        self._frame_featurizer = \
                            self.config.frame_featurizer.build()
                        self._frame_featurizer.start()

                    if self._frame_preprocessor is not None:
                        # Pre-process and then featurize the frame
                        _img = self._frame_preprocessor(img)
                        v = self._frame_featurizer.featurize(_img)
                    else:
                        # Featurize the frame
                        v = self._frame_featurizer.featurize(img)

                    # Write the feature to disk
                    np.savez_compressed(path, v=v)

                if returnX:
                    if X is None:
                        # Lazily build the GrowableArray now that we know the
                        # dimension of the features
                        X = GrowableArray(len(v))
                    X.update(v)

        if self._frame_featurizer and not self._keep_alive:
            # Stop the frame featurizer
            self._frame_featurizer.stop()
            self._frame_featurizer = None

        return X.finalize() if returnX else None

    def featurized_frame_path(self, frame_number):
        '''Returns the backing path for the given frame number.'''
        return os.path.join(
            self._backing_path, self._frame_string % frame_number)

    def flush_backing(self):
        '''Deletes all existing feautres on disk in the current backing path.
        The backing directory itself is not deleted.
        '''
        files = [
            f for f in os.listdir(self._backing_path) if f.endswith(".npz")]
        for f in files:
            os.remove(os.path.join(self._backing_path, f))

    def _stop(self):
        if self._frame_featurizer:
            self._frame_featurizer.stop()
            self._frame_featurizer = None

    def update_backing_path(self, backing_path):
        '''Update the backing path and create the directory tree, if needed.'''
        self._backing_path = backing_path
        try:
            os.makedirs(self._backing_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class ORBFeaturizer(Featurizer):
    '''ORB (Oriented FAST and rotated BRIEF features) Featurizer.

    Reference:
        http://www.willowgarage.com/sites/default/files/orb_final.pdf
    '''

    def __init__(self, num_keypoints=128):
        '''Constructs a new ORB Featurizer instance.'''
        super(ORBFeaturizer, self).__init__()
        self.num_keypoints = num_keypoints

        try:
            # OpenCV 3
            self.orb = cv2.ORB_create(nfeatures=num_keypoints)
        except AttributeError:
            # OpenCV 2
            self.orb = cv2.ORB(nfeatures=num_keypoints)

    def dim(self):
        '''Return the dimension of the features.'''
        return 32 * self.num_keypoints

    def _featurize(self, img):
        gray = etai.rgb_to_gray(img)
        return self.orb.detectAndCompute(gray, None)[1].flatten()


class RandFeaturizer(Featurizer):
    '''Random Featurizer that returns a feature vector with uniformly random
    entries regardless of the input data.
    '''

    def __init__(self, dim=1024):
        '''Constructs a RandFeaturizer instance.

        Args:
            dim: the desired embedding dimension. The default value is 1024
        '''
        super(RandFeaturizer, self).__init__(self)
        self._dim = dim

    def dim(self):
        '''Returns the dimension of the features extracted by this
        Featurizer.
        '''
        return self._dim

    def _featurize(self, _):
        return np.random.rand(self._dim)
