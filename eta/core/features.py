'''
Core interfaces, data structures, and methods for feature extraction in images.
features.py

Copyright 2017, Voxel51, LLC
voxel51.com

Jason Corso, jason@voxel51.com
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

import errno
import logging
import os
import shutil
import tempfile

import numpy as np

from eta.core.config import Config, Configurable
from eta.core.numutils import GrowableArray
import eta.core.video as etav
import eta.core.utils as etau


logger = logging.getLogger(__name__)


class FeaturizerConfig(Config):
    '''Featurizer configuration settings.

    d['type'] is the fully-qualified class name of the Featurizer, e.g.
    "eta.core.features.VideoFeaturizer". The parent module is loaded if
    necessary.
    '''

    def __init__(self, d):
        self.type = self.parse_string(d, "type")
        self._featurizer_cls, config_cls = Configurable.parse(self.type)
        self.config = self.parse_object(d, "config", config_cls)

    def build(self):
        '''Factory method to build the featurizer instance by the class
        name.'''
        return self._featurizer_cls(self.config)


class Featurizer(Configurable):
    '''Interface for feature extraction methods.

    Note that subclasses of Featurizer MUST call
    `super(SUBCLASSNAME, self).__init__()` in their __init__() in order for the
    Featurizer state management to be set up properly.  (This is necessitated
    by the loose inheritance implementation in Python.)

    @todo Sub-class to an ImageFeaturizer for typing concerns.
    '''

    def __init__(self):
        '''Initializes any featurizer by setting up internal state.'''
        self._started = False
        self._keep_alive = False

    def dim(self):
        '''Return the dimension of the features extracted by this method.'''
        raise NotImplementedError("subclass must implement dim().")

    def _start(self):
        '''Actual start code that subclasses need to override.  The public
        start function will set up state and then invoke this one.
        '''
        pass

    def start(self, warn_on_restart=True, keep_alive=True):
        '''Called by featurize before it starts in case any environment needs
        to be set up by subclasses.  It can be explicitly called by users of
        subclasses; state will be managed nonetheless.  The default parameter
        settings handle the case where an outside user of the class wants to
        `start()` the process once and then will featurizer many cases until
        they explicitly call `stop()`.
        '''
        if warn_on_restart and self._started:
            logger.warning('featurizer start called when already started.')

        if self._started:
            return

        self._started = True
        self._keep_alive = keep_alive
        self._start()

    def _stop(self):
        '''Actual stop code that subclasses need to override.  The public stop
        function will manage state and invoke this one.
        '''
        pass

    def stop(self):
        '''Called by featurize after it finishes to handle state management.
        If the user called start themselves, this should BE called by them as
        well; the Featurizer is not able to know the process has ended in this
        case.
        '''
        if not self._started:
            return

        self._stop()
        self._started = False
        self._keep_alive = False

    def _featurize(self, data):
        '''The core feature extraction routine that subclasses need to
        implement.'''
        raise NotImplementedError("subclass must implement _featurize().")

    def featurize(self, data):
        '''The core feature extraction routine to be called by users of the
        featurizer.  It appropriately interacts with the featurizer state
        management.'''
        self.start(warn_on_restart=False, keep_alive=False)
        v = self._featurize(data)
        if self._keep_alive is False:
            self.stop()
        return v


class CanFeaturize(object):
    '''Class exposes the ability to featurize data by storing a featurizer,
    allowing it to be set, get and used.
    '''

    def __init__(self, featurizer=None, hint_featurize=False):
        '''Initialize CanFeaturize instance.'''
        self.featurizer = featurizer
        self.hint_featurize = hint_featurize

    @staticmethod
    def featurize_if_needed(*args, **kwargs):
        '''This decorator function will check the `argX_to_featurize` to see if
        it needs to be featurized and if so, it will featurize it using the arg
        featurizer (already defined).  Note that argX_to_featurize is either
        the numeric index of the argument to featurize in *args or a named
        argument.  The code tries to reconcile one of them, ultimately failing
        if it cannot find one.

        The method we use to tell if the `argX_to_featurize` needs to be
        featurized is by checking if it is a string that points to a file on
        the disk.

        You can decorate with either just `@CanFeaturize.featurize_if_needed`
        or by specifying specific names of arguments to operate on
        `@CanFeaturize.featurize_if_needed(argname_to_featurize="foo")`
        or just
        `@CanFeaturize.featurize_if_needed("foo")` and this will be a name not
        an index.

        Args:
            argname_to_featurize ("X") specifies the name of the argument
            passed to the original function that you want to featurize

            argindex_to_featurize (0) specifies the index of the argument
            passed to the original function that you want to featurize.  The
            argname takes precedence over the argindex.
        '''
        # Handling the various types of invocations of the decorator.
        arg = None
        if args:
            arg = args[0]

        # Default argument settings are set here.
        argname_to_featurize = "X"
        # This is 1 and not 0 because we assume this is being used to annotate
        # a class member and not a generic function.
        argindex_to_featurize = 1

        if not callable(arg):
            n = len(args)
            if n >= 1:
                argname_to_featurize = args[0]
            elif 'argname_to_featurize' in kwargs:
                argname_to_featurize = kwargs['argname_to_featurize']

            if n >= 2:
                argindex_to_featurize = args[1]
            elif 'argindex_to_featurize' in kwargs:
                argindex_to_featurize = kwargs['argindex_to_featurize']
        # At this point, we have processed all possible invocations of the
        # annotation (the decorator) and we have the arguments to use.

        def decorated_(caller):
            '''Just the outside decorator that will pop the caller off the
            argument list.
            '''
            def decorated(*args, **kwargs):
                '''The main decorator function that handles featurization.'''
                #args[0] is the "self", the calling object.
                cfobject = args[0]
                assert isinstance(cfobject, CanFeaturize), \
                    "featurize_if_needed only decorates CanFeaturize methods"

                if not cfobject.featurizer:
                    # Cannot featurize if there is no featurizer
                    # This is also a potential efficiency option: do not set
                    # the featurizer if you want this decorator to early exit.
                    return caller(*args, **kwargs)

                needs_featurize = cfobject.hint_featurize

                # Here, have a featurizer and are not forced to featurize.
                data = None
                used_name = False
                used_index = False
                if argname_to_featurize in kwargs:
                    data = kwargs[argname_to_featurize]
                    used_name = True
                elif len(args) >= argindex_to_featurize:
                    data = args[argindex_to_featurize]
                    used_index = True
                else:
                    logger.warning('CanFeaturize: skipping test; unknown arg')

                if not needs_featurize and (used_name or used_index):
                    if isinstance(data, str):
                        if os.path.exists(data):
                            needs_featurize = True
                        else:
                            # If it is a string but not a file, it may be a
                            # video. Test that with our video library.
                            needs_featurize = etav.is_video(data)

                if needs_featurize:
                    data = cfobject.featurizer.featurize(data)
                    # Replace the call-structure before calling.
                    if used_name:
                        kwargs[argname_to_featurize] = data
                    if used_index:
                        targs = list(args)
                        targs[argindex_to_featurize] = data
                        args = tuple(targs)

                return caller(*args, **kwargs)
            return decorated

        # Be careful how to return; need to check the way we were invoked.
        # If arg is callable then we called it just with @featurize_if_needed.
        # Otherwise, we gave it parameters or even just parentheses.
        if callable(arg):
            return decorated_(arg)
        return decorated_

    def get_featurizer(self):
        return self.featurizer

    @property
    def has_featurizer(self):
        return bool(self.featurizer)

    def set_featurizer(self, featurizer):
        self.featurizer = featurizer


class FeaturizedFrameNotFoundError(OSError):
    '''Error for case when the featurized frame is not yet computed.'''
    pass


class VideoFramesFeaturizerConfig(Config):
    '''Specifies the configuration settings for the VideoFeaturizer class.'''

    def __init__(self, d):
        self.backing_path = self.parse_string(d, "backing_path",
                default="/tmp")
        self.backing_manager = self.parse_string(d, "backing_manager",
                default="random")
        self.backing_manager_remove_random = self.parse_bool(d,
                "backing_manager_remove_random", default=True)
        self.backing_manager_path_replace = self.parse_array(d,
                "backing_manager_path_replace", default=[])
        # This is any valid frames string for eta.
        self.frames = self.parse_string(d, "frames", default="*")
        # frame_featurizer is the sub-featurizer that does that work per frame
        self.frame_featurizer = self.parse_object(
                d, "frame_featurizer", FeaturizerConfig)


class VideoFramesFeaturizer(Featurizer):
    '''Class that encapsulates creating features from frames in a video and
    storing them to disk.

    Needs the following to be specified:
        backing_path: location on disk where featurized frames will be cached
            (this is a directory and not a printf like path)
        frame_featurizer: this is any image featurizer
    These are immutable quantities and are hence provided as a config.

    Frames are also part of the config, but these can be overridden at
    call-time.

    Featurized frames are store indexed by frame_number as compressed pickles.

    The backing_path stores the featurized frames to disk as a cache.  It uses
    `/tmp` as the default location for the backing store.  **WARNING** if you
    do not set the backing path per video and then use this featurizer for
    multiple videos, then your features will be invalid (they will just be
    computed on the first such video).  To help with this management, we
    provide an optional field `backing_manager` that has three possible
    settings (strings):
        manual: this is the fully manual setting and a use-at-your-own-risk
        random (default): this will create a new random location for the
            backing_path each time the featurize method is called.  By
            construction, this will force the featurization to be recomputed.
            The randomly created backing store will be removed after
            featurization, unless config field
            `backing_manager_remove_random` is set to false.
        replace: this uses the config field "backing_manager_path_replace"
            to replace substrings in the new input filename to be featurized
            and *hopefully* yield a new, unique output path.  The
            `eta.core.utils.replace_strings` function is used to actually
            perform the operation.

    Design note: a VideoFramesFeaturizer is a Featurizer that also has a
    Featurizer member, which processes each frame.  This frame featurizer gets
    instantiated and deleted on each start() and stop() call, so implement
    accordingly.  (Remember that featurizers can handle start and stop outside
    of a featurize loop.)

    @todo Should objects from this class are serializable.  If you initialize
    with the same video and the same config then it is essentially the same
    instance.  How to handle this well?

    @todo: can be generalized to not rely only on pickles (subclassed for
    pickles, eg).  I can imagine a featurizer might actually output an image.
    Think about this.

    '''

    def __init__(self, video_featurizer_config):
        '''Initializes the featurizer and creates the storage path.

        Implementing sub-classes need to explicitly call this :(
        super(SubClass, self).__init__(config)

        Member self.frame_preprocessor is a "function pointer" that takes in a
        frame and returns a preprocessed frame.  It is by default None and
        hence not call. You could, of course, chain together two video
        featurizers (once the class handles general output types) instead of
        this preprocessor function.  But, this is included her for speed (and
        it's what I needed initially).
        '''
        super(VideoFramesFeaturizer, self).__init__()

        self._frame_string = "%08d.npz"
        self.most_recent_frame = -1
        self._frame_preprocessor = None
        self.config = video_featurizer_config
        self._frame_featurizer = None
        self._backing_path = None

        self._backing_manager_lookup = {
                "manual": self._backing_manager_manual,
                "random": self._backing_manager_random,
                "replace": self._backing_manager_replace
            }

        self.update_backing_path(self.config.backing_path)
        self._backing_manager_random_last_tempdir = None


    @property
    def frame_preprocessor(self):
        '''Access to the frame_preprocessor attribute.'''
        return self._frame_preprocessor

    @frame_preprocessor.setter
    def frame_preprocessor(self, f):
        self._frame_preprocessor = f

    @frame_preprocessor.deleter
    def frame_preprocessor(self):
        self._frame_preprocessor = None

    def _backing_manager_manual(self, data, is_featurize_start=True):
        ''' Manual manager for the backing store on a new featurization call.
        '''
        pass

    def _backing_manager_random(self, data, is_featurize_start=True):
        ''' Manual manager for the backing store on a new featurization call.
        '''
        if is_featurize_start:
            td = tempfile.mkdtemp(dir="/tmp", prefix="eta.backing.")
            self._backing_manager_random_last_tempdir = td
            self.update_backing_path(td)
            return

        # not is_featurize_start  (->is_featurize_stop)
        if self.config.backing_manager_remove_random:
            shutil.rmtree(self._backing_manager_random_last_tempdir)
        self.update_backing_path(self.config.backing_path)

    def _backing_manager_replace(self, data, is_featurize_start=True):
        ''' Manual manager for the backing store on a new featurization call.
        '''
        if is_featurize_start:
            rp = etau.replace_strings(data,
                    self.config.backing_manager_path_replace)
            self._backing_manager_random_last_tempdir = rp
            self.update_backing_path(rp)
            return

        # not is_featurize_start  (->is_featurize_stop)
        self.update_backing_path(self.config.backing_path)

    def dim(self):
        ''' Return the dim of the underlying frame featurizer. '''
        if not self._frame_featurizer:
            self._frame_featurizer = self.config.frame_featurizer.build()
            d = self._frame_featurizer.dim()
            self._frame_featurizer = None
        else:
            d = self._frame_featurizer.dim()
        return d

    def is_featurized(self, frame_number):
        ''' Checks the backing store to determine whether or not the frame
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
            raise FeaturizedFrameNotFoundError()

        f = np.load(p)
        return f["v"]

    def featurize(self, data, frames=None, returnX=True):
        '''The core feature extraction routine to be called by users of the
        featurizer.  It appropriately interacts with the featurizer state
        management.

        Need to overload the Featurizer.featurize because we want to add some
        extra arguments.  It will still work as default if no arguments are
        expected.

        This is the place that we manage the `backing_path` based on the
        setting for the `backing_manager`
        '''

        if not frames:
            frames = self.config.frames

        backing_manager = \
            self._backing_manager_lookup[self.config.backing_manager]
        backing_manager(data)

        self.start(warn_on_restart=False, keep_alive=False)
        v = self._featurize(data, frames, returnX)
        if self._keep_alive is False:
            self.stop()

        backing_manager(data, False)

        return v

    def _featurize(self, data, frames=None, returnX=True):
        '''Main general use driver.  Works through a frame range to retrieve a
        matrix X of the features.

        data is the video *path*

        X.shape is [n_frames,featurized_length]

        Will do the featurization if needed.

        '''

        if not frames:
            frames = self.config.frames

        logger.debug("Featurizing frames %s" % frames)

        if returnX:
            X = None

        with etav.VideoProcessor(data, frames) as p:
            for img in p:
                self.most_recent_frame = p.frame_number
                f = self.featurized_frame_path(p.frame_number)

                try:
                    v = self.retrieve_featurized_frame(p.frame_number)
                except FeaturizedFrameNotFoundError:
                    logger.debug("Featurizing frame %d" % p.frame_number)
                    if not self._frame_featurizer:
                        self._frame_featurizer = \
                            self.config.frame_featurizer.build()
                        self._frame_featurizer.start()

                    # The frame is not yet cached, so compute and cache it.
                    if self._frame_preprocessor is not None:
                        v = self._frame_featurizer.featurize(
                            self._frame_preprocessor(img))
                    else:
                        v = self._frame_featurizer.featurize(img)
                    np.savez_compressed(f, v=v)
                finally:
                    if returnX:
                        if X is None:
                            X = GrowableArray(len(v))
                        X.update(v)

        # only stop and kill the frame featurizer if we are not keep_alive
        if self._frame_featurizer and not self._keep_alive:
            self._frame_featurizer.stop()
            self._frame_featurizer = None

        if returnX:
            return X.finalize()
        return None

    def featurized_frame_path(self, frame_number):
        '''Returns the backing path for the given frame number.'''
        return os.path.join(
            self._backing_path, self._frame_string % frame_number)


    def featurize_to_disk(self, data, frames="*"):
        '''Featurize all frames in the specified range by calling
        featurize_frame() for each frame and storing results directly to disk.
        Does not keep the results in memory.
        '''
        self.start(warn_on_restart=False, keep_alive=False)
        v = self._featurize(data, frames, returnX=False)
        if self._keep_alive is False:
            self.stop()
        return v

    def flush_backing(self):
        '''CAUTION! Deletes all featurized files on disk but not the
        directory.
        '''
        filelist = [
            f for f in os.listdir(self._backing_path)
            if f.endswith(".npz")
        ]
        for f in filelist:
            os.remove(os.path.join(self._backing_path, f))

    def _stop(self):
        ''' Internal stop that handles the removal of an active
        _frame_featurizer.
        '''
        if self._frame_featurizer:
            self._frame_featurizer.stop()
            self._frame_featurizer = None

    def update_backing_path(self, backing_path):
        '''Update the backing path and create the directory tree if needed.'''
        self._backing_path = backing_path
        try:
            os.makedirs(self._backing_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
