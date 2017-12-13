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
import importlib
import os

import numpy as np

from eta.core.config import Config, Configurable
from eta.core.numutils import GrowableArray
import eta.core.video as etav


class FeaturizerConfig(Config):
    '''Featurizer configuration settings.

    d['type'] is the class name for the object.  This is somewhat flexible.  If
    it is a dotted-moduled-path like (foo.bar.foobar) then we handle parsing it
    and loading foo.bar and invoking foobar when needed.  Otherwise, if it is
    just a class name foobar, then we assume it is defined in this module.
    '''

    def __init__(self, d):
        self.type = self.parse_string(d, "type")

        tlookup = self.type.rsplit('.', 1)
        if len(tlookup) == 1:
            self._featurizer_cls, config_cls = Configurable.parse(self.type,
                                                                  __name__)
        else:
            mname = tlookup[0]
            cname = tlookup[1]
            importlib.import_module(mname)
            self._featurizer_cls, config_cls = Configurable.parse(cname, mname)
        self.config = self.parse_object(d, "config", config_cls)

    def build(self):
        '''Factory method to build the featurizer instance by the class
        name.'''
        return self._featurizer_cls(self.config)


class Featurizer(Configurable):
    '''Interface for feature extraction methods.'''

    def dim(self):
        '''Return the dimension of the features extracted by this method.'''
        raise NotImplementedError("subclass must implement dim().")

    def featurize_start(self):
        '''Called by featurize before it starts in case any environment needs
        to be set up by subclasses.
        '''
        raise NotImplementedError("subclass must implement featurize_start().")

    def featurize_end(self):
        '''Called by featurize after it end in case any environment needs to be
        cleaned up by subclasses.
        '''
        raise NotImplementedError("subclass must implement featurize_end().")

    def featurize(self, data):
        '''The core feature extraction routine.'''
        raise NotImplementedError("subclass must implement featurize().")


class FeaturizedFrameNotFoundError(OSError):
    '''Error for case when the featurized frame is not yet computed.'''
    pass


class VideoFeaturizerConfig(Config):
    '''Specifies the configuration settings for the VideoFeaturizer class.'''

    def __init__(self, d):
        self.backing_path = self.parse_string(d, "backing_path")
        self.video_path = self.parse_string(d, "video_path")


class VideoFeaturizer(Featurizer):
    '''Class that encapsulates creating features from frames in a video and
    storing them to disk.

    Needs the following to be specified:
        backingPath: location on disk where featurized frames will be cached
            (this is a directory and not a printf like path)
        videoPath: location of the input video (suitable as input to
            VideoProcessor)
    These are immutable quantities and are hence provided as a config.

    Frames are not part of the config as a new VideoProcessor can be created
    with a different frame range depending on execution needs.

    Featurized frames are store indexed by frame_number as compressed pickles.

    @todo Should objects from this class are serializable.  If you initialize
    with the same video and the same config then it is essentially the same
    instance.  How to handle this well?

    @todo: can be generalized to not rely only on pickles (subclassed for
    pickles, eg).  I can imagine a featurizer might actually output an image.
    Think about this.

    @todo: REFACTOR a videofeaturizer is a featurizer that works on videos but
    also has a featurizer that does the actual work.  This will reduce total
    LOC.  See the vgg16.py where we have two VGG16 featurizers, but really
    there is only one bit of functionality.
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

        self._frame_string = "%08d.npz"
        self.most_recent_frame = -1
        self._frame_preprocessor = None
        self.config = video_featurizer_config

        try:
            os.makedirs(self.config.backing_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

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

    # def serialize(self):

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

    def featurize(self, frames="*"):
        '''Main general use driver.  Works through a frame range to retrieve a
        matrix X of the features.

        X.shape is [n_frames,featurized_length]

        Will do the featurization if needed.

        '''
        has_started = False

        X = None

        with etav.VideoProcessor(self.config.video_path, frames) as p:
            for img in p:
                self.most_recent_frame = p.frame_number
                f = self.featurized_frame_path(p.frame_number)

                try:
                    v = self.retrieve_featurized_frame(p.frame_number)
                except FeaturizedFrameNotFoundError:
                    if not has_started:
                        has_started = True
                        self.featurize_start()

                    # The frame is not yet cached, so compute and cache it.
                    if self._frame_preprocessor is not None:
                        v = self.featurize_frame(self._frame_preprocessor(img))
                    else:
                        v = self.featurize_frame(img)
                    np.savez_compressed(f, v=v)
                finally:
                    if X is None:
                        X = GrowableArray(len(v))
                    X.update(v)

        if has_started:
            self.featurize_end()

        return X.finalize()

    def featurized_frame_path(self, frame_number):
        '''Returns the backing path for the given frame number.'''
        return os.path.join(
            self.config.backing_path, self._frame_string % frame_number)

    def featurize_frame(self, frame):
        '''The actual featurization. Should return a numpy vector.'''
        raise NotImplementedError("subclass must implement featurize_frame()")

    def featurize_start(self):
        '''Called by featurize before it starts in case any environment needs
        to be set up by subclasses.
        '''
        pass

    def featurize_end(self):
        '''Called by featurize after it end in case any environment needs to be
        cleaned up by subclasses.
        '''
        pass

    def featurize_to_disk(self, frames="*"):
        '''Featurize all frames in the specified range by calling
        featurize_frame() for each frame and storing results directly to disk.
        Does not keep the results in memory.

        Creates a VideoProcessor on the fly for this.

        Checks the backing store to see if the frame is already featurized.

        @todo Check for refactorization possibilities with retrieve() as there
        is some redundant code.
        '''
        self.featurize_start()

        with etav.VideoProcessor(self.config.video_path, frames) as p:
            for img in p:
                self.most_recent_frame = p.frame_number
                f = self.featurized_frame_path(p.frame_number)

                try:
                    self.retrieve_featurized_frame(p.frame_number)
                except FeaturizedFrameNotFoundError:
                    # The frame is not yet cached, so compute it.
                    if self._frame_preprocessor is not None:
                        v = self.featurize_frame(self._frame_preprocessor(img))
                    else:
                        v = self.featurize_frame(img)
                    np.savez_compressed(f, v=v)

        self.featurize_end()

    def flush_backing(self):
        '''CAUTION! Deletes all featurized files on disk but not the
        directory.
        '''
        filelist = [
            f for f in os.listdir(self.config.backing_path)
            if f.endswith(".npz")
        ]
        for f in filelist:
            os.remove(os.path.join(self.config.backing_path, f))
