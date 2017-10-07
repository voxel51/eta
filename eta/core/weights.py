'''
ETA:  weights.py

Encapsulates an npz file that stores model weights locally
 in a cache and has the ability to pull a version of the 
 weights down from the net.

@todo: this should probably be (1) extended to allow for
       methods to visualized, display, debug weights and (2)
       refactored out so the actual backing store code 
       can be generally used.

Copyright 2017, Voxel51, LLC
voxel51.com

Jason Corso, jjc@voxel51.com
'''

import os

import numpy as np

from core.config import Config
import utils as ut


class WeightsConfig(Config):
    ''' Weights Config

        @todo: the weights_path should be relative to some global path that
               stores cached big-files.
    '''
    def __init__(self, d):
        self.weights_cache = self.parse_string(d,"weights_cache",default=None)
        if self.weights_cache is None:
            cdir = os.path.dirname(os.path.realpath(__file__)))
            cnam = 'cache'
            self.weights_cache = os.path.join(cdir,cnam)
        ut.ensure_dir(self.weights_cache)
        self.weights_filename = self.parse_string(d, "weights_filename")
        self.weights_path = os.path.join(self.weights_cache,
                                         self.weights_filename)
        self.weights_url = self.parse_string(d, "weights_url", default=None)
        self.weights_large_google_drive_file_flag = self.parse_bool(d,
                "weights_large_google_drive_file_flag",default=False)

class Weights(Config):
    ''' Weights class that encapsulates model weights and can
        load them from the net if needed (and if paths are provided.

        @todo: Would be great to make this class act like the actual
               dictionary it loads, by overloading/implementing the same
               methods.
    '''
    def __init__(self, d):
        self.config = d
        self.data   = None

        # check if the file is locally stored
        if not os.path.isfile(self.config.weights.weights_path):
            if self.config.weights_large_google_drive_file_flag:
                b = ut.download_large_google_drive_file(self.weights_url)
            else
                b = ut.download_file(self.weights_url)

            with open(self.config.weights_path,'wb') as f:
                f.write(b)

        # can this be ingested directly from 'b' if we downloaded it? 
        self.data = np.load(self.config.weights_path)



