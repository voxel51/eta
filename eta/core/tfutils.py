'''
Core utilities for working with TensorFlow models.

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

import tensorflow as tf

import eta.core.models as etam


class TensorFlowModelCheckpoint(etam.PublishedModel):
    '''Class that can load a published TensorFlow model checkpoint stored as a
    .model file.
    '''

    def __init__(self, model_name, sess):
        '''Initializes a TensorFlowModelCheckpoint instance.

        Args:
            model_name: the model to load
            sess: the tf.Session in which to load the checkpoint

        Raises:
            ModelError: if the model was not found
        '''
        super(TensorFlowModelCheckpoint, self).__init__(model_name)
        self._sess = sess

    def _load(self):
        saver = tf.train.Saver()
        saver.restore(self._sess, self.model_path)
