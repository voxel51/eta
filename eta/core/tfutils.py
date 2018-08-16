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
from future.utils import iteritems
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import copy
import logging

import tensorflow as tf

import eta
import eta.core.models as etam


logger = logging.getLogger(__name__)


def make_tf_session(config_proto=None):
    '''Makes a new tf.Session that inherits any config settings from the global
    `eta.config.tf_config`.

    Args:
        config_proto: an optional tf.ConfigProto from which to initialize the
            session config. By default, tf.ConfigProto() is used

    Returns:
        a tf.Session
    '''
    config = copy.copy(config_proto) if config_proto else tf.ConfigProto()
    # Apply ETA config settings
    _set_proto_fields(config, eta.config.tf_config)
    return tf.Session(config=config)


def _set_proto_fields(proto, d):
    def _split_field(field):
        chunks = field.split(".", 1)
        return tuple(chunks) if len(chunks) == 2 else (chunks[0], None)

    for field, val in iteritems(d):
        field, sub = _split_field(field)
        try:
            if sub:
                _set_proto_fields(getattr(proto, field), {sub: val})
            elif isinstance(val, dict):
                _set_proto_fields(getattr(proto, field), val)
            else:
                setattr(proto, field, val)
        except AttributeError as e:
            logger.warning(str(e))


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
