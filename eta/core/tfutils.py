'''
Core utilities for working with TensorFlow models.

Copyright 2018, Voxel51, Inc.
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
import os
import re

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
    config = make_tf_config(config_proto=config_proto)
    return tf.Session(config=config)


def make_tf_config(config_proto=None):
    '''Makes a new tf.ConfigProto that inherits any config settings from the
    global `eta.config.tf_config`.

    Args:
        config_proto: an optional tf.ConfigProto from which to initialize the
            config. By default, tf.ConfigProto() is used

    Returns:
        a tf.ConfigProto
    '''
    config = copy.copy(config_proto) if config_proto else tf.ConfigProto()

    if eta.config.tf_config:
        logger.debug(
            "Applying eta.tf_config settings: %s", str(eta.config.tf_config))
        _set_proto_fields(config, eta.config.tf_config)

    return config


def is_valid_tf_record_path(tf_record_path):
    '''Determines whether the provided tf.Record path is a valid path.

    Valid paths must either end in `.record` or `.tfrecord` or describe
        a sharded tf.Record.
    '''
    ext = os.path.splitext(tf_record_path)[1]
    return (ext == ".record"
            or ext == ".tfrecord"
            or is_sharded_tf_record_path(tf_record_path)


def is_sharded_tf_record_path(tf_record_path):
    '''Determines whether the given path is a sharded tf.Record path like
    "/path/to/tf.record-????-of-1000" OR
    "/path/to/<dataset_name>_<split_name>-?????-of-XXXXX.tfrecord".
    '''
    ext = os.path.splitext(tf_record_path)[1]
    filename = os.path.basename(tf_record_path)
    return re.match("\.record-\?+-of-\d+", ext) is not None or re.match(
            "\w+\w+\d+-of-\d+\.tfrecord", filename) is not None


def make_sharded_tf_record_path(base_path, num_shards):
    '''Makes a sharded tf.Record path with the given number of shards.

    Args:
        base_path: a path like "/path/to/tf.record"
        num_shards: the desired number of shards

    Returns:
        a sharded path like "/path/to/tf.record-????-of-1000"
    '''
    num_shards_str = str(num_shards)
    num_digits = len(num_shards_str)
    return base_path + "-" + "?" * num_digits + "-of-" + num_shards_str


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
