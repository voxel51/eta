'''
Core utilities for working with TensorFlow models.

Copyright 2017-2020, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
Yash Bhalgat, yash@voxel51.com
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

from copy import copy
import logging
import os
import re

import numpy as np
import tensorflow as tf

import eta
import eta.core.image as etai
import eta.core.models as etam
import eta.core.utils as etau


logger = logging.getLogger(__name__)


TF_RECORD_EXTENSIONS = [".record", ".tfrecord"]


def is_gpu_available():
    '''Determines whether TensorFlow can access a GPU.

    Returns:
        True/False
    '''
    return tf.test.is_gpu_available()


def load_graph(model_path, prefix=""):
    '''Loads the TF graph from the given `.pb` file.

    Args:
        model_path: the `.pb` file to load
        prefix: an optional prefix to prepend when importing the graph

    Returns:
        the loaded `tf.Graph`
    '''
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name=prefix)

    return graph


def export_frozen_graph(model_dir, output_node_names):
    '''Exports the TF model defined by the given model directory as a
    `frozen_model.pb` file in that directory.

    Only the subgraph defined by the specified output nodes is saved.

    Args:
        model_dir: the model directory, which will be passed to
            `tf.train.get_checkpoint_state()`
        output_node_names: a list of output node names to export

    Returns:
        the path to the frozen graph on disk
    '''
    if not os.path.isdir(model_dir):
        raise OSError(
            "Model directory '%s' does not exist" % model_dir)

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    frozen_model_name = "frozen_model.pb"

    with tf.Session(graph=tf.Graph()) as sess:
        # Import the meta graph
        saver = tf.train.import_meta_graph(
            input_checkpoint + ".meta", clear_devices=True)

        # Restore the weights
        saver.restore(sess, input_checkpoint)

        # Fix batch norm nodes
        # https://github.com/tensorflow/tensorflow/issues/3628#issuecomment-272149052
        gd = sess.graph.as_graph_def()
        for node in gd.node:
            if node.op == "RefSwitch":
                node.op = "Switch"
                for index in range(len(node.input)):
                    if "moving_" in node.input[index]:
                        node.input[index] = node.input[index] + "/read"
            elif node.op == "AssignSub":
                node.op = "Sub"
                if "use_locking" in node.attr:
                    del node.attr["use_locking"]

        # Export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, gd, output_node_names)

        # Write the frozen graph
        tf.train.write_graph(
            output_graph_def, model_dir, frozen_model_name, as_text=False)

        logger.info(
            "Exported %d ops in the frozen graph.", len(output_graph_def.node))

    output_path = os.path.join(model_dir, frozen_model_name)

    return output_path


def visualize_checkpoint(model_dir, log_dir=None, port=None):
    '''Visualizes the TF checkpoint in the given directory via TensorBoard.

    Specifically, this script performs the following actions:
        - loads the checkpoint via `tf.train.get_checkpoint_state()`
        - populates a TensorBoard log directory for the graph
        - launches a TensorBoard instance to visualize the graph

    Args:
        model_dir: the TF model directory
        log_dir: an optional log directory in which to write the TensorBoard
            files. By default, a temp directory is created
        port: an optional port on which to launch TensorBoard
    '''
    if not os.path.isdir(model_dir):
        raise OSError(
            "Model directory '%s' does not exist" % model_dir)

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    graph = tf.Graph()
    with graph.as_default():
        tf.train.import_meta_graph(
            input_checkpoint + ".meta", clear_devices=True)
        _launch_tensorboard(graph, log_dir=log_dir, port=port)


def visualize_frozen_graph(model_path, log_dir=None, port=None):
    '''Visualizes the TF graph from the given `.pb` file via TensorBoard.

    Specifically, this script performs the following actions:
        - loads the graph
        - populates a TensorBoard log directory for the graph
        - launches a TensorBoard instance to visualize the graph

    Args:
        model_path: the `.pb` file to load
        log_dir: an optional log directory in which to write the TensorBoard
            files. By default, a temp directory is created
        port: an optional port on which to launch TensorBoard
    '''
    graph = load_graph(model_path, prefix="import")
    _launch_tensorboard(graph, log_dir=log_dir, port=port)


def _launch_tensorboard(graph, log_dir=None, port=None):
    if log_dir is None:
        log_dir = etau.make_temp_dir()

    writer = tf.summary.FileWriter(log_dir)
    writer.add_graph(graph)
    logger.info("Model imported to '%s'", log_dir)

    args = ["tensorboard", "--logdir", log_dir]
    if port:
        args.extend(["--port", str(port)])

    logger.info(
        "Launching TensorBoard via the command:\n    %s\n", " ".join(args))
    etau.call(args)


class TFModelCheckpoint(etam.PublishedModel):
    '''Class that can load a published TensorFlow model checkpoint stored as a
    .model file.
    '''

    def __init__(self, model_name, sess):
        '''Initializes a TFModelCheckpoint instance.

        Args:
            model_name: the model to load
            sess: the tf.Session in which to load the checkpoint

        Raises:
            ModelError: if the model was not found
        '''
        super(TFModelCheckpoint, self).__init__(model_name)
        self._sess = sess

    def _load(self):
        saver = tf.train.Saver()
        saver.restore(self._sess, self.model_path)


class UsesTFSession(object):
    '''Mixin for classes that use one or more `tf.Session`s.

    It is highly recommended that all classes that use `tf.Session` use this
    mixin to ensure that:

        - all sessions have the settings from `eta.config.tf_config` applied
            to them

        - all sessions are appropriately closed

    To use this mixin, simply call `make_tf_session()` when you need to create
    a new session, and then either use the context manager interface or call
    the `close()` method to automatically clean up any sessions your instance
    was using.
    '''

    def __init__(self):
        self._sess_list = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def make_tf_session(self, config_proto=None, graph=None):
        '''Makes a new tf.Session that inherits any config settings from the
        global `eta.config.tf_config`.

        A reference to the session is stored internally so that it can be
        closed when `close()` is called.

        Args:
            config_proto: an optional tf.ConfigProto from which to initialize
                the session config. By default, tf.ConfigProto() is used
            graph: an optional tf.Graph to launch. If omitted, the default
                graph is used

        Returns:
            the tf.Session
        '''
        sess = make_tf_session(config_proto=config_proto, graph=graph)
        self._sess_list.append(sess)
        return sess

    def close(self):
        '''Closes any TensorFlow session(s) in use by this instance.'''
        for sess in self._sess_list:
            if sess:
                sess.close()

        self._sess_list = []


def make_tf_session(config_proto=None, graph=None):
    '''Makes a new tf.Session that inherits any config settings from the global
    `eta.config.tf_config`.

    Args:
        config_proto: an optional tf.ConfigProto from which to initialize the
            session config. By default, tf.ConfigProto() is used
        graph: an optional tf.Graph to launch. If omitted, the default graph
            is used

    Returns:
        the tf.Session
    '''
    config = make_tf_config(config_proto=config_proto)
    return tf.Session(config=config, graph=graph)


def make_tf_config(config_proto=None):
    '''Makes a new tf.ConfigProto that inherits any config settings from the
    global `eta.config.tf_config`.

    Args:
        config_proto: an optional tf.ConfigProto from which to initialize the
            config. By default, tf.ConfigProto() is used

    Returns:
        a tf.ConfigProto
    '''
    config = copy(config_proto) if config_proto else tf.ConfigProto()

    if eta.config.tf_config:
        tf_config = eta.config.tf_config
        if not is_gpu_available():
            # Remove GPU options, just for clarity
            tf_config = {
                k: v for k, v in iteritems(tf_config)
                if not k.startswith("gpu_options.")
            }

        if tf_config:
            logger.info(
                "Applying `tf_config` settings from ETA config: %s",
                str(tf_config))
            _set_proto_fields(config, tf_config)

    return config


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


def is_valid_tf_record_path(tf_record_path):
    '''Determines whether the provided tf.Record path is a valid path.

    Valid paths must either end in `.record` or `.tfrecord` or describe
        a sharded tf.Record.
    '''
    ext = os.path.splitext(tf_record_path)[1]
    return (
        ext in TF_RECORD_EXTENSIONS or
        is_sharded_tf_record_path(tf_record_path))


def is_sharded_tf_record_path(tf_record_path):
    '''Determines whether the given path is a sharded tf.Record path like
    "/path/to/data.record-????-of-XXXXX" or
    "/path/to/data-?????-of-XXXXX.tfrecord"
    '''
    ext_patt = "|".join(re.escape(e) for e in TF_RECORD_EXTENSIONS)
    shard_patt = "-\?+-of-\d+"

    # /path/to/data.record-????-of-XXXXX
    if re.search("(%s)%s$" % (ext_patt, shard_patt), tf_record_path):
        return True

    # /path/to/data-????-of-XXXXX.tfrecord
    if re.search("%s(%s)$" % (shard_patt, ext_patt), tf_record_path):
        return True

    return False


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


# Mean of ImageNet training set
IMG_MEAN_IMAGENET = np.array(
    (123.68, 116.779, 103.939), dtype=np.float32).reshape(1, 1, 3)

# Mean of PASCAL VOC training set
IMG_MEAN_VOC = np.array(
    (104.00698793, 116.66876762, 122.67891434),
    dtype=np.float32).reshape(1, 1, 3)


def float_preprocessing_numpy(imgs):
    '''Preprocesses the images by scaling them to [-1, 1] in float32 format.

    Args:
        imgs: a list of images

    Returns:
        the preprocessed images, in float32 format
    '''
    imgs_out = []
    for img in imgs:
        if etai.is_gray(img):
            img = etai.gray_to_rgb(img)
        elif etai.has_alpha(img):
            img = img[:, :, :3]

        img = 2.0 * (etai.to_float(img) - 0.5)
        imgs_out.append(img)

    return imgs_out


def inception_preprocessing_numpy(imgs, height, width):
    '''Performs Inception-style preprocessing of images using numpy.

    Specifically, the images are resized and then scaled to [-1, 1] in float32
    format.

    Args:
        imgs: a list of images
        height: desired image height after preprocessing
        width: desired image width after preprocessing

    Returns:
        the preprocessed images, in float32 format
    '''
    imgs_out = []
    for img in imgs:
        if etai.is_gray(img):
            img = etai.gray_to_rgb(img)
        elif etai.has_alpha(img):
            img = img[:, :, :3]

        img = etai.resize(img, height, width)
        img = 2.0 * (etai.to_float(img) - 0.5)
        imgs_out.append(img)

    return imgs_out


def voc_preprocessing_numpy(imgs):
    '''Performs PASCAL VOC-style preprocessing of images using numpy.

    Specifically, the images are centered by `IMG_MEAN_VOC` and returned in
    float32 format.

    Args:
        imgs: a list of images

    Returns:
        the preprocessed images, in float32 format
    '''
    imgs_out = []
    for img in imgs:
        if etai.is_gray(img):
            img = etai.gray_to_rgb(img)
        elif etai.has_alpha(img):
            img = img[:, :, :3]

        img = np.asarray(img, dtype=np.float32)
        img -= IMG_MEAN_VOC
        imgs_out.append(img)

    return imgs_out


def vgg_preprocessing_numpy(imgs, height, width):
    '''Performs VGG-style preprocessing of images using numpy.

    Specifically, the images are resized (aspect-preserving) to the desired
    size and then centered by `IMG_MEAN_IMAGENET`

    Args:
        imgs: a list of images
        height: desired height after preprocessing
        width: desired width after preprocessing

    Returns:
        the preprocessed images, in float32 format
    '''
    imgs_out = []
    for img in imgs:
        if etai.is_gray(img):
            img = etai.gray_to_rgb(img)
        elif etai.has_alpha(img):
            img = img[:, :, :3]

        # Aspect preserving resize
        if img.shape[0] < img.shape[1]:
            img = etai.resize(img, height=256)
        else:
            img = etai.resize(img, width=256)

        img = etai.central_crop(img, shape=(height, width))
        img = np.asarray(img, dtype=np.float32) - IMG_MEAN_IMAGENET
        imgs_out.append(img)

    return imgs_out
