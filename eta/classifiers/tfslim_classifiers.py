'''
Interface to the TF-Slim image classification library available at
https://github.com/tensorflow/models/tree/master/research/slim.

Copyright 2017-2019 Voxel51, Inc.
voxel51.com

Yash Bhalgat, yash@voxel51.com
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

import logging
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

import eta.constants as etac
from eta.core.config import Config, ConfigError
import eta.core.data as etad
import eta.core.learning as etal
import eta.core.models as etam
import eta.core.tfutils as etat
import eta.core.utils as etau

sys.path.insert(1, etac.TF_SLIM_DIR)
from preprocessing import preprocessing_factory
from nets import nets_factory


logger = logging.getLogger(__name__)


# Networks for which we provide default `output_name`s
_DEFAULT_OUTPUT_NAMES = {
    "resnet_v1_50": "resnet_v1_50/predictions/Reshape_1",
    "resnet_v2_50": "resnet_v2_50/predictions/Reshape_1",
    "mobilenet_v1_025": "MobilenetV1/Predictions/Reshape_1",
    "mobilenet_v2": "MobilenetV2/Predictions/Reshape_1",
    "inception_v3": "InceptionV3/Predictions/Reshape_1",
    "inception_v4": "InceptionV4/Logits/Predictions",
    "inception_resnet_v2": "InceptionResnetV2/Logits/Predictions"
}

# Networks for which we provide pre-processing implemented in numpy
_NUMPY_PREPROC_FUNCTIONS = {
    "resnet_v1_50": etat.vgg_preprocessing_numpy,
    "resnet_v2_50": etat.inception_preprocessing_numpy,
    "mobilenet_v2": etat.inception_preprocessing_numpy,
    "inception_v3": etat.inception_preprocessing_numpy,
    "inception_v4": etat.inception_preprocessing_numpy,
    "inception_resnet_v2": etat.inception_preprocessing_numpy,
}


class TFSlimClassifierConfig(Config, etal.HasDefaultDeploymentConfig):
    '''Configuration class for loading a TensorFlow classifier whose network
    architecture is defined in `tf.slim.nets`.

    Note that `labels_path` is passed through
    `eta.core.utils.fill_config_patterns` at load time, so it can contain
    patterns to be resolved.

    Note that this class implements the `HasDefaultDeploymentConfig` mixin, so
    if a published model is provided via the `model_name` attribute, then any
    omitted fields present in the default deployment config for the published
    model will be automatically populated.

    Attributes:
        model_name: the name of the published model to load. If this value is
            provided, `model_path` does not need to be
        model_path: the path to a TF SavedModel in `.pb` format to load. If
            this value is provided, `model_name` does not need to be
        attr_name: the name of the attribute that the classifier predicts
        network_name: the name of the network architecture from
            `tf.slim.nets.nets_factory`
        labels_path: the path to the labels map for the classifier
        preprocessing_fcn: the fully-qualified name of a pre-processing
            function to use. If omitted, the default pre-processing for the
            specified network architecture is used
        input_name: the name of the graph node to use as input. If omitted,
            the name "input" is used
        output_name: the name of the graph node to use as output. If omitted,
            the `_DEFAULT_OUTPUT_NAMES` is checked for a default value to use
    '''

    def __init__(self, d):
        self.model_name = self.parse_string(d, "model_name", default=None)
        self.model_path = self.parse_string(d, "model_path", default=None)

        # Loads any default deployment parameters, if possible
        if self.model_name:
            d = self.load_default_deployment_params(d, self.model_name)

        self.attr_name = self.parse_string(d, "attr_name")
        self.network_name = self.parse_string(d, "network_name")
        self.labels_path = etau.fill_config_patterns(
            self.parse_string(d, "labels_path"))
        self.preprocessing_fcn = self.parse_string(
            d, "preprocessing_fcn", default=None)
        self.input_name = self.parse_string(d, "input_name", default="input")
        self.output_name = self.parse_string(d, "output_name", default=None)

        self._validate()

    def _validate(self):
        if not self.model_name and not self.model_path:
            raise ConfigError(
                "Either `model_name` or `model_path` must be provided")


class TFSlimClassifier(etal.ImageClassifier, etat.UsesTFSession):
    '''Interface for the TF-Slim image classification library at
    https://github.com/tensorflow/models/tree/master/research/slim.

    This class uses `eta.core.tfutils.UsesTFSession` to create TF sessions, so
    it automatically applies settings in your `eta.config.tf_config`.

    Instances of this class must either use the context manager interface or
    manually call `close()` when finished to release memory.
    '''

    def __init__(self, config):
        '''Creates a TFSlimClassifier instance.

        Args:
            config: a TFSlimClassifierConfig instance
        '''
        self.config = config
        etat.UsesTFSession.__init__(self)

        if self.config.model_name:
            # Downloads the published model, if necessary
            model_path = etam.download_model(self.config.model_name)
        else:
            model_path = self.config.model_path

        # Load graph
        self._graph = self._build_graph(model_path)
        self._sess = self.make_tf_session(graph=self._graph)

        # Load labels map
        self.labels_map = etal.load_labels_map(self.config.labels_path)

        # Get network
        network_name = self.config.network_name
        network_fn = nets_factory.get_network_fn(
            network_name, num_classes=len(self.labels_map), is_training=False)
        self.img_size = network_fn.default_image_size

        # Parse input name
        self.input_name = self.config.input_name
        self._input_op = self._graph.get_operation_by_name(
            "prefix/" + self.input_name)

        # Parse output name
        if self.config.output_name:
            self.output_name = self.config.output_name
        else:
            self.output_name = _DEFAULT_OUTPUT_NAMES.get(network_name, None)
            if self.output_name is None:
                raise ValueError(
                    "`output_name` was not provided and network `%s` was not "
                    "found in default outputs map" % network_name)
        self._output_op = self._graph.get_operation_by_name(
            "prefix/" + self.output_name)

        # Setup pre-processing
        self._preprocessing_fcn = None
        self._preprocessing_sess = None
        self.preprocessing_fcn = self._make_preprocessing_fcn(
            network_name, self.config.preprocessing_fcn)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def predict(self, img):
        '''Peforms prediction on the given image.

        Args:
            img: the image to classify

        Returns:
            an `eta.core.data.AttributeContainer` instance containing the
                predictions
        '''
        # Perform pre-processing
        network_inputs = self.preprocessing_fcn([img])

        probs = self._sess.run(
            self._output_op.outputs[0],
            {self._input_op.outputs[0]: network_inputs})

        probs = probs[0, :]
        idx = np.argmax(probs)
        label = self.labels_map[idx]
        confidence = probs[idx]

        return self._package_attr(label, confidence)

    @staticmethod
    def _build_graph(model_path):
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="prefix")
        return graph

    def _make_preprocessing_fcn(self, network_name, preprocessing_fcn):
        # Use user-specified pre-processing, if provided
        if preprocessing_fcn:
            logger.info(
                "Using user-provided pre-processing function '%s'",
                preprocessing_fcn)
            preproc_fcn_user = etau.get_function(preprocessing_fcn)
            return lambda imgs: preproc_fcn_user(
                imgs, self.img_size, self.img_size)

        # Use numpy-based pre-processing if supported
        preproc_fcn_np = _NUMPY_PREPROC_FUNCTIONS.get(network_name, None)
        if preproc_fcn_np is not None:
            logger.info(
                "Found numpy-based pre-processing implementation for network "
                "'%s'", network_name)
            return lambda imgs: preproc_fcn_np(
                imgs, self.img_size, self.img_size)

        # Fallback to TF-slim pre-processing
        logger.info(
            "Using TF-based pre-processing from pre-processing_factory for "
            "network '%s'", network_name)
        self._preprocessing_fcn = preprocessing_factory.get_preprocessing(
            network_name, is_training=False)
        self._preprocessing_sess = self.make_tf_session()

        return self._builtin_preprocessing_tf

    def _builtin_preprocessing_tf(self, imgs):
        _imgs = tf.placeholder("uint8", [None, None, 3])
        _imgs_proc = tf.expand_dims(
            self._preprocessing_fcn(_imgs, self.img_size, self.img_size), 0)

        imgs_out = []
        for img in imgs:
            imgs_out.append(
                self._preprocessing_sess.run(
                    _imgs_proc, feed_dict={_imgs: img}))

        return imgs_out

    def _package_attr(self, label, confidence):
        attrs = etad.AttributeContainer()
        attr = etad.CategoricalAttribute(
            self.config.attr_name, label, confidence=confidence)
        attrs.add(attr)
        return attrs


def export_frozen_inference_graph(
        checkpoint_path, network_name, output_path, num_classes=None,
        labels_map_path=None, output_node=None):
    '''Exports the given TF-Slim model checkpoint as a frozen inference graph
    suitable for running inference.

    Either `num_classes` or `labels_map_path` must be provided.

    Args:
        checkpoint_path: path to the training checkpoint to export
        network_name: the name of the network architecture from
            `tf.slim.nets.nets_factory`
        output_path: the path to write the frozen graph `.pb` file
        num_classes: the number of output classes for the model. If specified,
            `labels_map_path` is ignored
        labels_map_path: the path to the labels map for the classifier; used to
            determine the number of output classes. Must be provided if
            `num_classes` is not provided
        output_node: the name of the output node from which to extract
            predictions. By default, this value is loaded from
            `_DEFAULT_OUTPUT_NAMES`
    '''
    if num_classes is None:
        if labels_map_path is None:
            raise ValueError(
                "Must provide a `labels_map_path` when `num_classes` is not "
                "specified")
        else:
            num_classes = len(etal.load_labels_map(labels_map_path))

    output_node = _DEFAULT_OUTPUT_NAMES.get(network_name, None)
    if output_node is None:
        raise ValueError(
            "No 'output_node' manually provided and no default output found " +
            "for network '%s'" % network_name)

    with tf.Graph().as_default() as graph:
        graph_def = _get_graph_def(graph, network_name, num_classes)
        freeze_graph.freeze_graph_with_def_protos(
            graph_def, None, checkpoint_path, output_node, None, None,
            output_path, True, "")


def _get_graph_def(graph, network_name, num_classes):
    # Adapted from `tensorflow/models/research/slim/export_inference_graph.py`
    network_fn = nets_factory.get_network_fn(
        network_name, num_classes=num_classes, is_training=False)
    img_size = network_fn.default_image_size
    input_shape = [1, img_size, img_size, 3]
    placeholder = tf.placeholder(
        name="input", dtype=tf.float32, shape=input_shape)
    network_fn(placeholder)

    return graph.as_graph_def()
