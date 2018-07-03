'''
TensorFlow implementation of the popular C3D network.

This module extends and adds functionality to the original C3D implementation
by Hou Xin.

C3D implementation in TensorFlow:
https://github.com/hx173149/C3D-tensorflow
Hou Xin, 2016

Copyright 2018, Voxel51, LLC
voxel51.com

Yixin Jin, yixin@voxel51.com
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

import numpy as np
import tensorflow as tf

from eta.core.config import Config
from eta.core.features import Featurizer
from eta.core.tfutils import TensorFlowModelWeights
import eta.core.video as etav


class C3DConfig(Config):
    '''Configuration settings for the C3D network.'''

    def __init__(self, d):
        self.model = self.parse_string(d, "model", default="C3D-UCF101")


class C3D(object):
    '''TensorFlow implementation of the C3D network architecture for the
    101 classes from UCF101.

    This implementation is hard-coded to process an tensor of video clips of
    size [XXXX, 16, 112, 112, 3].
    '''

    def __init__(self, config=None, sess=None, clips=None):
        '''Builds a new C3D network

        Args:
            config: an optional C3DConfig instance. If omitted, the default
                ETA configuration will be used
            sess: an optional tf.Session to use. If none is provided, a new
                tf.Session instance is created, and you are responsible for
                scalling the close() method of this class when you are done
                computing
            clips: an optional tf.placeholder of size [XXXX, 16, 112, 112, 3]
        '''
        self.config = config or C3DConfig.default()
        self.sess = sess or tf.Session()
        self.clips = clips or tf.placeholder(
            tf.float32, [None, 16, 112, 112, 3])

        # The source (https://github.com/hx173149/C3D-tensorflow) of the models
        # we use picked this variable scope, so we must use it too
        with tf.variable_scope("var_name"):
            self._build_conv_layers()
            self._build_fc_layers()
            self._build_output_layer()

        self._load_model(self.config.model)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def evaluate(self, clips, layer=None):
        '''Feed-forward evaluation through the network.

        Args:
            clips: an array of size [XXXX, 16, 112, 112, 3] containing clips(s)
                to feed into the network
            layer: an optional layer whose output to return. By default, the
                output softmax layer (i.e., the class probabilities) is
                returned

        Returns:
            an array of same size as the requested layer. The first dimension
                will always be XXXX
        '''
        if layer is None:
            layer = self.probs
        return self.sess.run(layer, feed_dict={self.clips: clips})

    def close(self):
        '''Closes the TensorFlow session used by this instance, if necessary.

        Users who did not pass their own tf.Session to the constructor **must**
        call this method to free up the network.
        '''
        if self.sess:
            self.sess.close()
            self.sess = None

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        var = tf.get_variable(
            name, shape,
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd is not None:
            weight_decay = tf.nn.l2_loss(var) * wd
            tf.add_to_collection("losses", weight_decay)
        return var

    def _conv3d(self, l_input, w, b):
        return tf.nn.bias_add(
            tf.nn.conv3d(
                l_input, w, strides=[1, 1, 1, 1, 1], padding="SAME"), b)

    def _max_pool(self, name, l_input, k):
        return tf.nn.max_pool3d(
            l_input, ksize=[1, k, 2, 2, 1],
            strides=[1, k, 2, 2, 1], padding="SAME", name=name)

    def _build_conv_layers(self):
        with tf.name_scope("conv1") as scope:
            weights = self._variable_with_weight_decay(
                "wc1", [3, 3, 3, 3, 64], 0.04, 0.00)
            biases = self._variable_with_weight_decay(
                "bc1", [64], 0.04, 0.0)
            conv = self._conv3d(self.clips, weights, biases)
            self.conv1 = tf.nn.relu(conv, name=scope)
            self.pool1 = self._max_pool("pool1", self.conv1, k=1)

        with tf.name_scope("conv2") as scope:
            weights = self._variable_with_weight_decay(
                "wc2", [3, 3, 3, 64, 128], 0.04, 0.00)
            biases = self._variable_with_weight_decay(
                "bc2", [128], 0.04, 0.0)
            conv = self._conv3d(self.pool1, weights, biases)
            self.conv2 = tf.nn.relu(conv, name=scope)
            self.pool2 = self._max_pool("pool2", self.conv2, k=2)

        with tf.name_scope("conv3a") as scope:
            weights = self._variable_with_weight_decay(
                "wc3a", [3, 3, 3, 128, 256], 0.04, 0.00)
            biases = self._variable_with_weight_decay(
                "bc3a", [256], 0.04, 0.0)
            conv = self._conv3d(self.pool2, weights, biases)
            self.conv3a = tf.nn.relu(conv, name=scope)

        with tf.name_scope("conv3b") as scope:
            weights = self._variable_with_weight_decay(
                "wc3b", [3, 3, 3, 256, 256], 0.04, 0.00)
            biases = self._variable_with_weight_decay(
                "bc3b", [256], 0.04, 0.0)
            conv = self._conv3d(self.conv3a, weights, biases)
            self.conv3b = tf.nn.relu(conv, name=scope)
            self.pool3 = self._max_pool("pool3", self.conv3b, k=2)

        with tf.name_scope("conv4a") as scope:
            weights = self._variable_with_weight_decay(
                "wc4a", [3, 3, 3, 256, 512], 0.04, 0.00)
            biases = self._variable_with_weight_decay(
                "bc4a", [512], 0.04, 0.0)
            conv = self._conv3d(self.pool3, weights, biases)
            self.conv4a = tf.nn.relu(conv, name=scope)

        with tf.name_scope("conv4b") as scope:
            weights = self._variable_with_weight_decay(
                "wc4b", [3, 3, 3, 512, 512], 0.04, 0.00)
            biases = self._variable_with_weight_decay(
                "bc4b", [512], 0.04, 0.0)
            conv = self._conv3d(self.conv4a, weights, biases)
            self.conv4b = tf.nn.relu(conv, name=scope)
            self.pool4 = self._max_pool("pool4", self.conv4b, k=2)

        with tf.name_scope("conv5a") as scope:
            weights = self._variable_with_weight_decay(
                "wc5a", [3, 3, 3, 512, 512], 0.04, 0.00)
            biases = self._variable_with_weight_decay(
                "bc5a", [512], 0.04, 0.0)
            conv = self._conv3d(self.pool4, weights, biases)
            self.conv5a = tf.nn.relu(conv, name=scope)

        with tf.name_scope("conv5b") as scope:
            weights = self._variable_with_weight_decay(
                "wc5b", [3, 3, 3, 512, 512], 0.04, 0.00)
            biases = self._variable_with_weight_decay(
                "bc5b", [512], 0.04, 0.0)
            conv = self._conv3d(self.conv5a, weights, biases)
            self.conv5b = tf.nn.relu(conv, name=scope)
            self.pool5 = self._max_pool("pool5", self.conv5b, k=2)

    def _build_fc_layers(self):
        with tf.name_scope("fc1") as scope:
            inputs = tf.reshape(self.pool5, [-1, 8192])
            weights = self._variable_with_weight_decay(
                "wd1", [8192, 4096], 0.04, 0.001)
            biases = self._variable_with_weight_decay(
                "bd1", [4096], 0.04, 0.0)
            self.fc1l = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
            self.fc1 = tf.nn.relu(self.fc1l, name=scope)
            #self.fc1 = tf.nn.dropout(self.fc1, 0.6)  # training only

        with tf.name_scope("fc2") as scope:
            weights = self._variable_with_weight_decay(
                "wd2", [4096, 4096], 0.04, 0.002)
            biases = self._variable_with_weight_decay(
                "bd2", [4096], 0.04, 0.0)
            self.fc2l = tf.nn.bias_add(tf.matmul(self.fc1, weights), biases)
            self.fc2 = tf.nn.relu(self.fc2l, name=scope)
            #self.fc2 = tf.nn.dropout(self.fc2, 0.6)  # training only

        with tf.name_scope("fc3") as scope:
            weights = self._variable_with_weight_decay(
                "wout", [4096, 101], 0.04, 0.005)
            biases = self._variable_with_weight_decay(
                "bout", [101], 0.04, 0.0)
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, weights), biases)

    def _build_output_layer(self):
        self.probs = tf.nn.softmax(self.fc3l)

    def _load_model(self, model):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        TensorFlowModelWeights(model, self.sess).load()


class C3DFeaturizerConfig(C3DConfig):
    '''Configuration settings for a C3DFeaturizer.

    Attributes:
        model: the C3D UCF101 model to use
        sample_method: the frame sampling method to use. The possible values
            are "first", "uniform", and "sliding_window"
        stride: the stride to use when the sampling method is "sliding_window"
    '''

    def __init__(self, d):
        super(C3DFeaturizerConfig, self).__init__(d)
        self.sample_method = self.parse_string(
            d, "sample_method", default="sliding_window")
        self.stride = self.parse_number(d, "stride", default=8)


class C3DFeaturizer(Featurizer):
    '''Featurizer that embeds videos into the C3D feature space.'''

    def __init__(self, config=None):
        super(C3DFeaturizer, self).__init__()
        self.config = config or C3DFeaturizerConfig.default()
        self.validate(self.config)
        self.c3d = None

    def dim(self):
        '''The dimension of the features extracted by this Featurizer.'''
        return 4096

    def _start(self):
        '''Starts a TensorFlow session and loads the network.'''
        if self.c3d is None:
            self.c3d = C3D(self.config)

    def _stop(self):
        '''Closes the TensorFlow session and frees up the network.'''
        if self.c3d:
            self.c3d.close()
            self.c3d = None

    def _featurize(self, video_path):
        '''Featurizes the input video using C3D.

        The frames are resized to 112 x 112 internally, if necessary.

        Attributes:
            video_path: the input video path

        Returns:
            the feature vector, a 1D array of length 4096
        '''
        clips = self._sample_clips(video_path)

        features = self.c3d.evaluate(clips, layer=self.c3d.fc2l)
        if self.config.sample_method == "sliding_window":
            # Average over sliding window clips
            features = np.mean(features, axis=0)
            features /= np.linalg.norm(features)
        else:
            features = features.reshape(-1)

        return features

    def _sample_clips(self, video_path):
        sample_method = self.config.sample_method
        stride = self.config.stride
        size = (112, 112)

        if sample_method == "first":
            clips = [etav.sample_first_frames(video_path, 16, size=size)]
        elif sample_method == "uniform":
            clips = [etav.uniformly_sample_frames(video_path, 16, size=size)]
        elif sample_method == "sliding_window":
            clips = etav.sliding_window_sample_frames(
                video_path, 16, stride, size=size)
        else:
            raise ValueError("Invalid sample_method '%s'" % sample_method)

        return clips
