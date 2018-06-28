'''
TensorFlow implementation of the popular C3D network.

This module extends and add functionality to the original C3D implementation
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

import logging
import math
import os

import numpy as np
from sklearn.preprocessing import normalize
import tensorflow as tf

from eta.core.config import Config
import eta.core.image as etai
from eta.core.features import Featurizer
from eta.core.tfutils import TensorFlowModelWeights
import eta.core.video as etav


logger = logging.getLogger(__name__)


class C3DConfig(Config):
    '''Configuration settings for the C3D network.'''

    def __init__(self, d):
        self.model = self.parse_string(d, "model", default="C3D-UCF101")
        self.batchsize = self.parse_number(d, "batchsize", default=1)
        self.inpath = self.parse_string(d, "inpath", default="")
        self.sample_method = self.parse_string(
            d, "sample_method", default="sliding_window_k_size_n_step")
        self.stride = self.parse_number(d, "stride", default=8)


class C3D(object):
    '''C3D network structure in TensorFlow.'''

    def __init__(self, config=None, sess=None, clips=None):
        '''Builds a new C3D network

        Args:
            config: an optional C3DConfig instance. If omitted, the default
                ETA configuration will be used
            sess: an optional tf.Session to use. If none is provided, a new
                tf.Session instance is created, and you are responsible for
                scalling the close() method of this class when you are done
                computing
            clips: an optional tf.placeholder of size [XXXX,
        '''
        self.config = config or C3DConfig.default()
        self.sess = sess or tf.Session()
        self.clips = clips or tf.placeholder(
            tf.float32, [None, 16, 112, 112, 3])
        self._build_conv_layers()
        self._build_fc_layers()
        self._build_output_layer()
        self._load_model(self.config.model)

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        var = tf.get_variable(
            name, shape,
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd is not None:
            weight_decay = tf.nn.l2_loss(var) * wd
            tf.add_to_collection('losses', weight_decay)
        return var

    def _conv3d(self, l_input, w, b):
        return tf.nn.bias_add(
            tf.nn.conv3d(
                l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'), b)

    def _max_pool(self, name, l_input, k):
        return tf.nn.max_pool3d(
            l_input, ksize=[1, k, 2, 2, 1],
            strides=[1, k, 2, 2, 1], padding='SAME', name=name)

    def evaluate(self, clips, layer=None):
        '''Feed-forward evaluation through the net.

        Args:
            imgs: an array of size [XXXX, 112, 112, 3] containing image(s) to
                feed into the network
            layer: an optional layer whose output to return. By default, the
                output softmax layer (i.e., the class probabilities) is
                returned
        '''
        if layer is None:
            layer = self.out
        return self.sess.run(layer, feed_dict={self.clips: [clips]})[0]

    def close(self):
        '''Closes the TensorFlow session and frees up the network.'''
        self.sess.close()
        self.sess = None

    def _build_conv_layers(self):
        clips = self.clips
        with tf.variable_scope('var_name') as var_scope:
            with tf.name_scope("conv1") as scope:
                weights = self._variable_with_weight_decay(
                        'wc1', [3, 3, 3, 3, 64], 0.04, 0.00)
                biases = self._variable_with_weight_decay(
                        'bc1', [64], 0.04, 0.0)
                conv = self._conv3d(clips, weights, biases)
                self.conv1 = tf.nn.relu(conv, name=scope)

            self.pool1 = self._max_pool('pool1', self.conv1, k=1)

            with tf.name_scope("conv2") as scope:
                weights = self._variable_with_weight_decay(
                        'wc2', [3, 3, 3, 64, 128], 0.04, 0.00)
                biases = self._variable_with_weight_decay(
                        'bc2', [128], 0.04, 0.0)
                conv = self._conv3d(self.pool1, weights, biases)
                self.conv2 = tf.nn.relu(conv, name=scope)

            self.pool2 = self._max_pool('pool2', self.conv2, k=2)

            with tf.name_scope("conv3a") as scope:
                weights = self._variable_with_weight_decay(
                        'wc3a', [3, 3, 3, 128, 256], 0.04, 0.00)
                biases = self._variable_with_weight_decay(
                        'bc3a', [256], 0.04, 0.0)
                conv = self._conv3d(self.pool2, weights, biases)
                self.conv3_1 = tf.nn.relu(conv, name=scope)

            with tf.name_scope("conv3b") as scope:
                weights = self._variable_with_weight_decay(
                        'wc3b', [3, 3, 3, 256, 256], 0.04, 0.00)
                biases = self._variable_with_weight_decay(
                        'bc3b', [256], 0.04, 0.0)
                conv = self._conv3d(self.conv3_1, weights, biases)
                self.conv3_2 = tf.nn.relu(conv, name=scope)

            self.pool3 = self._max_pool('pool3', self.conv3_2, k=2)

            with tf.name_scope("conv4a") as scope:
                weights = self._variable_with_weight_decay(
                        'wc4a', [3, 3, 3, 256, 512], 0.04, 0.00)
                biases = self._variable_with_weight_decay(
                        'bc4a', [512], 0.04, 0.0)
                conv = self._conv3d(self.pool3, weights, biases)
                self.conv4_1 = tf.nn.relu(conv, name=scope)

            with tf.name_scope("conv4b") as scope:
                weights = self._variable_with_weight_decay(
                        'wc4b', [3, 3, 3, 512, 512], 0.04, 0.00)
                biases = self._variable_with_weight_decay(
                        'bc4b', [512], 0.04, 0.0)
                conv = self._conv3d(self.conv4_1, weights, biases)
                self.conv4_2 = tf.nn.relu(conv, name=scope)

            self.pool4 = self._max_pool('pool4', self.conv4_2, k=2)

            with tf.name_scope("conv5a") as scope:
                weights = self._variable_with_weight_decay(
                        'wc5a', [3, 3, 3, 512, 512], 0.04, 0.00)
                biases = self._variable_with_weight_decay(
                        'bc5a', [512], 0.04, 0.0)
                conv = self._conv3d(self.pool4, weights, biases)
                self.conv5_1 = tf.nn.relu(conv, name=scope)

            with tf.name_scope("conv5b") as scope:
                weights = self._variable_with_weight_decay(
                        'wc5b', [3, 3, 3, 512, 512], 0.04, 0.00)
                biases = self._variable_with_weight_decay(
                        'bc5b', [512], 0.04, 0.0)
                conv = self._conv3d(self.conv5_1, weights, biases)
                self.conv5_2 = tf.nn.relu(conv, name=scope)

            self.pool5 = self._max_pool('pool5', self.conv5_2, k=2)

    def _build_fc_layers(self):
        with tf.variable_scope('var_name') as var_scope:
            with tf.name_scope("fc1") as scope:
                self.dense1 = tf.reshape(self.pool5, [1, 8192])
                weights = self._variable_with_weight_decay(
                        'wd1', [8192, 4096], 0.04, 0.001)
                biases = self._variable_with_weight_decay(
                        'bd1', [4096], 0.04, 0.0)
                self.dense1 = tf.matmul(self.dense1, weights) + biases
                self.fc6 = tf.nn.relu(self.dense1, name='fc6')
                self.dense1 = tf.nn.dropout(self.fc6, 0.6)

            with tf.name_scope("fc2") as scope:
                weights = self._variable_with_weight_decay(
                        'wd2', [4096, 4096], 0.04, 0.002)
                biases = self._variable_with_weight_decay(
                        'bd2', [4096], 0.04, 0.0)
                self.dense2 =  tf.nn.relu(
                    tf.matmul(self.dense1, weights) + biases, name='fc7')
                self.dense2 = tf.nn.dropout(self.dense2, 0.6)

    def _build_output_layer(self):
        with tf.variable_scope('var_name') as var_scope:
            with tf.name_scope("output") as scope:
                weights = self._variable_with_weight_decay(
                        'wout', [4096, 101], 0.04, 0.005)
                biases = self._variable_with_weight_decay(
                        'bout', [101], 0.04, 0.0)
                self.out = tf.matmul(self.dense2, weights) + biases

    def _load_model(self, model):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        TensorFlowModelWeights(model, self.sess).load()


class C3DFeaturizerConfig(C3DConfig):
    ''' C3D Featurization configuration settings that works on images'''
    pass


class C3DFeaturizer(Featurizer):
    '''Featurizer for videos using the C3D network structure.'''

    def __init__(self, config=None):
        super(C3DFeaturizer,self).__init__()
        self.config = config or C3DFeaturizerConfig.default()
        self.validate(self.config)
        self.sample_method = self.config.sample_method
        self.c3d  = None

    def dim(self):
        '''The dimension of the features extracted by this Featurizer.'''
        return 4096

    def sample_imgs(self, inpath):
        input_path = inpath or self.config.inpath
        if self.sample_method == 'get_first_k_frames':
            input_imgs = get_first_k_frames(input_path)
        elif self.sample_method == 'uniformly_sample_k_frames':
            input_imgs = uniformly_sample_k_frames(input_path)
        else:
            input_imgs = sliding_window_k_size_n_step(input_path,
                self.config.stride)
        return input_imgs

    def _start(self):
        '''Starts a TensorFlow session and loads the network.'''
        if self.c3d is None:
            self.c3d = C3D(self.config)

    def _stop(self):
        self.c3d.close()
        self.c3d = None

    def _featurize(self, img):
        if self.sample_method == 'sliding_window_k_size_n_step':
            clips = img.shape[0]
            tmp_feature = np.zeros(self.dim())
            for i in range(clips):
                tmp_imgs = img[i]
                tmp_feature += self.c3d.evaluate(tmp_imgs, layer=self.c3d.fc6)
            tmp_feature /= clips
            normalized_avg_feature = normalize(tmp_feature.reshape(1,-1))
            return normalized_avg_feature
        else:
            return self.c3d.evaluate(img, layer=self.c3d.fc6)
