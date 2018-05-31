'''
Tensorflow implementation of the popular VGG-16 network.

This implementation is hard-coded for the model architecture and weights that
Frossard originally trained for the 1000 classes from ImageNet.

VGG-16 implementation in TensorFlow:
http://www.cs.toronto.edu/~frossard/post/vgg16/
David Frossard, 2016

Model architecture:
https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

Model weights (from Caffe):
https://github.com/ethereon/caffe-tensorflow

Copyright 2017-2018, Voxel51, LLC
voxel51.com

Jason Corso, jjc@voxel51.com
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
import os

import cv2
import numpy as np
import tensorflow as tf

from eta import constants
from eta.core.config import Config, Configurable
from eta.core.features import Featurizer
import eta.core.image as im
from eta.core.weights import Weights, WeightsConfig


logger = logging.getLogger(__name__)


DEFAULT_VGG16_CONFIG = os.path.join(constants.CONFIGS_DIR, "vgg16-config.json")


class VGG16Config(Config):
    '''Configuration settings for the VGG-16 network.'''

    def __init__(self, d):
        self.weights = self.parse_object(d, "weights", WeightsConfig)

    @classmethod
    def load_default(cls):
        '''Loads the default config file.'''
        return cls.from_json(DEFAULT_VGG16_CONFIG)


class VGG16(object):
    '''TensorFlow implementation of the VGG-16 network originally trained for
    the 1000 classes from ImageNet.

    Reference:
        http://www.cs.toronto.edu/~frossard/post/vgg16/
        David Frossard, 2016
    '''

    def __init__(self, sess, config=None):
        '''Builds a new VGG-16 network.

        Args:
            sess: a tf.Session to use
            config: an optional VGG16Config instance. If omitted, the default ETA
                configuration will be used.
        '''
        self.sess = sess
        self.config = config or VGG16Config.load_default()

        self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self._build_conv_layers()
        self._build_fc_layers()

        self.probs = tf.nn.softmax(self.fc3l)

        self._load_weights(self.config.weights)

    def evaluate(self, imgs, layer=None):
        '''Feed-forward evaluation through the net.

        Args:
            imgs: an array of size [XXXX, 224, 224, 3] containing image(s) to
                feed into the network
            layer: an optional layer whose output to return. By default, the
                output of the last fully-connected layer (i.e., the class
                predictions) are returned
        '''
        if layer is None:
            layer = self.fc3l

        return self.sess.run(layer, feed_dict={self.imgs: [imgs]})[0]

    def _build_conv_layers(self):
        self.parameters = []

        with tf.name_scope('preprocess') as scope:
            mean = tf.constant(
                [123.68, 116.779, 103.939],
                dtype=tf.float32,
                shape=[1, 1, 1, 3],
                name='img_mean',
            )
            images = self.imgs - mean

        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[64], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 64, 64], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(
                self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[64], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        self.pool1 = tf.nn.max_pool(
            self.conv1_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool1',
        )

        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 64, 128], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(
                self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[128], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 128, 128], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(
                self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[128], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        self.pool2 = tf.nn.max_pool(
            self.conv2_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool2',
        )

        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 128, 256], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(
                self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[256], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(
                self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[256], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(
                self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[256], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        self.pool3 = tf.nn.max_pool(
            self.conv3_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool3',
        )

        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 256, 512], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(
                self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(
                self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(
                self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        self.pool4 = tf.nn.max_pool(
            self.conv4_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool4',
        )

        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(
                self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(
                self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            conv = tf.nn.conv2d(
                self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(0.0, shape=[512], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        self.pool5 = tf.nn.max_pool(
            self.conv5_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool4',
        )

    def _build_fc_layers(self):
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(
                tf.truncated_normal(
                    [shape, 4096], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            fc1b = tf.Variable(
                tf.constant(1.0, shape=[4096], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(
                tf.truncated_normal(
                    [4096, 4096], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            fc2b = tf.Variable(
                tf.constant(1.0, shape=[4096], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2l = fc2l
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(
                tf.truncated_normal(
                    [4096, 1000], dtype=tf.float32, stddev=1e-1),
                name='weights',
            )
            fc3b = tf.Variable(
                tf.constant(1.0, shape=[1000], dtype=tf.float32),
                trainable=True,
                name='biases',
            )
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def _load_weights(self, weights_config):
        weights = Weights(weights_config)
        for i, k in enumerate(sorted(weights)):
            logger.debug("%s %s %s", i, k, np.shape(weights[k]))
            self.sess.run(self.parameters[i].assign(weights[k]))


class VGG16FeaturizerConfig(Config):
    '''Configuration settings for a VGG16Featurizer that works on images.'''

    def __init__(self, d):
        self.weights = self.parse_object(
            d, "weights", WeightsConfig, default=None)

        if self.weights is None:
            self.weights = VGG16Config.load_default().weights


class VGG16Featurizer(Featurizer):
    '''Featurizer for images or frames using the VGG16 network structure.'''

    def __init__(self, config=None):
        super(VGG16Featurizer, self).__init__()

        self.config = config or VGG16FeaturizerConfig({})
        self.validate(self.config)

        self.sess = None
        self.vgg16 = None

    def dim(self):
        '''The dimension of the features extracted by this Featurizer.'''
        return 4096

    def _start(self):
        '''Starts the TensorFlow session and loads the network.'''
        self.sess = tf.Session()
        self.vgg16 = VGG16(self.sess, self.config)

    def _stop(self):
        '''Closes the TensorFlow session and frees up the network.'''
        self.sess.close()
        self.sess = None
        self.vgg16 = None

    def _featurize(self, img):
        '''Featurizes the image using the VGG-16 network.'''
        if len(img.shape) == 2:
            # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            # RGBA
            img = img[:, :, :3]

        img = im.resize(img, 224, 224)
        return self.vgg16.evaluate(img, layer=self.vgg16.fc2l)
