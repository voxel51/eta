'''
Implements the popular VGG16 network that can be used for classification and
embedding of images.

Extends and add functionality to the original VGG16 implementation by Frossard.

David Frossard, 2016
VGG16 implementation in TensorFlow
http://www.cs.toronto.edu/~frossard/post/vgg16/

Model from:
https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

Weights from Caffe converted using:
https://github.com/ethereon/caffe-tensorflow

This is not a generic network that can have its layers manipulated.  It is
hardcoded to be the exact implementation of the VGG16 network that outputs to
1000 classes.

Copyright 2017, Voxel51, LLC
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

from eta.core.config import Config
from eta import constants
from eta.core.features import Featurizer
import eta.core.image as im
from eta.core.weights import Weights, WeightsConfig


logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = os.path.join(constants.CONFIGS_DIR, 'vgg16-config.json')


class VGG16Config(Config):
    '''Configuration settings for the VGG16 network.'''

    def __init__(self, d):
        self.weights = self.parse_object(d, "weights", WeightsConfig)

    @classmethod
    def load_default(cls):
        '''Loads the default config file.'''
        return cls.from_json(DEFAULT_CONFIG_PATH)


class VGG16(object):
    '''VGG16 network structure hardcoded in TensorFlow.

    @todo allow sess to be None, in which case we should start up a tf Session
    and manage it.

    Args:
        imgs: a tf.Variable of shape [XXXX, 224, 224, 3] containing images to
            embed
        sess: a tf.Session to use
        config: an optional VGG16Config instance. If omitted, the default ETA
            configuration will be used.
    '''
    def __init__(self, imgs, sess, config=None):
        assert sess is not None, 'None sessions are not currently allowed!'

        self.imgs = imgs
        self._build_conv_layers()
        self._build_fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        self.config = config or VGG16Config.load_default()

        self._load_weights(self.config.weights, sess)

    def _build_conv_layers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant(
                [123.68, 116.779, 103.939],
                dtype=tf.float32,
                shape=[1, 1, 1, 3],
                name='img_mean',
            )
            images = self.imgs - mean

        # conv1_1
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

        # conv1_2
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

        # pool1
        self.pool1 = tf.nn.max_pool(
            self.conv1_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool1',
        )

        # conv2_1
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

        # conv2_2
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

        # pool2
        self.pool2 = tf.nn.max_pool(
            self.conv2_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool2',
        )

        # conv3_1
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

        # conv3_2
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

        # conv3_3
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

        # pool3
        self.pool3 = tf.nn.max_pool(
            self.conv3_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool3',
        )

        # conv4_1
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

        # conv4_2
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

        # conv4_3
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

        # pool4
        self.pool4 = tf.nn.max_pool(
            self.conv4_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool4',
        )

        # conv5_1
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

        # conv5_2
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

        # conv5_3
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

        # pool5
        self.pool5 = tf.nn.max_pool(
            self.conv5_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool4',
        )

    def _build_fc_layers(self):
        # fc1
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

        # fc2
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

        # fc3
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

    def _load_weights(self, weights_config, sess):
        weights = Weights(weights_config)
        for i, k in enumerate(sorted(weights)):
            logger.debug("%s %s %s", i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))


class VGG16FeaturizerConfig(Config):
    '''Configuration settings for a VGG16Featurizer that works on images.'''

    def __init__(self, d):
        self.weights = self.parse_object(
            d, "weights", WeightsConfig, default=None)
        if self.weights is None:
            self.default_config = VGG16Config.load_default()
            self.weights = self.default_config.weights


class VGG16Featurizer(Featurizer):
    '''Featurizer for images or frames using the VGG16 network structure.'''

    def __init__(self, config=VGG16FeaturizerConfig({})):
        super(VGG16Featurizer, self).__init__()

        self.validate(config)
        self.config = config

        self.sess = None
        self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.vgg = None

    def dim(self):
        '''This returns the known size of the output embedding layer; remember
        this class and its instances instantiate the known VGG16 network.
        Hence this embedding dimension is known and can be hard-coded.
        '''
        return 4096

    def _start(self):
        '''Starts the TF session and loads network.'''
        self.sess = tf.Session()
        self.vgg = VGG16(self.imgs, self.sess, self.config)

    def _stop(self):
        '''Closes the session and frees up network.'''
        self.sess.close()
        self.sess = None
        self.vgg = None

    def _featurize(self, data):
        '''Featurize the data (image) through the VGG16 network.'''
        if len(data.shape) == 2:
            # GRAY input
            t = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
            data = t
            del t
        if data.shape[2] == 4:
            # RGBA input
            data = data[:, :, :3]
        img1 = im.resize(data, 224, 224)
        return self.sess.run(
            self.vgg.fc2l, feed_dict={self.vgg.imgs: [img1]})[0]
            
            
class FeatVGG16(Featurizer):
    ''' VGG16 Featurizer. '''

    def __init__(self):
        Featurizer.__init__(self)
        self.name = "VGG16Featurizer"
        self.vconfig = VGG16FeaturizerConfig({})
        self.vfeaturizer = VGG16Featurizer(self.vconfig)

    def _start(self):
        ''' start the Featurizer '''
        Featurizer._start(self)
        self.vfeaturizer.start(warn_on_restart=True, keep_alive=True)

    def _featurize(self, data_in):
        ''' encode an image using VGG features. '''
        embedded_vector = self.vfeaturizer.featurize(data_in)
        logger.info("image embedded to vector of length %d", len(embedded_vector))
        logger.info("%s", embedded_vector)
        return embedded_vector

    def _stop(self):
        ''' stop the Featurizer '''
        Featurizer._stop(self)
        self.vfeaturizer.stop()


