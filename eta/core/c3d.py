'''
Extends and add functionality to the original C3D implementation by Frossard.

David Frossard, 2016
VGG16 implementation in TensorFlow
http://www.cs.toronto.edu/~frossard/post/vgg16/

Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow

Copyright 2017, Voxel51, LLC
voxel51.com

Yixin Jin, yixin@voxel51.com
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

import numpy as np
from scipy.misc import imresize
import tensorflow as tf

from eta.core.config import Config
from eta.core.features import Featurizer
import eta.core.video as etav
import input_data

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = os.path.join(constants.CONFIGS_DIR, 'c3d-config.json')


class C3dConfig(Config):
    '''Configuration settings for the C3D network.'''

    def __init__(self, d):
         self.model = self.parse_string(d, "model", default="C3D")


class C3D(object):
    '''C3D network structure hardcoded in TensorFlow.


    '''
    def __init__(self, config=None, sess=None, imgs=None):
        '''Builds a new C3D network

        Args:
            config: an optional C3DConfig instance. If omitted, the default
                ETA configuration will be used
            sess: an optional tf.Session to use. If none is provided, a new
                tf.Session instance is created, and you are responsible for
                scalling the close() method of this class when you are done
                computing
            imgs: an optional tf.placeholder of size [XXXX, 112, 112, 3] to
                use. By default, a placeholder of size [None, 112, 112, 3] is
                used so you can evaluate any number of images at once
        '''
        self.imgs = imgs
        self.build_c3d()

     def conv3d(name, l_input, w, b):
        return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
          b)

     def max_pool(name, l_input, k):
        return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1],
            strides=[1, k, 2, 2, 1], padding='SAME', name=name)

     def build_c3d(_X, _dropout, batch_size, _weights, _biases):

          # Convolution Layer
          conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
          conv1 = tf.nn.relu(conv1, 'relu1')
          pool1 = max_pool('pool1', conv1, k=1)

          # Convolution Layer
          conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
          conv2 = tf.nn.relu(conv2, 'relu2')
          pool2 = max_pool('pool2', conv2, k=2)

          # Convolution Layer
          conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
          conv3 = tf.nn.relu(conv3, 'relu3a')
          conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
          conv3 = tf.nn.relu(conv3, 'relu3b')
          pool3 = max_pool('pool3', conv3, k=2)

          # Convolution Layer
          conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
          conv4 = tf.nn.relu(conv4, 'relu4a')
          conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
          conv4 = tf.nn.relu(conv4, 'relu4b')
          pool4 = max_pool('pool4', conv4, k=2)

          # Convolution Layer
          conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
          conv5 = tf.nn.relu(conv5, 'relu5a')
          conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
          conv5 = tf.nn.relu(conv5, 'relu5b')
          pool5 = max_pool('pool5', conv5, k=2)

          # Fully connected layer
          pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
          dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
          dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

          self.fc6 = tf.nn.relu(dense1, name='fc1') # Relu activation
          dense1 = tf.nn.dropout(self.fc6, _dropout)

          dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
          dense2 = tf.nn.dropout(dense2, _dropout)

          # Output: class prediction
          out = tf.matmul(dense2, _weights['out']) + _biases['out']


class C3DFeaturizerConfig(Config):
    ''' C3D Featurization configuration settings that works on a video'''

    def __init__(self, d):
        self.weights = self.parse_object(
            d, "weights", WeightsConfig, default=None)
        if self.weights is None:
            self.default_config = C3dConfig.load_default()
            self.weights = self.default_config.weights


class C3DFeaturizer(etav.VideoFeaturizer):
    ''' Implements the C3D network as a VideoFeaturizer.  Let's the user specify
        which layer to embed.
        Embeds fc6 layer nearest the final activations (named c3d.fc1)
    '''

    def __init__(self,config):
        super(C3DFeaturizer,self).__init__(config.video_featurizer)
        self.model_path = config.model_path
        self.sample_method = config.sample_method
        self.sess = None
        self.c3d  = None

    def featurize_start(self):
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.c3d = build_c3d(images_placeholder, self.sess)

    def featurize_end(self):
        self.sess.close()
        self.sess = None
        self.c3d = None

    def featurize_frame(self,frame):
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver.restore(sess, self.model_path)
        if self.sample_method == 'get_first_k_frames':
            input_imgs = get_first_k_frames(config.inpath, config.frames)
        elif self.sample_method == 'uniformly_sample_k_frames':
            input_imgs = uniformly_sample_k_frames(config.inpath, config.frames)
        else:
            input_imgs = sliding_window_k_size_n_step(config.inpath, config.frames,
                config.stride)
        return self.sess.run(self.c3d.fc6, feed_dict={self.imgs: input_imgs})
