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


def get_first_k_frames(inpath, k, embedding_frame_size):
    '''Sample first k frames in a video. Return a numpy array of size
    [k, embedding_frame_siz, embedding_frame_size]

    Args:
        inpath: path to the input video
        k: number of frames to extract for the input video
        embedding_frame_size: size of the output frames
    '''
    data = []
    num_frames = etav.get_frame_count(inpath)
    assert k < num_frames
    with etav.FFmpegVideoReader(inpath, "1-%d" % k) as vr:
        for img in vr:
            img = etai.resize(img, embedding_frame_size, embedding_frame_size)
            data.append(img)
    np_arr_data = np.array(data).astype(np.float32)
    return np_arr_data


def uniformly_sample_k_frames(inpath, k, embedding_frame_size):
    '''Uniformly sample k frames, always including the first frame. Return
    a numpy array of size [k, embedding_frame_siz, embedding_frame_size]

    Args:
        inpath: path to the input video
        k: number of frames to extract for the input video
        embedding_frame_size: dimension to use when embedding the frames
    '''
    data = []
    num_frames = etav.get_frame_count(inpath)
    assert k < num_frames

    stride = int(math.ceil(1.0 * num_frames / k))
    rng = range(num_frames)
    rng = list(rng[::stride])
    if len(rng) < k:
        rng.append(num_frames - 1)
    elif len(rng) > k:
        rng.pop()
    assert len(rng) == k
    rng = [x + 1 for x in rng]
    with etav.FFmpegVideoReader(inpath, frames=rng) as vr:
        for img in vr:
            img = etai.resize(img, embedding_frame_size,
                embedding_frame_size)
            data.append(img)
    np_arr_data = np.array(data).astype(np.float32)
    return np_arr_data


def sliding_window_k_size_n_step(inpath, k, n, embedding_frame_size):
    '''Sample video clips using sliding window of size k and stride n. Return a
    numpy array [k, embedding_frame_siz, embedding_frame_size]

    Args:
        inpath: path to the input video
        k: number of frames to extract for the input video
        n: the stride for sliding window
        embedding_frame_size: size of the output frames
    '''
    data = []
    num_frames = etav.get_frame_count(inpath)
    assert k < num_frames
    i_first = 1
    i_last = num_frames
    i_count = num_frames
    logger.debug("sample_length %d", k)
    logger.debug("sample_stride %d", n)
    o_count = round((i_last - k + 1.0) / n)
    o_firsts = [i_first + n*x for x in range(0,o_count)]
    clips = zip(o_firsts,
                [x + k - 1 for x in o_firsts])
    del o_firsts

    with etav.FFmpegVideoReader(inpath) as vr:
        for img in vr:
            img = etai.resize(img, embedding_frame_size,
                embedding_frame_size)
            data.append(img)
    sampled_clips = []
    for clip in clips:
        first = clip[0] - 1
        last = clip[1]
        tmp_clip = data[first:last]
        sampled_clips.append(tmp_clip)
    np_arr_data = np.array(sampled_clips).astype(np.float32)
    return np_arr_data


class C3DConfig(Config):
    '''Configuration settings for the C3D network.'''

    def __init__(self, d):
        self.model = self.parse_string(d, "model", default="C3D-UCF101")
        self.dropout = self.parse_number(d, "dropout", default=0.6)
        self.batchsize = self.parse_number(d, "batchsize", default=1)
        self.inpath = self.parse_string(d, "inpath", default="")
        self.embedding_frame_size = self.parse_number(d,
            "embedding_frame_size", default=112)
        self.sample_method = self.parse_string(d, "sample_method",
            default="sliding_window_k_size_n_step")
        self.num_frames_per_clip = self.parse_number(d, "num_frames_per_clip",
            default=16)
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
            num_frames_per_clip,embedding_frame_size, embedding_frame_size, 3]
            to use. By default, a placeholder of size [None,
            num_frames_per_clip, embedding_frame_size, embedding_frame_size, 3]
            is used so you can evaluate any number of images at once
        '''
        self.config = config or C3DConfig.default()
        self.sess = sess or tf.Session()
        self.clips = clips or tf.placeholder(tf.float32, [None,
            self.config.num_frames_per_clip,self.config.embedding_frame_size,
            self.config.embedding_frame_size, 3])
        self.build_c3d(self.clips, self.config.dropout, self.config.batchsize)
        self._load_model(self.config.model)

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        var = tf.get_variable(name, shape,
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd is not None:
            weight_decay = tf.nn.l2_loss(var) * wd
            tf.add_to_collection('losses', weight_decay)
        return var

    def _conv3d(self, name, l_input, w, b):
        return tf.nn.bias_add(
            tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
            b)

    def _max_pool(self, name, l_input, k):
        return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1],
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

    def build_c3d(self, _X, _dropout, batch_size,):
        with tf.variable_scope('var_name') as var_scope:
            _weights = {
                'wc1': self._variable_with_weight_decay('wc1',
                    [3, 3, 3, 3, 64],0.04, 0.00),
                'wc2': self._variable_with_weight_decay('wc2',
                    [3, 3, 3, 64, 128],0.04, 0.00),
                'wc3a': self._variable_with_weight_decay('wc3a',
                    [3, 3, 3, 128, 256], 0.04, 0.00),
                'wc3b': self._variable_with_weight_decay('wc3b',
                    [3, 3, 3, 256, 256], 0.04, 0.00),
                'wc4a': self._variable_with_weight_decay('wc4a',
                    [3, 3, 3, 256, 512], 0.04, 0.00),
                'wc4b': self._variable_with_weight_decay('wc4b',
                    [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5a': self._variable_with_weight_decay('wc5a',
                    [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5b': self._variable_with_weight_decay('wc5b',
                    [3, 3, 3, 512, 512], 0.04, 0.00),
                'wd1': self._variable_with_weight_decay('wd1',
                    [8192, 4096], 0.04, 0.001),
                'wd2': self._variable_with_weight_decay('wd2',
                    [4096, 4096], 0.04, 0.002),
                'out': self._variable_with_weight_decay('wout',
                    [4096, 101], 0.04, 0.005)
                }
            _biases = {
                'bc1': self._variable_with_weight_decay('bc1',
                    [64], 0.04, 0.0),
                'bc2': self._variable_with_weight_decay('bc2',
                    [128], 0.04, 0.0),
                'bc3a': self._variable_with_weight_decay('bc3a',
                    [256], 0.04, 0.0),
                'bc3b': self._variable_with_weight_decay('bc3b',
                    [256], 0.04, 0.0),
                'bc4a': self._variable_with_weight_decay('bc4a',
                    [512], 0.04, 0.0),
                'bc4b': self._variable_with_weight_decay('bc4b',
                    [512], 0.04, 0.0),
                'bc5a': self._variable_with_weight_decay('bc5a',
                    [512], 0.04, 0.0),
                'bc5b': self._variable_with_weight_decay('bc5b',
                    [512], 0.04, 0.0),
                'bd1': self._variable_with_weight_decay('bd1',
                    [4096], 0.04, 0.0),
                'bd2': self._variable_with_weight_decay('bd2',
                    [4096], 0.04, 0.0),
                'out': self._variable_with_weight_decay('bout',
                    [101], 0.04, 0.0),
                }
        # Convolution Layer
        conv1 = self._conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
        conv1 = tf.nn.relu(conv1, 'relu1')
        pool1 = self._max_pool('pool1', conv1, k=1)

        # Convolution Layer
        conv2 = self._conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
        conv2 = tf.nn.relu(conv2, 'relu2')
        pool2 = self._max_pool('pool2', conv2, k=2)

        # Convolution Layer
        conv3 = self._conv3d('conv3a', pool2, _weights['wc3a'],
            _biases['bc3a'])
        conv3 = tf.nn.relu(conv3, 'relu3a')
        conv3 = self._conv3d('conv3b', conv3, _weights['wc3b'],
            _biases['bc3b'])
        conv3 = tf.nn.relu(conv3, 'relu3b')
        pool3 = self._max_pool('pool3', conv3, k=2)

        # Convolution Layer
        conv4 = self._conv3d('conv4a', pool3, _weights['wc4a'],
            _biases['bc4a'])
        conv4 = tf.nn.relu(conv4, 'relu4a')
        conv4 = self._conv3d('conv4b', conv4, _weights['wc4b'],
            _biases['bc4b'])
        conv4 = tf.nn.relu(conv4, 'relu4b')
        pool4 = self._max_pool('pool4', conv4, k=2)

        # Convolution Layer
        conv5 = self._conv3d('conv5a', pool4, _weights['wc5a'],
            _biases['bc5a'])
        conv5 = tf.nn.relu(conv5, 'relu5a')
        conv5 = self._conv3d('conv5b', conv5, _weights['wc5b'],
            _biases['bc5b'])
        conv5 = tf.nn.relu(conv5, 'relu5b')
        pool5 = self._max_pool('pool5', conv5, k=2)

        # Fully connected layer
        pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
        dense1 = tf.reshape(pool5, [1,
            _weights['wd1'].get_shape().as_list()[0]])
        # Reshape conv3 output to fit dense layer input
        dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

        self.fc6 = tf.nn.relu(dense1, name='fc1') # Relu activation
        dense1 = tf.nn.dropout(self.fc6, _dropout)

        # Relu activation
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) +
            _biases['bd2'], name='fc2')

        dense2 = tf.nn.dropout(dense2, _dropout)

        # Output: class prediction
        self.out = tf.matmul(dense2, _weights['out']) + _biases['out']

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
            input_imgs = get_first_k_frames(input_path,
                self.config.num_frames_per_clip,
                self.config.embedding_frame_size)
        elif self.sample_method == 'uniformly_sample_k_frames':
            input_imgs = uniformly_sample_k_frames(input_path,
                self.config.num_frames_per_clip,
                self.config.embedding_frame_size)
        else:
            input_imgs = sliding_window_k_size_n_step(input_path,
                self.config.num_frames_per_clip, self.config.stride,
                self.config.embedding_frame_size)
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