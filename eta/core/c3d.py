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
            imgs: an optional tf.placeholder of size [XXXX, 16, 112, 112, 3] to
                use. By default, a placeholder of size [None, 16, 112, 112, 3] is
                used so you can evaluate any number of images at once
        '''
        self.config = config or C3DConfig.default()
        self.sess = sess or tf.Session()
        self.imgs = imgs or tf.placeholder(tf.float32, [None, 16, 112, 112, 3])
        with tf.variable_scope('var_name') as var_scope:
            weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
                'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
                'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
                }
            biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
                'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
                'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
                'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
                }
        self.build_c3d(imgs, self.config.dropout, self.config.batchsize,
            weights, biases)
    def evaluate(self, imgs, layer=None):
        '''Feed-forward evaluation through the net.
        Args:
            imgs: an array of size [XXXX, 112, 112, 3] containing image(s) to
                feed into the network
            layer: an optional layer whose output to return. By default, the
                output softmax layer (i.e., the class probabilities) is
                returned
        '''
        if layer is None:
            layer = self.probs

        return self.sess.run(layer, feed_dict={self.imgs: [imgs]})[0]
    def close(self):
        '''Closes the tf.Session used by this instance.
        Users who did not pass their own tf.Session to the constructor **must**
        call this method.
        '''
        self.sess.close()
        self.sess = None

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

    def _load_model(self):
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver.restore(sess, self.config.model_path)

    #sample first k frames in a video
    def get_first_k_frames(inpath, k, out_size=112):
        data = []
        num_frames = etav.get_frames_count(inpath)
        assert k < num_frames
        with etav.VideoProcessor(inpath, k, out_size=out_size) as p:
            for img in p:
                p.write(img)
                data.append(img)
        np_arr_data = np.array(data).astype(np.float32)
        return np_arr_data

    #uniformly sample k frames, always including the first frame
    def uniformly_sample_k_frames(inpath, k, out_size=112):
        data = []
        num_frames = etav.get_frames_count(inpath)
        assert k < num_frames

        stride = math.ceil(1.0 * num_frames / k)
        rng = range(num_frames)
        rng = rng[::stride]
        with etav.VideoProcessor(inpath, num_frames, out_size=out_size) as p:
            for counter, img in enumerate(p):
                if counter in rng:
                    p.write(img)
                    data.append(img)
        np_arr_data = np.array(data).astype(np.float32)
        return np_arr_data

#sample video clips using sliding window of size k and stride ns
def sliding_window_k_size_n_step(inpath, k=16, n=8, out_size=112):
    data = []
    num_frames = etav.get_frames_count(inpath)
    assert k < num_frames
    i_first = 1
    i_last = num_frames
    i_count = num_frames
    logger.debug(
        "sampling: %s; found %d frames", data_config.input_path,
        i_count)
    logger.debug("sample_length %d", k)
    logger.debug("sample_stride %d", n)
    o_count = round((i_last - k + 1.0) / n)
    o_firsts = [i_first + n*x for x in range(0,o_count)]
    clips = zip(o_firsts,
                [x + k - 1 for x in o_firsts])
    del o_firsts
    with etav.VideoProcessor(inpath, num_frames, out_size=out_size) as p:
        for img in p:
            p.write(img)
            data.append(img)
    sampled_clips = []
    for clip in clips:
        first = clip[0] - 1
        last = clip[1]
        tmp_clip = data[first:last]
        sampled_clips.append(tmp_clip)
    np_arr_data = np.array(sampled_clips).astype(np.float32)
    return np_arr_data

class C3DFeaturizerConfig(Config):
    ''' C3D Featurization configuration settings that works on images'''
    pass


class C3DFeaturizer(Featurizer):
  '''Featurizer for images or frames using the C3D network structure.'''

    def __init__(self, config=None):
        super(C3DFeaturizer,self).__init__()
        self.config = config or C3DFeaturizerConfig.default()
        self.validate(self.config)
        self.sample_method = self.config.sample_method
        self.c3d  = None

    def dim(self):
        '''The dimension of the features extracted by this Featurizer.'''
        return 4096

    def _start(self):
        '''Starts a TensorFlow session and loads the network.'''
        if self.c3d is None:
            self.c3d = C3D(self.config)

    def _stops(self):
        self.sess.close()
        self.sess = None
        self.c3d = None

    def featurize_frame(self,frame):

        if self.sample_method == 'get_first_k_frames':
            input_imgs = get_first_k_frames(config.inpath, config.frames)
        elif self.sample_method == 'uniformly_sample_k_frames':
            input_imgs = uniformly_sample_k_frames(config.inpath, config.frames)
        else:
            input_imgs = sliding_window_k_size_n_step(config.inpath, config.frames,
                config.stride)
        return self.sess.run(self.c3d.fc6, feed_dict={self.imgs: input_imgs})
