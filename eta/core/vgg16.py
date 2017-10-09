'''
Implements the popular VGG16 network that can be used for classification and
embedding of images.

Extends and add functionality to the original VGG16 implementation by Frossard.

David Frossard, 2016
VGG16 implementation in TensorFlow
http://www.cs.toronto.edu/~frossard/post/vgg16/

Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
These weights are stored at XXX add this.

This is not a generic network that can have its layers manipulated.  It is
hardcoded to be the exact implementation of the VGG16 network that outputs
to 1000 classes.

Copyright 2017, Voxel51, LLC
voxel51.com

Jason Corso, jjc@voxel51.com
'''
import os

import numpy as np
from scipy.misc import imresize
import tensorflow as tf

from config import Config
from eta import constants
import video as vd
import weights as wt
import image as im


DEFAULT_CONFIG_PATH = os.path.join(constants.CONFIGS_DIR, "vgg16-config.json")


class VGG16Config(Config):
    '''VGG16 Model Config.

    This implements the configuration settings for the VGG16 network.

    A default configuration is included in ETA and will be loaded if no
    configuration is provided by the invoker of VGG16.
    '''
    def __init__(self, d):
        self.weights_config = self.parse_object(d, "weights", wt.WeightsConfig)

    @classmethod
    def load_default(cls):
        '''Loads the default config file from disk.'''
        return cls.from_json(DEFAULT_CONFIG_PATH)


class VGG16(object):
    '''VGG16 Network structure in TensorFlow.  Hardcoded.

    @todo: generalize to a None tf session--> if it is none, then start one up
    and manage it until this object is del'd.
    '''
    def __init__(self, imgs, sess, config=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        self.config = config

        assert sess is not None, 'None sessions are not allowed'

        if config is None:
            config = VGG16Config.load_default()

        self.load_weights(config.weights_config, sess)

    def convlayers(self):
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

    def fc_layers(self):
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

    def load_weights(self, weight_config, sess):
        weights = wt.Weights(weight_config)
        keys = sorted(weights.data.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights.data[k])
            sess.run(self.parameters[i].assign(weights.data[k]))


class VGG16FeaturizerConfig(Config):
    '''VGG16 Featurization configuration settings.

    Allows you to do a standard parse of the video featurizer config.

    If you have already parsed the config in a different place, then you can
    optionally pass the vfconfig object here and it will be used rather than a
    parsed one.
    '''
    def __init__(self, d, vfconfig=None):
        if vfconfig is None:
            self.video_featurizer = self.parse_object(
                d, "video_featurizer", vd.VideoFeaturizerConfig)
        else:
            self.video_featurizer = vfconfig
        # Note that if this is None, then the VGG16 class itself will load the
        # defult config.
        self.vgg16config = self.parse_object(
            d, "vgg16", VGG16Config, default=None)


class VGG16Featurizer(vd.VideoFeaturizer):
    '''Implements the VGG16 network as a VideoFeaturizer.

    Embeds fc layer nearest the final activations (named VGG16.fc21)

    @todo It is probably more efficient to send multiple frames to the gpu at
    once. Is this doable?
    '''

    def __init__(self, config):
        super(VGG16Featurizer,self).__init__(config.video_featurizer)
        self.vgg16config = config.vgg16config
        self.sess = None
        self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.vgg  = None

    def featurize_start(self):
        self.sess = tf.Session()
        self.vgg = VGG16(self.imgs, self.sess, self.vgg16config)

    def featurize_end(self):
        self.sess.close()
        self.sess = None
        self.vgg = None

    def featurize_frame(self, frame):
        # @todo this resize needs to be changed and more adaptable to the needs
        # allow a function plugin functionality?
        img1 = im.resize(frame, 224, 224)
        return self.sess.run(
            self.vgg.fc2l, feed_dict={self.vgg.imgs: [img1]})[0]

