#!/usr/bin/env python
'''
Example Code: Embed an image via VGG16.
This example shows direct use of the VGG16 Class.
The embed_video.py shows the use of the VideoFeaturization 
 classes that are the preferred approach for ETA modules
 since they maintain the on-disk backing store for the class
 which is used to communication between modules.

Also shows an example of starting up a tensor flow session.

Copyright 2017, Voxel51, LLC
voxel51.com

Jason Corso, jjc@voxel51.com
'''
import errno
import os
import sys

import tensorflow as tf
import numpy as np

from eta.core.config import Config
import eta.core.image as im
import eta.core.vgg16 as vgg



def embed_image(image_path):
    '''
        Uses the default weights specified in the default config.
        Embeds the image the VGG16-net.  Store the embedded vectors as
        a npz-->Uses a VideoFeaturizer to handle IO and storage on disk

        @param config Path to the config file for the VGG16 Featurizer, which
        contains information for the network, video, and the backing location
        for the featurized video
    '''

    image = im.read(image_path)
    if image is None:
        print "could not load image: %s"%(image_path)
        sys.exit(2)

    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    sess = tf.Session()
    vggn = vgg.VGG16(imgs, sess)

    rmage = im.resize(image,224,224)
    
    embedded_vector = sess.run(vggn.fc2l, feed_dict={vggn.imgs: [rmage]})[0]

    print "image embedded to vector of length %d"%(len(embedded_vector))
    print embedded_vector
    np.savez_compressed('result_embed_image.npz',v=embedded_vector)
    print "result saved to result_embed_image.npz"


if __name__ == '__main__':
    impath = '../data/water.jpg'
    if len(sys.argv) == 2:
        impath = sys.argv[1]
    embed_image(impath)

