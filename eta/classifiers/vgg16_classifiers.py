'''
Classifier interface to the VGG-16 implementation from the `eta.core.vgg16`
module.

Copyright 2017-2019, Voxel51, Inc.
voxel51.com

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

from eta.core.config import Config
import eta.core.data as etad
import eta.core.learning as etal
from eta.core.vgg16 import VGG16, VGG16Config


class VGG16ClassifierConfig(Config):
    '''VGG16Classifier configuration settings.

    Attributes:
        attr_name: the name of the attribute that the classifier predicts
        config: an `eta.core.vgg16.VGG16Config` specifying the VGG-16 model to
            use
    '''

    def __init__(self, d):
        self.attr_name = self.parse_string(d, "attr_name", default="imagenet")
        self.config = self.parse_object(d, "config", VGG16Config, default=None)
        if self.config is None:
            self.config = VGG16Config.default()


class VGG16Classifier(etal.ImageClassifier):
    '''Classifier interface for evaluating an `eta.core.vgg16.VGG16` instance
    on images.

    Instances of this class must either use the context manager interface or
    manually call `close()` when finished to release memory.
    '''

    def __init__(self, config=None):
        '''Creates a VGG16Classifier instance.

        Args:
            config: an optional VGG16ClassifierConfig instance. If omitted, the
                default VGG16ClassifierConfig is used
        '''
        self.config = config
        self._vgg16 = VGG16(config=config.config)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        '''Closes the session and releases any memory.'''
        self._vgg16.close()
        self._vgg16 = None

    def predict(self, img):
        '''Peforms prediction on the given image.

        Args:
            img: the image to classify

        Returns:
            an `eta.core.data.AttributeContainer` instance containing the
                predictions
        '''
        imgs = [VGG16.preprocess_image(img)]
        probs = self._vgg16.evaluate(imgs)[0]
        return self._process_probs(probs)

    def predict_all(self, imgs):
        '''Performs prediction on the given tensor of images.

        Args:
            imgs: a list (or d x ny x nx x 3 tensor) of images to classify

        Returns:
            a list of `eta.core.data.AttributeContainer` instances describing
                the predictions for each image
        '''
        imgs = [VGG16.preprocess_image(img) for img in imgs]
        all_probs = self._vgg16.evaluate(imgs)
        return [self._process_probs(probs) for probs in all_probs]

    def _process_probs(self, probs):
        idx = np.argmax(probs)
        label = self._vgg16.get_label(idx + 1)  # 1-based indexing
        confidence = probs[idx]
        return self._package_attr(label, confidence)

    def _package_attr(self, label, confidence):
        attrs = etad.AttributeContainer()
        attr = etad.CategoricalAttribute(
            self.config.attr_name, label, confidence=confidence)
        attrs.add(attr)
        return attrs
