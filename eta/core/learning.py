#!/usr/bin/env python
'''
Core learning infrastructure.

Copyright 2019, Voxel51, Inc.
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

from eta.core.config import Config, Configurable


class ClassifierConfig(Config):
    '''Configuration class that encapsulates the name of a Classifier and
    an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the Classifier
        config: an instance of the Config class associated with the specified
            Classifier
    '''

    def __init__(self, d):
        self.type = self.parse_string(d, "type")
        self._classifier_cls, config_cls = Configurable.parse(self.type)
        self.config = self.parse_object(d, "config", config_cls, default=None)
        if not self.config:
            # Try to load the default config for the classifier
            self.config = config_cls.load_default()

    def build(self):
        '''Factory method that builds the Classifier instance from the
        config specified by this class.
        '''
        return self._classifier_cls(self.config)


class Classifier(Configurable):
    '''Base class for all classifiers.

    Subclasses of Classifier must implement the `predict()` method.
    '''

    def predict(self, arg):
        '''Peforms prediction on the given input.

        Args:
            arg: the input to classify. Depending on the classifier, this may
                be an image, video, or embedding

        Returns:
            attributes: An `eta.core.data.AttributeContainer` describing the
                predictions for the image
        '''
        raise NotImplementedError("subclass must implement predict()")


class ObjectDetectorConfig(Config):
    '''Configuration class that encapsulates the name of an ObjectDetector and
    an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the ObjectDetector
        config: an instance of the Config class associated with the specified
            Detector
    '''

    def __init__(self, d):
        self.type = self.parse_string(d, "type")
        self._detector_cls, config_cls = Configurable.parse(self.type)
        self.config = self.parse_object(d, "config", config_cls, default=None)
        if not self.config:
            # Try to load the default config for the detector
            self.config = config_cls.load_default()

    def build(self):
        '''Factory method that builds the ObjectDetector instance from the
        config specified by this class.
        '''
        return self._detector_cls(self.config)


class ObjectDetector(Configurable):
    '''Base class for all object detectors.

    Subclasses of ObjectDetctor must implement the `detect()` method.
    '''

    def detect(self, img):
        '''Performs object detection on the input image.

        Args:
            img: an image

        Returns:
            objects: An `eta.core.objects.DetectedObjectContainer` describing
                the detected objects in the image
        '''
        raise NotImplementedError("subclass must implement detect()")
