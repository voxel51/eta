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
import eta.core.data as etad


class ClassifierConfig(Config):
    '''Configuration class that encapsulates the name of a `Classifier` and
    an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `Classifier`
        config: an instance of the Config class associated with the specified
            `Classifier`
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
    '''Interface for classifiers.

    Subclasses of `Classifier` must implement the `predict()` method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown.
    '''

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def predict(self, arg):
        '''Peforms prediction on the given argument.

        Args:
            arg: the data to classify

        Returns:
            an `eta.core.data.AttributeContainer` describing the predictions
        '''
        raise NotImplementedError("subclasses must implement predict()")


class ImageClassifier(Classifier):
    '''Base class for all classifiers that operate on single images.

    `ImageClassifier`s may output single or multiple labels per image.

    Subclasses of `ImageClassifier` must implement the `predict()` method, and
    they can optionally provide custom (efficient) implementations of the
    `predict_all()` method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown, e.g., operating a `Featurizer`
    that featurizes the input images.
    '''

    def predict(self, img):
        '''Peforms prediction on the given image.

        Args:
            img: the image to classify

        Returns:
            an `eta.core.data.AttributeContainer` instance containing the
                predictions
        '''
        raise NotImplementedError("subclasses must implement predict()")

    def predict_all(self, imgs):
        '''Performs prediction on the given tensor of images.

        Subclasses can override this method to increase efficiency, but, by
        default, this method simply iterates over the images and predicts each.

        Args:
            imgs: a list (or d x ny x nx x 3 tensor) of images to classify

        Returns:
            a list of `eta.core.data.AttributeContainer` instances describing
                the predictions for each image
        '''
        return [self.predict(img) for img in imgs]


class VideoFramesClassifier(Classifier):
    '''Base class for all classifiers that operate directly on videos
    represented as tensors of images.

    `VideoFramesClassifier`s may output single or multiple labels per video
    clip.

    Subclasses of `VideoFramesClassifier` must implement the `predict()` method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown, e.g., operating a `Featurizer`
    that featurizes the input frames.
    '''

    def predict(self, imgs):
        '''Peforms prediction on the given video represented as a tensor of
        images.

        Args:
            imgs: a list (or d x ny x nx x 3 tensor) of images defining the
                video to classify

        Returns:
            an `eta.core.data.AttributeContainer` instance describing
                the predictions for the input
        '''
        raise NotImplementedError("subclasses must implement predict()")


class VideoFramesVotingClassifierConfig(Config):
    '''Configuration settings for a VideoFramesVotingClassifier.'''

    def __init__(self, d):
        self.image_classifier = self.parse_object(
            d, "image_classifier", ClassifierConfig)
        self.confidence_weighted_vote = self.parse_bool(
            d, "confidence_weighted_vote", default=False)


class VideoFramesVotingClassifier(VideoFramesClassifier):
    '''A video frames classifier that uses an `ImageClassifier` to classify
    each image and then votes on each attribute to determine the predictions
    for the video.

    Note that all attributes are combined into a single vote. Thus, even if the
    `ImageClassifier` is a multilabel classifier, each prediction will contain
    the single most prevelant label.
    '''

    def __init__(self, config):
        '''Creates a VideoFramesVotingClassifier instance.

        Args:
            config: a VideoFramesVotingClassifierConfig instance
        '''
        self.config = config
        self.image_classifier = config.image_classifier.build()

        if not isinstance(self.image_classifier, ImageClassifier):
            raise ValueError("image_classifier must be a %s", ImageClassifier)

    def __enter__(self):
        self.image_classifier.__enter__()
        return self

    def __exit__(self, *args):
        self.image_classifier.__exit__(*args)

    def predict(self, imgs):
        '''Peforms prediction on the given video represented as a tensor of
        images.

        Args:
            imgs: a list (or d x ny x nx x 3 tensor) of images defining the
                video to classify

        Returns:
            an `eta.core.data.AttributeContainer` instance describing
                the predictions for the input
        '''
        frame_attrs = self.image_classifier.predict_all(imgs)
        return etad.majority_vote_categorical_attrs(
            frame_attrs,
            confidence_weighted=self.config.confidence_weighted_vote)


class VideoClassifier(Classifier):
    '''Base class for all classifiers that operate on entire videos.

    `VideoClassifier`s may output single or multiple (video-level) labels per
    video.

    Subclasses of `VideoClassifier` must implement the `predict()` method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown, e.g., operating a `Featurizer`
    that featurizes the frames of the input video.
    '''

    def predict(self, video_path):
        '''Peforms prediction on the given video.

        Args:
            video_path: the path to the video

        Returns:
            an `eta.core.data.AttributeContainer` instance containing the
                predictions
        '''
        raise NotImplementedError("subclasses must implement predict()")


class VideoModel(object):
    '''Base class for generic models that process entire videos and perform
    arbitrary predictions and detections.

    `VideoModel` is useful when implementing a highly customized model that
    does not fit any of the concrete classifier/detector interfaces.
    '''

    def process(self, video_path):
        '''Generates labels for the given video.

        Args:
            video_path: the path to the video

        Returns:
            an `eta.core.video.VideoLabels` instance containing the labels
                generated for the given video
        '''
        raise NotImplementedError("subclasses must implement process()")


class ObjectDetectorConfig(Config):
    '''Configuration class that encapsulates the name of an `ObjectDetector`
    and an instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the `ObjectDetector`
        config: an instance of the Config class associated with the specified
            detector
    '''

    def __init__(self, d):
        self.type = self.parse_string(d, "type")
        self._detector_cls, config_cls = Configurable.parse(self.type)
        self.config = self.parse_object(d, "config", config_cls, default=None)
        if not self.config:
            # Try to load the default config for the detector
            self.config = config_cls.load_default()

    def build(self):
        '''Factory method that builds the `ObjectDetector` instance from the
        config specified by this class.
        '''
        return self._detector_cls(self.config)


class ObjectDetector(Configurable):
    '''Base class for all object detectors.

    Subclasses of `ObjectDetctor` must implement the `detect()` method.

    Subclasses can optionally implement the context manager interface to
    perform any necessary setup and teardown.
    '''

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def detect(self, img):
        '''Detects objects in the given image.

        Args:
            img: an image

        Returns:
            an `eta.core.objects.DetectedObjectContainer` instance describing
                the detected objects
        '''
        raise NotImplementedError("subclass must implement detect()")
