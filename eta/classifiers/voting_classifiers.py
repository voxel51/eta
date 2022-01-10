"""
A collection of classifiers that use voting to generate predictions.

Copyright 2017-2022, Voxel51, Inc.
voxel51.com
"""
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

from eta.core.config import Config
import eta.core.data as etad
import eta.core.learning as etal


class VideoFramesVotingClassifierConfig(Config):
    """Configuration settings for a VideoFramesVotingClassifier."""

    def __init__(self, d):
        self.image_classifier = self.parse_object(
            d, "image_classifier", etal.ImageClassifierConfig
        )
        self.confidence_weighted_vote = self.parse_bool(
            d, "confidence_weighted_vote", default=False
        )


class VideoFramesVotingClassifier(etal.VideoFramesClassifier):
    """A video frames classifier that uses an `ImageClassifier` to classify
    each image and then votes on each attribute to determine the predictions
    for the video.

    Note that all attributes are combined into a single vote. Thus, even if the
    `ImageClassifier` is a multilabel classifier, each prediction will contain
    the single most prevelant label.
    """

    def __init__(self, config):
        """Creates a VideoFramesVotingClassifier instance.

        Args:
            config: a VideoFramesVotingClassifierConfig instance
        """
        self.config = config
        self.image_classifier = config.image_classifier.build()

    def __enter__(self):
        self.image_classifier.__enter__()
        return self

    def __exit__(self, *args):
        self.image_classifier.__exit__(*args)

    @property
    def is_multilabel(self):
        """Whether the classifier generates single labels (False) or multiple
        labels (True) per prediction.
        """
        return self.image_classifier.is_multilabel

    def predict(self, imgs):
        """Peforms prediction on the given video represented as a tensor of
        images.

        Args:
            imgs: a list (or n x h x w x 3 tensor) of images defining the
                video to classify

        Returns:
            an `eta.core.data.AttributeContainer` instance describing
                the predictions for the input
        """
        frame_attrs = self.image_classifier.predict_all(imgs)
        return etad.majority_vote_categorical_attrs(
            frame_attrs,
            confidence_weighted=self.config.confidence_weighted_vote,
        )
