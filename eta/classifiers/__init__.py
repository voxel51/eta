"""
Classifiers package declaration.

Copyright 2017-2022, Voxel51, Inc.
voxel51.com
"""

# Import all classifiers into the `eta.classifiers` namespace
from .tfslim_classifiers import (
    TFSlimClassifier,
    TFSlimClassifierConfig,
    TFSlimFeaturizer,
    TFSlimFeaturizerConfig,
)
from .vgg16_classifiers import VGG16Classifier, VGG16ClassifierConfig
from .voting_classifiers import (
    VideoFramesVotingClassifier,
    VideoFramesVotingClassifierConfig,
)
