'''
Classifiers package declaration.

Copyright 2017-2019, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
'''

# Import all classifiers into the `eta.classifiers` namespace
from .tfslim.tfslim_classifiers import TFSlimClassifier, TFSlimClassifierConfig
from .vgg16_classifiers import VGG16Classifier, VGG16ClassifierConfig
from .voting_classifiers import VideoFramesVotingClassifier, \
                                VideoFramesVotingClassifierConfig
