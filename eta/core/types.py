'''
ETA core type system.

Copyright 2018, Voxel51, LLC
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


class Type(object):
    DESCRIPTION = "Base class for all types"


###### Data types ##############################################################


class Data(Type):
    DESCRIPTION = "The base type for data"


class Builtin(Data):
    DESCRIPTION = "A builtin data type"


class Image(Data):
    DESCRIPTION = "An image"


class Video(Data):
    DESCRIPTION = "The base type for videos"


class VideoFile(Video):
    DESCRIPTION = "A video represented as a single (encoded) video file"


class ImageSequence(Video):
    DESCRIPTION = "A video represented as a sequence of images in a directory"


class Weights(Data):
    DESCRIPTION = "A model weights file"


class Frame(Data):
    DESCRIPTION = "Detected objects in a frame"


class FrameSequence(Data):
    DESCRIPTION = "Detected objects in a video represented as a sequence " + \
        "of Frame JSON files"


###### Module types ############################################################


class Module(Type):
    DESCRIPTION = "The base type for modules"


###### Pipeline types ##########################################################


class Pipeline(Type):
    DESCRIPTION = "The base type for pipelines"

