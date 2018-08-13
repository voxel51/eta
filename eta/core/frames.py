'''
Core data structures for describing attributes of frames in videos.

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

from collections import defaultdict

from eta.core.data import DataContainer
from eta.core.serial import Serializable


class FrameLabel(Serializable):
    '''A frame label in a video.'''

    def __init__(self, category, label, frame_number, confidence=None):
        '''Constructs a FrameLabel.

        Args:
            category: the category of the label
            label: the frame label
            frame_number: the frame number for the label
            confidence: (optional) the confidence of the label, in [0, 1]
        '''
        self.category = category
        self.label = label
        self.frame_number = frame_number
        self.confidence = confidence

    def attributes(self):
        '''Returns the list of attributes to serialize.

        Optional attributes that were not provided (e.g. are None) are omitted
        from this list.
        '''
        _attrs = ["category", "label", "frame_number"]
        if self.confidence is not None:
            _attrs.append("confidence")
        return _attrs

    @classmethod
    def from_dict(cls, d):
        '''Constructs an FrameLabel from a JSON dictionary.'''
        return cls(
            d["category"], d["label"], d["frame_number"],
            confidence=d.get("confidence", None),
        )


class FrameLabelContainer(DataContainer):
    '''A container for frame labels.'''

    _ELE_CLS = FrameLabel
    _ELE_CLS_FIELD = "_LABEL_CLS"
    _ELE_ATTR = "labels"

    def category_set(self):
        '''Returns the set of categories in the container.'''
        return set(fl.category for fl in self)

    def label_set(self):
        '''Returns the set of labels in the container.'''
        return set(fl.label for fl in self)

    def get_labels_for_frame(self, frame_number):
        '''Returns a FrameLabelContainer containing labels for the given frame
        number.
        '''
        return self.get_matches([lambda fl: fl.frame_number == frame_number])

    def get_frames_map(self):
        '''Returns a dict mapping frame numbers to FrameLabelContainers
        containing the labels for each frame.
        '''
        flm = defaultdict(lambda: FrameLabelContainer())
        for fl in self:
            flm[fl.frame_number].add(fl)
        return flm
