'''
TODO

Copyright 2017-2020, Voxel51, Inc.
voxel51.com

Tyler Ganter, tyler@voxel51.com
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from future.utils import iteritems, itervalues
import six
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict, OrderedDict
import errno
import logging
import os
from subprocess import Popen, PIPE
import threading

import cv2
import dateutil.parser
import numpy as np

from eta.core.config import Config, ConfigBuilder, ConfigError, Configurable
from eta.core.data import AttributeContainer, AttributeContainerSchema
from eta.core.events import EventContainer
import eta.core.frames as etaf
import eta.core.gps as etag
import eta.core.image as etai
from eta.core.objects import DetectedObjectContainer
from eta.core.serial import load_json, Serializable, Set, BigSet
import eta.core.utils as etau
import eta.core.data as etad


logger = logging.getLogger(__name__)




'''
find match
modify in place
add to "to remove"
add to "to add"
'''


d = {
    # delete attr
    "<frame attr>:*:asdf": "<delete>",

    # frame attr -> video attr
    "<frame attr>:*:time of day": "<video attr>:<categorical>:time of day",
    # video attr -> frame attr
    "<video attr>:*:scene type": "<frame attr>:<categorical>:scene type",

    # obj label -> new obj label + obj attr value
    "<object>:bus": "<object>:vehicle:<categorical>:type:bus",
    # obj label + obj attr value -> new obj label
    "<object>:vehicle:<categorical>:type:motorcycle": "<object>:motorcycle",
    # change attr type
    "<object>:<categorical>:occluded": "<object>:<boolean>:occluded",
    # rename string
    "<object>:<categorical>:road object:type:compost bin": "<object>:<categorical>:road object:type:bin compost"
}

ANY = "*"

VIDEO_ATTR = "<video attr>"
FRAME_ATTR = "<frame attr>"
IMAGE_ATTR = "<image attr>"
OBJECT = "<object>"
EVENT = "<event>"

DELETE = "<delete>"

BOOLEAN_ATTR = "<boolean>"
CATEGORICAL_ATTR = "<categorical>"
NUMERIC_ATTR = "<numeric>"

attr_type_map = {
    BOOLEAN_ATTR:     etau.get_class_name(etad.BooleanAttribute),
    CATEGORICAL_ATTR: etau.get_class_name(etad.CategoricalAttribute),
    NUMERIC_ATTR:     etau.get_class_name(etad.NumericAttribute)
}


class LabelsFilter(Serializable):

    @property
    def type(self):
        return self._type

    def __init__(self):
        self._type = etau.get_class_name(self)

class AttrFilter(LabelsFilter):

    @property
    def attr_type(self):
        return self._attr_type

    @property
    def attr_name(self):
        return self._attr_name

    @property
    def attr_value(self):
        return self._attr_value

    def __init__(self, attr_type=ANY, attr_name=ANY, attr_value=ANY):
        super(AttrFilter, self).__init__()
        self._attr_type = attr_type
        self._attr_name = attr_name
        self._attr_value = attr_value

class VideoAttrFilter(AttrFilter):
    pass

class FrameAttrFilter(AttrFilter):
    pass

class ImageAttrFilter(AttrFilter):
    pass

class _ThingWithLabelFilter(LabelsFilter):

    @property
    def label(self):
        return self._label

    def __init__(self, label=ANY):
        super(_ThingWithLabelFilter, self).__init__()
        self._label = label

class ObjectFilter(_ThingWithLabelFilter):
    pass

class EventFilter(_ThingWithLabelFilter):
    pass

class _AttrOfThingWithLabelFilter(LabelsFilter):

    @property
    def label(self):
        return self._label

    @property
    def attr_type(self):
        return self._attr_type

    @property
    def attr_name(self):
        return self._attr_name

    @property
    def attr_value(self):
        return self._attr_value

    def __init__(self, label=ANY, attr_type=ANY, attr_name=ANY, attr_value=ANY):
        super(_AttrOfThingWithLabelFilter, self).__init__()
        self._label = label
        self._attr_type = attr_type
        self._attr_name = attr_name
        self._attr_value = attr_value

    @classmethod
    def from_filters(cls, thing_with_label_filter, attr_filter):
        return cls(
            label=thing_with_label_filter.label,
            attr_type=attr_filter.attr_type,
            attr_name=attr_filter.attr_name,
            attr_value=attr_filter.attr_value
        )

class ObjectAttrFilter(_AttrOfThingWithLabelFilter):
    pass

class EventAttrFilter(_AttrOfThingWithLabelFilter):
    pass




class LabelsMapperConfig(Config):
    '''TODO'''

    def __init__(self, d):
        self.maps = d


class LabelsMapper(Configurable):
    '''TODO'''

    def __init__(self, config):
        self.config = config

    def _clear_state(self, map_in=None, map_out=None):
        # TODO TEMP
        map_in =  "<frame attr>:<categorical>:time of day"
        map_out = "<video attr>:<categorical>:time of day"

        self._cur_map_in = map_in
        self._cur_map_out = map_out
        self._to_remove = {
            "video attrs": [],
            "frame attrs": []
        }
        self._to_add = {
            "video attrs": [],
            "frame attrs": []
        }

    def map_labels(self, labels):
        for map_in, map_out in self.config.maps.items():
            self._clear_state(map_in, map_out)
            self._map_labels(labels)
            break

    def _map_labels(self, labels):
        pattern_parts = self._cur_map_in.split(":")
        qualifier = pattern_parts.pop(0)

        if qualifier == VIDEO_ATTR:
            if pattern_parts:
                pattern_parts[0] = (
                    pattern_parts[0]
                    if pattern_parts[0] == "*"
                    else attr_type_map[pattern_parts[0]]
                )
            for video_attr in labels.iter_video_attrs(*pattern_parts):
                self._process_video_attr(video_attr)
        elif qualifier == FRAME_ATTR:
            if pattern_parts:
                pattern_parts[0] = (
                    pattern_parts[0]
                    if pattern_parts[0] == "*"
                    else attr_type_map[pattern_parts[0]]
                )
            for frame_attr in labels.iter_frame_attrs(*pattern_parts):
                self._process_frame_attr(frame_attr)
        elif qualifier == OBJECT:
            raise NotImplementedError("TODO")
        elif qualifier == EVENT:
            raise NotImplementedError("TODO")
        else:
            raise ValueError("Invalid pattern qualifier %s" % qualifier)

        self._add_and_remove(labels)

    def _process_video_attr(self, video_attr):
        raise NotImplementedError("TODO")

    def _process_frame_attr(self, frame_attr):
        if self._cur_map_out == DELETE:
            self._to_remove["frame attrs"].append(frame_attr)
            return

        pattern_parts = self._cur_map_out.split(":")

        # Qualifier

        qualifier = pattern_parts.pop(0)

        if qualifier == VIDEO_ATTR:
            self._to_remove["frame attrs"].append(frame_attr)
            self._to_add["video attrs"].append(frame_attr)
        elif qualifier != FRAME_ATTR:
            raise ValueError(
                "Cannot map from %s to %s" % (FRAME_ATTR, qualifier))

        if not pattern_parts:
            return

        # Attribute Type

        attr_type = pattern_parts.pop(0)
        if attr_type != "*":
            try:
                new_type = attr_type_map[attr_type]
            except KeyError:
                raise KeyError("Invalid attr type '%s'" % attr_type)

            frame_attr.value = self._map_attr_value(
                frame_attr.type, new_type, frame_attr.value)
            frame_attr.type = new_type

        if not pattern_parts:
            return

        # Attribute Name

        attr_name = pattern_parts.pop(0)
        if attr_name != "*":
            frame_attr.name = attr_name

        if not pattern_parts:
            return

        # Attribute Value

        attr_value = pattern_parts.pop(0)
        if attr_value != "*":
            if frame_attr.type == attr_type_map[BOOLEAN_ATTR]:
                attr_value = is_true(attr_value)
            elif frame_attr.type == attr_type_map[NUMERIC_ATTR]:
                attr_value = float(attr_value)
            frame_attr.value = attr_value

    def _map_attr_value(self, prev_type, new_type, value):
        if prev_type == new_type:
            return value
        if new_type == attr_type_map[BOOLEAN_ATTR]:
            return is_true(value)
        if new_type == attr_type_map[NUMERIC_ATTR]:
            try:
                return float(value)
            except ValueError:
                raise ValueError("Failed to convert %s attr to type %s"
                                 % prev_type, new_type)
        return value

    def _add_and_remove(self, labels):
        labels.attrs.filter_elements(filters=[
            lambda el: el not in self._to_remove["video attrs"]
        ])

        for frame in labels.iter_frames():
            frame.attrs.filter_elements(filters=[
                lambda el: el not in self._to_remove["frame attrs"]
            ])

        for attr in self._to_add["video attrs"]:
            labels.attrs.add(attr)

        for attr in self._to_add["frame attrs"]:
            for frame in labels.iter_frames():
                frame.attrs.add(attr)


def is_true(thing_to_test):
    '''Cast an arg from client to native boolean'''
    if type(thing_to_test) == bool:
        return thing_to_test
    elif type(thing_to_test) == int or type(thing_to_test) == float:
        return thing_to_test == 1
    elif type(thing_to_test) == str:
        return thing_to_test.lower() == 'true'
    else:
        # make a best guess? hopefully you should never get here
        return bool(thing_to_test)


