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
from typing import Any, Optional, Union

import cv2
import dateutil.parser
import numpy as np
from typeguard import typechecked

from eta.core.config import Config, ConfigBuilder, ConfigError, Configurable
from eta.core.data import AttributeContainer, AttributeContainerSchema
from eta.core.events import EventContainer
import eta.core.frames as etaf
import eta.core.gps as etag
import eta.core.image as etai
from eta.core.objects import DetectedObjectContainer
from eta.core.serial import Serializable, Container
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

MATCHANY = "*"


class SchemaFilter(Serializable):

    @property
    def type(self):
        return self._type

    def __init__(self):
        self._type = etau.get_class_name(self)

    def iter_matches(self, labels):
        raise NotImplementedError("Subclass must implement")

    def attributes(self):
        return super(SchemaFilter, self).attributes() + ["type"]

    @classmethod
    def from_dict(cls, d, *args, **kwargs):
        attr_cls = etau.get_class(d["type"])
        return attr_cls._from_dict(d)

class AttrFilter(SchemaFilter):

    @property
    def attr_type(self):
        return self._attr_type

    @property
    def attr_name(self):
        return self._attr_name

    @property
    def attr_value(self):
        return self._attr_value

    @typechecked
    def __init__(self, attr_type: str = MATCHANY, attr_name: str = MATCHANY,
                 attr_value=MATCHANY):
        super(AttrFilter, self).__init__()
        self._attr_type = attr_type
        self._attr_name = attr_name
        self._attr_value = attr_value

    def attributes(self):
        return super(AttrFilter, self).attributes() \
               + ["attr_type", "attr_name", "attr_value"]

    @classmethod
    def _from_dict(cls, d):
        attr_type = d.get("attr_type", MATCHANY)
        attr_name = d.get("attr_name", MATCHANY)
        attr_value = d.get("attr_value", MATCHANY)

        return cls(attr_type=attr_type, attr_name=attr_name,
                   attr_value=attr_value)

class VideoAttrFilter(AttrFilter):

    def iter_matches(self, labels):
        for attr in labels.iter_video_attrs(
                attr_type=self.attr_type,
                attr_name=self.attr_name,
                attr_value=self.attr_value
        ):
            yield attr

class FrameAttrFilter(AttrFilter):

    def iter_matches(self, labels):
        for attr in labels.iter_frame_attrs(
                attr_type=self.attr_type,
                attr_name=self.attr_name,
                attr_value=self.attr_value
        ):
            yield attr

class ImageAttrFilter(AttrFilter):
    pass

class ThingWithLabelFilter(SchemaFilter):

    @property
    def label(self):
        return self._label

    def __init__(self, label=MATCHANY):
        super(ThingWithLabelFilter, self).__init__()
        self._label = label

    def attributes(self):
        return super(ThingWithLabelFilter, self).attributes() + ["label"]

    @classmethod
    def _from_dict(cls, d):
        label = d.get("label", MATCHANY)

        return cls(label=label)

class ObjectFilter(ThingWithLabelFilter):
    pass

class EventFilter(ThingWithLabelFilter):
    pass

class AttrOfThingWithLabelFilter(SchemaFilter):

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

    def __init__(self, label=MATCHANY, attr_type=MATCHANY, attr_name=MATCHANY,
                 attr_value=MATCHANY):
        super(AttrOfThingWithLabelFilter, self).__init__()
        self._label = label
        self._attr_type = attr_type
        self._attr_name = attr_name
        self._attr_value = attr_value

    def attributes(self):
        return super(AttrOfThingWithLabelFilter, self).attributes() \
               + ["label", "attr_type", "attr_name", "attr_value"]

    @classmethod
    def _from_dict(cls, d):
        label = d.get("label", MATCHANY)
        attr_type = d.get("attr_type", MATCHANY)
        attr_name = d.get("attr_name", MATCHANY)
        attr_value = d.get("attr_value", MATCHANY)

        return cls(label=label, attr_type=attr_type, attr_name=attr_name,
                   attr_value=attr_value)

    @classmethod
    @typechecked
    def from_filters(cls, thing_with_label_filter: ThingWithLabelFilter,
                     attr_filter: AttrFilter):
        return cls(
            label=thing_with_label_filter.label,
            attr_type=attr_filter.attr_type,
            attr_name=attr_filter.attr_name,
            attr_value=attr_filter.attr_value
        )

class ObjectAttrFilter(AttrOfThingWithLabelFilter):
    pass

class EventAttrFilter(AttrOfThingWithLabelFilter):
    pass

class SchemaMap(Serializable):

    @property
    def filter(self):
        return self._filter

    @property
    def output_map(self):
        return self._output_map

    @typechecked
    def __init__(self, filter: SchemaFilter,
                 output_map: Union[SchemaFilter, None]=None):
        self._filter = filter
        self._output_map = output_map

    def attributes(self):
        return super(SchemaMap, self).attributes() + ["filter", "output_map"]

    @classmethod
    def from_dict(cls, d, *args, **kwargs):
        '''Constructs a VideoLabels from a JSON dictionary.'''
        filter_dict = d.get("filter", None)
        filter = SchemaFilter.from_dict(filter_dict) if filter_dict else None
        if filter is None:
            raise ValueError("Missing field dict field 'filter'")

        output_map_dict = d.get("output_map", None)
        output_map = (SchemaFilter.from_dict(output_map_dict)
                      if output_map_dict else None)

        return cls(filter=filter, output_map=output_map)

class SchemaMapContainer(Container):
    _ELE_CLS = SchemaMap
    _ELE_CLS_FIELD = "_MAP_CLS"
    _ELE_ATTR = "maps"


class LabelsMapper():
    '''TODO'''

    _FRAME_ATTRS = "frame attrs"
    _VIDEO_ATTRS = "video attrs"

    @property
    def maps(self):
        return self._maps.maps

    @typechecked
    def __init__(self, maps: SchemaMapContainer):
        self._maps = maps

    def _clear_state(self, filter=None, map_out=None):
        self._cur_filter = filter
        self._cur_map_out = map_out
        self._to_remove = {
            self._VIDEO_ATTRS: [],
            self._FRAME_ATTRS: []
        }
        self._to_add = {
            self._VIDEO_ATTRS: [],
            self._FRAME_ATTRS: []
        }

    def map_labels(self, labels):
        for schema_map in self.maps:
            self._clear_state(schema_map.filter, schema_map.output_map)
            self._map_labels(labels)

    def _map_labels(self, labels):
        if isinstance(self._cur_filter, VideoAttrFilter):
            for video_attr in self._cur_filter.iter_matches(labels):
                self._process_video_attr(video_attr)
        if isinstance(self._cur_filter, FrameAttrFilter):
            for frame_attr in self._cur_filter.iter_matches(labels):
                self._process_frame_attr(frame_attr)
        elif isinstance(self._cur_filter, ObjectFilter):
            raise NotImplementedError("TODO")
        elif isinstance(self._cur_filter, EventFilter):
            raise NotImplementedError("TODO")
        elif isinstance(self._cur_filter, ObjectAttrFilter):
            raise NotImplementedError("TODO")
        elif isinstance(self._cur_filter, EventAttrFilter):
            raise NotImplementedError("TODO")
        else:
            raise ValueError("Invalid filter %s"
                             % etau.get_class_name(self._cur_filter))

        self._add_and_remove(labels)

    def _process_video_attr(self, video_attr):
        raise NotImplementedError("TODO")

    @typechecked
    def _process_frame_attr(self, frame_attr: etad.Attribute):
        if self._cur_map_out is None:
            self._to_remove[self._FRAME_ATTRS].append(frame_attr)
            return

        # Map Type

        if isinstance(self._cur_map_out, VideoAttrFilter):
            self._to_remove[self._FRAME_ATTRS].append(frame_attr)
            self._to_add[self._VIDEO_ATTRS].append(frame_attr)
        elif not isinstance(self._cur_map_out, FrameAttrFilter):
            raise ValueError(
                "Cannot map from %s to %s" % (
                    etau.get_class_name(self._cur_map_out),
                    etau.get_class_name(FrameAttrFilter))
            )

        # Attribute Type

        if self._cur_map_out.attr_type != MATCHANY:
            frame_attr.value = self._map_attr_value(
                frame_attr.type, self._cur_map_out.attr_type, frame_attr.value)
            frame_attr.type = self._cur_map_out.attr_type

        # Attribute Name

        if self._cur_map_out.attr_name != MATCHANY:
            frame_attr.name = self._cur_map_out.attr_name

        # Attribute Value

        if self._cur_map_out.attr_value != MATCHANY:
            if frame_attr.type == etau.get_class_name(etad.BooleanAttribute):
                frame_attr.value = is_true(self._cur_map_out.attr_value)
            elif frame_attr.type == etau.get_class_name(etad.BooleanAttribute):
                frame_attr.value = float(self._cur_map_out.attr_value)
            else:
                frame_attr.value = self._cur_map_out.attr_value

    def _map_attr_value(self, prev_type, new_type, value):
        if prev_type == new_type:
            return value
        if new_type == etau.get_class_name(etad.BooleanAttribute):
            return is_true(value)
        if new_type == etau.get_class_name(etad.NumericAttribute):
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


