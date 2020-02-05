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
import eta.core.objects as etao
from eta.core.serial import Serializable, Container
import eta.core.utils as etau
import eta.core.data as etad


logger = logging.getLogger(__name__)


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

    def create_attr(self):
        if any(x == MATCHANY for x in
               (self.attr_type, self.attr_name, self.attr_value)):
            raise ValueError("Cannot create attribute if all fields are not"
                             "explicit")

        return etau.get_class(self.attr_type)(name=self.attr_name,
                                              value=self.attr_value)

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

    def iter_matches(self, labels):
        for obj in labels.iter_objects(label=self.label):
            yield obj

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

    def create_attr(self):
        if any(x == MATCHANY for x in
               (self.attr_type, self.attr_name, self.attr_value)):
            raise ValueError("Cannot create attribute if all fields are not"
                             "explicit")

        return etau.get_class(self.attr_type)(name=self.attr_name,
                                              value=self.attr_value)

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

    def iter_matches(self, labels):
        for attr in labels.iter_object_attrs(
                attr_type=self.attr_type,
                attr_name=self.attr_name,
                attr_value=self.attr_value
        ):
            yield attr


class EventAttrFilter(AttrOfThingWithLabelFilter):

    def iter_matches(self, labels):
        for attr in labels.iter_event_attrs(
                attr_type=self.attr_type,
                attr_name=self.attr_name,
                attr_value=self.attr_value
        ):
            yield attr


class SchemaMapper(Serializable):
    '''

    Examples:
        delete attr
            {
                "filter": {
                    "type": "eta.core.labels_mapping.FrameAttrFilter",
                    "attr_name": "time of day"

                },
                "output_map": null
            }

            OR

            {
                "filter": {
                    "type": "eta.core.labels_mapping.FrameAttrFilter",
                    "attr_name": "time of day"

                }
            }

        frame attr -> video attr
            {
                "filter": {
                    "type": "eta.core.labels_mapping.FrameAttrFilter",
                    "attr_name": "time of day"

                },
                "output_map": {
                    "type": "eta.core.labels_mapping.VideoAttrFilter"
                }
            }

        video attr -> frame attr
            {
                "filter": {
                    "type": "eta.core.labels_mapping.VideoAttrFilter",
                    "attr_name": "time of day"

                },
                "output_map": {
                    "type": "eta.core.labels_mapping.FrameAttrFilter"
                }
            }

    '''

    _VIDEO_ATTRS = "video attrs"
    _FRAME_ATTRS = "frame attrs"
    _OBJECTS = "objects"
    _OBJECT_ATTRS = "object attrs"

    _VALID_CLASS_MAPS = {
        VideoAttrFilter: [FrameAttrFilter],
        FrameAttrFilter: [VideoAttrFilter],
        ObjectFilter: [ObjectAttrFilter],
        EventFilter: [EventAttrFilter],
        ObjectAttrFilter: [ObjectFilter],
        EventAttrFilter: [EventFilter]
    }

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

        self._validate_map()

    def map_labels(self, labels):
        self._clear_state()

        if isinstance(self.filter, VideoAttrFilter):
            for video_attr in self.filter.iter_matches(labels):
                self._process_video_attr(video_attr)
        if isinstance(self.filter, FrameAttrFilter):
            for frame_attr in self.filter.iter_matches(labels):
                self._process_frame_attr(frame_attr)
        elif isinstance(self.filter, ObjectFilter):
            for obj in self.filter.iter_matches(labels):
                self._process_object(obj)
        elif isinstance(self.filter, EventFilter):
            raise NotImplementedError("TODO")
        elif isinstance(self.filter, ObjectAttrFilter):
            for obj, attr in self.filter.iter_matches(labels):
                self._process_object_attr(obj, attr)
        elif isinstance(self.filter, EventAttrFilter):
            raise NotImplementedError("TODO")
        else:
            raise ValueError("Invalid filter %s"
                             % etau.get_class_name(self.filter))

        self._add_and_remove(labels)

    def attributes(self):
        return super(SchemaMapper, self).attributes() + ["filter", "output_map"]

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

    # PRIVATE

    def _validate_map(self):
        if self.output_map is None:
            # anything can be deleted
            return

        if self.filter.type == self.output_map.type:
            # anything can be mapped to the same type
            return

        filter_map_cls = etau.get_class(self.filter.type)
        output_map_cls = etau.get_class(self.output_map.type)

        valid_classes = self._VALID_CLASS_MAPS[filter_map_cls]

        if not any(issubclass(output_map_cls, cls) for cls in valid_classes):
            raise ValueError(
                "Invalid schema map from %s to %s"
                % (self.filter.type, self.output_map.type)
            )

    def _clear_state(self):
        self._to_remove = {
            self._VIDEO_ATTRS: [],
            self._FRAME_ATTRS: [],
            self._OBJECTS: [],
            self._OBJECT_ATTRS: []
        }
        self._to_add = {
            self._VIDEO_ATTRS: [],
            self._FRAME_ATTRS: []
        }

    def _process_video_attr(self, attr):
        raise NotImplementedError("TODO")

    @typechecked
    def _process_frame_attr(self, attr: etad.Attribute):
        # Delete

        if self.output_map is None:
            self._to_remove[self._FRAME_ATTRS].append(attr)
            return

        # Map Type

        if isinstance(self.output_map, VideoAttrFilter):
            self._to_remove[self._FRAME_ATTRS].append(attr)
            self._to_add[self._VIDEO_ATTRS].append(attr)

        # Attribute

        self._process_attr(attr)

    @typechecked
    def _process_object(self, obj: etao.DetectedObject):
        # Delete

        if self.output_map is None:
            self._to_remove[self._OBJECTS].append(obj)
            return

        # Map Type (create attribute)

        if isinstance(self.output_map, ObjectAttrFilter):
            obj.add_attribute(self.output_map.create_attr())

        # Object Label

        self._process_thing_with_label(obj)

    @typechecked
    def _process_object_attr(
            self, obj: etao.DetectedObject, attr: etad.Attribute):
        # Delete

        if self.output_map is None:
            self._to_remove[self._OBJECT_ATTRS].append(attr)
            return

        # Map Type (create attribute)

        if isinstance(self.output_map, ObjectFilter):
            # set the new object label
            obj.label = self.output_map.label

            # and remove this attribute
            self._to_remove[self._OBJECT_ATTRS].append(attr)

        # Object Label

        self._process_thing_with_label(obj)

        # Attribute

        self._process_attr(attr)

    def _process_thing_with_label(self, thing_with_label):
        if (hasattr(self.output_map, "label")
                and self.output_map.label != MATCHANY):
            thing_with_label.label = self.output_map.label

    def _process_attr(self, attr: etad.Attribute):
        # Attribute Type

        if (hasattr(self.output_map, "attr_type")
                and self.output_map.attr_type != MATCHANY):
            attr.value = self._map_attr_value(
                attr.type, self.output_map.attr_type, attr.value)
            attr.type = self.output_map.attr_type

        # Attribute Name

        if (hasattr(self.output_map, "attr_name")
                and self.output_map.attr_name != MATCHANY):
            attr.name = self.output_map.attr_name

        # Attribute Value

        if (hasattr(self.output_map, "attr_value")
                and self.output_map.attr_value != MATCHANY):
            attr.value = self.output_map.attr_value

    def _add_and_remove(self, labels):
        self._remove_video_attrs(labels)
        self._remove_frame_attrs(labels)
        self._remove_objects(labels)
        self._remove_object_attrs(labels)
        self._add_video_attrs(labels)
        self._add_frame_attrs(labels)

    def _remove_video_attrs(self, labels):
        labels.attrs.filter_elements(filters=[
            lambda el: el not in self._to_remove[self._VIDEO_ATTRS]
        ])

    def _remove_frame_attrs(self, labels):
        for frame in labels.iter_frames():
            frame.attrs.filter_elements(filters=[
                lambda el: el not in self._to_remove[self._FRAME_ATTRS]
            ])

    def _remove_objects(self, labels):
        for frame in labels.iter_frames():
            frame.objects.filter_elements(filters=[
                lambda el: el not in self._to_remove[self._OBJECTS]
            ])

    def _remove_object_attrs(self, labels):
        for frame in labels.iter_frames():
            for obj in frame.objects:
                obj.attrs.filter_elements(filters=[
                    lambda el: el not in self._to_remove[self._OBJECT_ATTRS]
                ])

    def _add_video_attrs(self, labels):
        for attr in self._to_add[self._VIDEO_ATTRS]:
            labels.attrs.add(attr)

    def _add_frame_attrs(self, labels):
        for attr in self._to_add[self._FRAME_ATTRS]:
            for frame in labels.iter_frames():
                frame.attrs.add(attr)

    @staticmethod
    def _map_attr_value(prev_type, new_type, value):
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


class SchemaMapperContainer(Container):
    _ELE_CLS = SchemaMapper
    _ELE_CLS_FIELD = "_MAP_CLS"
    _ELE_ATTR = "maps"

    def map_labels(self, labels):
        for mapper in self:
            mapper.map_labels(labels)

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


