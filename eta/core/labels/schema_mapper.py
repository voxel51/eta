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
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging
from typing import Union

from typeguard import typechecked

import eta.core.data as etad
import eta.core.events as etae
from eta.core.image import ImageLabels
import eta.core.objects as etao
import eta.core.serial as etas
from eta.core.utils import MATCH_ANY
import eta.core.utils as etau
from eta.core.video import VideoLabels

from .schema_filters import SchemaFilter, VideoAttrFilter, \
    FrameAttrFilter, ImageAttrFilter, ObjectFilter, ObjectAttrFilter, \
    EventFilter, EventAttrFilter
from .utils import is_true


logger = logging.getLogger(__name__)


class SchemaMapper(etas.Serializable):
    '''@todo(Tyler)

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

    _IMAGE_ATTRS = "image attrs"
    _VIDEO_ATTRS = "video attrs"
    _FRAME_ATTRS = "frame attrs"
    _OBJECTS = "objects"
    _OBJECT_ATTRS = "object attrs"
    _EVENTS = "events"
    _EVENT_ATTRS = "event attrs"

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
    def __init__(self, filter: SchemaFilter, output_map: SchemaFilter=None):
        self._filter = filter
        self._output_map = output_map

        self._validate_map()

    @typechecked
    def map_labels(self, labels: Union[ImageLabels, VideoLabels]):
        self._clear_state()

        if isinstance(self.filter, ImageAttrFilter):
            for image_attr in self.filter.iter_matches(labels):
                self._process_image_attr(image_attr)
        elif isinstance(self.filter, ObjectFilter):
            for obj in self.filter.iter_matches(labels):
                self._process_object(obj)
        elif isinstance(self.filter, ObjectAttrFilter):
            for obj, attr in self.filter.iter_matches(labels):
                self._process_object_attr(obj, attr)

        elif isinstance(self.filter, VideoAttrFilter):
            for video_attr in self.filter.iter_matches(labels):
                self._process_video_attr(video_attr)
        elif isinstance(self.filter, FrameAttrFilter):
            for frame_attr in self.filter.iter_matches(labels):
                self._process_frame_attr(frame_attr)
        elif isinstance(self.filter, EventFilter):
            for event in self.filter.iter_matches(labels):
                self._process_event(event)
        elif isinstance(self.filter, EventAttrFilter):
            for event, attr in self.filter.iter_matches(labels):
                self._process_event_attr(event, attr)
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
            self._OBJECT_ATTRS: [],
            self._EVENTS: [],
            self._EVENT_ATTRS: []
        }
        self._to_add = {
            self._VIDEO_ATTRS: [],
            self._FRAME_ATTRS: []
        }

    def _process_image_attr(self, attr):
        # Delete

        if self.output_map is None:
            self._to_remove[self._IMAGE_ATTRS].append(attr)
            return

        # Attribute

        self._process_attr(attr)

    def _process_video_attr(self, attr):
        # Delete

        if self.output_map is None:
            self._to_remove[self._VIDEO_ATTRS].append(attr)
            return

        # Map Type

        if isinstance(self.output_map, FrameAttrFilter):
            self._to_remove[self._VIDEO_ATTRS].append(attr)
            self._to_add[self._FRAME_ATTRS].append(attr)

        # Attribute

        self._process_attr(attr)

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
    def _process_event(self, event: etae.Event):
        # Delete

        if self.output_map is None:
            self._to_remove[self._EVENTS].append(event)
            return

        # Map Type (create attribute)

        if isinstance(self.output_map, EventAttrFilter):
            event.add_attribute(self.output_map.create_attr())

        # Event Label

        self._process_thing_with_label(event)

    @typechecked
    def _process_object_attr(
            self, obj: etao.DetectedObject, attr: etad.Attribute):
        # Delete

        if self.output_map is None:
            self._to_remove[self._OBJECT_ATTRS].append(attr)
            return

        # Map Type (create attribute)

        if isinstance(self.output_map, ObjectFilter):
            # and remove this attribute
            self._to_remove[self._OBJECT_ATTRS].append(attr)

        # Object Label

        self._process_thing_with_label(obj)

        # Attribute

        self._process_attr(attr)

    @typechecked
    def _process_event_attr(
            self, event: etae.Event, attr: etad.Attribute):
        # Delete

        if self.output_map is None:
            self._to_remove[self._EVENT_ATTRS].append(attr)
            return

        # Map Type (create attribute)

        if isinstance(self.output_map, EventFilter):
            # and remove this attribute
            self._to_remove[self._EVENT_ATTRS].append(attr)

        # Event Label

        self._process_thing_with_label(event)

        # Attribute

        self._process_attr(attr)

    def _process_thing_with_label(self, thing_with_label):
        if (hasattr(self.output_map, "label")
                and self.output_map.label != MATCH_ANY):
            thing_with_label.label = self.output_map.label

    def _process_attr(self, attr: etad.Attribute):
        # Attribute Type

        if (hasattr(self.output_map, "attr_type")
                and self.output_map.attr_type != MATCH_ANY):
            attr.value = self._map_attr_value(
                attr.type, self.output_map.attr_type, attr.value)
            attr.type = self.output_map.attr_type

        # Attribute Name

        if (hasattr(self.output_map, "attr_name")
                and self.output_map.attr_name != MATCH_ANY):
            attr.name = self.output_map.attr_name

        # Attribute Value

        if (hasattr(self.output_map, "attr_value")
                and self.output_map.attr_value != MATCH_ANY):
            attr.value = self.output_map.attr_value

    def _add_and_remove(self, labels):
        if isinstance(labels, ImageLabels):
            self._remove_things(labels, self._IMAGE_ATTRS)

        elif isinstance(labels, VideoLabels):
            self._remove_things(labels, self._VIDEO_ATTRS)
            self._remove_things(labels, self._FRAME_ATTRS)
            self._remove_things(labels, self._EVENTS)
            self._remove_things(labels, self._EVENT_ATTRS)
            self._add_video_attrs(labels)
            self._add_frame_attrs(labels)

        self._remove_things(labels, self._OBJECTS)
        self._remove_things(labels, self._OBJECT_ATTRS)

    def _remove_things(self, labels, labels_type):
        for container in self._iterate_containers(labels, labels_type):
            container.filter_elements(filters=[
                lambda el: el not in self._to_remove[labels_type]
            ])

    def _iterate_containers(self, labels, label_type):
        if isinstance(labels, ImageLabels):
            if label_type == self._IMAGE_ATTRS:
                yield labels.attrs
            elif label_type == self._OBJECTS:
                yield labels.objects
            elif label_type == self._OBJECT_ATTRS:
                for obj in labels.objects:
                    yield obj.attrs
            else:
                raise ValueError("Unexpected label_type '%s'" % label_type)

        elif isinstance(labels, VideoLabels):
            if label_type == self._VIDEO_ATTRS:
                yield labels.attrs
            elif label_type == self._FRAME_ATTRS:
                for frame in labels.iter_frames():
                    yield frame.attrs
            elif label_type == self._OBJECTS:
                for frame in labels.iter_frames():
                    yield frame.objects
            elif label_type == self._EVENTS:
                yield labels.events
            elif label_type == self._OBJECT_ATTRS:
                for frame in labels.iter_frames():
                    for obj in frame.objects:
                        yield obj.attrs
            elif label_type == self._EVENT_ATTRS:
                for event in labels.iter_events():
                    yield event.attrs
            else:
                raise ValueError("Unexpected label_type '%s'" % label_type)

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


class SchemaMapperContainer(etas.Container):
    '''@todo(Tyler)'''
    _ELE_CLS = SchemaMapper
    _ELE_CLS_FIELD = "_MAP_CLS"
    _ELE_ATTR = "maps"

    @typechecked
    def map_labels(self, labels: Union[ImageLabels, VideoLabels]):
        for mapper in self:
            mapper.map_labels(labels)
