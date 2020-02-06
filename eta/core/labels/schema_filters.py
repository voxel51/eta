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

from typeguard import typechecked

from eta.core.serial import Serializable
from eta.core.utils import MATCH_ANY
import eta.core.utils as etau


logger = logging.getLogger(__name__)


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
    def __init__(self, attr_type: str = MATCH_ANY, attr_name: str = MATCH_ANY,
                 attr_value=MATCH_ANY):
        super(AttrFilter, self).__init__()
        self._attr_type = attr_type
        self._attr_name = attr_name
        self._attr_value = attr_value

    def create_attr(self):
        if any(x == MATCH_ANY for x in
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
        attr_type = d.get("attr_type", MATCH_ANY)
        attr_name = d.get("attr_name", MATCH_ANY)
        attr_value = d.get("attr_value", MATCH_ANY)

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

    def iter_matches(self, labels):
        for attr in labels.iter_image_attrs(
                attr_type=self.attr_type,
                attr_name=self.attr_name,
                attr_value=self.attr_value
        ):
            yield attr


class ThingWithLabelFilter(SchemaFilter):

    @property
    def label(self):
        return self._label

    def __init__(self, label=MATCH_ANY):
        super(ThingWithLabelFilter, self).__init__()
        self._label = label

    def attributes(self):
        return super(ThingWithLabelFilter, self).attributes() + ["label"]

    @classmethod
    def _from_dict(cls, d):
        label = d.get("label", MATCH_ANY)

        return cls(label=label)


class ObjectFilter(ThingWithLabelFilter):

    def iter_matches(self, labels):
        for obj in labels.iter_objects(label=self.label):
            yield obj


class EventFilter(ThingWithLabelFilter):

    def iter_matches(self, labels):
        for obj in labels.iter_events(label=self.label):
            yield obj


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

    def __init__(self, label=MATCH_ANY, attr_type=MATCH_ANY,
                 attr_name=MATCH_ANY, attr_value=MATCH_ANY):
        super(AttrOfThingWithLabelFilter, self).__init__()
        self._label = label
        self._attr_type = attr_type
        self._attr_name = attr_name
        self._attr_value = attr_value

    def create_attr(self):
        if any(x == MATCH_ANY for x in
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
        label = d.get("label", MATCH_ANY)
        attr_type = d.get("attr_type", MATCH_ANY)
        attr_name = d.get("attr_name", MATCH_ANY)
        attr_value = d.get("attr_value", MATCH_ANY)

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
