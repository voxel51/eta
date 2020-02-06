'''
Core data structures for working with events in videos.

Copyright 2017-2019, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
Jason Corso, jason@voxel51.com
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

from eta.core.data import AttributeContainer
import eta.core.frames as etaf
from eta.core.serial import Container, Serializable
from eta.core.utils import MATCH_ANY


class Event(Serializable):
    '''An event in a video.

    Attributes:
        label: event label
        frames: a FrameRanges instance describing the frames in the event
        confidence: (optional) the detection confidence in [0, 1]
        index: (optional) an index assigned to the event
        score: (optional) a score assigned to the event
        attrs: (optional) an AttributeContainer of attributes for the event
            attributes of the event
    '''

    def __init__(
            self, label, frames, confidence=None, index=None, score=None,
            attrs=None):
        '''Creates an Event instance.

        Args:
            label: the event label
            frames: a FrameRanges instance describing the frames in the event
            confidence: (optional) the detection confidence in [0, 1]
            index: (optional) an index assigned to the event
            score: (optional) a score assigned to the event
            attrs: (optional) an AttributeContainer of attributes for the event
        '''
        self.label = label
        self.frames = frames
        self.confidence = confidence
        self.index = index
        self.score = score
        self.attrs = attrs or AttributeContainer()

    @property
    def is_contiguous(self):
        '''Whether the event is contiguous, i.e., whether it consists of a
        single `FrameRange`.

        If you want to ensure that the event does not contain trivial adjacent
        `FrameRange`s, then call `self.frames.simplify()` first.
        '''
        return self.frames.is_contiguous

    @property
    def has_attributes(self):
        '''Whether the event has attributes.'''
        return bool(self.attrs)

    def clear_attributes(self):
        '''Removes all attributes from the event.'''
        self.attrs = AttributeContainer()

    def add_attribute(self, attr):
        '''Adds the Attribute to the event.

        Args:
            attr: an Attribute
        '''
        self.attrs.add(attr)

    def add_attributes(self, attrs):
        '''Adds the AttributeContainer of attributes to the event.

        Args:
            attrs: an AttributeContainer
        '''
        self.attrs.add_container(attrs)

    def attributes(self):
        '''Returns the list of attributes to serialize.

        Returns:
            a list of attrinutes
        '''
        _attrs = ["label", "frames"]
        _optional_attrs = ["confidence", "index", "score"]
        _attrs.extend(
            [a for a in _optional_attrs if getattr(self, a) is not None])
        if self.attrs:
            _attrs.append("attrs")
        return _attrs

    @staticmethod
    def build_simple(first, last, label, confidence=None, index=None):
        '''Creates a simple contiguous `Event`.

        Args:
            first: the first frame of the event
            last: the last frame of the event
            label: the event label
            confidence: (optional) confidence in [0, 1]
            index: (optional) index for the event

        Returns:
             an Event
        '''
        frames = etaf.FrameRanges.build_simple(first, last)
        return Event(label, frames, confidence=confidence, index=index)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an Event from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an Event
        '''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = AttributeContainer.from_dict(attrs)

        return cls(
            d["label"],
            etaf.FrameRanges.from_dict(d["frames"]),
            confidence=d.get("confidence", None),
            index=d.get("index", None),
            score=d.get("score", None),
            attrs=attrs,
        )


class EventContainer(Container):
    '''Base class for containers that store lists of `Event`s.'''

    _ELE_CLS = Event
    _ELE_CLS_FIELD = "_EVENT_CLS"
    _ELE_ATTR = "events"

    def iter_events(self, label=MATCH_ANY):
        for event in self:
            if label != MATCH_ANY and event.label != label:
                continue
            yield event

    def iter_event_attrs(self, label=MATCH_ANY, attr_type=MATCH_ANY,
                         attr_name=MATCH_ANY, attr_value=MATCH_ANY):
        for event in self.iter_events(label):
            for attr in event.attrs.iter_attrs(
                    attr_type, attr_name, attr_value):
                yield event, attr

    def get_labels(self):
        '''Returns a set containing the labels of the `Event`s.

        Returns:
            a set of labels
        '''
        return set(event.label for event in self)

    def sort_by_confidence(self, reverse=False):
        '''Sorts the `Event`s by confidence.

        `Event`s whose confidence is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("confidence", reverse=reverse)

    def sort_by_index(self, reverse=False):
        '''Sorts the `Event`s by index.

        `Event`s whose index is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("index", reverse=reverse)

    def sort_by_score(self, reverse=False):
        '''Sorts the `Event`s by score.

        `Event`s whose score is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("score", reverse=reverse)

    def filter_by_schema(self, schema):
        '''Filters the events/attributes from this container that are not
        compliant with the given schema.

        Args:
            schema: a VideoLabelsSchema
        '''
        filter_func = lambda event: event.label in schema.events
        self.filter_elements([filter_func])
        for event in self:
            if event.has_attributes:
                event.attrs.filter_by_schema(schema.events[event.label])

    def remove_events_without_attrs(self, labels=None):
        '''Filters the events from this container that do not have attributes.

        Args:
            labels: an optional list of event `label` strings to which to
                restrict attention when filtering. By default, all event are
                processed
        '''
        filter_func = lambda event: (
            (labels is not None and event.label not in labels)
            or event.has_attributes)
        self.filter_elements([filter_func])
