'''
Core data structures for working with events in videos.

This module has two "layers" of events.
1.  The "primitive" layer of events includes Event, EventSeries, and
EventDetection.  These are basic ways of representing asemantic ranges of
content.
2.  The "semantic" layer of events, includes DetectedEvent and
DetectedEventContainer.  These include capabilities of discontiguous events,
multi-class/attribute-based events and capture metadata of an event.  These use
the primitive layer's elements.

Copyright 2017-2019 Voxel51, Inc.
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

import numpy as np

from eta.core.config import Config, Configurable
from eta.core.data import AttributeContainer
from eta.core.serial import Container, Serializable
import eta.core.video as etav


class DetectedEvent(Serializable):
    '''A detected event in a video.

    Attributes:
        label: event label
        frame_ranges: a FrameRanges instance describing the frames in the event
        confidence: (optional) the detection confidence in [0, 1]
        index: (optional) an index assigned to the event
        score: (optional) a score assigned to the event
        attrs: (optional) an AttributeContainer of attributes for the event
            attributes of the event
    '''

    def __init__(
            self, label, frame_ranges, confidence=None, index=None, score=None,
            attrs=None):
        '''Creates a DetectedEvent instance.

        Args:
            label: the event label
            frame_ranges: a FrameRanges instance describing the frame numbers
                in the event
            confidence: (optional) the detection confidence in [0, 1]
            index: (optional) an index assigned to the event
            score: (optional) a score assigned to the event
            attrs: (optional) an AttributeContainer of attributes for the event
        '''
        self.label = label
        self.frame_ranges = frame_ranges
        self.confidence = confidence
        self.index = index
        self.score = score
        self.attrs = attrs or AttributeContainer()

    @property
    def is_contiguous(self):
        '''Whether the event is contiguous, i.e., whether it consists of a
        single `FrameRange`.

        If you want to ensure that the event does not contain trivial adjacent
        `FrameRange`s, then call `self.frame_ranges.simplify()` first.
        '''
        return self.frame_ranges.is_contiguous

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
        _attrs = ["label", "frame_ranges"]
        _optional_attrs = ["confidence", "index", "score"]
        _attrs.extend(
            [a for a in _optional_attrs if getattr(self, a) is not None])
        if self.attrs:
            _attrs.append("attrs")
        return _attrs

    @staticmethod
    def build_simple(first, last, label, confidence=None, index=None):
        '''Creates a simple contiguous `DetectedEvent` that has a label and
        optional confidence and index.

        Args:
            first: the first frame of the event
            last: the last frame of the event
            label: the event label
            confidence: (optional) confidence in [0, 1]
            index: (optional) index for the event

        Returns:
             a DetectedEvent
        '''
        frame_ranges = etav.FrameRanges.build_simple(first, last)
        return DetectedEvent(
            label, frame_ranges, confidence=confidence, index=index)

    @classmethod
    def from_dict(cls, d):
        '''Constructs a DetectedEvent from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a DetectedEvent
        '''
        attrs = d.get("attrs", None)
        if attrs is not None:
            attrs = AttributeContainer.from_dict(attrs)

        return cls(
            d["label"],
            etav.FrameRanges.from_dict(d["frame_ranges"]),
            confidence=d.get("confidence", None),
            index=d.get("index", None),
            score=d.get("score", None),
            attrs=attrs,
        )


class DetectedEventContainer(Container):
    '''Base class for containers that store lists of `DetectedEvent`s.'''

    _ELE_CLS = DetectedEvent
    _ELE_CLS_FIELD = "_EVENT_CLS"
    _ELE_ATTR = "events"

    def get_labels(self):
        '''Returns a set containing the labels of the DetectedEvents.

        Returns:
            a set of labels
        '''
        return set(event.label for event in self)

    def sort_by_confidence(self, reverse=False):
        '''Sorts the `DetectedEvent`s by confidence.

        `DetectedEvent`s whose confidence is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("confidence", reverse=reverse)

    def sort_by_index(self, reverse=False):
        '''Sorts the `DetectedEvent`s by index.

        `DetectedEvent`s whose index is None are always put last.

        Args:
            reverse: whether to sort in descending order. The default is False
        '''
        self.sort_by("index", reverse=reverse)

    def sort_by_score(self, reverse=False):
        '''Sorts the `DetectedEvent`s by score.

        `DetectedEvent`s whose score is None are always put last.

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
            labels: an optional list of DetectedEvent label strings to which
                to restrict attention when filtering. By default, all event
                are processed
        '''
        filter_func = lambda event: (
            (labels is not None and event.label not in labels)
            or event.has_attributes
        )
        self.filter_elements([filter_func])


class Event(Serializable):
    '''An event in a video.'''

    def __init__(self, start, stop):
        '''Initializes an Event.'''
        self.start = start
        self.stop = stop

    def to_str(self):
        '''Converts the Event to a string.'''
        return "%d-%d" % (self.start, self.stop)

    @classmethod
    def from_dict(cls, d):
        '''Constructs an Event from a JSON dictionary.'''

        return cls(d["start"], d["stop"])


class EventSeries(Serializable):
    '''A series of events in a video.'''

    def __init__(self, events=None):
        '''Initializes an EventSeries.

        Args:
            events: optional list of events in the video.
        '''
        self.events = events or []

    def add(self, event):
        '''Adds an Event to the series.'''
        self.events.append(event)

    def to_str(self):
        '''Converts the EventSeries to a string.'''
        return ",".join([e.to_str() for e in self.events])

    @classmethod
    def from_dict(cls, d):
        '''Constructs an EventSeries from a JSON dictionary.'''
        return cls(events=[Event.from_dict(de) for de in d["events"]])


class EventDetection(Serializable):
    '''A per-frame binary event detection.'''

    def __init__(self, bools=None):
        '''Constructs an EventDetection instance from a list of per-frame
        detections.

        Args:
            bools: an optional list (or 1D numpy array) of per-frame
                detections. The values can be any type convertable to boolean
                via bool()
        '''
        if bools is None:
            bools = []
        self.bools = [bool(b) for b in list(bools)]

    def add(self, b):
        '''Adds a detection to the series.'''
        self.bools.append(bool(b))

    def serialize(self, reflective=False):
        '''Serializes the EventDetection into a dictionary.

        Args:
            reflective: whether to include reflective attributes when
                serializing the object. By default, this is False

        Returns:
            a JSON dictionary representation of the object
        '''
        d = self._prepare_serial_dict(reflective)
        for idx, b in enumerate(self.bools, 1):
            d["%d" % idx] = b
        return d

    def to_series(self):
        '''Converts the EventDetection into an EventSeries.'''
        events = EventSeries()
        start = None
        in_event = False
        for idx, b in enumerate(self.bools, 1):
            if in_event:
                if not b:
                    events.add(Event(start, idx - 1))
                    in_event = False
            elif b:
                start = idx
                in_event = True
        if in_event:
            events.add(Event(start, len(self.bools)))
        return events

    @classmethod
    def from_dict(cls, d):
        '''Constructs a EventDetection from a JSON dictionary.'''
        return cls(bools=[bool(d[k]) for k in sorted(d.keys(), key=int)])


class FilterConfig(Config):
    '''Detection filter configuration settings.'''

    def __init__(self, d):
        self.type = self.parse_string(d, "type")
        self._filter_cls, _config_cls = Configurable.parse(self.type)
        self.config = self.parse_object(d, "config", _config_cls)

    def build(self):
        return self._filter_cls(self.config)


class Filter(Configurable):
    '''Interface for detection filters.'''

    def apply(self, detection):
        raise NotImplementedError("subclass must implement apply()")


class HysteresisFilterConfig(Config):
    '''Configuration settings for a HysteresisFilter.'''

    def __init__(self, d):
        self.start_window = int(self.parse_number(d, "start_window"))
        self.start_density = float(self.parse_number(d, "start_density"))
        self.stop_window = int(self.parse_number(d, "stop_window"))
        self.stop_density = float(self.parse_number(d, "stop_density"))


class HysteresisFilter(Filter):
    '''A simple hysteresis filter.'''

    def __init__(self, config):
        self.validate(config)
        self.config = config

    def apply(self, detection):
        '''Filters the EventDetection.'''
        vals = np.array(detection.bools, dtype=float)
        filt = np.zeros_like(vals, dtype=bool)
        in_event = False
        for idx in range(len(vals)):
            if in_event:
                vi = vals[idx:(idx + self.config.stop_window)]
                in_event = (vi.mean() >= self.config.stop_density)
            else:
                vi = vals[idx:(idx + self.config.start_window)]
                in_event = (vi.mean() >= self.config.start_density)
            filt[idx] = in_event
        return EventDetection(bools=filt)
