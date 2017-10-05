'''
Core data structures for working with events in videos.

@todo generalize to allow frame ranges.

@todo generalize to allow multi-class event detections/series.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
import collections
import sys

import numpy as np

from config import Config, Configurable
from serial import Serializable
import utils


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
            bools: an optional list (or 1D numpy array) of per-frame detections.
                The values can be any type convertable to boolean via bool()
        '''
        if bools is None:
            bools = []
        self.bools = [bool(b) for b in list(bools)]

    def add(self, b):
        '''Adds a detection to the series.'''
        self.bools.append(bool(b))

    def serialize(self):
        '''Serializes the EventDetection into a dictionary.'''
        return collections.OrderedDict(
            ("%d" % idx, b) for idx, b in enumerate(self.bools, 1)
        )

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
        self._filter_cls, config_cls = Configurable.parse(__name__, self.type)
        self.config = self.parse_object(d, "config", config_cls)

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

