#!/usr/bin/env python
'''
Core utilities for working with GPS coordinates.

Copyright 2019, Voxel51, Inc.
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

import datetime
import math

import scipy.interpolate as spi

from eta.core.serial import Serializable


EARTH_RADIUS_MILES = 3959
EARTH_RADIUS_METERS = 6378000


class GPSWaypoints(Serializable):
    '''Class encapsulating GPS waypoints for a video.

    Attributes:
        latitude: the latitude, in degrees
        longitude: the longitude, in degrees
        frame_number: the associated frame number in the video
    '''

    def __init__(self, points=None):
        '''Creates a GPSWaypoints instance.

        Args:
            points: a list of GPSWaypoint instances
        '''
        self.points = points or []
        self._flat = None
        self._flon = None
        self._init_gps()

    def get_location(self, frame_number):
        '''Gets the (lat, lon) coordinates at the given frame number in the
        video.

        Nearest neighbors is used to interpolate between waypoints, if
        necessary.

        Args:
            frame_number: the frame number of interest

        Returns:
            the (lat, lon) at the given frame in the video
        '''
        return self._flat(frame_number), self._flon(frame_number)

    def _init_gps(self):
        frames = [loc.frame_number for loc in self.points]
        lats = [loc.latitude for loc in self.points]
        lons = [loc.longitude for loc in self.points]
        self._flat = self._make_interp(frames, lats)
        self._flon = self._make_interp(frames, lons)

    @staticmethod
    def _make_interp(x, y):
        return spi.interp1d(
            x, y, kind="nearest", bounds_error=False, fill_value="extrapolate")

    @classmethod
    def from_dict(cls, d):
        '''Constructs a GPSWaypoints from a JSON dictionary.'''
        points = d.get("points", None)
        if points is not None:
            points = [GPSWaypoint.from_dict(p) for p in points]

        return cls(points=points)


class GPSWaypoint(Serializable):
    '''Class encapsulating a GPS waypoint in a video.

    Attributes:
        latitude: the latitude, in degrees
        longitude: the longitude, in degrees
        frame_number: the associated frame number in the video
    '''

    def __init__(self, latitude, longitude, frame_number):
        '''Constructs a GPSWaypoint instance.

        Args:
            latitude: the latitude, in degrees
            longitude: the longitude, in degrees
            frame_number: the associated frame number in the video
        '''
        self.latitude = latitude
        self.longitude = longitude
        self.frame_number = frame_number

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.'''
        return ["latitude", "longitude", "frame_number"]

    @classmethod
    def from_dict(cls, d):
        '''Constructs a GPSWaypoint from a JSON dictionary.'''
        return cls(
            latitude=d["latitude"],
            longitude=d["longitude"],
            frame_number=d["frame_number"],
        )


def lat_lon_distance(lat1, lon1, lat2, lon2, in_miles=False):
    '''Computes the distance between two points on earth.

    Args:
        (lat1, lon1): latitudue and longitude of the first point, in degrees
        (lat2, lon2): latitudue and longitude of the second point, in degrees
        in_miles: whether to return the distance in miles (True) or meters
            (False). By default, this is False

    Returns:
        the distance (arc length) between the two points
    '''
    dlat = degrees_to_radians(lat2 - lat1)
    dlon = degrees_to_radians(lon2 - lon1)
    lat1r = degrees_to_radians(lat1)
    lat2r = degrees_to_radians(lat2)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + (
        math.sin(dlon / 2) * math.sin(dlon / 2) *
        math.cos(lat1r) * math.cos(lat2r))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_MILES * c if in_miles else EARTH_RADIUS_METERS * c


def degrees_to_radians(deg):
    '''Converts degrees to radians.'''
    return deg * math.pi / 180


def seconds_since_epoch_to_datetime(secs):
    '''Converts seconds since the epoch to a datetime.'''
    return datetime.datetime.fromtimestamp(secs)


def ms_since_epoch_to_datetime(ms):
    '''Converts milliseconds since the epoch to a datetime.'''
    return seconds_since_epoch_to_datetime(ms / 1000.0)
