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


EARTH_RADIUS_MILES = 3959


def lat_lon_distance(lat1, lon1, lat2, lon2):
    '''Computes the distance, in miles, between two points on earth.

    Args:
        (lat1, lon1): latitudue and longitude of the first point, in degrees
        (lat2, lon2): latitudue and longitude of the second point, in degrees

    Returns:
        the distance (arc length) between the two points, in miles
    '''
    dlat = degrees_to_radians(lat2 - lat1)
    dlon = degrees_to_radians(lon2 - lon1)
    lat1r = degrees_to_radians(lat1)
    lat2r = degrees_to_radians(lat2)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + (
        math.sin(dlon / 2) * math.sin(dlon / 2) *
        math.cos(lat1r) * math.cos(lat2r))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_MILES * c


def degrees_to_radians(deg):
    '''Converts degrees to radians.'''
    return deg * math.pi / 180


def seconds_since_epoch_to_datetime(secs):
    '''Converts seconds since the epoch to a datetime.'''
    return datetime.datetime.fromtimestamp(secs)


def ms_since_epoch_to_datetime(ms):
    '''Converts milliseconds since the epoch to a datetime.'''
    return seconds_since_epoch_to_datetime(ms / 1000.0)
