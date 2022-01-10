"""
Core utilities for working with GPS coordinates.

Copyright 2017-2022, Voxel51, Inc.
voxel51.com
"""
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

import eta.core.frameutils as etaf
import eta.core.serial as etas
import eta.core.utils as etau

spi = etau.lazy_import("scipy.interpolate")


EARTH_RADIUS_MILES = 3959
EARTH_RADIUS_METERS = 6378000


class GPSWaypoints(etas.Serializable):
    """Class encapsulating GPS waypoints for a video.

    Attributes:
        latitude: the latitude, in degrees
        longitude: the longitude, in degrees
        frame_number: the associated frame number in the video
    """

    def __init__(self, points=None):
        """Creates a GPSWaypoints instance.

        Args:
            points: a list of GPSWaypoint instances
        """
        self.points = points or []
        self._flat = None
        self._flon = None
        self._init_gps()

    def __len__(self):
        """The number of waypoints in this instance."""
        return len(self.points)

    def __bool__(self):
        """Whether this instance contains any waypoints."""
        return bool(self.points)

    @property
    def has_points(self):
        """Returns True/False if this object contains any waypoints."""
        return len(self.points) > 1

    def add_point(self, waypoint):
        """Adds the GPSWaypoint to this object."""
        self.points.append(waypoint)
        self._init_gps()

    def add_points(self, waypoints):
        """Adds the list of the GPSWaypoint instances to this object."""
        self.points.extend(waypoints)
        self._init_gps()

    def get_location(self, frame_number):
        """Gets the (lat, lon) coordinates at the given frame number in the
        video.

        Nearest neighbors is used to interpolate between waypoints, if
        necessary.

        Args:
            frame_number: the frame number of interest

        Returns:
            the (lat, lon) at the given frame in the video, or (None, None) if
                no coordinates are available
        """
        if not self.has_points:
            return None, None
        return self._flat(frame_number), self._flon(frame_number)

    def _init_gps(self):
        if not self.has_points:
            return
        frames = [loc.frame_number for loc in self.points]
        lats = [loc.latitude for loc in self.points]
        lons = [loc.longitude for loc in self.points]
        self._flat = self._make_interp(frames, lats)
        self._flon = self._make_interp(frames, lons)

    @staticmethod
    def _make_interp(x, y):
        return spi.interp1d(
            x, y, kind="nearest", bounds_error=False, fill_value="extrapolate"
        )

    def attributes(self):
        """Returns the list of class attributes that will be serialized."""
        return ["points"] if self.has_points else []

    @classmethod
    def from_dict(cls, d):
        """Constructs a GPSWaypoints from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a GPSWaypoints instance
        """
        points = d.get("points", None)
        if points is not None:
            points = [GPSWaypoint.from_dict(p) for p in points]

        return cls(points=points)


class GPSWaypoint(etas.Serializable):
    """Class encapsulating a GPS waypoint in a video.

    Attributes:
        latitude: the latitude, in degrees
        longitude: the longitude, in degrees
        frame_number: the associated frame number in the video
    """

    def __init__(self, latitude, longitude, frame_number):
        """Constructs a GPSWaypoint instance.

        Args:
            latitude: the latitude, in degrees
            longitude: the longitude, in degrees
            frame_number: the associated frame number in the video
        """
        self.latitude = latitude
        self.longitude = longitude
        self.frame_number = frame_number

    def attributes(self):
        """Returns the list of class attributes that will be serialized."""
        return ["latitude", "longitude", "frame_number"]

    @classmethod
    def from_dict(cls, d):
        """Constructs a GPSWaypoint from a JSON dictionary."""
        return cls(
            latitude=d["latitude"],
            longitude=d["longitude"],
            frame_number=d["frame_number"],
        )


def lat_lon_distance(lat1, lon1, lat2, lon2, in_miles=False):
    """Computes the distance between two points on earth.

    Args:
        (lat1, lon1): latitudue and longitude of the first point, in degrees
        (lat2, lon2): latitudue and longitude of the second point, in degrees
        in_miles: whether to return the distance in miles (True) or meters
            (False). By default, this is False

    Returns:
        the distance (arc length) between the two points
    """
    dlat = degrees_to_radians(lat2 - lat1)
    dlon = degrees_to_radians(lon2 - lon1)
    lat1r = degrees_to_radians(lat1)
    lat2r = degrees_to_radians(lat2)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + (
        math.sin(dlon / 2)
        * math.sin(dlon / 2)
        * math.cos(lat1r)
        * math.cos(lat2r)
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_MILES * c if in_miles else EARTH_RADIUS_METERS * c


def parse_gopro_gps5(gps5_path, video_metadata):
    """Constructs a GPSWaypoints from a GoPro GPS5 JSON file.

    This implementation assumes that the input JSON file follows the schema
    below::

        {
            "1": {
                "streams": {
                    "GPS5": {
                        "samples": [
                            {
                                "value": [<lat-degrees>, <lon-degrees>, ...],
                                "cts": <ms-since-first-frame>,
                                ...
                            },
                            ...
                        ]
                    }
                }
            }
        }

    Args:
        gps5_path: the path to a GoPro GPS5 JSON file
        video_metadata: a VideoMetadata for the video

    Returns:
        a GPSWaypoints instance
    """
    # Load GPS5 data
    g = etas.load_json(gps5_path)
    samples = g["1"]["streams"]["GPS5"]["samples"]

    # Convert to GPSWaypoints
    points = []
    for sample in samples:
        lat = sample["value"][0]
        lon = sample["value"][1]

        timestamp = sample["cts"] / 1000.0  # cts = ms since first frame
        frame_number = etaf.timestamp_to_frame_number(
            timestamp,
            video_metadata.duration,
            video_metadata.total_frame_count,
        )

        points.append(GPSWaypoint(lat, lon, frame_number))

    return GPSWaypoints(points=points)


def parse_gopro_geojson(geojson_path, video_metadata):
    """Constructs a GPSWaypoints from a GoPro GeoJSON file.

    This implementation assumes that the input JSON file follows the schema
    below::

        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [<lon-degrees>, <lat-degrees>, ...],
                    ...
                ]
            },
            "properties": {
                "RelativeMicroSec": [<ms-since-first-frame>, ...],
                ...
            }
        }

    Args:
        geojson_path: the path to a GeoJSON file
        video_metadata: a VideoMetadata for the video

    Returns:
        a GPSWaypoints instance
    """
    # Load GeoJSON
    g = etas.load_json(geojson_path)
    coordinates = g["geometry"]["coordinates"]
    # Note that, despite `MicroSec` in the name, this seems to be expressed in
    # milliseconds...
    timestamps = g["properties"]["RelativeMicroSec"]

    # Convert to GPSWaypoints
    points = []
    for coords, timestamp in zip(coordinates, timestamps):
        lat = coords[1]
        lon = coords[0]

        timestamp /= 1000.0  # convert to seconds
        frame_number = etaf.timestamp_to_frame_number(
            timestamp,
            video_metadata.duration,
            video_metadata.total_frame_count,
        )

        points.append(GPSWaypoint(lat, lon, frame_number))

    return GPSWaypoints(points=points)


def degrees_to_radians(deg):
    """Converts degrees to radians.

    Args:
        deg: degrees

    Returns:
        radians
    """
    return deg * math.pi / 180


def seconds_since_epoch_to_datetime(secs):
    """Converts seconds since the epoch to a datetime.

    Args:
        secs: seconds since the epoch

    Returns:
        a datetime
    """
    return datetime.datetime.fromtimestamp(secs)


def milliseconds_since_epoch_to_datetime(ms):
    """Converts milliseconds since the epoch to a datetime.

    Args:
        ms: milliseconds since the epoch

    Returns:
        a datetime
    """
    return seconds_since_epoch_to_datetime(ms / 1000.0)
