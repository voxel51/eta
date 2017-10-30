'''
Core data structures for working with geometric objects.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
import eta.core.image as im
from eta.core.serial import Serializable


class BoundingBox(Serializable):
    '''A bounding box in an image.

    The bounding box is represented with two RelativePoint instances, one for
    the top left corner of the box, and the other for the bottom right corner.
    '''

    def __init__(self, top_left, bottom_right):
        '''Constructs a BoundingBox from two RelativePoint instances.'''
        self.top_left = top_left
        self.bottom_right = bottom_right

    def coords_in(self, **kwargs):
        '''Returns the coordinates of the bounding box in the specified image.

        Args:
            **kwargs: a valid argument for im.to_frame_size()

        Returns:
            box: a (top-left-x, top-left-y, width, height) tuple describing the
                bounding box
        '''
        tl = self.top_left.coords_in(**kwargs)
        br = self.bottom_right.coords_in(**kwargs)
        return tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]

    def extract_from(self, img, force_square=False):
        '''Extracts the subimage defined by this bounding box from the image.

        Args:
            img: an image
            force_square: whether to (minimally) manipulate the bounding box
                during extraction so that the returned subimage is square
        '''
        x1, y1 = self.top_left.coords_in(img=img)
        x2, y2 = self.bottom_right.coords_in(img=img)
        x = slice(x1, x2)
        y = slice(y1, y2)
        if force_square:
            h, w = img.shape[:2]
            x, y = make_square(x, y, w, h)
        return img[y, x, ...]

    @classmethod
    def from_dict(cls, d):
        '''Constructs a BoundingBox from a JSON dictionary.'''
        return cls(
            RelativePoint.from_dict(d["top_left"]),
            RelativePoint.from_dict(d["bottom_right"]),
        )


class RelativePoint(Serializable):
    '''A point in an image, represented as (x, y) coordinates in [0, 1].'''

    def __init__(self, x, y):
        '''Construct a relative point from (x, y) coordinates.

        Args:
            (x, y): floating point values in [0, 1]
        '''
        self.x = float(x)
        self.y = float(y)

    def coords_in(self, **kwargs):
        '''Returns the absolute (x, y) coordinates of this point in the
        specified image.

        Args:
            **kwargs: a valid argument for im.to_frame_size()

        Returns:
            (x, y): the absolute x, y coordinates of this point
        '''
        w, h = im.to_frame_size(**kwargs)
        return int(w * 1.0 * self.x), int(h * 1.0 * self.y)

    @classmethod
    def from_abs(cls, x, y, **kwargs):
        '''Constructs a RelativePoint from absolute (x, y) pixel coordinates.

        Args:
            **kwargs: a valid argument for im.to_frame_size()
        '''
        w, h = im.to_frame_size(**kwargs)
        x /= 1.0 * w
        y /= 1.0 * h
        return cls(x, y)

    @classmethod
    def from_dict(cls, d):
        '''Constructs a RelativePoint from a JSON dictionary.'''
        return cls(d["x"], d["y"])


def make_square(x, y, w, h):
    '''Force the x, y slices into a square by expanding the smaller dimension.

    If the smaller dimension can't be expanded enough and still fit
    in the maximum allowed size, the larger dimension is contracted as needed.

    Args:
        x, y: slice objects
        w, h: the (width, height) of the maximum allowed size

    Returns:
        x and y slices that define a square
    '''
    ws = x.stop - x.start
    hs = y.stop - y.start
    dx = hs - ws
    if dx < 0:
        return make_square(y, x, h, w)[::-1]
    # subimage is now always skinny

    def pad(z, dz, zmax):
        dz1 = int(0.5 * dz)
        dz2 = dz - dz1
        ddz = max(0, dz1 - z.start) - max(0, z.stop + dz2 - zmax)
        return slice(z.start - dz1 + ddz, z.stop + dz2 + ddz)

    dy = min(0, w - dx - ws)
    return pad(x, dx + dy, w), pad(y, dy, h)
