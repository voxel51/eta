'''
Core image processing tools.

ETA uses OpenCV for some of its image-related processing.  OpenCV stores its
images in BGR format.  ETA stores its images in RGB format.  This module's
contract is that it expects RGB to be passed to it and RGB to be expected from
it.

Copyright 2017-2018, Voxel51, LLC
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

import errno
import os
from subprocess import Popen, PIPE

import cv2
import numpy as np

import eta
import eta.core.utils as etau
import eta.core.web as etaw


SUPPORTED_IMAGE_FORMATS = [".png", ".jpg", ".jpeg", ".gif", ".tiff", ".bmp"]


def is_supported_image(filepath):
    '''Determines whether the given file has a supported image type.'''
    return os.path.splitext(filepath)[1] in SUPPORTED_IMAGE_FORMATS


def glob_images(dir_):
    '''Returns an iterator over all supported image files in the directory.'''
    return etau.multiglob(
        *SUPPORTED_IMAGE_FORMATS, root=os.path.join(dir_, "*"))


def make_image_sequence_patt(basedir, basename="", patt=None, ext=None):
    '''Makes an image sequence pattern of the following form:

    <basedir>/<basename>-<patt><ext>

    where the "-" is omitted

    Args:
        basedir: the base directory
        basename: an optional base filename. If omitted, the hyphen is also
            omitted
        patt: an optional image pattern to use. If omitted, the default pattern
            `eta.config.default_sequence_idx` is used
        ext: an optional image extension to use. If omitted, the default image
            extension `eta.config.default_image_ext`

    Returns:
        the image sequence pattern
    '''
    name = basename + "-" if basename else ""
    patt = patt or eta.config.default_sequence_idx
    ext = ext or eta.config.default_image_ext
    return os.path.join(basedir, name + patt + ext)


###### Image I/O ##############################################################


def decode(b, flag=cv2.IMREAD_UNCHANGED):
    '''Decodes an image from raw bytes.

    Args:
        bytes: the raw bytes of an image (e.g. from read() or from a web
            download)
        flag: an optional OpenCV image format flag. By default, the image is
            returned in its native format (color, grayscale, transparent, ...)

    Returns:
        A uint8 numpy array containing the image
    '''
    vec = np.asarray(bytearray(b), dtype=np.uint8)
    return _exchange_rb(cv2.imdecode(vec, flag))


def download(url, flag=cv2.IMREAD_UNCHANGED):
    '''Downloads an image from a URL.

    Args:
        url: the URL of the image
        flag: an optional OpenCV image format flag. By default, the image is
            returned in its raw format

    Returns:
        A uint8 numpy array containing the image
    '''
    return decode(etaw.download_file(url), flag=flag)


def read(path, flag=cv2.IMREAD_UNCHANGED):
    '''Reads image from path.

    Args:
        path: the path to the image on disk
        flag: an optional OpenCV image format flag. By default, the image is
            returned in its native format (color, grayscale, transparent, ...)

    Returns:
        A uint8 numpy array containing the image
        '''
    return _exchange_rb(cv2.imread(path, flag))


def write(img, path):
    '''Writes image to file. The output directory is created if necessary.

    Args:
        img: a numpy array
        path: the output path
    '''
    etau.ensure_basedir(path)
    cv2.imwrite(path, _exchange_rb(img))


###### Image Manipulation #####################################################


def create(width, height, background=None):
    '''Creates a blank image and optionally fills it with a color.

    Args:
        width (int): width of the image to create in pixels
        height (int): height of the image to create in pixels
        background (string): hex RGB (eg, "#ffffff")
    '''
    image = np.zeros((height, width, 3), dtype=np.uint8)

    if background:
        image[:] = hex_to_rgb(background)

    return image


def overlay(im1, im2, x0=0, y0=0):
    '''Overlays im2 onto im1 at the specified coordinates.

    *** Caution: im1 will be modified in-place if possible. ***

    Args:
        im1: a non-transparent image
        im2: a possibly-transparent image
        (x0, y0): the top-left coordinate of im2 in im1 after overlaying, where
            (0, 0) corresponds to the top-left of im1. This coordinate may lie
            outside of im1, in which case some (even all) of im2 may be omitted

    Returns:
        the overlaid image
    '''
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    # Active slice of im1
    y1t = np.clip(y0, 0, h1)
    y1b = np.clip(y0 + h2, 0, h1)
    x1l = np.clip(x0, 0, w1)
    x1r = np.clip(x0 + w2, 0, w1)
    y1 = slice(y1t, y1b)
    x1 = slice(x1l, x1r)

    # Active slice of im2
    y2t = np.clip(y1t - y0, 0, h2)
    y2b = y2t + y1b - y1t
    x2l = np.clip(x1l - x0, 0, w2)
    x2r = x2l + x1r - x1l
    y2 = slice(y2t, y2b)
    x2 = slice(x2l, x2r)

    if im2.shape[2] == 4:
        # Mix transparent image
        im1 = to_double(im1)
        im2 = to_double(im2)
        alpha = im2[y2, x2, 3][:, :, np.newaxis]
        im1[y1, x1, :] *= (1 - alpha)
        im1[y1, x1, :] += alpha * im2[y2, x2, :3]
        im1 = np.uint8(255 * im1)
    else:
        # Insert opaque image
        im1[y1, x1, :] = im2[y2, x2, :]

    return im1


def rasterize(vector_path, width):
    '''Renders a vector image as a raster image with the given width,
    in pixels.
    '''
    with etau.TempDir() as d:
        try:
            png_path = os.path.join(d, "tmp.png")
            Convert(
                in_opts=["-density", "1200", "-trim"],
                out_opts=["-resize", str(width)],
            ).run(vector_path, png_path)
            return read(png_path)
        except Exception:
            # Fail gracefully
            return None

    # @todo why is it slightly blurry this way?
    # try:
    #     out = Convert(
    #         in_opts=["-density", "1200", "-trim"],
    #         out_opts=["-resize", str(width)],
    #     ).run(vector_path, "png:-")
    #     return read(out)
    # except Exception:
    #     # Fail gracefully
    #     return None


def resize(img, width=None, height=None, *args, **kwargs):
    '''Resizes the given image to the given width and height. At most one
    dimension can be None, in which case the aspect-preserving value is used.
    '''
    if height is None:
        height = int(round(img.shape[0] * (width * 1.0 / img.shape[1])))
    if width is None:
        width = int(round(img.shape[1] * (height * 1.0 / img.shape[0])))
    return cv2.resize(img, (width, height), *args, **kwargs)


def to_double(img):
    '''Converts img to a double precision image with values in [0, 1].'''
    return img.astype(np.float) / np.iinfo(img.dtype).max


class Convert(object):
    '''Interface for the ImageMagick convert binary.'''

    def __init__(
            self,
            executable="convert",
            in_opts=None,
            out_opts=None,
        ):
        '''Constructs a convert command, minus the input/output paths.

        Args:
            executable: the system path to the convert binary
            in_opts: a list of input options for convert
            out_opts: a list of output options for convert
        '''
        self._executable = executable
        self._in_opts = in_opts or []
        self._out_opts = out_opts or []
        self._args = None
        self._p = None

    @property
    def cmd(self):
        '''The last executed convert command string, or None if run() has not
        yet been called.
        '''
        return " ".join(self._args) if self._args else None

    def run(self, inpath, outpath):
        '''Run the convert binary with the specified input/outpath paths.

        Args:
            inpath: the input path
            outpath: the output path. Use "-" or a format like "png:-" to pipe
                output to STDOUT

        Returns:
            out: STDOUT of the convert binary

        Raises:
            ExecutableNotFoundError: if the convert binary cannot be found
            ExecutableRuntimeError: if the convert binary raises an error
                during execution
        '''
        self._args = (
            [self._executable] +
            self._in_opts + [inpath] +
            self._out_opts + [outpath]
        )

        try:
            self._p = Popen(self._args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        except EnvironmentError as e:
            if e.errno == errno.ENOENT:
                raise etau.ExecutableNotFoundError(self._executable)
            else:
                raise

        out, err = self._p.communicate()
        if self._p.returncode != 0:
            raise etau.ExecutableRuntimeError(self.cmd, err)

        return out


###### Image Properties and Representations ###################################


def has_alpha(img):
    '''Checks if the image has an alpha channel.'''
    return not is_gray(img) and img.shape[2] == 4


def is_gray(img):
    '''Checks if the image is grayscale and return True if so.

    The check is performed by counting the number of bands.
    '''
    return len(img.shape) == 2


def to_frame_size(frame_size=None, shape=None, img=None):
    '''Converts an image size representation to a (width, height) tuple.

    Pass *one* keyword argument to compute the frame size.

    Args:
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape
        img: the image itself

    Returns:
        a (width, height) frame size tuple

    Raises:
        TypeError: if none of the keyword arguments were passed
    '''
    if img is not None:
        shape = img.shape
    if shape is not None:
        return shape[1], shape[0]
    elif frame_size is not None:
        return tuple(frame_size)
    else:
        raise TypeError("A valid keyword argument must be provided")


def aspect_ratio(**kwargs):
    '''Computes the aspect ratio of the image.

    Args:
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape
        img: the image itself

    Returns:
        the aspect ratio of the image
    '''
    fs = to_frame_size(**kwargs)
    return fs[0] / fs[1]


def parse_frame_size(frame_size):
    '''Parses the given frame size, ensuring that it is valid.

    Args:
        a (width, height) tuple or list, optionally with dimensions that are
            -1 to indicate "fill-in" dimensions

    Returns:
        the frame size converted to a tuple, if necessary

    Raises:
        ValueError: if the frame size was invalid
    '''
    if isinstance(frame_size, list):
        frame_size = tuple(frame_size)
    if not isinstance(frame_size, tuple):
        raise ValueError(
            "Frame size must be a tuple or list; found '%s'" % str(frame_size))
    if len(frame_size) != 2:
        raise ValueError(
            "frame_size must be a be a (width, height) tuple; found '%s'" %
            str(frame_size))
    return frame_size


def infer_missing_dims(frame_size, ref_size):
    '''Infers the missing entries (if any) of the given frame size.

    Args:
        frame_size: a (width, height) tuple. One or both dimensions can be -1,
            in which case the input aspect ratio is preserved
        ref_size: the reference (width, height)

    Returns:
        the concrete (width, height) with no negative values
    '''
    width, height = frame_size
    kappa = ref_size[0] / ref_size[1]
    if width < 0:
        if height < 0:
            return ref_size
        width = int(round(height * kappa))
    elif height < 0:
        height = int(round(width / kappa))
    return width, height


def scale_frame_size(frame_size, scale):
    '''Scales the frame size by the given factor.

    Args:
        frame_size: a (width, height) tuple
        scale: the desired scale factor

    Returns:
        the scaled (width, height)
    '''
    return tuple(int(round(scale * d)) for d in frame_size)


def clamp_frame_size(frame_size, max_size):
    '''Clamps the frame size to the given maximum size

    Args:
        frame_size: a (width, height) tuple
        max_size: a (max width, max height) tuple. One or both dimensions can
            be -1, in which case no constraint is applied that dimension

    Returns:
        the (width, height) scaled down if necessary so that width <= max width
            and height <= max height
    '''
    alpha = 1
    if max_size[0] > 0:
        alpha = min(alpha, max_size[0] / frame_size[0])
    if max_size[1] > 0:
        alpha = min(alpha, max_size[1] / frame_size[1])
    return scale_frame_size(frame_size, alpha)


class Length(object):
    '''Represents a length along a specified dimension of an image as a
    relative percentage or an absolute pixel count.
    '''

    def __init__(self, length_str, dim):
        '''Builds a Length object.

        Args:
            length_str: a string of the form '<float>%' or '<int>px' describing
                a relative or absolute length, respectively
            dim: the dimension to measure length along
        '''
        self.dim = dim
        if length_str.endswith("%"):
            self.relunits = True
            self.rellength = 0.01 * float(length_str[:-1])
            self.length = None
        elif length_str.endswith("px"):
            self.relunits = False
            self.rellength = None
            self.length = int(length_str[:-2])
        else:
            raise TypeError(
                "Expected '<float>%%' or '<int>px', received '%s'" %
                str(length_str)
            )

    def render(self, frame_size=None, shape=None, img=None):
        '''Returns the length in pixels for the given frame size/shape/img.

        Pass any *one* of the keyword arguments to render the length.

        Args:
            frame_size: the (width, height) of the image
            shape: the (height, width, ...) of the image, e.g. from img.shape
            img: the image itself

        Raises:
            LengthError: if none of the keyword arguments were passed
        '''
        if img is not None:
            shape = img.shape
        elif frame_size is not None:
            shape = frame_size[::-1]
        elif shape is None:
            raise LengthError("One keyword argument must be provided")

        if self.relunits:
            return int(round(self.rellength * shape[self.dim]))
        return self.length


class LengthError(Exception):
    pass


class Width(Length):
    '''Represents the width of an image as a relative percentage or an absolute
    pixel count.
    '''

    def __init__(self, width_str):
        '''Builds a Width object.

        Args:
            width_str: a string of the form '<float>%' or '<int>px' describing
                a relative or absolute width, respectively
        '''
        super(Width, self).__init__(width_str, 1)


class Height(Length):
    '''Represents the height of an image as a relative percentage or an
    absolute pixel count.
    '''

    def __init__(self, height_str):
        '''Builds a Height object.

        Args:
            height_str: a string of the form '<float>%' or '<int>px' describing
                a relative or absolute height, respectively
        '''
        super(Height, self).__init__(height_str, 0)


class Location(object):
    '''Represents a location in an image.'''

    # Valid loc strings
    TOP_LEFT = ["top-left", "tl"]
    TOP_RIGHT = ["top-right", "tr"]
    BOTTOM_RIGHT = ["bottom-right", "br"]
    BOTTOM_LEFT = ["bottom-left", "bl"]

    def __init__(self, loc):
        '''Constructs a Location object.

        Args:
            loc: a (case-insenstive) string specifying a location
                ["top-left", "top-right", "bottom-right", "bottom-left"]
                ["tl", "tr", "br", "bl"]
        '''
        self._loc = loc.lower()

    @property
    def is_top_left(self):
        '''True if the location is top left, otherwise False.'''
        return self._loc in self.TOP_LEFT

    @property
    def is_top_right(self):
        '''True if the location is top right, otherwise False.'''
        return self._loc in self.TOP_RIGHT

    @property
    def is_bottom_right(self):
        '''True if the location is bottom right, otherwise False.'''
        return self._loc in self.BOTTOM_RIGHT

    @property
    def is_bottom_left(self):
        '''True if the location is bottom left, otherwise False.'''
        return self._loc in self.BOTTOM_LEFT


###### Image Composition ######################################################


def best_tiling_shape(n, kappa=1.777, **kwargs):
    '''Computes the (width, height) of the best tiling of n images in a grid
    such that the composite image would have roughly the specified aspect
    ratio.

    The returned tiling always satisfies width * height >= n.

    Args:
        n: the number of images to tile
        kappa: the desired aspect ratio of the composite image. By default,
            this is 1.777
        **kwargs: a valid keyword argument for to_frame_size(). By default,
            square images are assumed

    Returns:
        the (width, height) of the best tiling
    '''
    alpha = aspect_ratio(**kwargs) if kwargs else 1.0

    def _cost(w, h):
        return (alpha * w - kappa * h) ** 2 + (w * h - n) ** 2

    def _best_width_for_height(h):
        w = np.arange(int(np.ceil(n / h)), n + 1)
        return w[np.argmin(_cost(w, h))]

    # Caution: this is O(n^2)
    hh = np.arange(1, n + 1)
    ww = np.array([_best_width_for_height(h) for h in hh])
    idx = np.argmin(_cost(ww, hh))
    return  ww[idx], hh[idx]


def tile_images(imgs, width, height, fill_value=0):
    '''Tiles the images in the given array into a grid of the given width and
    height (row-wise).

    If fewer than width * height images are provided, the remaining tiles are
    filled with blank images.

    Args:
        imgs: a list (or num_images x height x width x num_channels numpy
            array) of same-size images
        width: the desired grid width
        height: the desired grid height
        fill_value: a value to fill any blank chips in the tiled image

    Returns:
        the tiled image
    '''
    # Parse images
    imgs = np.asarray(imgs)
    num_imgs = len(imgs)
    if num_imgs == 0:
        raise ValueError("Must have at least one image to tile")

    # Pad with blank images, if necessary
    num_blanks = width * height - num_imgs
    if num_blanks < 0:
        raise ValueError(
            "Cannot tile %d images in a %d x %d grid" %
            (num_imgs, width, height))
    if num_blanks > 0:
        blank = np.full_like(imgs[0], fill_value)
        blanks = np.repeat(blank[np.newaxis, ...], num_blanks, axis=0)
        imgs = np.concatenate((imgs, blanks), axis=0)

    # Tile images
    rows = [
        np.concatenate(imgs[(i * width):((i + 1) * width)], axis=1)
        for i in range(height)
    ]
    return np.concatenate(rows, axis=0)


###### Color Conversions ######################################################


def hex_to_rgb(value):
    '''Converts "#rrbbgg" to a (red, green, blue) tuple.'''
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_bgr(value):
    '''Converts "#rrbbgg" to a (blue, green, red) tuple.'''
    return hex_to_rgb(value)[::-1]


def rgb_to_hex(red, green, blue):
    '''Converts (red, green, blue) to a "#rrbbgg" string.'''
    return "#%02x%02x%02x" % (red, green, blue)


def bgr_to_hex(blue, green, red):
    '''Converts (blue, green, red) to a "#rrbbgg" string.'''
    return rgb_to_hex(red, green, blue)


def rgb_to_gray(img):
    '''Converts the input RGB image to a grayscale image.'''
    if is_gray(img):
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def bgr_to_gray(img):
    '''Converts the input BGR image to a grayscale image.'''
    if is_gray(img):
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gray_to_bgr(img):
    '''Convert a grayscale image to an BGR image.'''
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def gray_to_rgb(img):
    '''Convert a grayscale image to an RGB image.'''
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def rgb_to_bgr(img):
    '''Converts an RGB image to a BGR image (supports alpha).'''
    return _exchange_rb(img)


def bgr_to_rgb(img):
    '''Converts a BGR image to an RGB image (supports alpha).'''
    return _exchange_rb(img)


def _exchange_rb(img):
    '''Converts an image from opencv format (BGR) to/from eta format (RGB) by
    exchanging the red and blue channels.

    This is a symmetric procedure and hence only needs one function.

    Handles gray (pass-through), 3- and 4-channel images.
    '''
    if is_gray(img):
        return img
    return img[..., [2, 1, 0] + list(range(3, img.shape[2]))]
