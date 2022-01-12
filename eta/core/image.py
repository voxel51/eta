"""
Core tools and data structures for working with images.

Notes::

    [image format] ETA stores images exclusively in RGB format. In contrast,
        OpenCV stores its images in BGR format, so all images that are read or
        produced outside of this library must be converted to RGB. This
        conversion can be done via `eta.core.image.bgr_to_rgb()`

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

import colorsys
import errno
import itertools
import logging
import os
import operator
from subprocess import Popen, PIPE

import cv2
import numpy as np
from skimage import measure

import eta
from eta.core.frames import FrameLabels, FrameLabelsSchema
import eta.core.geometry as etag
import eta.core.labels as etal
import eta.core.polylines as etap
import eta.core.serial as etas
import eta.core.utils as etau
import eta.core.web as etaw


logger = logging.getLogger(__name__)


#
# The file extensions of supported image files. Use LOWERCASE!
#
# In practice, any image that `cv2.imread` can read will be supported.
# Nonetheless, we enumerate this list here so that the ETA type system can
# verify the extension of an image provided to a pipeline at build time.
#
# This list was taken from
# https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#imread
#
SUPPORTED_IMAGE_FORMATS = {
    ".bmp",
    ".dib",
    ".jp2",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".pbm",
    ".pgm",
    ".png",
    ".ppm",
    ".ras",
    ".sr",
    ".tif",
    ".tiff",
}


def is_image_mime_type(filepath):
    """Determines whether the given file has a `image` MIME type.

    Args:
        filepath: the path to the file

    Returns:
        True/False
    """
    return etau.guess_mime_type(filepath).startswith("image")


def is_supported_image(filepath):
    """Determines whether the given file has a supported image type."""
    return os.path.splitext(filepath)[1].lower() in SUPPORTED_IMAGE_FORMATS


def glob_images(dir_):
    """Returns an iterator over all supported image files in the directory."""
    return etau.multiglob(
        *SUPPORTED_IMAGE_FORMATS, root=os.path.join(dir_, "*")
    )


def make_image_sequence_patt(basedir, basename="", patt=None, ext=None):
    """Makes an image sequence pattern of the following form:

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
    """
    name = basename + "-" if basename else ""
    patt = patt or eta.config.default_sequence_idx
    ext = ext or eta.config.default_image_ext
    return os.path.join(basedir, name + patt + ext)


class ImageLabels(FrameLabels):
    """Class encapsulating labels for an image.

    ImageLabels are spatial concepts that describe a collection of information
    about a specific image. ImageLabels can have frame-level attributes,
    object detections, event detections, and segmentation masks.

    Attributes:
        filename: (optional) the filename of the image
        metadata: (optional) an ImageMetadata describing metadata about the
            image
        mask: (optiona) a segmentation mask for the image
        mask_index: (optional) a MaskIndex describing the semantics of the
            segmentation mask
        attrs: an AttributeContainer of attributes of the image
        objects: a DetectedObjectContainer of objects in the image
        keypoints: a KeypointsContainer of keypoints in the image
        polylines: a PolylineContainer of polylines in the image
        events: a DetectedEventContainer of events in the image
    """

    def __init__(self, filename=None, metadata=None, **kwargs):
        """Creates an ImageLabels instance.

        Args:
            filename: (optional) the filename of the image
            metadata: (optional) an ImageMetadata instance describing metadata
                about the image
            **kwargs: valid keyword arguments for FrameLabels(**kwargs)
        """
        self.filename = filename
        self.metadata = metadata
        kwargs.pop(
            "frame_number", None
        )  # ImageLabels don't use `frame_number`
        super(ImageLabels, self).__init__(**kwargs)

    @property
    def has_filename(self):
        """Whether the image has a filename."""
        return self.filename is not None

    @property
    def has_metadata(self):
        """Whether the image has metadata."""
        return self.metadata is not None

    def merge_labels(self, frame_labels, reindex=False):
        """Merges the given FrameLabels into this labels.

        Args:
            frame_labels: a FrameLabels
            reindex: whether to offset the `index` fields of objects and events
                in `frame_labels` before merging so that all indices are
                unique. The default is False
        """
        super(ImageLabels, self).merge_labels(frame_labels, reindex=reindex)

        if isinstance(frame_labels, ImageLabels):
            if frame_labels.has_filename and not self.has_filename:
                self.filename = frame_labels.filename
            if frame_labels.has_metadata and not self.has_metadata:
                self.metadata = frame_labels.metadata

    @classmethod
    def from_frame_labels(cls, frame_labels, filename=None, metadata=None):
        """Constructs an ImageLabels from a FrameLabels.

        Args:
            frame_labels: a FrameLabels instance
            filename: an optional filename for the image
            metadata: an optional ImageMetadata instance for the image

        Returns:
            an ImageLabels instance
        """
        return cls(
            filename=filename,
            metadata=metadata,
            mask=frame_labels.mask,
            mask_index=frame_labels.mask_index,
            attrs=frame_labels.attrs,
            objects=frame_labels.objects,
            keypoints=frame_labels.keypoints,
            polylines=frame_labels.polylines,
            events=frame_labels.events,
        )

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        """
        _attrs = []
        if self.filename:
            _attrs.append("filename")
        if self.metadata:
            _attrs.append("metadata")
        _attrs.extend(super(ImageLabels, self).attributes())
        return _attrs

    @classmethod
    def from_dict(cls, d):
        """Constructs an ImageLabels from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            an ImageLabels
        """
        filename = d.get("filename", None)

        metadata = d.get("metadata", None)
        if metadata is not None:
            metadata = ImageMetadata.from_dict(metadata)

        return super(ImageLabels, cls).from_dict(
            d, filename=filename, metadata=metadata
        )


class ImageLabelsSchema(FrameLabelsSchema):
    """Schema describing the content of one or more ImageLabels.

    Attributes:
        attrs: an AttributeContainerSchema describing constant attributes of
            the image(s)
        frames: an AttributeContainerSchema describing frame-level attributes
            of the image(s)
        objects: an ObjectContainerSchema describing the objects in the
            image(s)
        keypoints: a KeypointsContainerSchema describing the keypoints in the
            image(s)
        polylines: a PolylineContainerSchema describing the polylines in the
            image(s)
        events: an EventContainerSchema describing the events in the image(s)
    """

    pass


class ImageSetLabels(etal.LabelsSet):
    """Class encapsulating labels for a set of images.

    ImageSetLabels support item indexing by the `filename` of the ImageLabels
    instances in the set.

    ImageSetLabels instances behave like defaultdicts: new ImageLabels
    instances are automatically created if a non-existent filename is accessed.

    ImageLabels without filenames may be added to the set, but they cannot be
    accessed by `filename`-based lookup.

    Attributes:
        images: an OrderedDict of ImageLabels with filenames as keys
        schema: an ImageLabelsSchema describing the schema of the labels
    """

    _ELE_ATTR = "images"
    _ELE_KEY_ATTR = "filename"
    _ELE_CLS = ImageLabels
    _ELE_CLS_FIELD = "_LABELS_CLS"

    def sort_by_filename(self, reverse=False):
        """Sorts the ImageLabels in this instance by filename.

        ImageLabels without filenames are always put at the end of the set.

        Args:
            reverse: whether to sort in reverse order. By default, this is
                False
        """
        self.sort_by("filename", reverse=reverse)

    def clear_frame_attributes(self):
        """Removes all frame attributes from all ImageLabels in the set."""
        for image_labels in self:
            image_labels.clear_frame_attributes()

    def clear_objects(self):
        """Removes all `DetectedObject`s from all ImageLabels in the set."""
        for image_labels in self:
            image_labels.clear_objects()

    def get_filenames(self):
        """Returns the set of filenames of ImageLabels in the set.

        Returns:
            the set of filenames
        """
        return set(il.filename for il in self if il.filename)

    def remove_objects_without_attrs(self, labels=None):
        """Removes `DetectedObject`s from the ImageLabels in the set that do
        not have attributes.

        Args:
            labels: an optional list of DetectedObject label strings to which
                to restrict attention when filtering. By default, all objects
                are processed
        """
        for image_labels in self:
            image_labels.remove_objects_without_attrs(labels=labels)

    @classmethod
    def from_image_labels_patt(cls, image_labels_patt):
        """Creates an ImageSetLabels from a pattern of ImageLabels files.

        Args:
             image_labels_patt: a pattern with one or more numeric sequences
                for ImageLabels files on disk

        Returns:
            an ImageSetLabels instance
        """
        return cls.from_labels_patt(image_labels_patt)


class BigImageSetLabels(ImageSetLabels, etas.BigSet):
    """An `eta.core.serial.BigSet` of ImageLabels.

    Behaves identically to ImageSetLabels except that each ImageLabels is
    stored on disk.

    BigImageSetLabels store a `backing_dir` attribute that specifies the path
    on disk to the serialized elements. If a backing directory is explicitly
    provided, the directory will be maintained after the BigImageSetLabels
    object is deleted; if no backing directory is specified, a temporary
    backing directory is used and is deleted when the BigImageSetLabels
    instance is garbage collected.

    Attributes:
        images: an OrderedDict whose keys are filenames and whose values are
            uuids for locating ImageLabels on disk
        schema: an ImageLabelsSchema describing the schema of the labels
        backing_dir: the backing directory in which the ImageLabels
            are/will be stored
    """

    def __init__(self, images=None, schema=None, backing_dir=None):
        """Creates a BigImageSetLabels instance.

        Args:
            images: an optional dictionary or list of (key, uuid) tuples for
                elements in the set
            schema: an optional ImageLabelsSchema to enforce on the object.
                By default, no schema is enforced
            backing_dir: an optional backing directory in which the ImageLabels
                are/will be stored. If omitted, a temporary backing directory
                is used
        """
        self.schema = schema
        etas.BigSet.__init__(self, backing_dir=backing_dir, images=images)

    def empty_set(self):
        """Returns an empty in-memory ImageSetLabels version of this
        BigImageSetLabels.

        Returns:
            an empty ImageSetLabels
        """
        return ImageSetLabels(schema=self.schema)

    def filter_by_schema(self, schema):
        """Filters the labels in the set by the given schema.

        Args:
            schema: an ImageLabelsSchema
        """
        for key in self.keys():
            image_labels = self[key]
            image_labels.filter_by_schema(schema)
            self[key] = image_labels

    def remove_objects_without_attrs(self, labels=None):
        """Removes `DetectedObject`s from the ImageLabels in the set that do
        not have attributes.

        Args:
            labels: an optional list of DetectedObject label strings to which
                to restrict attention when filtering. By default, all objects
                are processed
        """
        for key in self.keys():
            image_labels = self[key]
            image_labels.remove_objects_without_attrs(labels=labels)
            self[key] = image_labels


def decode(b, include_alpha=False, flag=None):
    """Decodes an image from raw bytes.

    By default, images are returned as color images with no alpha channel.

    Args:
        bytes: the raw bytes of an image, e.g., from read() or from a web
            download
        include_alpha: whether to include the alpha channel of the image, if
            present, in the returned array. By default, this is False
        flag: an optional OpenCV image format flag to use. If provided, this
            flag takes precedence over `include_alpha`

    Returns:
        a uint8 numpy array containing the image
    """
    flag = _get_opencv_imread_flag(flag, include_alpha)
    vec = np.asarray(bytearray(b), dtype=np.uint8)
    return _exchange_rb(cv2.imdecode(vec, flag))


def download(url, include_alpha=False, flag=None):
    """Downloads an image from a URL.

    By default, images are returned as color images with no alpha channel.

    Args:
        url: the URL of the image
        include_alpha: whether to include the alpha channel of the image, if
            present, in the returned array. By default, this is False
        flag: an optional OpenCV image format flag to use. If provided, this
            flag takes precedence over `include_alpha`

    Returns:
        a uint8 numpy array containing the image
    """
    bytes = etaw.download_file(url)
    return decode(bytes, include_alpha=include_alpha, flag=flag)


def read(path_or_url, include_alpha=False, flag=None):
    """Reads image from a file path or URL.

    By default, images are returned as color images with no alpha channel.

    Args:
        path_or_url: the file path or URL to the image
        include_alpha: whether to include the alpha channel of the image, if
            present, in the returned array. By default, this is False
        flag: an optional OpenCV image format flag to use. If provided, this
            flag takes precedence over `include_alpha`

    Returns:
        a uint8 numpy array containing the image
    """
    if etaw.is_url(path_or_url):
        return download(path_or_url, include_alpha=include_alpha, flag=flag)

    flag = _get_opencv_imread_flag(flag, include_alpha)
    img_bgr = cv2.imread(path_or_url, flag)
    if img_bgr is None:
        raise OSError("Image not found '%s'" % path_or_url)

    return _exchange_rb(img_bgr)


def write(img, path):
    """Writes image to file. The output directory is created if necessary.

    Args:
        img: a numpy array
        path: the output path
    """
    etau.ensure_basedir(path)
    cv2.imwrite(path, _exchange_rb(img))


def encode(img, ext):
    """Encodes the given image.

    Args:
        img: a numpy array
        ext: the image extension, e.g., `".jpg"`

    Returns:
        the encoded image bytes
    """
    return cv2.imencode(ext, _exchange_rb(img))[1].tobytes()


def _get_opencv_imread_flag(flag, include_alpha):
    if flag is not None:
        return flag
    if include_alpha:
        return cv2.IMREAD_UNCHANGED
    return cv2.IMREAD_COLOR


class ImageMetadata(etas.Serializable):
    """Class encapsulating metadata about an image.

    Attributes:
        frame_size: the [width, height] of the image
        num_channels: the number of channels in the image
        size_bytes: the size of the image file on disk, in bytes
        mime_type: the MIME type of the image
    """

    def __init__(
        self,
        frame_size=None,
        num_channels=None,
        size_bytes=None,
        mime_type=None,
    ):
        """Constructs an ImageMetadata instance. All args are optional.

        Args:
            frame_size: the [width, height] of the image
            num_channels: the number of channels in the image
            size_bytes: the size of the image file on disk, in bytes
            mime_type: the MIME type of the image
        """
        self.frame_size = frame_size
        self.num_channels = num_channels
        self.size_bytes = size_bytes
        self.mime_type = mime_type

    def attributes(self):
        """Returns the list of class attributes that will be serialized."""
        _attrs = ["frame_size", "num_channels", "size_bytes", "mime_type"]
        # Exclude attributes that are None
        return [a for a in _attrs if getattr(self, a) is not None]

    @classmethod
    def build_for(cls, filepath):
        """Builds an ImageMetadata object for the given image.

        Args:
            filepath: the path to the image on disk

        Returns:
            an ImageMetadata instance
        """
        img = read(filepath, include_alpha=True)
        return cls(
            frame_size=to_frame_size(img=img),
            num_channels=img.shape[2] if len(img.shape) > 2 else 1,
            size_bytes=os.path.getsize(filepath),
            mime_type=etau.guess_mime_type(filepath),
        )

    @classmethod
    def from_dict(cls, d):
        """Constructs an ImageMetadata from a JSON dictionary."""
        return cls(
            frame_size=d.get("frame_size", None),
            num_channels=d.get("num_channels", None),
            size_bytes=d.get("size_bytes", None),
            mime_type=d.get("mime_type", None),
        )


def create(width, height, background=None):
    """Creates a blank image and optionally fills it with a color.

    Args:
        width: the width of the image, in pixels
        height: the height of the image, in pixels
        background: hex RGB (e.g., "#ffffff")

    Returns:
        the image
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if background:
        img[:] = hex_to_rgb(background)

    return img


def overlay(im1, im2, x0=0, y0=0):
    """Overlays im2 onto im1 at the specified coordinates.

    Args:
        im1: a non-transparent image
        im2: a possibly-transparent image
        (x0, y0): the top-left coordinate of im2 in im1 after overlaying, where
            (0, 0) corresponds to the top-left of im1. This coordinate may lie
            outside of im1, in which case some (even all) of im2 may be omitted

    Returns:
        a copy of im1 with im2 overlaid
    """
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
        im1[y1, x1, :] *= 1 - alpha
        im1[y1, x1, :] += alpha * im2[y2, x2, :3]
        im1 = np.uint8(255 * im1)
    else:
        # Insert opaque image
        im1 = np.copy(im1)
        im1[y1, x1, :] = im2[y2, x2, :]

    return im1


def rasterize(vector_path, width, include_alpha=True, flag=None):
    """Renders a vector image as a raster image with the given pixel width.

    By default, the image is returned with an alpha channel, if possible.

    Args:
        vector_path: the path to the vector image
        width: the desired image width
        include_alpha: whether to include the alpha channel of the image, if
            present, in the returned array. By default, this is True
        flag: an optional OpenCV image format flag to use. If provided, this
            flag takes precedence over `include_alpha`

    Returns:
        a uint8 numpy array containing the rasterized image
    """
    with etau.TempDir() as d:
        try:
            png_path = os.path.join(d, "tmp.png")
            Convert(
                in_opts=["-density", "1200", "-trim"],
                out_opts=["-resize", str(width)],
            ).run(vector_path, png_path)
            return read(png_path, include_alpha=include_alpha, flag=flag)
        except Exception:
            # Fail gracefully
            return None

    # @todo why is it slightly blurry this way?
    # try:
    #     out = Convert(
    #         in_opts=["-density", "1200", "-trim"],
    #         out_opts=["-resize", str(width)],
    #     ).run(vector_path, "png:-")
    #     return read(out, include_alpha=include_alpha, flag=flag)
    # except Exception:
    #     # Fail gracefully
    #     return None


def resize(img, width=None, height=None, *args, **kwargs):
    """Resizes the given image to the given width and height.

    At most one dimension can be None or negative, in which case the
    aspect-preserving value is used.

    Args:
        img: an image
        width: the desired image width
        height: the desired image height
        *args: valid positional arguments for `cv2.resize()`
        **kwargs: valid keyword arguments for `cv2.resize()`

    Returns:
        the resized image
    """
    ih, iw = img.shape[:2]

    if height is None or height < 0:
        height = int(round(ih * (width * 1.0 / iw)))

    if width is None or width < 0:
        width = int(round(iw * (height * 1.0 / ih)))

    return cv2.resize(img, (width, height), *args, **kwargs)


def resize_to_fit_max(img, max_dim, *args, **kwargs):
    """Resizes the given image, if necessary, so that its largest dimension is
    exactly equal to the specified value.

    The aspect ratio of the input image is preserved.

    Args:
        img: an image
        max_dim: the maximum dimension
        *args: valid positional arguments for `cv2.resize()`
        **kwargs: valid keyword arguments for `cv2.resize()`

    Returns:
        the fitted image
    """
    width, height = to_frame_size(img=img)

    alpha = max_dim * 1.0 / max(width, height)
    width = int(round(alpha * width))
    height = int(round(alpha * height))
    return resize(img, width=width, height=height, *args, **kwargs)


def resize_to_fit_min(img, min_dim, *args, **kwargs):
    """Resizes the given image, if necessary, so that its smallest dimension is
    exactly equal to the specified value.

    The aspect ratio of the input image is preserved.

    Args:
        img: an image
        min_dim: the minimum dimension
        *args: valid positional arguments for `cv2.resize()`
        **kwargs: valid keyword arguments for `cv2.resize()`

    Returns:
        the fitted image
    """
    width, height = to_frame_size(img=img)

    alpha = min_dim * 1.0 / min(width, height)
    width = int(round(alpha * width))
    height = int(round(alpha * height))
    return resize(img, width=width, height=height, *args, **kwargs)


def resize_to_even(img, *args, **kwargs):
    """Minimally resizes the given image, if necessary, so that its dimensions
    are even.

    Args:
        img: an image
        *args: valid positional arguments for `cv2.resize()`
        **kwargs: valid keyword arguments for `cv2.resize()`

    Returns:
        the resized image with even dimensions
    """
    width, height = to_frame_size(img=img)

    should_resize = False

    if width % 2:
        width -= 1
        should_resize = True

    if height % 2:
        height -= 1
        should_resize = True

    if should_resize:
        img = resize(img, width=width, height=height, *args, **kwargs)

    return img


def expand(
    img, min_width=None, min_height=None, min_dim=None, *args, **kwargs
):
    """Resizes the given image, if necesary, so that its width and height are
    greater than or equal to the specified minimum values.

    The aspect ratio of the input image is preserved.

    Args:
        img: an image
        min_width: the minimum width
        min_height: the minimum height
        min_dim: the minimum width and height
        *args: valid positional arguments for `cv2.resize()`
        **kwargs: valid keyword arguments for `cv2.resize()`

    Returns:
        the expanded (if necessary) image
    """
    if min_dim is not None:
        min_width = min_dim
        min_height = min_dim

    iw, ih = to_frame_size(img=img)
    ow, oh = iw, ih

    if ow < min_width:
        oh = int(round(oh * (min_width / ow)))
        ow = min_width

    if oh < min_height:
        ow = int(round(ow * (min_height / oh)))
        oh = min_height

    if (ow > iw) or (oh > ih):
        img = resize(img, width=ow, height=oh, *args, **kwargs)

    return img


def contract(
    img, max_width=None, max_height=None, max_dim=None, *args, **kwargs
):
    """Resizes the given image, if necesary, so that its width and height are
    less than or equal to the specified maximum values.

    The aspect ratio of the input image is preserved.

    Args:
        img: an image
        max_width: the maximum width
        max_height: the maximum height
        max_dim: the maximum width and width
        *args: valid positional arguments for `cv2.resize()`
        **kwargs: valid keyword arguments for `cv2.resize()`

    Returns:
        the contracted (if necessary) image
    """
    if max_dim is not None:
        max_width = max_dim
        max_height = max_dim

    iw, ih = to_frame_size(img=img)
    ow, oh = iw, ih

    if ow > max_width:
        oh = int(round(oh * (max_width / ow)))
        ow = max_width

    if oh > max_height:
        ow = int(round(ow * (max_height / oh)))
        oh = max_height

    if (ow < iw) or (oh < ih):
        img = resize(img, width=ow, height=oh, *args, **kwargs)

    return img


def central_crop(img, frame_size=None, shape=None):
    """Extracts a centered crop of the required size from the given image.

    The image is resized as necessary if the requested size is larger than the
    resolution of the input image.

    Pass *one* keyword argument to this function.

    Args:
        img: the input image
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape

    Returns:
        A cropped portion of the image of height `h` and width `w`.
    """
    width, height = to_frame_size(frame_size=frame_size, shape=shape)

    # Expand image, if necessary
    img = expand(img, min_width=width, min_height=height)

    # Extract central crop
    bounding = (height, width)
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def render_frame_mask(
    mask, frame_size=None, shape=None, img=None, as_bool=False
):
    """Renders the given frame mask for an image of the given dimensions.

    The pixel values of the frame mask will be preserved (i.e., no new pixel
    values will be added).

    One of `frame_size`, `shape`, and `img` must be provided.

    Args:
        mask: the frame mask
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape
        img: the image itself
        as_bool: whether to return the mask as a boolean image (True) or a
            uint8 image (False). The default is False

    Returns:
        the resized frame mask
    """
    width, height = to_frame_size(frame_size=frame_size, shape=shape, img=img)

    mask = np.asarray(mask, dtype=np.uint8)
    mask = resize(
        mask, width=width, height=height, interpolation=cv2.INTER_NEAREST
    )

    if as_bool:
        mask = mask.astype(bool)

    return mask


def render_instance_mask(
    mask, bounding_box, frame_size=None, shape=None, img=None, as_bool=True
):
    """Renders the given instance mask for an image of the given dimensions
    such that it can be inscribed in the given bounding box.

    One of `frame_size`, `shape`, and `img` must be provided.

    Args:
        mask: the instance mask
        bounding_box: the BoundingBox in which to inscribe the mask
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape
        img: the image itself
        as_bool: whether to return the mask as a boolean image (True) or a
            uint8 image (False). The default is True

    Returns:
        (mask, offset), where `mask` is a rendered version of the input mask
        that can be directly inscribed in the bounding box, and
        `offset = (tlx, tly)` are the coordinates of the top-left corner of the
        mask within the image
    """
    tlx, tly, width, height = bounding_box.coords_in(
        frame_size=frame_size, shape=shape, img=img
    )
    offset = (tlx, tly)

    mask = np.asarray(mask, dtype=np.uint8)

    # Can consider using `interpolation=cv2.INTER_NEAREST` here
    mask = resize(mask, width=width, height=height)

    if as_bool:
        mask = mask.astype(bool)

    return mask, offset


def render_instance_image(
    mask, bounding_box, frame_size=None, shape=None, img=None
):
    """Renders a binary image of the specified size containing the given
    instance mask inscribed in the given bounding box.

    One of `frame_size`, `shape`, and `img` must be provided.

    Args:
        mask: an boolean numpy array defining the instance mask
        bounding_box: the BoundingBox in which to inscribe the mask
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape
        img: the image itself

    Returns:
        a binary instance mask of the specified size
    """
    w, h = to_frame_size(frame_size=frame_size, shape=shape, img=img)
    mask, offset = render_instance_mask(mask, bounding_box, frame_size=(w, h))
    x0, y0 = offset
    dh, dw = mask.shape

    img_mask = np.zeros((h, w), dtype=bool)
    img_mask[y0 : (y0 + dh), x0 : (x0 + dw)] = mask
    return img_mask


def get_contour_band_mask(mask, bandwidth):
    """Returns a mask that traces the contours of the given frame mask with
    thickness specified by the given bandwidth.

    Args:
        mask: a frame mask
        bandwidth: the bandwidth to use to trace each contour in the output
            mask. A typical value for this parameter is 5 pixels

    Returns:
        a binary mask indicating the contour bands
    """
    mask = np.asarray(mask)
    band_mask = np.zeros(mask.shape, dtype=np.uint8)

    for value in np.unique(mask):
        mask_value = (mask == value).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_value, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(band_mask, contours, -1, 1, bandwidth)

    return band_mask.astype(bool)


def convert_object_to_polygon(dobj, tolerance=2, filled=True):
    """Converts the DetectedObject to a Polyline.

    If the object an instance mask, the polyline will trace the boundary of the
    mask; otherwise, the polyline will trace the bounding box itself.

    Args:
        dobj: a DetectedObject
        tolerance (2): a tolerance, in pixels, when generating an approximate
            polyline for the instance mask
        filled (True): whether the polyline should be filled

    Returns:
        a Polyline
    """
    if dobj.has_mask:
        mask_polygons = _mask_to_polygons(dobj.mask, tolerance=tolerance)

        x0 = dobj.bounding_box.top_left.x
        y0 = dobj.bounding_box.top_left.y
        w_box = dobj.bounding_box.width()
        h_box = dobj.bounding_box.height()

        h_mask, w_mask = dobj.mask.shape[:2]

        points = []
        for mask_polygon in mask_polygons:
            ppoints = []
            for x, y in mask_polygon:
                xp = x0 + (x / w_mask) * w_box
                yp = y0 + (y / h_mask) * h_box

                ppoints.append((xp, yp))

            points.append(ppoints)
    else:
        tlx, tly, brx, bry = dobj.bounding_box.to_coords()
        points = [[(tlx, tly), (brx, tly), (brx, bry), (tlx, bry)]]

    return etap.Polyline(
        name=dobj.name,
        label=dobj.label,
        index=dobj.index,
        points=points,
        closed=True,
        filled=filled,
        attrs=dobj.attrs,
    )


def _mask_to_polygons(mask, tolerance=2):
    # Pad mask to close contours of shapes which start and end at an edge
    padded_mask = np.pad(mask, pad_width=1, mode="constant", constant_values=0)

    contours = measure.find_contours(padded_mask, 0.5)
    contours = [c - 1 for c in contours]  # undo padding

    polygons = []
    for contour in contours:
        contour = _close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue

        # After padding and subtracting 1 there may be -0.5 points
        contour = np.maximum(contour, 0)

        contour = contour[:-1]  # store as open contour
        contour = np.flip(contour, axis=1)
        polygons.append(contour.tolist())

    return polygons


def _close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))

    return contour


def render_bounding_box(polyline):
    """Renders a tight BoundingBox around the given Polyline.

    Args:
        polyline: a Polyline

    Returns:
        a BoundingBox
    """
    xx, yy = zip(*list(itertools.chain(*polyline.points)))
    xtl = min(xx)
    ytl = min(yy)
    xbr = max(xx)
    ybr = max(yy)
    return etag.BoundingBox.from_coords(xtl, ytl, xbr, ybr)


def render_bounding_box_and_mask(polyline, mask_size):
    """Renders a tight BoundingBox and instance mask for the given Polyline.

    Args:
        polyline: a Polyline
        mask_size: the `(width, height)` at which to render the mask

    Returns:
        a `(BoundingBox, mask)` tuple
    """
    # Compute bounding box
    bounding_box = render_bounding_box(polyline)

    # Compute absolute coordinates within `mask_size` image on bounding box
    xtl, ytl, xbr, ybr = bounding_box.to_coords()
    w_box = xbr - xtl
    h_box = ybr - ytl
    w_mask, h_mask = mask_size
    abs_points = []
    for shape in polyline.points:
        abs_shape = []
        for x, y in shape:
            xabs = int(round(((x - xtl) / w_box) * w_mask))
            yabs = int(round(((y - ytl) / h_box) * h_mask))
            abs_shape.append((xabs, yabs))

        abs_points.append(abs_shape)

    # Render mask

    mask = np.zeros((h_mask, w_mask), dtype=np.uint8)
    abs_points = [np.array(shape, dtype=np.int32) for shape in abs_points]

    if polyline.filled:
        # Note: this function handles closed vs not closed automatically
        mask = cv2.fillPoly(mask, abs_points, 255)
    else:
        mask = cv2.polylines(
            mask, abs_points, polyline.closed, 255, thickness=1
        )

    mask = mask.astype(bool)

    return bounding_box, mask


def to_double(img):
    """Converts the given image to a double precision (float) image with values
    in [0, 1].

    Args:
        img: an image

    Returns:
        a copy of the image in double precision format
    """
    return img.astype(np.float) / np.iinfo(img.dtype).max


def to_float(img):
    """Converts the given image to a single precision (float32) image with
    values in [0, 1].

    Args:
        img: an image

    Returns:
        a copy of the image in single precision format
    """
    return img.astype(np.float32) / np.iinfo(img.dtype).max


class Convert(object):
    """Interface for the ImageMagick convert binary."""

    def __init__(
        self, executable="convert", in_opts=None, out_opts=None,
    ):
        """Constructs a convert command, minus the input/output paths.

        Args:
            executable: the system path to the convert binary
            in_opts: a list of input options for convert
            out_opts: a list of output options for convert
        """
        self._executable = executable
        self._in_opts = in_opts or []
        self._out_opts = out_opts or []
        self._args = None
        self._p = None

    @property
    def cmd(self):
        """The last executed convert command string, or None if run() has not
        yet been called.
        """
        return " ".join(self._args) if self._args else None

    def run(self, inpath, outpath):
        """Run the convert binary with the specified input/outpath paths.

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
        """
        self._args = (
            [self._executable]
            + self._in_opts
            + [inpath]
            + self._out_opts
            + [outpath]
        )

        try:
            self._p = Popen(self._args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        except EnvironmentError as e:
            if e.errno == errno.ENOENT:
                raise etau.ExecutableNotFoundError(exe=self._executable)
            raise

        out, err = self._p.communicate()
        if self._p.returncode != 0:
            raise etau.ExecutableRuntimeError(self.cmd, err)

        return out


def has_alpha(img):
    """Checks if the image has an alpha channel.

    Args:
        img: an image

    Returns:
        True/False
    """
    return img.ndim == 4


def is_gray(img):
    """Checks if the image is grayscale, i.e., has exactly two channels.

    Args:
        img: an image

    Returns:
        True/False
    """
    return img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)


def is_color(img):
    """Checks if the image is color, i.e., has at least three channels.

    Args:
        img: an image

    Returns:
        True/False
    """
    return img.ndim > 2


def to_frame_size(frame_size=None, shape=None, img=None):
    """Converts an image size representation to a (width, height) tuple.

    Pass *one* keyword argument to compute the frame size.

    Args:
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape
        img: the image itself

    Returns:
        a (width, height) frame size tuple

    Raises:
        TypeError: if none of the keyword arguments were passed
    """
    if img is not None:
        shape = img.shape

    if shape is not None:
        return shape[1], shape[0]

    if frame_size is not None:
        return tuple(frame_size)

    raise TypeError("A valid keyword argument must be provided")


def aspect_ratio(**kwargs):
    """Computes the aspect ratio of the image.

    Args:
        frame_size: the (width, height) of the image
        shape: the (height, width, ...) of the image, e.g. from img.shape
        img: the image itself

    Returns:
        the aspect ratio of the image
    """
    fs = to_frame_size(**kwargs)
    return fs[0] / fs[1]


def parse_frame_size(frame_size):
    """Parses the given frame size, ensuring that it is valid.

    Args:
        a (width, height) tuple or list, optionally with dimensions that are
            -1 to indicate "fill-in" dimensions

    Returns:
        the frame size converted to a tuple, if necessary

    Raises:
        ValueError: if the frame size was invalid
    """
    if isinstance(frame_size, list):
        frame_size = tuple(frame_size)
    if not isinstance(frame_size, tuple):
        raise ValueError(
            "Frame size must be a tuple or list; found '%s'" % str(frame_size)
        )
    if len(frame_size) != 2:
        raise ValueError(
            "frame_size must be a be a (width, height) tuple; found '%s'"
            % str(frame_size)
        )
    return frame_size


def infer_missing_dims(frame_size, ref_size):
    """Infers the missing entries (if any) of the given frame size.

    Args:
        frame_size: a (width, height) tuple. One or both dimensions can be -1,
            in which case the input aspect ratio is preserved
        ref_size: the reference (width, height)

    Returns:
        the concrete (width, height) with no negative values
    """
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
    """Scales the frame size by the given factor.

    Args:
        frame_size: a (width, height) tuple
        scale: the desired scale factor

    Returns:
        the scaled (width, height)
    """
    return tuple(int(round(scale * d)) for d in frame_size)


def clip_frame_size(frame_size, min_size=None, max_size=None):
    """Clips the frame size to the given minimum and maximum sizes, if
    necessary.

    The aspect ratio of the input frame size is preserved.

    Args:
        frame_size: a (width, height) tuple
        min_size: an optional (min width, min height) tuple. One or both
            dimensions can be -1, in which case no constraint is applied that
            dimension
        max_size: an optional (max width, max height) tuple. One or both
            dimensions can be -1, in which case no constraint is applied that
            dimension

    Returns:
        the (width, height) scaled if necessary so that
            min width <= width <= max width and
            min height <= height <= max height
    """
    alpha = 1

    if min_size is not None:
        if min_size[0] > 0:
            alpha = max(alpha, min_size[0] / frame_size[0])

        if min_size[1] > 0:
            alpha = max(alpha, min_size[1] / frame_size[1])

    if max_size is not None:
        if max_size[0] > 0:
            alpha = min(alpha, max_size[0] / frame_size[0])

        if max_size[1] > 0:
            alpha = min(alpha, max_size[1] / frame_size[1])

    return scale_frame_size(frame_size, alpha)


class Length(object):
    """Represents a length along a specified dimension of an image as a
    relative percentage or an absolute pixel count.
    """

    def __init__(self, length_str, dim):
        """Creates a Length instance.

        Args:
            length_str: a string of the form '<float>%' or '<int>px' describing
                a relative or absolute length, respectively
            dim: the dimension to measure length along
        """
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
                "Expected '<float>%%' or '<int>px', received '%s'"
                % str(length_str)
            )

    def render_for(self, frame_size=None, shape=None, img=None):
        """Returns the length in pixels for the given frame size/shape/img.

        Pass any *one* of the keyword arguments to render the length.

        Args:
            frame_size: the (width, height) of the image
            shape: the (height, width, ...) of the image, e.g. from img.shape
            img: the image itself

        Raises:
            LengthError: if none of the keyword arguments were passed
        """
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
    """Error raised when an invalid Length is encountered."""

    pass


class Width(Length):
    """Represents the width of an image as a relative percentage or an absolute
    pixel count.
    """

    def __init__(self, width_str):
        """Creates a Width instance.

        Args:
            width_str: a string of the form '<float>%' or '<int>px' describing
                a relative or absolute width, respectively
        """
        super(Width, self).__init__(width_str, 1)


class Height(Length):
    """Represents the height of an image as a relative percentage or an
    absolute pixel count.
    """

    def __init__(self, height_str):
        """Creates a Height instance.

        Args:
            height_str: a string of the form '<float>%' or '<int>px' describing
                a relative or absolute height, respectively
        """
        super(Height, self).__init__(height_str, 0)


class Location(object):
    """Represents a location in an image."""

    # Valid loc strings
    TOP_LEFT = ["top-left", "tl"]
    TOP_RIGHT = ["top-right", "tr"]
    BOTTOM_RIGHT = ["bottom-right", "br"]
    BOTTOM_LEFT = ["bottom-left", "bl"]

    def __init__(self, loc):
        """Creates a Location instance.

        Args:
            loc: a (case-insenstive) string specifying a location
                ["top-left", "top-right", "bottom-right", "bottom-left"]
                ["tl", "tr", "br", "bl"]
        """
        self._loc = loc.lower()

    @property
    def is_top_left(self):
        """True if the location is top left, otherwise False."""
        return self._loc in self.TOP_LEFT

    @property
    def is_top_right(self):
        """True if the location is top right, otherwise False."""
        return self._loc in self.TOP_RIGHT

    @property
    def is_bottom_right(self):
        """True if the location is bottom right, otherwise False."""
        return self._loc in self.BOTTOM_RIGHT

    @property
    def is_bottom_left(self):
        """True if the location is bottom left, otherwise False."""
        return self._loc in self.BOTTOM_LEFT


def best_tiling_shape(n, kappa=1.777, **kwargs):
    """Computes the (width, height) of the best tiling of n images in a grid
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
    """
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
    return ww[idx], hh[idx]


def tile_images(imgs, width, height, fill_value=0):
    """Tiles the images in the given array into a grid of the given width and
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
    """
    # Parse images
    imgs = np.asarray(imgs)
    num_imgs = len(imgs)
    if num_imgs == 0:
        raise ValueError("Must have at least one image to tile")

    # Pad with blank images, if necessary
    num_blanks = width * height - num_imgs
    if num_blanks < 0:
        raise ValueError(
            "Cannot tile %d images in a %d x %d grid"
            % (num_imgs, width, height)
        )
    if num_blanks > 0:
        blank = np.full_like(imgs[0], fill_value)
        blanks = np.repeat(blank[np.newaxis, ...], num_blanks, axis=0)
        imgs = np.concatenate((imgs, blanks), axis=0)

    # Tile images
    rows = [
        np.concatenate(imgs[(i * width) : ((i + 1) * width)], axis=1)
        for i in range(height)
    ]
    return np.concatenate(rows, axis=0)


#
# R, G, B: ints in [0, 255], [0, 255], [0, 255]
# B, G, R: ints in [0, 255], [0, 255], [0, 255]
# H, S, V: floats in [0, 1], [0, 1], [0, 1]
# H, L, S: floats in [0, 1], [0, 1], [0, 1]
#


def rgb_to_hsv(r, g, b):
    """Converts (red, green, blue) to a (hue, saturation, value) tuple.

    Args:
        r, g, b: the RGB values

    Returns:
        an H, S, V tuple
    """
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)


def rgb_to_hls(r, g, b):
    """Converts (red, green, blue) to a (hue, lightness, saturation) tuple.

    Args:
        r, g, b: the RGB values

    Returns:
        an H, L, S tuple
    """
    return colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)


def rgb_to_hex(r, g, b):
    """Converts (red, green, blue) to a "#rrbbgg" string.

    Args:
        r, g, b: the RGB values

    Returns:
        a hex string
    """
    return "#%02x%02x%02x" % (r, g, b)


def bgr_to_hsv(b, g, r):
    """Converts (blue, green, red) to a (hue, saturation, value) tuple.

    Args:
        b, g, r: the BGR values

    Returns:
        an H, S, V tuple
    """
    return rgb_to_hsv(r, g, b)


def bgr_to_hls(b, g, r):
    """Converts (blue, green, red) to a (hue, lightness, saturation) tuple.

    Args:
        b, g, r: the BGR values

    Returns:
        an H, L, S tuple
    """
    return rgb_to_hls(r, g, b)


def bgr_to_hex(b, g, r):
    """Converts (blue, green, red) to a "#rrbbgg" string.

    Args:
        b, g, r: the BGR values

    Returns:
        a hex string
    """
    return rgb_to_hex(r, g, b)


def hsv_to_rgb(h, s, v):
    """Converts a (hue, saturation, value) tuple to a (red, green blue) tuple.

    Args:
        h, s, v: the HSV values

    Returns:
        an R, G, B tuple
    """
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(255 * r), int(255 * g), int(255 * b))


def hsv_to_bgr(h, s, v):
    """Converts a (hue, saturation, value) tuple to a (blue, green red) tuple.

    Args:
        h, s, v: the HSV values

    Returns:
        a B, G, R tuple
    """
    return hsv_to_rgb(h, s, v)[::-1]


def hsv_to_hls(h, s, v):
    """Converts a (hue, saturation, value) tuple to a
    (hue, lightness, saturation) tuple.

    Args:
        h, s, v: the HSV values

    Returns:
        an H, L, S tuple
    """
    return rgb_to_hls(*hsv_to_rgb(h, s, v))


def hsv_to_hex(h, s, v):
    """Converts a (hue, saturation, value) tuple to a "#rrbbgg" string.

    Args:
        h, s, v: the HSV values

    Returns:
        a hex string
    """
    return rgb_to_hex(*hsv_to_rgb(h, s, v))


def hls_to_rgb(h, l, s):
    """Converts a (hue, lightness, saturation) tuple to a (red, green blue)
    tuple.

    Args:
        h, l, s: the HLS values

    Returns:
        an R, G, B tuple
    """
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(255 * r), int(255 * g), int(255 * b))


def hls_to_bgr(h, l, s):
    """Converts a (hue, lightness, saturation) tuple to a (blue, green red)
    tuple.

    Args:
        h, l, s: the HLS values

    Returns:
        a B, G, R tuple
    """
    return hls_to_rgb(h, l, s)[::-1]


def hls_to_hsv(h, l, s):
    """Converts a (hue, lightness, saturation) tuple to a
    (hue, saturation, value) tuple.

    Args:
        h, l, s: the HLS values

    Returns:
        an H, S, V tuple
    """
    return rgb_to_hls(*hls_to_rgb(h, l, s))


def hls_to_hex(h, l, s):
    """Converts a (hue, lightness, saturation) tuple to a "#rrbbgg" string.

    Args:
        h, l, s: the HLS values

    Returns:
        a hex string
    """
    return rgb_to_hex(*hls_to_rgb(h, l, s))


def hex_to_rgb(h):
    """Converts a "#rrbbgg" string to a (red, green, blue) tuple.

    Args:
        h: a hex string

    Returns:
        an R, G, B tuple
    """
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def hex_to_bgr(h):
    """Converts a "#rrbbgg" string to a (blue, green, red) tuple.

    Args:
        h: a hex string

    Returns:
        a B, G, R tuple
    """
    return hex_to_rgb(h)[::-1]


def hex_to_hsv(h):
    """Converts a "#rrbbgg" string to a (hue, saturation, value) tuple.

    Args:
        h: a hex string

    Returns:
        an H, S, V tuple
    """
    return rgb_to_hsv(*hex_to_rgb(h))


def hex_to_hls(h):
    """Converts a "#rrbbgg" string to a (hue, lightness, saturation) tuple.

    Args:
        h: a hex string

    Returns:
        an H, L, S tuple
    """
    return rgb_to_hls(*hex_to_rgb(h))


def rgb_to_gray(img):
    """Converts the input RGB image to a grayscale image.

    Args:
        img: an RGB image

    Returns:
        a grayscale image
    """
    if is_gray(img):
        return img

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def bgr_to_gray(img):
    """Converts the input BGR image to a grayscale image.

    Args:
        img: a BGR image

    Returns:
        a grayscale image
    """
    if is_gray(img):
        return img

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gray_to_bgr(img):
    """Convert a grayscale image to an BGR image.

    Args:
        img: a grayscale image

    Returns:
        a BGR image
    """
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def gray_to_rgb(img):
    """Convert a grayscale image to an RGB image.

    Args:
        img: a grayscale image

    Returns:
        an RGB image
    """
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def rgb_to_bgr(img):
    """Converts an RGB image to a BGR image (supports alpha).

    Args:
        img: an RGB image

    Returns:
        a BGR image
    """
    return _exchange_rb(img)


def bgr_to_rgb(img):
    """Converts a BGR image to an RGB image (supports alpha).

    Args:
        img: a BGR image

    Returns:
        an RGB image
    """
    return _exchange_rb(img)


def _exchange_rb(img):
    """Converts an image from BGR to/from RGB format by exchanging the red and
    blue channels.

    Handles gray (passthrough), 3-channel, and 4-channel images.

    Args:
        img: an image

    Returns:
        a copy of the input image with its first and third channels swapped
    """
    if is_gray(img):
        return img

    return img[..., [2, 1, 0] + list(range(3, img.shape[2]))]
