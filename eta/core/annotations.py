'''
Core utilities for rendering annotations on media.

Copyright 2018-2019, Voxel51, Inc.
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

import logging
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import eta.constants as etac
from eta.core.config import Config, Configurable
import eta.core.image as etai
import eta.core.logo as etal
import eta.core.video as etav


logger = logging.getLogger(__name__)


class ColormapConfig(Config):
    '''Configuration class that encapsulates the name of a Colormap and an
    instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the Colormap, e.g.,
            `core.utils.ManualColormap`
        config: an instance of the Config class associated with the specified
            Colormap (e.g., `core.utils.ManualColormapConfig`)
    '''

    def __init__(self, d):
        self.type = self.parse_string(d, "type")
        self._colormap_cls, config_cls = Configurable.parse(self.type)
        self.config = self.parse_object(d, "config", config_cls)

    def build(self):
        '''Factory method that builds the Colormap instance from the config
        specified by this class.
        '''
        return self._colormap_cls(self.config)


class Colormap(Configurable):
    '''Class encapsulating a colormap.'''

    @property
    def colors(self):
        '''The array of hex colors in the colormap.'''
        raise NotImplementedError("subclass must implement colors")

    def get_color(self, index):
        '''Gets the color for the given index. Modular arithmetic is used, so
        any integer is valid regardless of the colormap length.
        '''
        return self.colors[index % len(self.colors)]

    @classmethod
    def load_default(cls):
        '''Loads the default Colormap.'''
        config = ShuffledHLSColormapConfig.builder().set(num_colors=36).build()
        return ShuffledHLSColormap(config)


class ManualColormapConfig(Config):
    '''Class that configures a ManualColormap.'''

    def __init__(self, d):
        self.colors = self.parse_array(d, "colors")


class ManualColormap(Colormap):
    '''A colormap with manually specified colors.'''

    def __init__(self, config):
        '''Creates a ManualColormap instance.

        Args:
            config: a ManualColormapConfig instance
        '''
        self.validate(config)
        self.config = config

    @property
    def colors(self):
        '''The array of hex colors in the colormap.'''
        return self.config.colors


class ShuffledHLSColormapConfig(Config):
    '''Class that configures an ShuffledHLSColormap.'''

    def __init__(self, d):
        self.num_colors = self.parse_number(d, "num_colors")
        self.lightness = self.parse_number(d, "lightness", default=0.4)
        self.saturation = self.parse_number(d, "saturation", default=0.7)
        self.seed = self.parse_number(d, "seed", default=42)


class ShuffledHLSColormap(Colormap):
    '''A colormap with equally spaced (and randonly shuffled) hues in HLS
    colorspace.
    '''

    def __init__(self, config):
        '''Creates an ShuffledHLSColormap instance.

        Args:
            config: a ShuffledHLSColormapConfig instance
        '''
        self.validate(config)
        self.config = config
        self._colors = self._generate_colors(self.config.num_colors)

    @property
    def colors(self):
        '''The array of hex colors in the colormap.'''
        return self._colors

    def _generate_colors(self, num_hues):
        hues = np.linspace(0, 1, num_hues + 1)[:-1]
        colors = [
            etai.hls_to_hex(hue, self.config.lightness, self.config.saturation)
            for hue in hues
        ]
        rng = random.Random(self.config.seed)
        rng.shuffle(colors)
        return colors


# Annotation constants
_IS_PYTHON_2 = (sys.version_info[0] == 2)
_DEFAULT_ANNOTATION_ALPHA = 0.8
_DEFAULT_ANNOTATION_TEXT_COLOR = (255, 255, 255)
_DEFAULT_FRAME_ANNOTATION_BACKGROUND_COLOR = (0, 0, 0)
_DEFAULT_FRAME_ANNOTATION_GAP = 0.01
_DEFAULT_ANNOTATION_PAD = (2, 2)
_DEFAULT_ANNOTATION_LINEWIDTH = 2
_DEFAULT_COLORMAP = Colormap.load_default()
_DEFAULT_FONT = ImageFont.truetype(etac.DEFAULT_FONT_PATH, 16)
_DEFAULT_LOGO = etal.Logo.load_default()


def annotate_video(
        input_path, video_labels, output_path, add_logo=True, logo=None,
        colormap=None, font=None, alpha=None, text_color=None, pad=None,
        linewidth=None):
    '''Annotates the video with the given labels.

    Args:
        input_path: the path to the video to annotate
        video_labels: an `eta.core.video.VideoLabels` instance describing the
            content to annotate
        output_path: the path to write the output video
        add_logo: whether to add a logo to the output video. By default, this
            is True
        logo: an `eta.core.logo.Logo` to render on the video. If omitted, the
            default logo is used
        colormap: an optional `eta.core.annotations.Colormap` to use. If not
            provided, the default colormap is used
        font: an optional `PIL.ImageFont` to use. If not provided, the default
            font is used
        alpha: an optional transparency to use for the annotation
            boxes/backgrounds
        text_color: an optional text color to use
        pad: an optional (padx, pady) to use to pad the annotation text
        linewidth: an optional bounding box linewdith to use
    '''
    # Parse args
    if add_logo and logo is None:
        logo = _DEFAULT_LOGO
    if not colormap:
        colormap = _DEFAULT_COLORMAP
    if not font:
        font = _DEFAULT_FONT
    if alpha is None:
        alpha = _DEFAULT_ANNOTATION_ALPHA
    if text_color is None:
        text_color = _DEFAULT_ANNOTATION_TEXT_COLOR
    if pad is None:
        pad = _DEFAULT_ANNOTATION_PAD
    if linewidth is None:
        linewidth = _DEFAULT_ANNOTATION_LINEWIDTH

    # Annotate video
    with etav.VideoProcessor(input_path, out_video_path=output_path) as p:
        # Render logo for video, if necessary
        if add_logo:
            logo.render_for(frame_size=p.output_frame_size)

        # Get video-level attributes
        if video_labels.attrs:
            video_labels.attrs.sort_by_name()
            video_attrs = video_labels.attrs
        else:
            video_attrs = None

        # Annotate frames
        for img in p:
            logger.debug("Annotating frame %d", p.frame_number)
            frame_labels = video_labels[p.frame_number]
            img_anno = annotate_video_frame(
                img, frame_labels, video_attrs=video_attrs,
                add_logo=add_logo, logo=logo, colormap=colormap, font=font,
                alpha=alpha, text_color=text_color, pad=pad,
                linewidth=linewidth)
            p.write(img_anno)


def annotate_video_frame(img, frame_labels, video_attrs=None, add_logo=True,
        logo=None, colormap=None, font=None, alpha=None, text_color=None,
        pad=None, linewidth=None, attrs_bg_color=None, attrs_gap=None):
    '''Annotates the video frame with the given labels.

    Args:
        img: the video frame to annotate
        frame_labels: an `eta.core.video.VideoFrameLabels` instance describing
            the content to annotate
        video_attrs: an optional `eta.core.data.AttributeContainer` of video
            level attributes
        add_logo: whether to add a logo to the annotated frame. By default,
            this is True
        logo: an `eta.core.Logo` to render on the annotated frame. If omitted,
            the default logo is used
        colormap: an optional `eta.core.annotations.Colormap` to use. If not
            provided, the default colormap is used
        font: an optional `PIL.ImageFont` to use. If not provided, the default
            font is used
        alpha: an optional transparency to use for the annotation
            boxes/backgrounds
        text_color: an optional text color to use
        pad: an optional (padx, pady) to use to pad the annotation text
        linewidth: an optional bounding box linewdith to use
        attrs_bg_color: an optional background color to use for the frame
            attributes box. If omitted, the default background color is used
        attrs_gap: the relative gap (w.r.t. the image width) to leave between
            the frame attributes box and the upper left corner of the image. If
            omitted, the default gap is used

    Returns:
        the annotated image
    '''
    return _annotate_image(img, frame_labels, video_attrs, add_logo, logo,
        colormap, font, alpha, text_color, pad, linewidth, attrs_bg_color,
        attrs_gap)


def annotate_image(
        img, image_labels, add_logo=True, logo=None, colormap=None, font=None,
        alpha=None, text_color=None, pad=None, linewidth=None,
        attrs_bg_color=None, attrs_gap=None):
    '''Annotates the image with the given labels.

    Args:
        img: the image to annotate
        image_labels: an `eta.core.image.ImageLabels` instance describing
            the content to annotate
        add_logo: whether to add a logo to the annotated image. By default,
            this is True
        logo: an `eta.core.Logo` to render on the annotated image. If omitted,
            the default logo is used
        colormap: an optional `eta.core.annotations.Colormap` to use. If not
            provided, the default colormap is used
        font: an optional `PIL.ImageFont` to use. If not provided, the default
            font is used
        alpha: an optional transparency to use for the annotation
            boxes/backgrounds
        text_color: an optional text color to use
        pad: an optional (padx, pady) to use to pad the annotation text
        linewidth: an optional bounding box linewdith to use
        attrs_bg_color: an optional background color to use for the frame
            attributes box. If omitted, the default background color is used
        attrs_gap: the relative gap (w.r.t. the image width) to leave between
            the frame attributes box and the upper left corner of the image. If
            omitted, the default gap is used

    Returns:
        the annotated image
    '''
    return _annotate_image(img, image_labels, None, add_logo, logo, colormap,
        font, alpha, text_color, pad, linewidth, attrs_bg_color, attrs_gap)


def _annotate_image(
        img, labels, more_attrs, add_logo, logo, colormap, font, alpha,
        text_color, pad, linewidth, attrs_bg_color, attrs_gap):
    #
    # Assumption: labels has `objects` and `attrs` members. This covers both
    # ImageLabels and VideoFrameLabels
    #

    # Parse args
    if add_logo and logo is None:
        logo = _DEFAULT_LOGO
        logo.render_for(img=img)
    if not colormap:
        colormap = _DEFAULT_COLORMAP
    if not font:
        font = _DEFAULT_FONT
    if alpha is None:
        alpha = _DEFAULT_ANNOTATION_ALPHA
    if text_color is None:
        text_color = _DEFAULT_ANNOTATION_TEXT_COLOR
    if pad is None:
        pad = _DEFAULT_ANNOTATION_PAD
    if attrs_bg_color is None:
        attrs_bg_color = _DEFAULT_FRAME_ANNOTATION_BACKGROUND_COLOR
    if attrs_gap is None:
        attrs_gap = _DEFAULT_FRAME_ANNOTATION_GAP
    if linewidth is None:
        linewidth = _DEFAULT_ANNOTATION_LINEWIDTH

    # Render objects, if necessary
    logger.debug("Rendering %d objects", len(labels.objects))
    for obj in labels.objects:
        img = _annotate_object(
            img, obj, colormap, font, alpha, text_color, pad, linewidth)

    # Render frame attributes, if necessary
    if more_attrs is not None:
        attr_strs = [_render_attr_name_value(a) for a in more_attrs]
    else:
        attr_strs = []
    labels.attrs.sort_by_name()
    attr_strs.extend([_render_attr_name_value(a) for a in labels.attrs])
    if attr_strs:
        logger.debug("Rendering %d frame attributes", len(attr_strs))
        img = _annotate_attrs(
            img, attr_strs, font, alpha, text_color, attrs_bg_color, pad,
            attrs_gap)

    # Add logo
    if add_logo:
        img = logo.apply(img)

    return img


def _annotate_attrs(
        img, attr_strs, font, alpha, text_color, bg_color, pad, gap):
    num_attrs = len(attr_strs)
    gap_pixels = int(gap * img.shape[1])
    label_text_size = _compute_max_text_size(font, attr_strs)  # width, height

    overlay = img.copy()

    # Draw attribute background
    bgtlx = gap_pixels
    bgtly = gap_pixels
    bgbrx = bgtlx + label_text_size[0] + 2 * pad[0]
    bgbry = bgtly + num_attrs * label_text_size[1] + (num_attrs + 1) * pad[1]
    cv2.rectangle(overlay, (bgtlx, bgtly), (bgbrx, bgbry), bg_color, -1)

    # Overlay translucent box
    img_anno = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    img_pil = Image.fromarray(img_anno)
    draw = ImageDraw.Draw(img_pil)

    # Draw attributes
    for idx, attr_str in enumerate(attr_strs):
        txttlx = bgtlx + pad[0]
        txttly = bgtly + (idx + 1) * pad[1] + idx * label_text_size[1] - 1
        draw.text((txttlx, txttly), attr_str, font=font, fill=text_color)

    return np.asarray(img_pil)


def _annotate_object(
        img, obj, colormap, font, alpha, text_color, pad, linewidth):
    # Construct label string
    label_str = _render_object_label(obj)

    # Get box color
    color_index = label_str.__hash__()
    box_color = _parse_hex_color(colormap.get_color(color_index))

    overlay = img.copy()
    label_text_size = font.getsize(label_str)  # width, height

    # Draw bounding box
    objtlx, objtly = obj.bounding_box.top_left.coords_in(img=img)
    objbrx, objbry = obj.bounding_box.bottom_right.coords_in(img=img)
    cv2.rectangle(
        overlay, (objtlx, objtly), (objbrx, objbry), box_color, linewidth)

    # Draw label background
    bgtlx = objtlx - linewidth + 1
    bgbry = objtly - linewidth + 1
    bgbrx = bgtlx + label_text_size[0] + 2 * pad[0]
    bgtly = bgbry - label_text_size[1] - 2 * pad[1]
    cv2.rectangle(overlay, (bgtlx, bgtly), (bgbrx, bgbry), box_color, -1)

    # Overlay translucent box
    img_anno = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    img_pil = Image.fromarray(img_anno)
    draw = ImageDraw.Draw(img_pil)

    # Draw label
    txttlx = bgtlx + pad[0]
    txttly = bgtly + pad[1] - 1
    draw.text((txttlx, txttly), label_str, font=font, fill=text_color)

    # Add attributes, if necessary
    if obj.has_attributes:
        # Alphabetize attributes by name
        obj.attrs.sort_by_name()
        attrs_str = ", ".join([_render_attr_value(a) for a in obj.attrs])

        # Draw attributes
        atxttlx = objtlx + linewidth + pad[0]
        atxttly = objtly - 1 + pad[1]
        draw.text((atxttlx, atxttly), attrs_str, font=font, fill=text_color)

    return np.asarray(img_pil)


def _parse_hex_color(h):
    rgb = etai.hex_to_rgb(h)
    if _IS_PYTHON_2:
        # OpenCV hates `future` types, so we do this funny cast
        rgb = np.array(rgb)
    return rgb


def _compute_max_text_size(font, text_strs):
    sizes = [font.getsize(s) for s in text_strs]
    width, height = np.max(sizes, axis=0)
    return width, height


def _render_attr_value(attr):
    return _clean(attr.value)


def _render_attr_name_value(attr):
    name = _clean(attr.name)
    value = _clean(attr.value)
    return "%s: %s" % (name, value)


def _render_object_label(obj):
    label_str = _clean(obj.label).upper()
    if obj.index is None:
        return label_str
    return "%s %d" % (label_str, obj.index)


def _clean(s):
    return str(s).lower().replace("_", " ")
