'''
Core utilities for rendering annotations on media.

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
_DEFAULT_ANNOTATION_PAD = (2, 2)
_DEFAULT_ANNOTATION_LINEWIDTH = 2
_DEFAULT_COLORMAP = Colormap.load_default()
_DEFAULT_FONT = ImageFont.truetype(etac.DEFAULT_FONT_PATH, 16)
_DEFAULT_LOGO = etal.Logo.load_default()


def annotate_video(
        input_path, video_labels, output_path, add_logo=True, logo=None,
        colormap=None, font=None, alpha=None, text_color=None, pad=None,
        linewidth=None):
    '''Annotates the video with the given video labels.

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

        for img in p:
            logger.debug("Annotating frame %d", p.frame_number)
            frame_labels = video_labels[p.frame_number]
            img_anno = annotate_image(
                img, frame_labels, add_logo=add_logo, logo=logo,
                colormap=colormap, font=font, alpha=alpha,
                text_color=text_color, pad=pad, linewidth=linewidth)
            p.write(img_anno)


def annotate_image(
        img, frame_labels, add_logo=True, logo=None, colormap=None, font=None,
        alpha=None, text_color=None, pad=None, linewidth=None):
    '''Annotates the image with the given frame labels.

    Args:
        img: the image to annotate
        frame_labels: an `eta.core.video.VideoFrameLabels` instance describing
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

    Returns:
        the annotated image
    '''
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
    if linewidth is None:
        linewidth = _DEFAULT_ANNOTATION_LINEWIDTH

    # Render objects
    logger.debug("Rendering %d objects", len(frame_labels.objects))
    for obj in frame_labels.objects:
        img = _annotate_object(
            img, obj, colormap, font, alpha, text_color, pad, linewidth)

    # @todo support frame attributes

    # Add logo
    if add_logo:
        img = logo.apply(img)

    return img


def _annotate_object(
        img, obj, colormap, font, alpha, text_color, pad, linewidth):
    # Generate message
    label = str(obj.label).upper()
    if obj.index is not None:
        index = int(obj.index)
        msg = "%s %d" % (label, index)
    else:
        index = 0
        msg = label

    # Get box color
    color_index = label.__hash__() + index
    box_color = etai.hex_to_rgb(colormap.get_color(color_index))
    if _IS_PYTHON_2:
        # OpenCV hates `future` types, so we do this funny cast
        box_color = np.array(box_color)

    overlay = img.copy()
    text_size = font.getsize(msg)  # width, height

    # Draw bounding box
    objtlx, objtly = obj.bounding_box.top_left.coords_in(img=img)
    objbrx, objbry = obj.bounding_box.bottom_right.coords_in(img=img)
    cv2.rectangle(overlay, (objtlx, objtly), (objbrx, objbry), box_color, linewidth)

    # Draw message background
    bgtlx = objtlx - linewidth + 1
    bgbry = objtly - linewidth + 1
    bgbrx = bgtlx + text_size[0] + 2 * pad[0]
    bgtly = bgbry - text_size[1] - 2 * pad[1]
    cv2.rectangle(overlay, (bgtlx, bgtly), (bgbrx, bgbry), box_color, -1)

    # Construct attribute text
    attr_msg = ""
    for attr in obj.attrs.attrs:
        attr_msg += attr.value+" "
    attr_text_size = font.getsize(msg)

    # Draw attribute message background
    abgtlx = objtlx - linewidth + 1
    abgbrx = abgtlx + attr_text_size[0] + 2 * pad[0]
    abgtly = objbry + linewidth - 1
    abgbry = abgtly + attr_text_size[1] + 2 * pad[1]
    cv2.rectangle(overlay, (abgtlx, abgtly), (abgbrx, abgbry), box_color, -1)

    # Overlay translucent box
    img_anno = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Draw messages
    img_pil = Image.fromarray(img_anno)
    draw = ImageDraw.Draw(img_pil)
    txttlx = bgtlx + pad[0]
    txttly = bgtly + pad[1] - 1
    draw.text((txttlx, txttly), msg, font=font, fill=text_color)
    atxttlx = abgtlx + pad[0]
    atxttly = abgtly + pad[1] - 1
    draw.text((atxttlx, atxttly), attr_msg, font=font, fill=text_color)

    return np.asarray(img_pil)
