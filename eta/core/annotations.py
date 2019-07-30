'''
Core utilities for rendering annotations on media.

Copyright 2017-2019, Voxel51, Inc.
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

import eta
import eta.constants as etac
from eta.core.config import Config, Configurable
import eta.core.image as etai
import eta.core.logo as etal
import eta.core.video as etav


logger = logging.getLogger(__name__)


class AnnotationConfig(Config):
    '''Configuration class that controls the look-and-feel of the annotations
    rendered on images/videos.

    Attributes:
        colormap_config: the `eta.core.annotations.ColormapConfig` to use to
            render the annotation boxes
        show_object_confidences: whether to show object label confidences, if
            available
        show_object_attr_confidences: whether to show object attribute
            confidences, if available
        show_frame_attr_confidences: whether to show frame/video attribute
            confidences, if available
        show_all_confidences: whether to show all confidences, if available.
            If set to `True`, the other confidence-rendering flags are ignored
        font_path: the path to the `PIL.ImageFont` to use
        font_size: the font size to use
        linewidth: the linewidth, in pixels, of the object bounding boxes
        alpha: the transparency of the object bounding boxes and frame
            attributes
        text_color: the annotation text color
        object_text_pad_pixels: the padding, in pixels, around the text in the
            object labels
        attrs_bg_color: the background color for the frame attributes box
        attrs_box_gap: the gap between the frame attributes box and the upper
            left corner of the image
        attrs_text_pad_pixels: the padding, in pixels, around the text in the
            frame attributes box
        attrs_text_line_spacing_pixels: the padding, in pixels, between each
            line of text in the frame attributes box
        add_logo: whether to add a logo to the output video
        logo_config: the `eta.core.logo.LogoConfig` describing the logo to use
    '''

    def __init__(self, d):
        self.colormap_config = self.parse_object(
            d, "colormap_config", ColormapConfig, default=None)

        self.show_object_confidences = self.parse_bool(
            d, "show_object_confidences", default=False)
        self.show_object_attr_confidences = self.parse_bool(
            d, "show_object_attr_confidences", default=False)
        self.show_frame_attr_confidences = self.parse_bool(
            d, "show_frame_attr_confidences", default=False)
        self.show_all_confidences = self.parse_bool(
            d, "show_all_confidences", default=False)

        self.font_path = self.parse_string(
            d, "font_path", default=etac.DEFAULT_FONT_PATH)
        self.font_size = self.parse_number(d, "font_size", default=16)
        self.linewidth = self.parse_number(d, "linewidth", default=2)
        self.alpha = self.parse_number(d, "alpha", default=0.75)
        self.text_color = self.parse_string(d, "text_color", default="#FFFFFF")
        self.object_text_pad_pixels = self.parse_number(
            d, "object_text_pad_pixels", default=2)
        self.attrs_bg_color = self.parse_string(
            d, "attrs_bg_color", default="#000000")
        self.attrs_box_gap = self.parse_string(
            d, "attrs_box_gap", default="1%")
        self.attrs_text_pad_pixels = self.parse_number(
            d, "attrs_text_pad_pixels", default=5)
        self.attrs_text_line_spacing_pixels = self.parse_number(
            d, "attrs_text_line_spacing_pixels", default=1)

        self.add_logo = self.parse_bool(d, "add_logo", default=True)
        self.logo_config = self.parse_object(
            d, "logo_config", etal.LogoConfig, default=None)

        if self.colormap_config is not None:
            self._colormap = self.colormap_config.build()
        else:
            self._colormap = Colormap.load_default()

        self._font = ImageFont.truetype(self.font_path, self.font_size)

        if self.logo_config is not None:
            self._logo = etal.Logo(self.logo_config)
        elif self.add_logo:
            self._logo = etal.Logo.load_default()
        else:
            self._logo = None

    @property
    def colormap(self):
        return self._colormap

    @property
    def font(self):
        return self._font

    @property
    def logo(self):
        return self._logo


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


def annotate_video(
        input_path, video_labels, output_path, annotation_config=None):
    '''Annotates the video with the given labels.

    Args:
        input_path: the path to the video to annotate
        video_labels: an `eta.core.video.VideoLabels` instance describing the
            content to annotate
        output_path: the path to write the output video
        annotation_config: an optional AnnotationConfig specifying how to
            render the annotations. If omitted, the default config is used
    '''
    if annotation_config is None:
        annotation_config = _DEFAULT_ANNOTATION_CONFIG

    # Annotate video
    with etav.VideoProcessor(input_path, out_video_path=output_path) as p:
        # Render logo for video, if necessary
        if annotation_config.add_logo:
            annotation_config.logo.render_for(frame_size=p.output_frame_size)

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
                annotation_config=annotation_config)
            p.write(img_anno)


def annotate_video_frame(
        img, frame_labels, video_attrs=None, annotation_config=None):
    '''Annotates the video frame with the given labels.

    Args:
        img: the video frame to annotate
        frame_labels: an `eta.core.video.VideoFrameLabels` instance describing
            the content to annotate
        video_attrs: an optional `eta.core.data.AttributeContainer` of video
            level attributes
        annotation_config: an optional AnnotationConfig specifying how to
            render the annotations. If omitted, the default config is used

    Returns:
        the annotated image
    '''
    if annotation_config is None:
        annotation_config = _DEFAULT_ANNOTATION_CONFIG
        if annotation_config.add_logo:
            annotation_config.logo.render_for(img=img)

    return _annotate_image(img, frame_labels, video_attrs, annotation_config)


def annotate_image(img, image_labels, annotation_config=None):
    '''Annotates the image with the given labels.

    Args:
        img: the image to annotate
        image_labels: an `eta.core.image.ImageLabels` instance describing
            the content to annotate
        annotation_config: an optional AnnotationConfig specifying how to
            render the annotations. If omitted, the default config is used

    Returns:
        the annotated image
    '''
    if annotation_config is None:
        annotation_config = _DEFAULT_ANNOTATION_CONFIG
        if annotation_config.add_logo:
            annotation_config.logo.render_for(img=img)

    return _annotate_image(img, image_labels, None, annotation_config)


def _annotate_image(img, labels, more_attrs, annotation_config):
    #
    # Assumption: labels has `objects` and `attrs` members. This covers both
    # ImageLabels and VideoFrameLabels
    #

    # Render objects, if necessary
    logger.debug("Rendering %d objects", len(labels.objects))
    for obj in labels.objects:
        img = _annotate_object(img, obj, annotation_config)

    # Render frame attributes, if necessary
    show_frame_attr_confidences = (
        annotation_config.show_frame_attr_confidences or
        annotation_config.show_all_confidences)
    if more_attrs is not None:
        attr_strs = [
            _render_attr_name_value(
                a, show_confidence=show_frame_attr_confidences)
            for a in more_attrs
        ]
    else:
        attr_strs = []
    labels.attrs.sort_by_name()
    attr_strs.extend([
        _render_attr_name_value(
            a, show_confidence=show_frame_attr_confidences)
        for a in labels.attrs
    ])
    if attr_strs:
        logger.debug("Rendering %d frame attributes", len(attr_strs))
        img = _annotate_attrs(img, attr_strs, annotation_config)

    # Add logo
    if annotation_config.add_logo:
        img = annotation_config.logo.apply(img)

    return img


def _annotate_attrs(img, attr_strs, annotation_config):
    # Parse config
    font = annotation_config.font
    alpha = annotation_config.alpha
    box_pad = annotation_config.attrs_text_pad_pixels
    line_gap = annotation_config.attrs_text_line_spacing_pixels
    text_size = _compute_max_text_size(font, attr_strs)  # width, height
    box_gap = etai.Width(annotation_config.attrs_box_gap).render_for(img=img)
    text_color = tuple(_parse_hex_color(annotation_config.text_color))
    bg_color = _parse_hex_color(annotation_config.attrs_bg_color)
    num_attrs = len(attr_strs)

    overlay = img.copy()

    # Draw attribute background
    bgtlx = box_gap
    bgtly = box_gap
    bgbrx = bgtlx + text_size[0] + 2 * (box_pad + _DX)
    bgbry = (
        bgtly + num_attrs * text_size[1] + (num_attrs - 1) * line_gap +
        2 * box_pad)
    cv2.rectangle(overlay, (bgtlx, bgtly), (bgbrx, bgbry), bg_color, -1)

    # Overlay translucent box
    img_anno = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    img_pil = Image.fromarray(img_anno)
    draw = ImageDraw.Draw(img_pil)

    # Draw attributes
    for idx, attr_str in enumerate(attr_strs):
        txttlx = bgtlx + box_pad + _DX
        txttly = bgtly + box_pad + idx * line_gap + idx * text_size[1] - 1
        draw.text((txttlx, txttly), attr_str, font=font, fill=text_color)

    return np.asarray(img_pil)


def _annotate_object(img, obj, annotation_config):
    # Parse config
    show_object_confidences = (
        annotation_config.show_object_confidences or
        annotation_config.show_all_confidences)
    show_object_attr_confidences = (
        annotation_config.show_object_attr_confidences or
        annotation_config.show_all_confidences)
    colormap = annotation_config.colormap
    font = annotation_config.font
    alpha = annotation_config.alpha
    linewidth = annotation_config.linewidth
    pad = annotation_config.object_text_pad_pixels
    text_color = tuple(_parse_hex_color(annotation_config.text_color))

    # Construct label string
    label_str, label_hash = _render_object_label(
        obj, show_confidence=show_object_confidences)

    # Get box color
    box_color = _parse_hex_color(colormap.get_color(label_hash))

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
    bgbrx = bgtlx + label_text_size[0] + 2 * (pad + _DX)
    bgtly = bgbry - label_text_size[1] - 2 * pad
    cv2.rectangle(overlay, (bgtlx, bgtly), (bgbrx, bgbry), box_color, -1)

    # Overlay translucent box
    img_anno = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    img_pil = Image.fromarray(img_anno)
    draw = ImageDraw.Draw(img_pil)

    # Draw label
    if label_str:
        txttlx = bgtlx + pad + _DX
        txttly = bgtly + pad - 1
        draw.text((txttlx, txttly), label_str, font=font, fill=text_color)

    # Add attributes, if necessary
    if obj.has_attributes:
        # Alphabetize attributes by name
        obj.attrs.sort_by_name()
        attrs_str = ", ".join([
            _render_attr_value(a, show_confidence=show_object_attr_confidences)
            for a in obj.attrs
        ])

        # Draw attributes
        if attrs_str:
            atxttlx = objtlx + linewidth + pad
            atxttly = objtly - 1 + pad
            draw.text(
                (atxttlx, atxttly), attrs_str, font=font, fill=text_color)

    return np.asarray(img_pil)


_DEFAULT_ANNOTATION_CONFIG = AnnotationConfig.default()
_DX = 2  # extra horizontal space needed for text width to be more "correct"


def _parse_hex_color(h):
    rgb = etai.hex_to_rgb(h)
    if eta.is_python2():
        # OpenCV hates `future` types, so we do this funny cast
        rgb = np.array(rgb)
    return rgb


def _compute_max_text_size(font, text_strs):
    sizes = [font.getsize(s) for s in text_strs]
    width, height = np.max(sizes, axis=0)
    return width, height


def _render_attr_value(attr, show_confidence=True):
    attr_str = _clean(attr.value)
    if show_confidence and attr.confidence is not None:
        attr_str += " (%.2f)" % attr.confidence
    return attr_str


def _render_attr_name_value(attr, show_confidence=True):
    name = _clean(attr.name)
    value = _clean(attr.value)
    attr_str = "%s: %s" % (name, value)
    if show_confidence and attr.confidence is not None:
        attr_str += " (%.2f)" % attr.confidence
    return attr_str


def _render_object_label(obj, show_confidence=True):
    label_str = _clean(obj.label).upper()

    add_confidence = show_confidence and obj.confidence is not None
    if add_confidence:
        label_str += " (%.2f)"

    if obj.index is not None:
        label_str += "     %d" % obj.index

    # Compute hash before rendering confidence so it doesn't affect coloring
    label_hash = label_str.__hash__()

    if add_confidence:
        label_str = label_str % obj.confidence

    return label_str, label_hash


def _clean(s):
    return str(s).lower().replace("_", " ")
