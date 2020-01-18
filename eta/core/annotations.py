'''
Core utilities for rendering annotations on media.

Copyright 2017-2020, Voxel51, Inc.
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

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import eta
import eta.constants as etac
from eta.core.config import Config, Configurable
import eta.core.data as etad
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
        show_object_boxes: whether to show object bounding boxes, if available.
            If this is false, all attributes, confidences, etc. are also hidden
        show_object_attrs: whether to show object attributes, if available
        show_object_confidences: whether to show object label confidences, if
            available
        show_object_attr_confidences: whether to show object attribute
            confidences, if available
        show_frame_attr_confidences: whether to show video/frame attribute
            confidences, if available
        show_all_confidences: whether to show all confidences, if available.
            If set to `True`, the other confidence-rendering flags are ignored
        show_object_indices: whether to show object indices, if available. By
            default, this is `True`
        show_object_masks: whether to show object segmentation masks, if
            available
        occluded_object_attr: the name of the boolean attribute indicating
            whether an object is occluded
        hide_occluded_objects: whether to hide objects when they are occluded
        hide_attr_values: an optional list of video/frame/object attribute
            values to NOT RENDER if they appear
        hide_false_boolean_attrs: whether to hide video/frame/object attributes
            when they are False
        font_path: the path to the `PIL.ImageFont` to use
        font_size: the font size to use
        linewidth: the linewidth, in pixels, of the object bounding boxes
        alpha: the transparency of the object bounding boxes
        confidence_scaled_alpha: True will scale `alpha` and `mask_fill_alpha`
            by the object confidence
        text_color: the annotation text color
        object_attrs_render_method: the method used to render object attributes
        object_text_pad_pixels: the padding, in pixels, around the text in the
            object labels
        attrs_bg_color: the background color for attributes boxes
        attrs_bg_alpha: the transparency of the video/frame/object attributes
            panel boxes
        attrs_box_gap: the gap between the frame attributes box and the upper
            left corner of the image
        attrs_text_pad_pixels: the padding, in pixels, around the text in the
            frame attributes box
        attrs_text_line_spacing_pixels: the padding, in pixels, between each
            line of text in the frame attributes box
        mask_border_thickness: the thickness, in pixels, to use when drawing
            the border of segmentation masks
        mask_fill_alpha: the transparency of the segmentation mask
        add_logo: whether to add a logo to the output video
        logo_config: the `eta.core.logo.LogoConfig` describing the logo to use
    '''

    def __init__(self, d):
        self.colormap_config = self.parse_object(
            d, "colormap_config", ColormapConfig, default=None)
        self.show_object_boxes = self.parse_bool(
            d, "show_object_boxes", default=True)
        self.show_object_attrs = self.parse_bool(
            d, "show_object_attrs", default=True)
        self.show_object_confidences = self.parse_bool(
            d, "show_object_confidences", default=False)
        self.show_object_attr_confidences = self.parse_bool(
            d, "show_object_attr_confidences", default=False)
        self.show_frame_attr_confidences = self.parse_bool(
            d, "show_frame_attr_confidences", default=False)
        self.show_all_confidences = self.parse_bool(
            d, "show_all_confidences", default=False)
        self.show_object_indices = self.parse_bool(
            d, "show_object_indices", default=True)
        self.show_object_masks = self.parse_bool(
            d, "show_object_masks", default=True)
        self.occluded_object_attr = self.parse_string(
            d, "occluded_object_attr", default="occluded")
        self.hide_occluded_objects = self.parse_bool(
            d, "hide_occluded_objects", default=False)
        self.hide_attr_values = self.parse_array(
            d, "hide_attr_values", default=None)
        self.hide_false_boolean_attrs = self.parse_bool(
            d, "hide_false_boolean_attrs", default=False)
        self.font_path = self.parse_string(
            d, "font_path", default=etac.DEFAULT_FONT_PATH)
        self.font_size = self.parse_number(d, "font_size", default=16)
        self.linewidth = self.parse_number(d, "linewidth", default=2)
        self.alpha = self.parse_number(d, "alpha", default=0.75)
        self.confidence_scaled_alpha = self.parse_bool(
            d, "confidence_scaled_alpha", default=False)
        self.text_color = self.parse_string(d, "text_color", default="#FFFFFF")
        self.object_attrs_render_method = self.parse_categorical(
            d, "object_attrs_render_method", ["list", "panel"],
            default="panel")
        self.object_text_pad_pixels = self.parse_number(
            d, "object_text_pad_pixels", default=2)
        self.attrs_bg_color = self.parse_string(
            d, "attrs_bg_color", default="#000000")
        self.attrs_bg_alpha = self.parse_number(
            d, "attrs_bg_alpha", default=0.5)
        self.attrs_box_gap = self.parse_string(
            d, "attrs_box_gap", default="1%")
        self.attrs_text_pad_pixels = self.parse_number(
            d, "attrs_text_pad_pixels", default=5)
        self.attrs_text_line_spacing_pixels = self.parse_number(
            d, "attrs_text_line_spacing_pixels", default=1)
        self.mask_border_thickness = self.parse_number(
            d, "mask_border_thickness", default=2)
        self.mask_fill_alpha = self.parse_number(
            d, "mask_fill_alpha", default=0.5)
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
    # Parse config
    hide_attr_values = annotation_config.hide_attr_values
    hide_false_boolean_attrs = annotation_config.hide_false_boolean_attrs
    show_frame_attr_confidences = (
        annotation_config.show_frame_attr_confidences or
        annotation_config.show_all_confidences)
    add_logo = annotation_config.add_logo
    logo = annotation_config.logo

    #
    # Assumption: labels has `objects` and `attrs` members. This covers both
    # ImageLabels and VideoFrameLabels
    #

    # Render objects, if necessary
    logger.debug("Rendering %d objects", len(labels.objects))
    for obj in labels.objects:
        img = _annotate_object(img, obj, annotation_config)

    attr_strs = []

    # Render `more_attrs`
    if more_attrs is not None:
        attr_strs.extend(
            _render_attrs(
                more_attrs, hide_attr_values, hide_false_boolean_attrs,
                show_frame_attr_confidences))

    # Render frame attributes
    labels.attrs.sort_by_name()  # alphabetize
    attr_strs.extend(
        _render_attrs(
            labels.attrs, hide_attr_values, hide_false_boolean_attrs,
            show_frame_attr_confidences))

    # Draw attributes panel
    if attr_strs:
        img = _annotate_frame_attrs(img, attr_strs, annotation_config)

    # Add logo
    if add_logo:
        img = logo.apply(img)

    return img


def _annotate_frame_attrs(img, attr_strs, annotation_config):
    logger.debug("Rendering %d frame attributes", len(attr_strs))

    # Compute upper-left corner of attrs panel
    offset = etai.Width(annotation_config.attrs_box_gap).render_for(img=img)
    top_left_coords = (offset, offset)

    img_anno = _draw_attrs_panel(
        img, attr_strs, top_left_coords, annotation_config)

    return img_anno


def _annotate_object(img, obj, annotation_config):
    # Parse config
    show_object_boxes = annotation_config.show_object_boxes
    show_object_attrs = annotation_config.show_object_attrs
    show_object_confidences = (
        annotation_config.show_object_confidences or
        annotation_config.show_all_confidences)
    show_object_attr_confidences = (
        annotation_config.show_object_attr_confidences or
        annotation_config.show_all_confidences)
    show_object_indices = annotation_config.show_object_indices
    occluded_object_attr = annotation_config.occluded_object_attr
    hide_occluded_objects = annotation_config.hide_occluded_objects
    show_object_masks = annotation_config.show_object_masks
    hide_attr_values = annotation_config.hide_attr_values
    hide_false_boolean_attrs = annotation_config.hide_false_boolean_attrs
    colormap = annotation_config.colormap
    font = annotation_config.font
    alpha = annotation_config.alpha
    confidence_scaled_alpha = annotation_config.confidence_scaled_alpha
    linewidth = annotation_config.linewidth
    attrs_render_method = annotation_config.object_attrs_render_method
    pad = annotation_config.object_text_pad_pixels
    text_color = tuple(_parse_hex_color(annotation_config.text_color))
    mask_border_thickness = annotation_config.mask_border_thickness
    mask_fill_alpha = annotation_config.mask_fill_alpha

    # Check for occluded objects
    if hide_occluded_objects:
        for attr in obj.attrs:
            if attr.name == occluded_object_attr and attr.value:
                # Skip occluded object
                return img

    # Scale alpha by confidence, if requested
    if confidence_scaled_alpha and obj.confidence is not None:
        alpha *= obj.confidence
        mask_fill_alpha *= obj.confidence

    # Construct label string
    label_str, label_hash = _render_object_label(
        obj, show_index=show_object_indices,
        show_confidence=show_object_confidences)

    box_color = _parse_hex_color(colormap.get_color(label_hash))  # box color
    label_text_size = font.getsize(label_str)  # width, height

    img_anno = img.copy()

    #
    # Draw segmentation mask
    #

    if obj.has_mask and show_object_masks:
        img_anno = _draw_object_mask(
            img_anno, obj, box_color, border_thickness=mask_border_thickness,
            border_alpha=alpha, fill_alpha=mask_fill_alpha)

    #
    # Draw bounding box
    #

    if not show_object_boxes:
        return img_anno

    overlay = img_anno.copy()

    # Bounding box
    objtlx, objtly = obj.bounding_box.top_left.coords_in(img=img)
    objbrx, objbry = obj.bounding_box.bottom_right.coords_in(img=img)
    cv2.rectangle(
        overlay, (objtlx, objtly), (objbrx, objbry), box_color, linewidth)

    #
    # Draw object label
    #

    # Label background
    bgtlx = objtlx - linewidth + 1
    bgbry = objtly - linewidth + 1
    bgbrx = bgtlx + label_text_size[0] + 2 * (pad + _DX)
    bgtly = bgbry - label_text_size[1] - 2 * pad
    cv2.rectangle(overlay, (bgtlx, bgtly), (bgbrx, bgbry), box_color, -1)

    # Overlay
    img_anno = cv2.addWeighted(overlay, alpha, img_anno, 1 - alpha, 0)

    img_pil = Image.fromarray(img_anno)
    draw = ImageDraw.Draw(img_pil)

    # Draw label
    if label_str:
        txttlx = bgtlx + pad + _DX
        txttly = bgtly + pad - 1
        draw.text((txttlx, txttly), label_str, font=font, fill=text_color)

    #
    # Render object attributes
    #

    if not show_object_attrs or not obj.has_attributes:
        return np.asarray(img_pil)

    # Render object attribute strings
    obj.attrs.sort_by_name()  # alphabetize by name
    attr_strs = _render_attrs(
        obj.attrs, hide_attr_values, hide_false_boolean_attrs,
        show_object_attr_confidences)
    if not attr_strs:
        return np.asarray(img_pil)

    logger.debug("Rendering %d object attributes", len(attr_strs))

    # Method 1: draw attributes as list
    if attrs_render_method == "list":
        atxttlx = objtlx + linewidth + pad
        atxttly = objtly - 1 + pad
        attrs_str = ", ".join(attr_strs)
        draw.text(
            (atxttlx, atxttly), attrs_str, font=font, fill=text_color)

    img_anno = np.asarray(img_pil)

    # Method 2: draw attributes as panel
    if attrs_render_method == "panel":
        # Compute upper-left corner of attrs panel
        atxttlx = objtlx + 2 * linewidth
        atxttly = objtly + 2 * linewidth - 1
        top_left_coords = (atxttlx, atxttly)

        img_anno = _draw_attrs_panel(
            img_anno, attr_strs, top_left_coords, annotation_config)

    return img_anno


def _draw_attrs_panel(img, attr_strs, top_left_coords, annotation_config):
    # Parse config
    font = annotation_config.font
    box_pad = annotation_config.attrs_text_pad_pixels
    line_gap = annotation_config.attrs_text_line_spacing_pixels
    text_size = _compute_max_text_size(font, attr_strs)  # width, height
    text_color = tuple(_parse_hex_color(annotation_config.text_color))
    bg_color = _parse_hex_color(annotation_config.attrs_bg_color)
    bg_alpha = annotation_config.attrs_bg_alpha
    num_attrs = len(attr_strs)

    overlay = img.copy()

    # Draw attribute background
    bgtlx, bgtly = top_left_coords
    bgbrx = bgtlx + text_size[0] + 2 * (box_pad + _DX)
    bgbry = (
        bgtly + num_attrs * text_size[1] + (num_attrs - 1) * line_gap +
        2 * box_pad)
    cv2.rectangle(overlay, (bgtlx, bgtly), (bgbrx, bgbry), bg_color, -1)

    # Overlay translucent box
    img_anno = cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0)

    img_pil = Image.fromarray(img_anno)
    draw = ImageDraw.Draw(img_pil)

    # Draw attributes
    for idx, attr_str in enumerate(attr_strs):
        txttlx = bgtlx + box_pad + _DX
        txttly = bgtly + box_pad + idx * line_gap + idx * text_size[1] - 1
        draw.text((txttlx, txttly), attr_str, font=font, fill=text_color)

    return np.asarray(img_pil)


def _draw_object_mask(
        img, obj, color, border_thickness=None, border_alpha=None,
        fill_alpha=None):
    mask, offset = etai.render_instance_mask(obj, img=img, as_bool=False)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=offset)

    img_anno = img.copy()

    if fill_alpha is not None and fill_alpha > 0:
        img_mask = img_anno.copy()
        cv2.drawContours(img_mask, contours, -1, color, cv2.FILLED)
        img_anno = cv2.addWeighted(
            img_mask, fill_alpha, img_anno, 1 - fill_alpha, 0)

    if (border_thickness is not None and border_thickness > 0 and
            border_alpha is not None and border_alpha > 0):
        img_border = img_anno.copy()
        cv2.drawContours(img_border, contours, -1, color, border_thickness)
        img_anno = cv2.addWeighted(
            img_border, border_alpha, img_anno, 1 - border_alpha, 0)

    return img_anno


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


def _render_attrs(
        attrs, hide_attr_values, hide_false_boolean_attrs, show_confidence):
    attr_strs = []
    for attr in attrs:
        if hide_attr_values is not None and attr.value in hide_attr_values:
            # Hide this attribute
            continue

        if (hide_false_boolean_attrs and
                isinstance(attr, etad.BooleanAttribute) and not attr.value):
            # Hide false boolean attribute
            continue

        attr_strs.append(
            _render_attr_name_value(attr, show_confidence=show_confidence))

    return attr_strs


def _render_attr_value(attr, show_confidence=True):
    if isinstance(attr, etad.NumericAttribute):
        attr_str = _render_numeric_attr_value(attr)
    else:
        attr_str = _clean_str(attr.value)

    if show_confidence and attr.confidence is not None:
        attr_str += " (%.2f)" % attr.confidence

    return attr_str


def _render_attr_name_value(attr, show_confidence=True):
    name = _clean_str(attr.name)

    if isinstance(attr, etad.NumericAttribute):
        value = _render_numeric_attr_value(attr)
    else:
        value = _clean_str(attr.value)

    attr_str = "%s: %s" % (name, value)
    if show_confidence and attr.confidence is not None:
        attr_str += " (%.2f)" % attr.confidence

    return attr_str


def _render_object_label(obj, show_index=True, show_confidence=True):
    add_confidence = show_confidence and obj.confidence is not None
    add_index = show_index and obj.index is not None

    label_str = _clean_str(obj.label).upper()

    if add_confidence:
        label_str += " (%.2f)"

    if add_index:
        label_str += "     %d" % obj.index

    # Compute hash before rendering confidence so it doesn't affect coloring
    label_hash = label_str.__hash__()

    if add_confidence:
        label_str = label_str % obj.confidence

    return label_str, label_hash


def _render_numeric_attr_value(attr):
    if isinstance(attr.value, int):
        return "%d" % attr.value

    return "%.2f" % attr.value


def _clean_str(s):
    return str(s).lower().replace("_", " ")
