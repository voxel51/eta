"""
Core utilities for rendering annotations on media.

@todo improve efficiency by minimizing number of times that images are copied
and rendered.

Copyright 2017-2020, Voxel51, Inc.
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

from copy import deepcopy
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
    """Configuration class that controls the look-and-feel of the annotations
    rendered on images/videos.

    Attributes:
        show_frame_attr_confidences: whether to render video/frame attribute
            confidences, if available
        frame_attrs_box_gap: the gap between the frame attributes box and the
            upper left corner of the image
        show_object_boxes: whether to render object bounding boxes, if available.
            If this is false, labels, confidences, attributes, etc. are also
            hidden
        show_object_labels: whether to render object labels, if available
        show_object_attrs: whether to render object attributes, if available
        show_object_confidences: whether to render object label confidences, if
            available
        show_object_attr_confidences: whether to render object attribute
            confidences, if available
        show_object_indices: whether to render object indices, if available
        show_object_masks: whether to render object segmentation masks, if
            available
        occluded_object_attr: the name of the boolean attribute indicating
            whether an object is occluded
        hide_occluded_objects: whether to hide objects when they are occluded
        object_labels_whitelist: an optional whitelist of object labels. If
            provided, only objects with labels in this list will be rendered
        object_labels_blacklist: an optional blacklist of object labels. If
            provided, object with labels in this list will not be rendered
        show_event_boxes: whether to render event bounding boxes, if available.
            If this is false, all attributes, confidences, etc. are also hidden
        show_event_labels: whether to render event labels, if available
        show_event_attrs: whether to render event attributes, if available
        show_event_confidences: whether to render event label confidences, if
            available
        show_event_attr_confidences: whether to render event attribute
            confidences, if available
        show_event_indices: whether to render event indices, if available. By
            default, this is `True`
        show_event_masks: whether to render event segmentation masks, if
            available
        show_event_label_on_objects: whether to render event labels as
            attributes ob objects that belong to events
        show_event_objects_in_same_color: whether to render objects that belong
             to events in the same color as their parent event
        occluded_event_attr: the name of the boolean attribute indicating
            whether an event is occluded
        hide_occluded_events: whether to hide events when they are occluded
        event_labels_whitelist: an optional whitelist of event labels. If
            provided, only events with labels in this list will be rendered
        event_labels_blacklist: an optional blacklist of event labels. If
            provided, events with labels in this list will not be rendered
        bbox_alpha: the transparency of bounding boxes
        bbox_label_text_pad_pixels: the padding, in pixels, around the text in
            bounding box labels
        bbox_linewidth: the linewidth, in pixels, of bounding boxes
        mask_border_thickness: the thickness, in pixels, to use when drawing
            the borders of segmentation masks
        mask_fill_alpha: the transparency of segmentation masks
        show_frame_mask_semantics: whether to render semantic labels for frame
            mask regions, when mask indexes are available
        attrs_box_render_method: the method used to render object attributes
        attrs_box_bg_color: the background color for attributes boxes
        attrs_box_bg_alpha: the transparency of attribute panel boxes
        attrs_box_text_pad_pixels: the padding, in pixels, around the text in
            attribute boxes
        attrs_box_text_line_spacing_pixels: the padding, in pixels, between
            each line of text in attribute boxes
        show_all_confidences: whether to render all confidences, if available.
            If set to `True`, this overrides all other confidence flags
        hide_attr_values: an optional list of attribute values (of any kind)
            to NOT RENDER
        hide_false_boolean_attrs: whether to hide attributes (of any kind)
            when they are False
        confidence_scaled_alpha: whether to scale alpha values of objects
            and events based on their associated confidences
        colormap_config: the `eta.core.annotations.ColormapConfig` to use to
            select colors for objects/event boxes
        text_color: the annotation text color
        font_path: the path to the `PIL.ImageFont` to use
        font_size: the font size to use
        scale_by_media_height: whether to scale font sizes and linewidths
            according to the height of the media (relative to a height of 720
            pixels)
        add_logo: whether to add a logo to the video
        logo_config: the `eta.core.logo.LogoConfig` describing the logo to use
    """

    def __init__(self, d):
        ##### FRAME ATTRIBUTES #####
        self.show_frame_attr_confidences = self.parse_bool(
            d, "show_frame_attr_confidences", default=False
        )
        self.frame_attrs_box_gap = self.parse_string(
            d, "frame_attrs_box_gap", default="1%"
        )

        ##### OBJECTS #####
        self.show_object_boxes = self.parse_bool(
            d, "show_object_boxes", default=True
        )
        self.show_object_labels = self.parse_bool(
            d, "show_object_labels", default=True
        )
        self.show_object_attrs = self.parse_bool(
            d, "show_object_attrs", default=True
        )
        self.show_object_confidences = self.parse_bool(
            d, "show_object_confidences", default=False
        )
        self.show_object_attr_confidences = self.parse_bool(
            d, "show_object_attr_confidences", default=False
        )
        self.show_object_indices = self.parse_bool(
            d, "show_object_indices", default=True
        )
        self.show_object_masks = self.parse_bool(
            d, "show_object_masks", default=True
        )
        self.occluded_object_attr = self.parse_string(
            d, "occluded_object_attr", default="occluded"
        )
        self.hide_occluded_objects = self.parse_bool(
            d, "hide_occluded_objects", default=False
        )
        self.object_labels_whitelist = self.parse_array(
            d, "object_labels_whitelist", default=None
        )
        self.object_labels_blacklist = self.parse_array(
            d, "object_labels_blacklist", default=None
        )

        ##### EVENTS #####
        self.show_event_boxes = self.parse_bool(
            d, "show_event_boxes", default=True
        )
        self.show_event_labels = self.parse_bool(
            d, "show_event_labels", default=True
        )
        self.show_event_attrs = self.parse_bool(
            d, "show_event_attrs", default=True
        )
        self.show_event_confidences = self.parse_bool(
            d, "show_event_confidences", default=False
        )
        self.show_event_attr_confidences = self.parse_bool(
            d, "show_event_attr_confidences", default=False
        )
        self.show_event_indices = self.parse_bool(
            d, "show_event_indices", default=True
        )
        self.show_event_masks = self.parse_bool(
            d, "show_event_masks", default=True
        )
        self.show_event_label_on_objects = self.parse_bool(
            d, "show_event_label_on_objects", default=True
        )
        self.show_event_objects_in_same_color = self.parse_bool(
            d, "show_event_objects_in_same_color", default=True
        )
        self.occluded_event_attr = self.parse_string(
            d, "occluded_event_attr", default="occluded"
        )
        self.hide_occluded_events = self.parse_bool(
            d, "hide_occluded_events", default=False
        )
        self.event_labels_whitelist = self.parse_array(
            d, "event_labels_whitelist", default=None
        )
        self.event_labels_blacklist = self.parse_array(
            d, "event_labels_blacklist", default=None
        )

        ##### BOUNDING BOXES #####
        self.bbox_alpha = self.parse_number(d, "bbox_alpha", default=0.75)
        self.bbox_label_text_pad_pixels = self.parse_number(
            d, "bbox_label_text_pad_pixels", default=2
        )
        self.bbox_linewidth = self.parse_number(d, "bbox_linewidth", default=2)

        ##### MASKS #####
        self.mask_border_thickness = self.parse_number(
            d, "mask_border_thickness", default=2
        )
        self.mask_fill_alpha = self.parse_number(
            d, "mask_fill_alpha", default=0.7
        )
        self.show_frame_mask_semantics = self.parse_bool(
            d, "show_frame_mask_semantics", default=True
        )

        ##### ATTRIBUTE BOXES #####
        self.attrs_box_render_method = self.parse_categorical(
            d, "attrs_box_render_method", ["list", "panel"], default="panel"
        )
        self.attrs_box_bg_color = self.parse_string(
            d, "attrs_box_bg_color", default="#000000"
        )
        self.attrs_box_bg_alpha = self.parse_number(
            d, "attrs_box_bg_alpha", default=0.5
        )
        self.attrs_box_text_pad_pixels = self.parse_number(
            d, "attrs_box_text_pad_pixels", default=5
        )
        self.attrs_box_text_line_spacing_pixels = self.parse_number(
            d, "attrs_box_text_line_spacing_pixels", default=1
        )

        ##### ALL LABELS #####
        self.show_all_confidences = self.parse_bool(
            d, "show_all_confidences", default=False
        )
        self.hide_attr_values = self.parse_array(
            d, "hide_attr_values", default=None
        )
        self.hide_false_boolean_attrs = self.parse_bool(
            d, "hide_false_boolean_attrs", default=False
        )
        self.confidence_scaled_alpha = self.parse_bool(
            d, "confidence_scaled_alpha", default=False
        )

        ##### FONTS AND COLORS #####
        self.colormap_config = self.parse_object(
            d, "colormap_config", ColormapConfig, default=None
        )
        self.text_color = self.parse_string(d, "text_color", default="#FFFFFF")
        self.font_path = self.parse_string(
            d, "font_path", default=etac.DEFAULT_FONT_PATH
        )
        self.font_size = self.parse_number(d, "font_size", default=16)
        self.scale_by_media_height = self.parse_bool(
            d, "scale_by_media_height", default=True
        )

        ##### LOGO #####
        self.add_logo = self.parse_bool(d, "add_logo", default=True)
        self.logo_config = self.parse_object(
            d, "logo_config", etal.LogoConfig, default=None
        )

        self._media_height = None
        self._logo = None
        self._font = None
        self._linewidth = None
        self.set_media_size(frame_size=(1280, 720))

        #
        # Load Logo _after_ setting media size to avoid unnecessary rendering
        # of the logo
        #
        if self.logo_config is not None:
            self._logo = etal.Logo(self.logo_config)
        elif self.add_logo:
            self._logo = etal.Logo.load_default()

        if self.colormap_config is not None:
            self._colormap = self.colormap_config.build()
        else:
            self._colormap = Colormap.load_default()

    @property
    def colormap(self):
        return self._colormap

    @property
    def media_height(self):
        return self._media_height

    @property
    def font(self):
        return self._font

    @property
    def linewidth(self):
        return self._linewidth

    @property
    def logo(self):
        return self._logo

    def set_media_size(self, frame_size=None, shape=None, img=None):
        """Sets the size of the media to the given value. This allows for
        optimizing font sizes, linewidths, and logo resolutions to suit the
        dimensions of the media being annotated.

        Exactly *one* keyword argument must be provided.

        Args:
            frame_size: the (width, height) of the image/video frame
            shape: the (height, width, ...) of the image/video frame, e.g. from
                img.shape
            img: an example image/video frame
        """
        frame_size = etai.to_frame_size(
            frame_size=frame_size, shape=shape, img=img
        )

        # Set media height
        self._media_height = frame_size[1]

        # Render logo, if necessary
        if self.add_logo and self.logo is not None:
            self._logo.render_for(frame_size=frame_size)

        # Render font
        font_size = int(self.font_size * self._get_media_scale_factor())
        self._font = ImageFont.truetype(self.font_path, font_size)

        # Render linewidth
        self._linewidth = int(
            self.bbox_linewidth * self._get_media_scale_factor()
        )

    def _get_media_scale_factor(self):
        if self.scale_by_media_height:
            return self.media_height / 720.0

        return 1.0


class ColormapConfig(Config):
    """Configuration class that encapsulates the name of a Colormap and an
    instance of its associated Config class.

    Attributes:
        type: the fully-qualified class name of the Colormap, e.g.,
            `core.utils.ManualColormap`
        config: an instance of the Config class associated with the specified
            Colormap (e.g., `core.utils.ManualColormapConfig`)
    """

    def __init__(self, d):
        self.type = self.parse_string(d, "type")
        self._colormap_cls, config_cls = Configurable.parse(self.type)
        self.config = self.parse_object(d, "config", config_cls)

    def build(self):
        """Factory method that builds the Colormap instance from the config
        specified by this class.
        """
        return self._colormap_cls(self.config)


class Colormap(Configurable):
    """Class encapsulating a colormap."""

    @property
    def colors(self):
        """The array of hex colors in the colormap."""
        raise NotImplementedError("subclass must implement colors")

    def get_color(self, index):
        """Gets the color for the given index. Modular arithmetic is used, so
        any integer is valid regardless of the colormap length.
        """
        return self.colors[index % len(self.colors)]

    @classmethod
    def load_default(cls):
        """Loads the default Colormap."""
        config = ShuffledHLSColormapConfig.builder().set(num_colors=36).build()
        return ShuffledHLSColormap(config)


class ManualColormapConfig(Config):
    """Class that configures a ManualColormap."""

    def __init__(self, d):
        self.colors = self.parse_array(d, "colors")


class ManualColormap(Colormap):
    """A colormap with manually specified colors."""

    def __init__(self, config):
        """Creates a ManualColormap instance.

        Args:
            config: a ManualColormapConfig instance
        """
        self.validate(config)
        self.config = config

    @property
    def colors(self):
        """The array of hex colors in the colormap."""
        return self.config.colors


class ShuffledHLSColormapConfig(Config):
    """Class that configures an ShuffledHLSColormap."""

    def __init__(self, d):
        self.num_colors = self.parse_number(d, "num_colors")
        self.lightness = self.parse_number(d, "lightness", default=0.4)
        self.saturation = self.parse_number(d, "saturation", default=0.7)
        self.seed = self.parse_number(d, "seed", default=42)


class ShuffledHLSColormap(Colormap):
    """A colormap with equally spaced (and randonly shuffled) hues in HLS
    colorspace.
    """

    def __init__(self, config):
        """Creates an ShuffledHLSColormap instance.

        Args:
            config: a ShuffledHLSColormapConfig instance
        """
        self.validate(config)
        self.config = config
        self._colors = self._generate_colors(self.config.num_colors)

    @property
    def colors(self):
        """The array of hex colors in the colormap."""
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


class Draw(object):
    """Context manager that allows for convenient, temporary conversion of a
    numpy image into a `ImageDraw.Draw` instance when inside the context.

    The input numpy array is modified in-place when the context manager exits.
    """

    def __init__(self, img):
        """Creates a Draw instance.

        Args:
            img: an numpy image
        """
        self._img = img
        self._img_pil = None

    def __enter__(self):
        self._img_pil = Image.fromarray(self._img)
        return ImageDraw.Draw(self._img_pil)

    def __exit__(self, *args):
        self._img[:] = np.asarray(self._img_pil)


def annotate_video(
    input_path, video_labels, output_path, annotation_config=None
):
    """Annotates the video with the given labels.

    Args:
        input_path: the path to the video to annotate
        video_labels: an `eta.core.video.VideoLabels` instance describing the
            content to annotate
        output_path: the path to write the output video
        annotation_config: an optional AnnotationConfig specifying how to
            render the annotations. If omitted, the default config is used
    """
    if annotation_config is None:
        annotation_config = _DEFAULT_ANNOTATION_CONFIG

    # Render framewise labels for annotation
    video_labels = video_labels.render_framewise()
    mask_index = video_labels.mask_index

    # Annotate video
    with etav.VideoProcessor(input_path, out_video_path=output_path) as p:
        # Set media size
        annotation_config.set_media_size(frame_size=p.output_frame_size)

        # Annotate frames
        for img in p:
            logger.debug("Annotating frame %d", p.frame_number)
            frame_labels = video_labels[p.frame_number]
            img_anno = _annotate_video_frame(
                img,
                frame_labels,
                mask_index=mask_index,
                annotation_config=annotation_config,
            )
            p.write(img_anno)


def _annotate_video_frame(
    img, frame_labels, mask_index=None, annotation_config=None
):
    if annotation_config is None:
        annotation_config = _DEFAULT_ANNOTATION_CONFIG
        annotation_config.set_media_size(img=img)

    return _annotate_image(
        img, frame_labels, annotation_config, mask_index=mask_index
    )


def annotate_image(img, frame_labels, annotation_config=None):
    """Annotates the image with the given labels.

    Args:
        img: the image to annotate
        frame_labels: a FrameLabels describing the content to annotate
        annotation_config: an optional AnnotationConfig specifying how to
            render the annotations. If omitted, the default config is used

    Returns:
        the annotated image
    """
    if annotation_config is None:
        annotation_config = _DEFAULT_ANNOTATION_CONFIG

    if etai.is_gray(img):
        img = etai.gray_to_rgb(img)

    # Set media size
    annotation_config.set_media_size(img=img)

    return _annotate_image(img, frame_labels, annotation_config)


def _annotate_image(img, frame_labels, annotation_config, mask_index=None):
    # Parse config
    hide_attr_values = annotation_config.hide_attr_values
    hide_false_boolean_attrs = annotation_config.hide_false_boolean_attrs
    show_frame_attr_confidences = (
        annotation_config.show_frame_attr_confidences
        or annotation_config.show_all_confidences
    )
    add_logo = annotation_config.add_logo

    # Parse inputs
    has_mask = frame_labels.has_mask
    has_events = frame_labels.has_events
    has_objects = frame_labels.has_objects
    has_attributes = frame_labels.has_attributes

    #
    # Draw frame mask
    #

    if has_mask:
        logger.debug("Rendering frame mask")

        mask = frame_labels.mask
        if frame_labels.has_mask_index:
            mask_index = frame_labels.mask_index

        img = _draw_frame_mask(
            img, mask, annotation_config, mask_index=mask_index
        )

    #
    # Draw events
    #

    event_attr_strs = []

    if has_events:
        logger.debug("Rendering %d events", len(frame_labels.events))
        for event in frame_labels.events:
            img, attr_strs = _draw_event(img, event, annotation_config)
            event_attr_strs.extend(attr_strs)

    #
    # Draw objects
    #

    if has_objects:
        logger.debug("Rendering %d objects", len(frame_labels.objects))
        for obj in frame_labels.objects:
            img = _draw_object(img, obj, annotation_config)

    #
    # Draw attributes panel
    #

    if has_attributes or event_attr_strs:
        attr_strs = []

        if has_attributes:
            # Alphabetize
            frame_labels.attrs.sort_by_name()

            # Render attributes
            attr_strs.extend(
                _render_attrs(
                    frame_labels.attrs,
                    hide_attr_values,
                    hide_false_boolean_attrs,
                    show_frame_attr_confidences,
                )
            )

        # Append any event-level attributes
        # @todo visually separate these from other attributes?
        attr_strs.extend(event_attr_strs)

        # Draw attributes
        if attr_strs:
            logger.debug("Rendering %d frame attributes", len(attr_strs))
            img = _draw_frame_attrs(img, attr_strs, annotation_config)

    #
    # Draw logo
    #

    if add_logo:
        logger.debug("Rendering logo")
        img = _draw_logo(img, annotation_config)

    return img


def _draw_logo(img, annotation_config):
    logo = annotation_config.logo
    return logo.apply(img)


def _draw_frame_attrs(img, attr_strs, annotation_config):
    # Compute upper-left corner of attrs panel
    width = etai.Width(annotation_config.frame_attrs_box_gap)
    offset = width.render_for(img=img)
    top_left_coords = (offset, offset)

    img_anno = _draw_attrs_panel(
        img, attr_strs, annotation_config, top_left_coords=top_left_coords
    )

    return img_anno


def _draw_event(img, event, annotation_config, color=None):
    #
    # Draw event box
    #

    # Parse config
    show_box = annotation_config.show_event_boxes
    show_label = annotation_config.show_event_labels
    show_attrs = annotation_config.show_event_attrs
    show_confidence = (
        annotation_config.show_event_confidences
        or annotation_config.show_all_confidences
    )
    show_attr_confidences = (
        annotation_config.show_event_attr_confidences
        or annotation_config.show_all_confidences
    )
    show_index = annotation_config.show_event_indices
    occluded_attr = annotation_config.occluded_event_attr
    hide_occluded = annotation_config.hide_occluded_events
    labels_whitelist = annotation_config.event_labels_whitelist
    labels_blacklist = annotation_config.event_labels_blacklist
    show_mask = annotation_config.show_event_masks
    show_event_label_on_objects = annotation_config.show_event_label_on_objects
    show_event_objects_in_same_color = (
        annotation_config.show_event_objects_in_same_color
    )

    # If the event has no bounding box, return event attributes for rendering
    # via another method
    return_attrs = not event.has_bounding_box

    img_anno, attr_strs, event_color = _draw_bbox_with_attrs(
        img,
        event,
        annotation_config,
        show_box,
        show_label,
        show_attrs,
        show_confidence,
        show_attr_confidences,
        show_index,
        occluded_attr,
        hide_occluded,
        labels_whitelist,
        labels_blacklist,
        show_mask,
        color=color,
        return_attrs=return_attrs,
    )

    #
    # Draw event objects
    #

    if event.has_objects:
        # Set object color to match event color, if requested
        if show_event_objects_in_same_color:
            obj_color = event_color or color
        else:
            obj_color = None

        # Add event info to object title, if requested
        if show_event_label_on_objects:
            event_attrs = _make_event_attr(event, show_index=show_index)
        else:
            event_attrs = None

        for obj in event.objects:
            img_anno = _draw_object(
                img_anno,
                obj,
                annotation_config,
                color=obj_color,
                pre_attrs=event_attrs,
            )

    return img_anno, attr_strs


def _draw_object(img, obj, annotation_config, color=None, pre_attrs=None):
    # Parse config
    show_box = annotation_config.show_object_boxes
    show_label = annotation_config.show_object_labels
    show_attrs = annotation_config.show_object_attrs
    show_confidence = (
        annotation_config.show_object_confidences
        or annotation_config.show_all_confidences
    )
    show_attr_confidences = (
        annotation_config.show_object_attr_confidences
        or annotation_config.show_all_confidences
    )
    show_index = annotation_config.show_object_indices
    occluded_attr = annotation_config.occluded_object_attr
    hide_occluded = annotation_config.hide_occluded_objects
    labels_whitelist = annotation_config.object_labels_whitelist
    labels_blacklist = annotation_config.object_labels_blacklist
    show_mask = annotation_config.show_object_masks

    img_anno, _, _ = _draw_bbox_with_attrs(
        img,
        obj,
        annotation_config,
        show_box,
        show_label,
        show_attrs,
        show_confidence,
        show_attr_confidences,
        show_index,
        occluded_attr,
        hide_occluded,
        labels_whitelist,
        labels_blacklist,
        show_mask,
        color=color,
        pre_attrs=pre_attrs,
    )

    return img_anno


def _draw_bbox_with_attrs(
    img,
    obj_or_event,
    annotation_config,
    show_box,
    show_label,
    show_attrs,
    show_confidence,
    show_attr_confidences,
    show_index,
    occluded_attr,
    hide_occluded,
    labels_whitelist,
    labels_blacklist,
    show_mask,
    color=None,
    return_attrs=False,
    pre_attrs=None,
):
    # Parse config
    hide_attr_values = annotation_config.hide_attr_values
    hide_false_boolean_attrs = annotation_config.hide_false_boolean_attrs
    colormap = annotation_config.colormap
    bbox_alpha = annotation_config.bbox_alpha
    confidence_scaled_alpha = annotation_config.confidence_scaled_alpha
    mask_border_thickness = annotation_config.mask_border_thickness
    mask_fill_alpha = annotation_config.mask_fill_alpha

    # Parse inputs
    has_bounding_box = obj_or_event.has_bounding_box
    has_mask = obj_or_event.has_mask
    has_attributes = obj_or_event.has_attributes

    #
    # Check for immediate return
    #

    return_now = False

    # Check labels whitelist
    if labels_whitelist is not None:
        if obj_or_event.label not in labels_whitelist:
            return_now = True

    # Check labels blacklist
    if labels_blacklist is not None:
        if obj_or_event.label in labels_blacklist:
            return_now = True

    # Check for occlusion
    if has_attributes and hide_occluded:
        for attr in obj_or_event.attrs:
            if attr.name == occluded_attr and attr.value:
                return_now = True

    if return_now:
        return img, [], None

    # Scale alpha by confidence, if requested
    if confidence_scaled_alpha and obj_or_event.confidence is not None:
        bbox_alpha *= obj_or_event.confidence
        mask_fill_alpha *= obj_or_event.confidence

    # Render title string
    title_str, title_hash = _render_bbox_title(
        obj_or_event,
        show_label=show_label,
        show_confidence=show_confidence,
        show_index=show_index,
    )

    # Choose box color
    if color:
        # Use manually specified color
        bbox_color = color
    else:
        # Choose random color based on hash of title
        bbox_color = _parse_hex_color(colormap.get_color(title_hash))

    img_anno = img.copy()

    #
    # Draw segmentation mask
    #

    if has_mask and has_bounding_box and show_mask:
        mask = obj_or_event.mask
        bounding_box = obj_or_event.bounding_box
        img_anno = _draw_instance_mask(
            img_anno,
            mask,
            bounding_box,
            bbox_color,
            border_thickness=mask_border_thickness,
            border_alpha=bbox_alpha,
            fill_alpha=mask_fill_alpha,
        )

    #
    # Draw bounding box
    #

    if has_bounding_box and show_box:
        bounding_box = obj_or_event.bounding_box
        img_anno = _draw_bounding_box(
            img_anno,
            bounding_box,
            title_str,
            bbox_alpha,
            bbox_color,
            annotation_config,
        )

    #
    # Draw attributes
    #

    if has_attributes or pre_attrs:
        # Alphabetize attributes by name
        obj_or_event.attrs.sort_by_name()

        if pre_attrs:
            attrs = deepcopy(pre_attrs)
            attrs.add_container(obj_or_event.attrs)
        else:
            attrs = obj_or_event.attrs

        # Render attribute strings
        attr_strs = _render_attrs(
            attrs,
            hide_attr_values,
            hide_false_boolean_attrs,
            show_attr_confidences,
        )

        # Return attributes instead of drawing them, if requested
        if return_attrs:
            return img_anno, attr_strs, bbox_color

        # Draw attributes
        if has_bounding_box and show_attrs and attr_strs:
            logger.debug("Rendering %d bbox attributes", len(attr_strs))

            bounding_box = obj_or_event.bounding_box
            img_anno = _draw_bbox_attrs(
                img_anno, bounding_box, attr_strs, annotation_config
            )

    return img_anno, [], bbox_color


def _draw_bbox_attrs(img, bounding_box, attr_strs, annotation_config):
    # Parse config
    font = annotation_config.font
    linewidth = annotation_config.linewidth
    attrs_render_method = annotation_config.attrs_box_render_method
    label_text_pad_pixels = annotation_config.bbox_label_text_pad_pixels
    text_color = tuple(_parse_hex_color(annotation_config.text_color))

    # Get upper-left corner bounding box
    boxtlx, boxtly = bounding_box.top_left.coords_in(img=img)

    # Method 1: comma-separated attributes list
    if attrs_render_method == "list":
        with Draw(img) as draw:
            atxttlx = boxtlx + linewidth + label_text_pad_pixels
            atxttly = boxtly - 1 + label_text_pad_pixels
            attrs_str = ", ".join(attr_strs)

            draw.text(
                (atxttlx, atxttly), attrs_str, font=font, fill=text_color
            )

    # Method 2: attribute panel
    if attrs_render_method == "panel":
        # Upper-left corner of attrs panel
        atxttlx = boxtlx + 2 * linewidth
        atxttly = boxtly + 2 * linewidth - 1
        top_left_coords = (atxttlx, atxttly)

        img = _draw_attrs_panel(
            img, attr_strs, annotation_config, top_left_coords=top_left_coords
        )

    return img


def _draw_bounding_box(
    img, bounding_box, title_str, bbox_alpha, bbox_color, annotation_config
):
    # Parse config
    font = annotation_config.font
    linewidth = annotation_config.linewidth
    label_text_pad_pixels = annotation_config.bbox_label_text_pad_pixels
    text_color = tuple(_parse_hex_color(annotation_config.text_color))

    overlay = img.copy()

    #
    # Draw bounding box
    #

    boxtlx, boxtly = bounding_box.top_left.coords_in(img=img)
    boxbrx, boxbry = bounding_box.bottom_right.coords_in(img=img)
    cv2.rectangle(
        overlay, (boxtlx, boxtly), (boxbrx, boxbry), bbox_color, linewidth
    )

    #
    # Draw box title
    #

    # Title background
    if title_str:
        textw, texth = font.getsize(title_str)
        bgtlx = boxtlx - linewidth + 1
        bgbry = boxtly - linewidth + 1
        bgbrx = bgtlx + textw + 2 * (label_text_pad_pixels + _DX)
        bgtly = bgbry - texth - 2 * label_text_pad_pixels
        cv2.rectangle(overlay, (bgtlx, bgtly), (bgbrx, bgbry), bbox_color, -1)

    img_anno = cv2.addWeighted(overlay, bbox_alpha, img, 1 - bbox_alpha, 0)

    # Title text
    if title_str:
        with Draw(img_anno) as draw:
            txttlx = bgtlx + label_text_pad_pixels + _DX
            txttly = bgtly + label_text_pad_pixels - 1
            draw.text((txttlx, txttly), title_str, font=font, fill=text_color)

    return img_anno


def _draw_attrs_panel(
    img, attr_strs, annotation_config, center_coords=None, top_left_coords=None
):
    # Parse config
    font = annotation_config.font
    box_pad = annotation_config.attrs_box_text_pad_pixels
    line_gap = annotation_config.attrs_box_text_line_spacing_pixels
    text_size = _compute_max_text_size(font, attr_strs)  # width, height
    text_color = tuple(_parse_hex_color(annotation_config.text_color))
    bg_color = _parse_hex_color(annotation_config.attrs_box_bg_color)
    bg_alpha = annotation_config.attrs_box_bg_alpha
    num_attrs = len(attr_strs)

    #
    # Compute box coordinates
    #

    bgw = text_size[0] + 2 * (box_pad + _DX)
    bgh = num_attrs * text_size[1] + (num_attrs - 1) * line_gap + 2 * box_pad
    if center_coords:
        cx, cy = center_coords
        top_left_coords = (int(cx - 0.5 * bgw), int(cy - 0.5 * bgh))
    if not top_left_coords:
        raise ValueError(
            "Either `center_coords` or `top_left_coords` must be provided"
        )

    bgtlx, bgtly = top_left_coords
    bgbrx = bgtlx + bgw
    bgbry = bgtly + bgh

    # Draw background
    overlay = img.copy()
    cv2.rectangle(overlay, (bgtlx, bgtly), (bgbrx, bgbry), bg_color, -1)
    img_anno = cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0)

    # Draw attributes
    with Draw(img_anno) as draw:
        for idx, attr_str in enumerate(attr_strs):
            txttlx = bgtlx + box_pad + _DX
            txttly = bgtly + box_pad + idx * line_gap + idx * text_size[1] - 1
            draw.text((txttlx, txttly), attr_str, font=font, fill=text_color)

    return img_anno


def _draw_frame_mask(img, mask, annotation_config, mask_index=None):
    # Parse config
    colormap = annotation_config.colormap
    fill_alpha = annotation_config.mask_fill_alpha
    show_semantics = annotation_config.show_frame_mask_semantics

    # Parse inputs
    has_mask_index = mask_index is not None
    if not has_mask_index:
        show_semantics = False

    # Lists of mask semantics to render
    center_coords = []
    attr_strs = []

    #
    # Draw frame mask
    #

    img_anno = img

    if fill_alpha is not None and fill_alpha > 0:
        overlay = img.copy()

        mask = etai.render_frame_mask(mask, img=img)
        for index in np.unique(mask):
            if has_mask_index:
                if index not in mask_index:
                    # When we have a `mask_index`, skip regions with no value
                    continue
            elif index == 0:
                # When no `mask_index` exists, treat 0 as background
                continue

            color = _parse_hex_color(colormap.get_color(51 * index))
            maski = mask == index
            overlay[maski] = color

            if show_semantics and index in mask_index:
                coords = _compute_region_centroids(maski)
                attr_str = mask_index[index].value
                center_coords.extend(coords)
                attr_strs.extend([attr_str] * len(coords))

        img_anno = cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0)

    #
    # Draw mask semantics
    #

    if show_semantics and center_coords:
        img_anno = _annotate_frame_mask_regions(
            img_anno, center_coords, attr_strs, annotation_config
        )

    return img_anno


def _compute_region_centroids(mask):
    mask = mask.astype(np.uint8)

    # Label the largest contour
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    c = max(contours, key=cv2.contourArea)

    # Extract centroid, if possible
    coords = []
    M = cv2.moments(c)
    if M["m00"] > 0:
        tlx = int(M["m10"] / M["m00"])
        tly = int(M["m01"] / M["m00"])
        coords.append((tlx, tly))

    return coords


def _annotate_frame_mask_regions(
    img, center_coords, attr_strs, annotation_config
):
    for coords, attr_str in zip(center_coords, attr_strs):
        img = _draw_attrs_panel(
            img, [attr_str], annotation_config, center_coords=coords
        )

    return img


def _draw_instance_mask(
    img,
    mask,
    bounding_box,
    color,
    border_thickness=None,
    border_alpha=None,
    fill_alpha=None,
):
    mask, offset = etai.render_instance_mask(
        mask, bounding_box, img=img, as_bool=False
    )
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=offset
    )

    img_anno = img.copy()

    if fill_alpha is not None and fill_alpha > 0:
        img_mask = img_anno.copy()
        cv2.drawContours(img_mask, contours, -1, color, cv2.FILLED)
        img_anno = cv2.addWeighted(
            img_mask, fill_alpha, img_anno, 1 - fill_alpha, 0
        )

    if (
        border_thickness is not None
        and border_thickness > 0
        and border_alpha is not None
        and border_alpha > 0
    ):
        img_border = img_anno.copy()
        cv2.drawContours(img_border, contours, -1, color, border_thickness)
        img_anno = cv2.addWeighted(
            img_border, border_alpha, img_anno, 1 - border_alpha, 0
        )

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


def _make_event_attr(event, show_index=True):
    value = event.label or ""
    if show_index and event.has_index:
        value += " %d" % event.index
    value = value.strip()

    attrs = etad.AttributeContainer()
    event_attr = etad.CategoricalAttribute(
        "event", value, confidence=event.confidence
    )
    attrs.add(event_attr)
    return attrs


def _render_attrs(
    attrs, hide_attr_values, hide_false_boolean_attrs, show_confidence
):
    attr_strs = []
    for attr in attrs:
        if hide_attr_values is not None and attr.value in hide_attr_values:
            # Hide this attribute
            continue

        if (
            hide_false_boolean_attrs
            and isinstance(attr, etad.BooleanAttribute)
            and not attr.value
        ):
            # Hide false boolean attribute
            continue

        attr_strs.append(
            _render_attr_name_value(attr, show_confidence=show_confidence)
        )

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


def _render_bbox_title(
    obj_or_event, show_label=True, show_confidence=False, show_index=True
):
    label = obj_or_event.label
    confidence = obj_or_event.confidence
    index = obj_or_event.index

    title_str = ""
    title_hash = 0

    if show_label and label is not None:
        _label = _clean_str(label).upper()
        title_str += _label
        title_hash += _label.__hash__()

    if show_confidence and confidence is not None:
        title_str += " (%.2f)" % confidence

    if show_index and index is not None:
        title_str += "     %d" % index
        title_hash += str(index).__hash__()

    return title_str, title_hash


def _render_numeric_attr_value(attr):
    if isinstance(attr.value, int):
        return "%d" % attr.value

    return "%.2g" % attr.value


def _clean_str(s):
    return str(s).lower().replace("_", " ")
