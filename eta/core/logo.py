"""
Tools for rendering logos on images.

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

import os
import sys

import eta.constants as etac
from eta.core.config import Config, Configurable
import eta.core.image as etai
import eta.core.utils as etau


class LogoConfig(Config):
    """Logo configuration settings.

    At least one of `vector_path` and `raster_path` must be provided.
    If a vector image is provided, it is always used. If the vector image
    fails to render or if none is provided, the raster image is used.

    Note that `vector_path` and `raster_path` are passed through
    `eta.core.utils.fill_config_patterns` at load time, so they can contain
    patterns to be resolved.

    Attributes:
        vector_path: path to a vector image
        raster_path: path to a raster image
        width: a string like "30%" or "256px" specifying the width of the
            rendered logo. Must be a valid input to `etai.Width(width)`
        gap: a string like "1%" or "10px" specifying the gap between the
            logo and the image border. Relative gaps are interpreted with
            respect to the image width. Must be a valid input to
            to etai.Width(width)
        gapx: a gap string only for the x coordinate
        gapy: a gap string only for the y coordinate
        loc: a string like "bl" or "top-right" specifying the location to
            apply the logo. Must be a valid input to `etai.Location(loc)`
    """

    def __init__(self, d):
        _vector_path = self.parse_string(d, "vector_path", default=None)
        if _vector_path:
            self.vector_path = etau.fill_config_patterns(_vector_path)
        else:
            self.vector_path = _vector_path

        _raster_path = self.parse_string(d, "raster_path", default=None)
        if _raster_path:
            self.raster_path = etau.fill_config_patterns(_raster_path)
        else:
            self.raster_path = _raster_path

        self.width = self.parse_string(d, "width", default="8%")
        self.gap = self.parse_string(d, "gap", default="1%")
        self.gapx = self.parse_string(d, "gapx", default=None)
        self.gapy = self.parse_string(d, "gapy", default=None)
        self.loc = self.parse_string(d, "loc", default="top-right")

    @classmethod
    def load_default(cls):
        """Loads the default LogoConfig."""
        return cls.from_json(etac.DEFAULT_LOGO_CONFIG_PATH)


class Logo(Configurable):
    """Class for rendering a vector/raster logo onto an image."""

    def __init__(self, config=None):
        """Constructs a Logo instance.

        Args:
            config: an LogoConfig instance. If omitted, the default LogoConfig
                is used
        """
        if config is None:
            config = LogoConfig.load_default()
        self.validate(config)
        self.config = config
        self._width = etai.Width(self.config.width)
        if self.config.gapx:
            self._gapx = etai.Width(self.config.gapx)
        else:
            self._gapx = etai.Width(self.config.gap)
        if self.config.gapy:
            self._gapy = etai.Width(self.config.gapy)
        else:
            self._gapy = etai.Width(self.config.gap)
        self._loc = etai.Location(self.config.loc)
        self._logo = None

    @classmethod
    def load_default(cls):
        """Loads the default Logo."""
        return cls.from_config(LogoConfig.load_default())

    def render_for(self, frame_size=None, shape=None, img=None):
        """Renders the logo for the given frame size/shape/image.

        Pass any *one* of the keyword arguments to render the logo.

        Args:
            frame_size: the (width, height) of the image
            shape: the (height, width, ...) of the image, e.g. from img.shape
            img: the image itself
        """
        # Compute width
        w = self._width.render_for(frame_size=frame_size, shape=shape, img=img)

        # Render vector image
        if self.config.vector_path:
            self._logo = etai.rasterize(self.config.vector_path, w)

        # Render raster image, if necessary
        if self._logo is None and self.config.raster_path:
            raw_logo = etai.read(self.config.raster_path, include_alpha=True)
            self._logo = etai.resize(raw_logo, width=w)

    def apply(self, img):
        """Applies the logo to the given image.

        If something goes wrong or the logo hasn't been rendered, returns the
        input image.
        """
        if self._logo is None:
            return img

        # Compute top-left coordinates of logo in img
        gapx = self._gapx.render_for(img=img)
        gapy = self._gapy.render_for(img=img)

        def offset(dim, gap):
            return img.shape[dim] - self._logo.shape[dim] - gap

        if self._loc.is_top_left:
            x0, y0 = gapx, gapy
        elif self._loc.is_top_right:
            x0, y0 = offset(1, gapx), gapy
        elif self._loc.is_bottom_right:
            x0, y0 = offset(1, gapx), offset(0, gapy)
        elif self._loc.is_bottom_left:
            x0, y0 = gapx, offset(0, gapy)
        else:
            return img

        # Overly logo
        return etai.overlay(img, self._logo, x0=x0, y0=y0)
