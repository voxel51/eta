#!/usr/bin/env python
"""
A module for generating clips from a video.

Info:
    type: eta.core.types.Module
    version: 0.1.0

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

import logging
import sys

from eta.core.config import Config
from eta.core.frameutils import FrameRanges
import eta.core.image as etai
import eta.core.module as etam
import eta.core.video as etav


logger = logging.getLogger(__name__)


class ModuleConfig(etam.BaseModuleConfig):
    """Clip configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    """

    def __init__(self, d):
        super(ModuleConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    """Data configuration settings.

    Inputs:
        input_path (eta.core.types.Video): The input video
        frame_ranges_path (eta.core.types.FrameRanges): [None] A FrameRanges
            instance specifying the clips to generate

    Outputs:
        output_video_clips_path (eta.core.types.VideoClips): [None] The output
            video clips
        output_frames_dir (eta.core.types.ImageSequenceDirectory): [None] A
            directory in which to write the sampled frames
        output_frames_path (eta.core.types.ImageSequence): [None] The output
            sampled frames
    """

    def __init__(self, d):
        self.input_path = self.parse_string(d, "input_path")
        self.frame_ranges_path = self.parse_string(
            d, "frame_ranges_path", default=None
        )
        self.output_video_clips_path = self.parse_string(
            d, "output_video_clips_path", default=None
        )
        self.output_frames_dir = self.parse_string(
            d, "output_frames_dir", default=None
        )
        self.output_frames_path = self.parse_string(
            d, "output_frames_path", default=None
        )

        self._validate()

    def _validate(self):
        Config.parse_mutually_exclusive_fields(
            {
                "output_video_clips_path": self.output_video_clips_path,
                "output_frames_dir": self.output_frames_dir,
                "output_frames_path": self.output_frames_path,
            }
        )


class ParametersConfig(Config):
    """Parameter configuration settings.

    Parameters:
        frames (eta.core.types.String): [None] A frames string specifying the
            clips to generate
    """

    def __init__(self, d):
        self.frames = self.parse_string(d, "frames", default=None)


def _clip_videos(clip_config):
    for data in clip_config.data:
        frames = _get_frames(data, clip_config.parameters)
        _clip_video(data, frames)


def _get_frames(data, parameters):
    if data.frame_ranges_path:
        return FrameRanges.from_json(data.frame_ranges_path)

    return parameters.frames


def _clip_video(data, frames):
    logger.info("Generating video clips for '%s'", data.input_path)

    # Collect output paths
    if data.output_frames_dir:
        out_images_path = etai.make_image_sequence_patt(data.output_frames_dir)
    else:
        out_images_path = data.output_frames_path
    out_clips_path = data.output_video_clips_path

    # Sample clips
    with etav.VideoProcessor(
        data.input_path,
        frames=frames,
        out_images_path=out_images_path,
        out_clips_path=out_clips_path,
    ) as p:
        for img in p:
            p.write(img)


def run(config_path, pipeline_config_path=None):
    """Run the clip_videos module.

    Args:
        config_path: path to a ModuleConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    """
    clip_config = ModuleConfig.from_json(config_path)
    etam.setup(clip_config, pipeline_config_path=pipeline_config_path)
    _clip_videos(clip_config)


if __name__ == "__main__":
    run(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
