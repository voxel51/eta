#!/usr/bin/env python
'''
A module for generating clips from a video.

Info:
    type: eta.core.types.Module
    version: 0.1.0

Copyright 2017-2018, Voxel51, LLC
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
import sys

from eta.core.config import Config
import eta.core.events as etae
import eta.core.image as etai
import eta.core.module as etam
import eta.core.video as etav


logger = logging.getLogger(__name__)


class ClipConfig(etam.BaseModuleConfig):
    '''Clip configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(ClipConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)

        self._validate()

    def _validate(self):
        for data in self.data:
            Config.parse_mutually_exclusive_fields({
                "event_detection_path": data.event_detection_path,
                "event_series_path": data.event_series_path,
                "frames": self.parameters.frames,
            })


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        input_path (eta.core.types.Video): The input video
        event_detection_path (eta.core.types.EventDetection): [None] Per-frame
            binary labels defining the clips to generate
        event_series_path (eta.core.types.EventSeries): [None] An EventSeries
            specifying the clips to generate

    Outputs:
        output_video_clips_path (eta.core.types.VideoClips): [None] The output
            video clips
        output_frames_dir (eta.core.types.ImageSequenceDirectory): [None] A
            directory in which to write the sampled frames
        output_frames_path (eta.core.types.ImageSequence): [None] The output
            sampled frames
    '''

    def __init__(self, d):
        self.input_path = self.parse_string(d, "input_path")
        self.event_detection_path = self.parse_string(
            d, "event_detection_path", default=None)
        self.event_series_path = self.parse_string(
            d, "event_series_path", default=None)
        self.output_video_clips_path = self.parse_string(
            d, "output_video_clips_path", default=None)
        self.output_frames_dir = self.parse_string(
            d, "output_frames_dir", default=None)
        self.output_frames_path = self.parse_string(
            d, "output_frames_path", default=None)

        self._validate()

    def _validate(self):
        Config.parse_mutually_exclusive_fields({
            "output_video_clips_path": self.output_video_clips_path,
            "output_frames_dir": self.output_frames_dir,
            "output_frames_path": self.output_frames_path,
        })


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        frames (eta.core.types.String): [None] A frames string specifying the
            clips to generate
    '''

    def __init__(self, d):
        self.frames = self.parse_string(d, "frames", default=None)


def _clip_videos(clip_config):
    for data in clip_config.data:
        frames = _get_frames(data, clip_config.parameters)
        _clip_video(data, frames)


def _get_frames(data, parameters):
    if data.event_detection_path:
        # Get frames from per-frame detections
        detections = etae.EventDetection.from_json(data.event_detection_path)
        frames = detections.to_series().to_str()
    elif data.event_series_path:
        # Get frames from clip series
        series = etae.EventSeries.from_json(data.event_series_path)
        frames = series.to_str()
    else:
        # Manually specified frames
        frames = parameters.frames

    return frames


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
            data.input_path, frames=frames, out_images_path=out_images_path,
            out_clips_path=out_clips_path) as p:
        for img in p:
            p.write(img)


def run(config_path, pipeline_config_path=None):
    '''Run the clip_videos module.

    Args:
        config_path: path to a ClipConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    clip_config = ClipConfig.from_json(config_path)
    etam.setup(clip_config, pipeline_config_path=pipeline_config_path)
    _clip_videos(clip_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
