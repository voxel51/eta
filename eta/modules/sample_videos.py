#!/usr/bin/env python
'''
A module for sampling frames from videos.

The sampled frames are written to a directory with filenames that encode their
frame number in the original video.

Info:
    type: eta.core.types.Module
    version: 0.1.0

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
import os
import sys

import numpy as np

import eta
from eta.core.config import Config
import eta.core.image as etai
import eta.core.module as etam
import eta.core.utils as etau
import eta.core.video as etav


logger = logging.getLogger(__name__)


class SampleVideosConfig(etam.BaseModuleConfig):
    '''Sample videos configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(SampleVideosConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        input_path (eta.core.types.Video): The input video

    Outputs:
        output_frames_dir (eta.core.types.ImageSequenceDirectory): A directory
            of sampled frames
    '''

    def __init__(self, d):
        self.input_path = self.parse_string(d, "input_path")
        self.output_frames_dir = self.parse_string(d, "output_frames_dir")


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        accel (eta.core.types.Number): [None] A desired acceleration factor to
            apply when sampling frames. For example, an acceleration of 2x
            would correspond to sampling every other frame
        fps (eta.core.types.Number): [None] The desired sampling rate, which
            must be less than the frame rate of the input video
        size (eta.core.types.Array): [None] A desired output (width, height)
            of the sampled frames. Dimensions can be -1, in which case the
            input aspect ratio is preserved
        max_fps (eta.core.types.Number): [None] The maximum sampling rate
            allowed for the output video. If this parameter is specified, the
            `accel` parameter will be increased if necessary to satisfy this
            constraint
        max_size (eta.core.types.Array): [None] A maximum (width, height)
            allowed for the sampled frames. Frames are resized as necessary to
            meet this limit. Dimensions can be -1, in which case no constraint
            is applied to them
        always_sample_last (eta.core.types.Boolean): [False] Whether to always
            sample the last frame of the video
        max_video_file_size (eta.core.types.Number): [None] The maximum file
            size of the input video in bytes. If size is greater than this,
            an error will be thrown
        max_video_duration (eta.core.types.Number): [None] The maximum duration
            of the input video in seconds. If duration is greater than this,
            an error will be thrown
    '''

    def __init__(self, d):
        self.accel = self.parse_number(d, "accel", default=None)
        self.fps = self.parse_number(d, "fps", default=None)
        self.size = self.parse_array(d, "size", default=None)
        self.max_fps = self.parse_number(d, "max_fps", default=None)
        self.max_size = self.parse_array(d, "max_size", default=None)
        self.always_sample_last = self.parse_bool(
            d, "always_sample_last", default=False)
        self.max_video_file_size = self.parse_number(
            d, "max_video_file_size", default=None)
        self.max_video_duration = self.parse_number(
            d, "max_video_duration", default=None)


def _sample_videos(config):
    parameters = config.parameters
    for data in config.data:
        _process_video(data.input_path, data.output_frames_dir, parameters)


def _process_video(input_path, output_frames_dir, parameters):
    stream_info = etav.VideoStreamInfo.build_for(input_path)

    _check_input_video_size(
        stream_info, parameters.max_video_file_size,
        parameters.max_video_duration)

    ifps = stream_info.frame_rate
    isize = stream_info.frame_size
    iframe_count = stream_info.total_frame_count

    #
    # Compute acceleration using the following strategy:
    #   (i) use `accel` if provided
    #   (ii) if `accel` is not provided, use `fps` to set `accel`
    #   (iii) if a `max_fps` is provided, always increase `accel` if necessary
    #
    accel = parameters.accel
    fps = parameters.fps
    max_fps = parameters.max_fps
    if accel is not None:
        if accel < 1:
            raise ValueError(
                "Acceleration factor must be greater than 1; found %d" % accel)

    elif fps is not None:
        if fps > ifps:
            raise ValueError(
                "Sampling frame rate (%g) cannot be greater than input frame "
                "rate (%g)" % (fps, ifps))

        accel = ifps / fps

    if max_fps is not None and max_fps > 0:
        min_accel = ifps / max_fps
        if accel is None or accel < min_accel:
            logger.warning(
                "Maximum frame rate %g requires acceleration of at least %g; "
                "setting `accel = %g` now", max_fps, min_accel, min_accel)

            accel = min_accel

    if accel is None:
        accel = 1.0

    # Determine frames to sample
    sample_pts = np.arange(1, iframe_count, accel)
    sample_frames = set(int(round(x)) for x in sample_pts)
    if parameters.always_sample_last:
        sample_frames.add(iframe_count)

    # Compute output frame size
    if parameters.size is not None:
        psize = etai.parse_frame_size(parameters.size)
        osize = etai.infer_missing_dims(psize, isize)
    else:
        osize = isize
    if parameters.max_size is not None:
        msize = etai.parse_frame_size(parameters.max_size)
        osize = etai.clamp_frame_size(osize, msize)

    # Avoid resizing if possible
    resize_frames = osize != isize
    if resize_frames:
        owidth, oheight = osize
        logger.info("Resizing frames to %d x %d", owidth, oheight)

    # Sample frames
    frames = sorted(sample_frames)
    output_patt = os.path.join(
        output_frames_dir,
        eta.config.default_sequence_idx + eta.config.default_image_ext)
    size = osize if resize_frames else None
    etav.sample_select_frames(
        input_path, frames, output_patt=output_patt, size=size, fast=True)


def _check_input_video_size(
        stream_info, max_video_file_size, max_video_duration):
    if (max_video_file_size is not None
            and stream_info.size_bytes > max_video_file_size):
        raise ValueError(
            "Input video file size must be less than %s; found %s" % (
                etau.to_human_bytes_str(max_video_file_size),
                etau.to_human_bytes_str(stream_info.size_bytes)))

    if (max_video_duration is not None
            and stream_info.duration > max_video_duration):
        raise ValueError(
            "Input video duration must be less than %s; found %s" % (
                etau.to_human_time_str(max_video_duration),
                etau.to_human_time_str(stream_info.duration)))


def run(config_path, pipeline_config_path=None):
    '''Run the sample_videos module.

    Args:
        config_path: path to a SampleVideosConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    config = SampleVideosConfig.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)
    _sample_videos(config)


if __name__ == "__main__":
    run(*sys.argv[1:])
