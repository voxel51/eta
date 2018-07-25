#!/usr/bin/env python
'''
A module for resizing videos.

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
import eta.core.image as etai
import eta.core.module as etam
import eta.core.utils as etau
import eta.core.video as etav
import eta.core.ziputils as etaz


logger = logging.getLogger(__name__)


class ResizeConfig(etam.BaseModuleConfig):
    '''Resize configuration settings.

    Attributes:
        data (DataConfig)
        parameters (ParametersConfig)
    '''

    def __init__(self, d):
        super(ResizeConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Inputs:
        input_path (eta.core.types.Video): [None] The input video
        input_zip (eta.core.types.ZippedVideoFileDirectory): [None] A zip file
            containing a directory of input video files

    Outputs:
        output_video_path (eta.core.types.VideoFile): [None] The output resized
            video
        output_frames_dir (eta.core.types.ImageSequenceDirectory): [None] A
            directory in which to write the resized frames
        output_frames_path (eta.core.types.ImageSequence): [None] The output
            resized frames pattern
        output_zip (eta.core.types.ZippedVideoFileDirectory): [None] A zip file
            containing a directory of resized video files
    '''

    def __init__(self, d):
        self.input_path = self.parse_string(d, "input_path", default=None)
        self.input_zip = self.parse_string(d, "input_zip", default=None)
        self.output_video_path = self.parse_string(
            d, "output_video_path", default=None)
        self.output_frames_dir = self.parse_string(
            d, "output_frames_dir", default=None)
        self.output_frames_path = self.parse_string(
            d, "output_frames_path", default=None)
        self.output_zip = self.parse_string(d, "output_zip", default=None)

        self._output_field = None
        self._output_val = None
        self._parse_outputs()

    @property
    def is_zip(self):
        return self.input_zip and self.output_zip

    @property
    def output_field(self):
        return self._output_field

    @property
    def output_path(self):
        return self._output_val

    def _parse_outputs(self):
        if self.is_zip:
            self._output_field = "output_zip"
            self._output_val = self.output_zip
            return

        field, val = Config.parse_mutually_exclusive_fields({
            "output_video_path": self.output_video_path,
            "output_frames_dir": self.output_frames_dir,
            "output_frames_path": self.output_frames_path,
        })
        if field == "output_frames_dir":
            val = etai.make_image_sequence_patt(val)
        self._output_field = field
        self._output_val = val


class ParametersConfig(Config):
    '''Parameter configuration settings.

    Parameters:
        scale (eta.core.types.Number): [None] A numeric scale factor to apply
            to the input resolution
        size (eta.core.types.Array): [None] A desired output (width, height)
            of the video. Dimensions can be -1, in which case the input aspect
            ratio is preserved
        max_size (eta.core.types.Array): [None] A maximum (width, height)
            allowed for the video. Dimensions can be -1, in which case no
            constraint is applied to them
        ffmpeg_out_opts (eta.core.types.Array): [None] An array of ffmpeg
            output options
    '''

    def __init__(self, d):
        self.scale = self.parse_number(d, "scale", default=None)
        self.size = self.parse_array(d, "size", default=None)
        self.max_size = self.parse_array(d, "max_size", default=None)
        self.ffmpeg_out_opts = self.parse_array(
            d, "ffmpeg_out_opts", default=None)


def _resize_videos(resize_config):
    parameters = resize_config.parameters
    for data in resize_config.data:
        if data.is_zip:
            _process_zip(data.input_zip, data.output_zip, parameters)
        else:
            _process_video(data.input_path, data.output_path, parameters)


def _process_zip(input_zip, output_zip, parameters):
    input_paths = etaz.extract_zip(input_zip)
    output_paths = etaz.make_parallel_files(output_zip, input_paths)

    # Iterate over videos
    for input_path, output_path in zip(input_paths, output_paths):
        _process_video(input_path, output_path, parameters)

    # Collect outputs
    etaz.make_zip(output_zip)


def _process_video(input_path, output_path, parameters):
    # Compute output frame size
    isize = etav.get_frame_size(input_path)
    if parameters.scale:
        osize = etai.scale_frame_size(isize, parameters.scale)
    elif parameters.size:
        psize = etai.parse_frame_size(parameters.size)
        osize = etai.infer_missing_dims(psize, isize)
    else:
        osize = isize

    # Apply size limit, if requested
    if parameters.max_size:
        msize = etai.parse_frame_size(parameters.max_size)
        osize = etai.clamp_frame_size(osize, msize)

    # Handle no-ops efficiently
    same_size = osize == isize
    if same_size and etav.is_same_video_file_format(input_path, output_path):
        logger.info(
            "Same frame size and video format detected, so no computation is "
            "required. Just symlinking %s to %s", output_path, input_path)
        etau.symlink_file(input_path, output_path)
        return

    # Resize video
    logger.info("Resizing video '%s' to size %s", input_path, str(osize))
    etav.FFmpegVideoResizer(
        size=osize, out_opts=parameters.ffmpeg_out_opts).run(
            input_path, output_path)


def run(config_path, pipeline_config_path=None):
    '''Run the resize_videos module.

    Args:
        config_path: path to a ResizeConfig file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    resize_config = ResizeConfig.from_json(config_path)
    etam.setup(resize_config, pipeline_config_path=pipeline_config_path)
    _resize_videos(resize_config)


if __name__ == "__main__":
    run(*sys.argv[1:])
