#!/usr/bin/env python
'''
Sample video frames.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
import sys

sys.path.insert(0, "..")
from core.config import Config
import core.events as ev
import core.video  as vd
import core.image  as im


def sample_video_by_fps(data_config):
    assert data_config.fps != -1, "Must provide 'fps'"
    print "Sampling video '%s' at %s fps" % (
        data_config.input_path, str(data_config.fps))

    vd.FFmpegVideoSampler(fps=data_config.fps).run(
        data_config.input_path,
        data_config.output_path,
    )


def sample_video_by_clips(data_config):
    assert data_config.clips_path != None, "Must provide 'clips_path'"
    print "Sampling video '%s' by clips '%s'" % (
        data_config.input_path, data_config.clips_path)

    detections = ev.EventDetection.from_json(data_config.clips_path)
    frames = detections.to_series().to_str()

    processor = vd.VideoProcessor(
        data_config.input_path,
        frames=frames,
        out_impath=data_config.output_path,
    )
    with processor:
        for img in processor:
            processor.write(img)


def sample_videos(sample_config):
    for data_config in sample_config.data:
        if data_config.fps != -1:
            sample_video_by_fps(data_config)
        else:
            sample_video_by_clips(data_config)


class SampleConfig(Config):
    '''Sampler configuration settings.'''

    def __init__(self, d):
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    '''Data configuration settings.

    Either `fps` or `clips_path` must be specified.

    @todo add a fps/clips module keyword to handle each case separately.
    '''

    def __init__(self, d):
        self.input_path = self.parse_string(d, "input_path")
        self.output_path = self.parse_string(d, "output_path")
        self.fps = self.parse_number(d, "fps", default=-1)
        self.clips_path = self.parse_string(d, "clips_path", default=None)


if __name__ == "__main__":
    sample_videos(SampleConfig.from_json(sys.argv[1]))
