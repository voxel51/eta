#!/usr/bin/env python
'''
Generate clips from a video.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
import sys

from eta.core.config import Config, ConfigError
import eta.core.events as ev
import eta.core.video as vd


def clip_videos(clip_config):
    for data_config in clip_config.data:
        print "Generating video clips for '%s'" % data_config.input_path
        with vd.VideoProcessor(
            data_config.input_path,
            frames=data_config.get_frames(),
            out_vidpath=data_config.output_path,
        ) as p:
            for img in p:
                p.write(img)


class ClipConfig(Config):
    '''Clip configuration settings.'''

    def __init__(self, d):
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    '''Data configuration settings.'''

    def __init__(self, d):
        self.input_path = self.parse_string(d, "input_path")
        self.output_path = self.parse_string(d, "output_path")
        self.events_json_path = self.parse_string(
            d, "events_json_path", default=None)
        self.frames = self.parse_string(d, "frames", default=None)

    def get_frames(self):
        if self.events_json_path:
            return ev.EventSeries.from_json(self.events_json_path).to_str()
        elif self.frames:
            return self.frames
        else:
            raise ConfigError("Expected 'events_json_path' or 'frames'")


if __name__ == "__main__":
    clip_videos(ClipConfig.from_json(sys.argv[1]))
