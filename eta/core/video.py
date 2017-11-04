'''
Core video processing tools.

Copyright 2017, Voxel51, LLC
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

import errno
import json
import os
from subprocess import Popen, PIPE
import threading

import cv2
import numpy as np

from eta.core.config import Config
import eta.core.image as im
from eta.core.numutils import GrowableArray
from eta.core import utils


def get_stream_info(inpath):
    '''Get stream info for the video using ffprobe.

    Args:
        inpath: video path

    Returns:
        stream: a dictionary of stream info

    Raises:
        VideoReaderError: if no stream info was found
    '''
    ffprobe = FFprobe(opts=[
        "-print_format", "json",     # return json string
        "-show_streams",             # get stream info
    ])
    out = ffprobe.run(inpath)

    try:
        info = json.loads(out)
        return info["streams"][0]  # assume first stream is main video
    except:
        raise VideoReaderError("Unable to get stream info for '%s'" % inpath)


def get_encoding_str(inpath, use_ffmpeg=True):
    '''Get the encoding string of the input video.

    Args:
        inpath: video path
        use_ffmpeg: whether to use ffmpeg (True) or OpenCV (False)
    '''
    r = FFmpegVideoReader(inpath) if use_ffmpeg else OpenCVVideoReader(inpath)
    with r:
        return r.encoding_str


def get_frame_rate(inpath, use_ffmpeg=True):
    '''Get the frame rate of the input video.

    Args:
        inpath: video path
        use_ffmpeg: whether to use ffmpeg (True) or OpenCV (False)
    '''
    r = FFmpegVideoReader(inpath) if use_ffmpeg else OpenCVVideoReader(inpath)
    with r:
        return r.frame_rate


def get_frame_size(inpath, use_ffmpeg=True):
    '''Get the frame (width, height) of the input video.

    Args:
        inpath: video path
        use_ffmpeg: whether to use ffmpeg (True) or OpenCV (False)
    '''
    r = FFmpegVideoReader(inpath) if use_ffmpeg else OpenCVVideoReader(inpath)
    with r:
        return r.frame_size


# @todo: once frame counts for directories of frames are resolved for
#        FFmpegVideoReader, use FFmpeg here
def get_frame_count(inpath, use_ffmpeg=False):
    '''Get the number of frames in the input video.

    Args:
        inpath: video path
        use_ffmpeg: whether to use ffmpeg (True) or OpenCV (False)
    '''
    r = FFmpegVideoReader(inpath) if use_ffmpeg else OpenCVVideoReader(inpath)
    with r:
        return r.total_frame_count


class VideoProcessor(object):
    '''Class for reading a video and writing a new video frame-by-frame.

    The typical usage is:
    ```
    with VideoProcessor(...) as p:
        for img in p:
            new_img = ... # process img
            p.write(new_img)
    ```
    '''

    # @todo: switch to in_use_ffmpeg=True
    def __init__(
            self,
            inpath,
            frames="*",
            in_use_ffmpeg=False,
            out_use_ffmpeg=True,
            out_impath=None,
            out_vidpath=None,
            out_fps=None,
            out_size=None,
            out_opts=None,
        ):
        '''Construct a new video processor.

        Args:
            inpath: path to the input video. Passed directly to a VideoReader

            frames: a string specifying the range(s) of frames to process.
                Passed directly to a VideoReader

            in_use_ffmpeg: whether to use FFmpegVideoReader to read input
                videos rather than OpenCVVideoReader

            out_use_ffmpeg: whether to use FFmpegVideoWriter to write output
                videos rather than OpenCVVideoWriter

            out_impath: a path like "/path/to/frames/%05d.png" with one
                placeholder that specifies where to save frames as individual
                images when the write() method is called. When out_impath is
                None or "", no images are written

            out_vidpath: a path like "/path/to/video/%05d-%05d.mp4" with two
                placeholders that specifies where to save output videos for
                each frame range when the write() method is called. When
                out_vidpath is None or "", no videos are written

            out_fps: a frame rate for the output video, if any. If the input
                source is a video and fps is None, the same frame rate is used

            out_size: the frame size for the output video, if any. If out_size
                is None, the input frame size is assumed

            out_opts: a list of output video options for ffmpeg. Passed
                directly to FFmpegVideoWriter. Only applicable when
                out_use_ffmpeg = True

        Raises:
            VideoProcessorError: if insufficient options are supplied to
                construct a VideoWriter
        '''
        if in_use_ffmpeg:
            self._reader = FFmpegVideoReader(inpath, frames=frames)
        else:
            self._reader = OpenCVVideoReader(inpath, frames=frames)
        self._video_writer = None
        self._write_images = out_impath is not None and out_impath != ""
        self._write_videos = out_vidpath is not None and out_vidpath != ""

        self.inpath = inpath
        self.frames = frames
        self.in_use_ffmpeg = in_use_ffmpeg
        self.out_use_ffmpeg = out_use_ffmpeg
        self.out_impath = out_impath
        self.out_vidpath = out_vidpath
        if out_fps is not None and out_fps > 0:
            self.out_fps = out_fps
        elif self._reader.frame_rate > 0:
            self.out_fps = self._reader.frame_rate
        else:
            raise VideoProcessorError((
                "The inferred frame rate '%s' cannot be used. You must " +
                "manually specify a frame rate"
            ) % str(self._reader.frame_rate))
        self.out_size = out_size if out_size else self._reader.frame_size
        self.out_opts = out_opts

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.process()

    @property
    def input_frame_size(self):
        '''The (width, height) of each input frame.'''
        return self._reader.frame_size

    @property
    def output_frame_size(self):
        '''The (width, height) of each output frame.'''
        return self.out_size

    @property
    def input_frame_rate(self):
        '''The input frame rate.'''
        return self._reader.frame_rate

    @property
    def output_frame_rate(self):
        '''The output frame rate.'''
        return self.out_fps

    @property
    def frame_number(self):
        '''The current frame number, or -1 if no frames have been read.'''
        return self._reader.frame_number

    @property
    def frame_range(self):
        '''The (first, last) frames for the current range, or (-1, -1) if no
        frames have been read.
        '''
        return self._reader.frame_range

    @property
    def is_new_frame_range(self):
        '''Whether the current frame is the first in a new range.'''
        return self._reader.is_new_frame_range

    @property
    def total_frame_count(self):
        '''The total number of frames in the video.'''
        return self._reader.total_frame_count

    def process(self):
        '''Returns the next frame.'''
        img = self._reader.read()
        if self._write_videos and self._reader.is_new_frame_range:
            self._init_new_video_writer()
        return img

    def write(self, img):
        '''Writes the given image to the output writer(s).'''
        if self._write_images:
            im.write(img, self.out_impath % self._reader.frame_number)
        if self._write_videos:
            self._video_writer.write(img)

    def close(self):
        '''Closes the video processor.'''
        self._reader.close()
        if self._video_writer is not None:
            self._video_writer.close()

    def _init_new_video_writer(self):
        # Close any existing writer
        if self._video_writer is not None:
            self._video_writer.close()

        # Open a new writer with outpath configured for the current frame range
        if self.out_use_ffmpeg:
            self._video_writer = FFmpegVideoWriter(
                self.out_vidpath % self._reader.frame_range,
                self.out_fps,
                self.out_size,
                out_opts=self.out_opts,
            )
        else:
            self._video_writer = OpenCVVideoWriter(
                self.out_vidpath % self._reader.frame_range,
                self.out_fps,
                self.out_size,
            )


class VideoProcessorError(Exception):
    pass


class VideoReader(object):
    '''Base class for reading videos.'''

    def __init__(self, inpath, frames):
        '''Initialize base VideoReader capabilities.'''
        if frames == "*":
            frames = "1-%d" % self.total_frame_count
        self._ranges = FrameRanges.from_str(frames)

        self.inpath = inpath
        self.frames = frames

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.read()

    @property
    def frame_number(self):
        '''The current frame number, or -1 if no frames have been read.'''
        return self._ranges.frame

    @property
    def frame_range(self):
        '''The (first, last) frames for the current range, or (-1, -1) if no
        frames have been read.
        '''
        return self._ranges.frame_range

    @property
    def is_new_frame_range(self):
        '''Whether the current frame is the first in a new range.'''
        return self._ranges.is_new_frame_range

    @property
    def encoding_str(self):
        raise NotImplementedError("subclass must implement encoding_str")

    @property
    def frame_size(self):
        raise NotImplementedError("subclass must implement frame_size")

    @property
    def frame_rate(self):
        raise NotImplementedError("subclass must implement frame_rate")

    @property
    def total_frame_count(self):
        raise NotImplementedError("subclass must implement total_frame_count")

    def read(self):
        raise NotImplementedError("subclass must implement read()")

    def close(self):
        raise NotImplementedError("subclass must implement close()")


class VideoReaderError(Exception):
    pass


class FFmpegVideoReader(VideoReader):
    '''Class for reading video using ffmpeg.

    The input video can be a standalone video file like "/path/to/video.mp4"
    or a directory of frames like "/path/to/frames/%5d.png". This path is
    passed directly to ffmpeg.

    A frames string like "1-5,10-15" can optionally be passed to only read
    certain frame ranges.

    This class uses 1-based indexing for all frame operations.
    '''

    def __init__(self, inpath, frames="*"):
        '''Constructs a new VideoReader with ffmpeg backend.

        Args:
            inpath: path to the input video, which can be a standalone video
                file like "/path/to/video.mp4" or a directory of frames like
                "/path/to/frames/%5d.png". This path is passed directly to
                ffmpeg

            frames: a string like "1-5,10-15" specifying the range(s) of frames
                in the input video to read. Set frames="*" to read all frames
        '''
        self._stream = get_stream_info(inpath)
        self._ffmpeg = FFmpeg(
            out_opts=[
                "-f", 'image2pipe',         # pipe frames to stdout
                "-vcodec", "rawvideo",      # output will be raw video
                "-pix_fmt", "bgr24",        # pixel format (BGR for OpenCV)
            ],
        )
        self._ffmpeg.run(inpath, "-")
        self._raw_frame = None

        super(FFmpegVideoReader, self).__init__(inpath, frames)

    @property
    def encoding_str(self):
        '''Return the video encoding string.'''
        return str(self._stream["codec_tag_string"])

    @property
    def frame_size(self):
        '''The (width, height) of each frame.'''
        return int(self._stream["width"]), int(self._stream["height"])

    @property
    def frame_rate(self):
        '''The frame rate.'''
        return eval(self._stream["avg_frame_rate"] + ".0")

    @property
    def total_frame_count(self):
        '''The total number of frames in the video, or 0 if it could not be
        determined.
        '''
        try:
            # this fails for directories of frames...
            return int(self._stream["nb_frames"])
        except KeyError:
            # @todo: this seems to work for directories of frames...
            #        can we count on it?
            return int(self._stream.get("duration_ts", 0))

    def read(self):
        '''Reads the next frame.

        Returns:
            img: the next frame

        Raises:
            StopIteration: if there are no more frames to process
            VideoReaderError: if unable to load the next frame from file
        '''
        for idx in range(max(0, self.frame_number), next(self._ranges)):
            if not self._grab():
                raise VideoReaderError(
                    "Failed to grab frame %d" % (idx + 1))
        return self._retrieve()

    def close(self):
        '''Closes the video reader.'''
        self._ffmpeg.close()

    def _grab(self):
        try:
            width, height = self.frame_size
            self._raw_frame = self._ffmpeg.read(width * height * 3)
            return True
        except:
            return False

    def _retrieve(self):
        width, height = self.frame_size
        vec = np.fromstring(self._raw_frame, dtype="uint8")
        return vec.reshape((height, width, 3))


class OpenCVVideoReader(VideoReader):
    '''Class for reading video using OpenCV.

    The input video can be a standalone video file like "/path/to/video.mp4"
    or a directory of frames like "/path/to/frames/%5d.png". This path is
    passed directly to cv2.VideoCapture. So, for example, if you specify a
    directory of frames, the frame numbering must start from 0-3.

    A frames string like "1-5,10-15" can optionally be passed to only read
    certain frame ranges.

    This class uses 1-based indexing for all frame operations.
    '''

    def __init__(self, inpath, frames="*"):
        '''Constructs a new VideoReader with OpenCV backend.

        Args:
            inpath: path to the input video, which can be a standalone video
                file like "/path/to/video.mp4" or a directory of frames like
                "/path/to/frames/%5d.png". This path is passed directly to
                cv2.VideoCapture.

            frames: a string like "1-5,10-15" specifying the range(s) of frames
                in the input video to read. Set frames = "*" to read all
                frames.

        Raises:
            VideoReaderError: if the input video could not be opened.
        '''
        self._cap = cv2.VideoCapture(inpath)
        if not self._cap.isOpened():
            raise VideoReaderError("Unable to open '%s'" % inpath)

        super(OpenCVVideoReader, self).__init__(inpath, frames)

    @property
    def encoding_str(self):
        '''Return the video encoding string.'''
        try:
            # OpenCV 3
            code = int(self._cap.get(cv2.CV_CAP_PROP_FOURCC))
        except AttributeError:
            # OpenCV 2
            code = int(self._cap.get(cv2.cv.CV_CAP_PROP_FOURCC))
        return FOURCC.int_to_str(code)

    @property
    def frame_size(self):
        '''The (width, height) of each frame.'''
        try:
            # OpenCV 3
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        except AttributeError:
            # OpenCV 2
            return (
                int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)),
            )

    @property
    def frame_rate(self):
        '''The frame rate.'''
        try:
            # OpenCV 3
            return float(self._cap.get(cv2.CAP_PROP_FPS))
        except AttributeError:
            # OpenCV 2
            return float(self._cap.get(cv2.cv.CV_CAP_PROP_FPS))

    @property
    def total_frame_count(self):
        '''The total number of frames in the video.'''
        try:
            # OpenCV 3
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except AttributeError:
            # OpenCV 2
            return int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    def read(self):
        '''Reads the next frame.

        Returns:
            img: the next frame

        Raises:
            StopIteration: if there are no more frames to process
            VideoReaderError: if unable to load the next frame from file
        '''
        for idx in range(max(0, self.frame_number), next(self._ranges)):
            if not self._cap.grab():
                raise VideoReaderError(
                    "Failed to grab frame %d" % (idx + 1))
        return self._cap.retrieve()[1]

    def close(self):
        '''Closes the video reader.'''
        self._cap.release()


class VideoWriter(object):
    '''Base class for writing videos.'''

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, img):
        raise NotImplementedError("subclass must implement write()")

    def close(self):
        raise NotImplementedError("subclass must implement close()")


class VideoWriterError(Exception):
    pass


class FFmpegVideoWriter(VideoWriter):
    '''Class for writing videos using ffmpeg.'''

    DEFAULT_OUT_OPTS = [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "slow", "-crf", "22", "-an",
    ]

    def __init__(self, outpath, fps, size, out_opts=None):
        '''Constructs a VideoWriter with ffmpeg backend.

        Args:
            outpath: the output video path, e.g., "/path/to/video.mp4".
                Existing files are overwritten, and the directory is created
                if needed
            fps: the frame rate
            size: the (width, height) of each frame
            out_opts: A list of output options for ffmpeg. If not specified,
                the DEFAULT_OUT_OPTS list is used
        '''
        self.outpath = outpath
        self.fps = fps
        self.size = size
        self.out_opts = out_opts or self.DEFAULT_OUT_OPTS

        self._ffmpeg = FFmpeg(
            in_opts=[
                "-f", "rawvideo",           # input will be raw video
                "-vcodec", "rawvideo",      # input will be raw video
                "-s", "%dx%d" % self.size,  # frame size
                "-pix_fmt", "bgr24",        # pixel format (BGR for OpenCV)
                "-r", str(self.fps),        # frame rate
            ],
            out_opts=self.out_opts,
        )
        self._ffmpeg.run("-", self.outpath)

    def write(self, img):
        '''Appends the image to the output video.

        Args:
            img: an image in OpenCV format, e.g., img = cv2.imread(...)
        '''
        self._ffmpeg.stream(img.tostring())

    def close(self):
        '''Closes the video writer.'''
        self._ffmpeg.close()


class OpenCVVideoWriter(VideoWriter):
    '''Class for writing videos using cv2.VideoWriter.

    Uses the default encoding scheme for the extension of the output path.
    '''

    def __init__(self, outpath, fps, size):
        '''Constructs a VideoWriter with OpenCV backend.

        Args:
            outpath: the output video path, e.g., "/path/to/video.mp4".
                Existing files are overwritten, and the directory is created
                if needed
            fps: the frame rate
            size: the (width, height) of each frame

        Raises:
            VideoWriterError: if the writer failed to open
        '''
        self.outpath = outpath
        self.fps = fps
        self.size = size
        self._writer = cv2.VideoWriter()

        utils.ensure_path(self.outpath)
        self._writer.open(self.outpath, -1, self.fps, self.size, True)
        if not self._writer.isOpened():
            raise VideoWriterError("Unable to open '%s'" % self.outpath)

    def write(self, img):
        '''Appends the image to the output video.

        Args:
            img: an image in OpenCV format, e.g., img = cv2.imread(...)
        '''
        self._writer.write(img)

    def close(self):
        '''Closes the video writer.'''
        # self._writer.release()  # warns to use a separate thread
        threading.Thread(target=self._writer.release, args=()).start()


class FFprobe(object):
    '''Interface for the ffprobe binary.'''

    DEFAULT_GLOBAL_OPTS = ["-loglevel", "error"]

    def __init__(
            self,
            executable="ffprobe",
            global_opts=None,
            opts=None,
        ):
        '''Constructs an ffprobe command, minus the input path.

        Args:
            executable: the system path to the ffprobe binary
            global_opts: a list of global options for ffprobe. By default,
                self.DEFAULT_GLOBAL_OPTS is used
            opts: a list of options for ffprobe
        '''
        self._executable = executable
        self._global_opts = global_opts or self.DEFAULT_GLOBAL_OPTS
        self._opts = opts or []

        self._args = None
        self._p = None

    @property
    def cmd(self):
        '''The last executed ffprobe command string, or None if run() has not
        yet been called.
        '''
        return " ".join(self._args) if self._args else None

    def run(self, inpath):
        '''Run the ffprobe binary with the specified input path.

        Args:
            inpath: the input path

        Returns:
            out: the stdout from the ffprobe binary

        Raises:
            ExecutableNotFoundError: if the ffprobe binary cannot be found
            ExecutableRuntimeError: if the ffprobe binary raises an error
                during execution
        '''
        self._args = (
            [self._executable] +
            self._global_opts +
            self._opts +
            ["-i", inpath]
        )

        try:
            self._p = Popen(
                self._args,
                stdout=PIPE,
                stderr=PIPE,
            )
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise utils.ExecutableNotFoundError(self._executable)
            else:
                raise

        out, err = self._p.communicate()
        if self._p.returncode != 0:
            raise utils.ExecutableRuntimeError(self.cmd, err)

        return out


class FFmpeg(object):
    '''Interface for the ffmpeg binary.'''

    DEFAULT_GLOBAL_OPTS = ["-loglevel", "error"]

    def __init__(
            self,
            executable="ffmpeg",
            global_opts=None,
            in_opts=None,
            out_opts=None,
        ):
        '''Constructs an ffmpeg command, minus the input/output paths.

        Args:
            executable: the system path to the ffmpeg binary
            global_opts: a list of global options for ffmpeg. By default,
                self.DEFAULT_GLOBAL_OPTS is used
            in_opts: a list of input options for ffmpeg
            out_opts: a list of output options for ffmpeg
        '''
        self.is_input_streaming = False
        self.is_output_streaming = False

        self._executable = executable
        self._global_opts = global_opts or self.DEFAULT_GLOBAL_OPTS
        self._in_opts = in_opts or []
        self._out_opts = out_opts or []
        self._args = None
        self._p = None

    @property
    def cmd(self):
        '''The last executed ffmpeg command string, or None if run() has not
        yet been called.
        '''
        return " ".join(self._args) if self._args else None

    def run(self, inpath, outpath):
        '''Run the ffmpeg binary with the specified input/outpath paths.

        Args:
            inpath: the input path. If inpath is "-", input streaming mode is
                activated and data can be passed via the stream() method
            outpath: the output path. Existing files are overwritten, and the
                directory is created if needed. If outpath is "-", output
                streaming mode is activated and data can be read via the
                read() method

        Raises:
            ExecutableNotFoundError: if the ffmpeg binary cannot be found
            ExecutableRuntimeError: if the ffmpeg binary raises an error during
                execution
        '''
        self.is_input_streaming = (inpath == "-")
        self.is_output_streaming = (outpath == "-")

        self._args = (
            [self._executable] +
            self._global_opts +
            self._in_opts + ["-i", inpath] +
            self._out_opts + [outpath]
        )

        if not self.is_output_streaming:
            utils.ensure_path(outpath)

        try:
            self._p = Popen(self._args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise utils.ExecutableNotFoundError(self._executable)
            else:
                raise

        # Run non-streaming jobs immediately
        if not (self.is_input_streaming or self.is_output_streaming):
            err = self._p.communicate()[1]
            if self._p.returncode != 0:
                raise utils.ExecutableRuntimeError(self.cmd, err)

    def stream(self, string):
        '''Writes the string to ffmpeg's stdin stream.

        Raises:
            FFmpegStreamingError: if input streaming mode is not active
        '''
        if not self.is_input_streaming:
            raise FFmpegStreamingError("Not currently input streaming")
        self._p.stdin.write(string)

    def read(self, num_bytes):
        '''Reads the given number of bytes from ffmpeg's stdout stream.

        Raises:
            FFmpegStreamingError: if output streaming mode is not active
        '''
        if not self.is_output_streaming:
            raise FFmpegStreamingError("Not currently output streaming")
        return self._p.stdout.read(num_bytes)

    def close(self):
        '''Closes a streaming ffmpeg program.

        Raises:
            FFmpegStreamingError: if a streaming mode is not active
        '''
        if not (self.is_input_streaming or self.is_output_streaming):
            raise FFmpegStreamingError("Not currently streaming")
        self._p.stdin.close()
        self._p.stdout.close()
        self._p.wait()
        self._p = None
        self.is_input_streaming = False
        self.is_output_streaming = False


class FFmpegStreamingError(Exception):
    pass


class FFmpegVideoResizer(FFmpeg):
    '''Class for resizing videos using ffmpeg.

    Example usage:
        resizer = FFmpegVideoResizer(size=(512, -1))
        resizer.run("/path/to/video.mp4", "/path/to/resized.mp4")
    '''

    DEFAULT_OUT_OPTS = [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "slow", "-crf", "22", "-an",
    ]

    def __init__(self, size=None, scale=None, scale_str=None, **kwargs):
        '''Constructs a video resizer with ffmpeg backend.

        Any one of the `size`, `scale`, and `scale_str` arguments can be
        specified to define the resizing operation.

        Args:
            size: the output (width, height) of each frame. At most one
                dimension can be -1, in which case the aspect ratio is
                preserved
            scale: a positive number by which to scale the input video
                (e.g., 0.5 or 2)
            scale_str: a string that directly specifies a valid ffmpeg scale=
                option
            **kwargs: optional keyword arguments for FFmpeg()

        Raises:
            FFmpegVideoResizerError: if a valid scale option could not be
                generated from the given args
        '''
        out_opts = kwargs.pop("out_opts", self.DEFAULT_OUT_OPTS) or []
        out_opts += [
            "-vf",
            "scale={0}".format(self._gen_scale_opt(
                size=size, scale=scale, scale_str=scale_str)),
        ]
        super(FFmpegVideoResizer, self).__init__(out_opts=out_opts, **kwargs)

    @staticmethod
    def _gen_scale_opt(size=None, scale=None, scale_str=None):
        if size:
            return "{0}:{1}".format(*size)
        elif scale:
            return "iw*{0}:ih*{0}".format(scale)
        elif scale_str:
            return scale_str
        else:
            raise FFmpegVideoResizerError("Invalid scale spec")


class FFmpegVideoResizerError(Exception):
    pass


class FFmpegVideoSampler(FFmpeg):
    '''Class for sampling videos using ffmpeg.

    Example usage:
        sampler = FFmpegVideoSampler(24)
        sampler.run("/path/to/video.mp4", "/path/to/frames/%05d.png")
    '''

    def __init__(self, fps, **kwargs):
        '''Constructs a video sampler with ffmpeg backend.

        Args:
            fps: the desired frame rate
            **kwargs: optional keyword arguments for FFmpeg()
        '''
        out_opts = kwargs.pop("out_opts", []) or []
        out_opts += ["-vf", "fps={0}".format(fps)]
        super(FFmpegVideoSampler, self).__init__(out_opts=out_opts, **kwargs)


class VideoFeaturizerConfig(Config):
    '''Specifies the configuration settings for the VideoFeaturizer class.'''

    def __init__(self, d):
        self.backing_path = self.parse_string(d, "backing_path")
        self.video_path = self.parse_string(d, "video_path")


class FeaturizedFrameNotFoundError(OSError):
    pass


class VideoFeaturizer(object):
    '''Class that encapsulates creating features from frames in a video and
    storing them to disk.

    Needs the following to be specified:
        backingPath: location on disk where featurized frames will be cached
            (this is a directory and not a printf like path)
        videoPath: location of the input video (suitable as input to
            VideoProcessor)
    These are immutable quantities and are hence provided as a config.

    Frames are not part of the config as a new VideoProcessor can be created
    with a different frame range depending on execution needs.

    Featurized frames are store indexed by frame_number as compressed pickles.

    @todo Should objects from this class are serializable.  If you initialize
    with the same video and the same config then it is essentially the same
    instance.  How to handle this well?

    @todo: can be generalized to not rely only on pickles (subclassed for
    pickles, eg).  I can imagine a featurizer might actually output an image.
    Think about this.
    '''

    def __init__(self, video_featurizer_config):
        '''Initializes the featurizer and creates the storage path.

        Implementing sub-classes need to explicitly call this :(
        super(SubClass, self).__init__(config)

        Member self.frame_preprocessor is a "function pointer" that takes in a
        frame and returns a preprocessed frame.  It is by default None and
        hence not call. You could, of course, chain together two video
        featurizers (once the class handles general output types) instead of
        this preprocessor function.  But, this is included her for speed (and
        it's what I needed initially).
        '''

        self._frame_string = "%08d.npz"
        self.most_recent_frame = -1
        self._frame_preprocessor = None
        self.config = video_featurizer_config

        try:
            os.makedirs(self.config.backing_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @property
    def frame_preprocessor(self):
        return self._frame_preprocessor

    @frame_preprocessor.setter
    def frame_preprocessor(self, f):
        self._frame_preprocessor = f

    @frame_preprocessor.deleter
    def frame_preprocessor(self):
        self._frame_preprocessor = None

    # def serialize(self):

    def is_featurized(self, frame_number):
        ''' Checks the backing store to determine whether or not the frame
        number is already featurized and stored to disk.
        '''
        return os.path.isfile(self.featurized_frame_path(frame_number))

    def retrieve_featurized_frame(self, frame_number):
        '''The frame_number here is rendered into a string for the filepath.
        So, if it is not available as part of the video, then it will be an
        error.

        No checking is explicitly done here. Careful about starting from
        0 or 1.
        '''
        p = self.featurized_frame_path(frame_number)

        if not os.path.isfile(p):
            raise FeaturizedFrameNotFoundError()

        f = np.load(p)
        return f["v"]

    def retrieve(self, frames="*"):
        '''Main general use driver.  Works through a frame range to retrieve a
        matrix X of the features.

        X.shape is [n_frames,featurized_length]

        Will do the featurization if needed.

        @todo Refactorization needed. featurize() does similar work.
        These can be combined to only featurize and cache in one place.
        '''
        has_started = False

        X = None

        with VideoProcessor(self.config.video_path, frames) as p:
            for img in p:
                self.most_recent_frame = p.frame_number
                f = self.featurized_frame_path(p.frame_number)

                try:
                    v = self.retrieve_featurized_frame(p.frame_number)
                except FeaturizedFrameNotFoundError:
                    if not has_started:
                        has_started = True
                        self.featurize_start()

                    # The frame is not yet cached, so compute and cache it.
                    if self._frame_preprocessor is not None:
                        v = self.featurize_frame(self._frame_preprocessor(img))
                    else:
                        v = self.featurize_frame(img)
                    np.savez_compressed(f, v=v)
                finally:
                    if X is None:
                        X = GrowableArray(len(v))
                    X.update(v)

        if has_started:
            self.featurize_end()

        return X.finalize()

    def featurized_frame_path(self, frame_number):
        '''Returns the backing path for the given frame number.'''
        return os.path.join(
            self.config.backing_path, self._frame_string % frame_number)

    def featurize_frame(self, frame):
        '''The actual featurization. Should return a numpy vector.'''
        raise NotImplementedError("subclass must implement featurize_frame()")

    def featurize_start(self):
        '''Called by featurize before it starts in case any environment needs
        to be set up by subclasses.
        '''
        pass

    def featurize_end(self):
        '''Called by featurize after it end in case any environment needs to be
        cleaned up by subclasses.
        '''
        pass

    def featurize(self, frames="*"):
        '''Featurize all frames in the specified range by calling
        featurize_frame() for each frame.

        Creates a VideoProcessor on the fly for this.

        Checks the backing store to see if the frame is already featurized.

        @todo Check for refactorization possibilities with retrieve() as there
        is some redundant code.
        '''
        self.featurize_start()

        with VideoProcessor(self.config.video_path, frames) as p:
            for img in p:
                self.most_recent_frame = p.frame_number
                f = self.featurized_frame_path(p.frame_number)

                try:
                    self.retrieve_featurized_frame(p.frame_number)
                except FeaturizedFrameNotFoundError:
                    # The frame is not yet cached, so compute it.
                    if self._frame_preprocessor is not None:
                        v = self.featurize_frame(self._frame_preprocessor(img))
                    else:
                        v = self.featurize_frame(img)
                    np.savez_compressed(f, v=v)

        self.featurize_end()

    def flush_backing(self):
        '''CAUTION! Deletes all featurized files on disk but not the
        directory.
        '''
        filelist = [
            f for f in os.listdir(self.config.backing_path)
            if f.endswith(".npz")
        ]
        for f in filelist:
            os.remove(os.path.join(self.config.backing_path, f))


class FOURCC(object):
    '''Class reprsesenting a FOURCC code.'''

    def __init__(self, _i=None, _s=None):
        '''Don't call this directly!'''
        if _i:
            self.int = _i
            self.str = FOURCC.int_to_str(_i)
        elif _s:
            self.int = FOURCC.str_to_int(_s)
            self.str = _s

    @classmethod
    def from_str(cls, s):
        '''Construct a FOURCC instance from a string.'''
        return cls(_s=s)

    @classmethod
    def from_int(cls, i):
        '''Construct a FOURCC instance from an integer.'''
        return cls(_i=i)

    @staticmethod
    def str_to_int(s):
        '''Convert the FOURCC string to an int.'''
        return cv2.cv.FOURCC(*s)

    @staticmethod
    def int_to_str(i):
        '''Convert the FOURCC int to a string.'''
        return chr((i & 0x000000FF) >> 0) + \
               chr((i & 0x0000FF00) >> 8) + \
               chr((i & 0x00FF0000) >> 16) + \
               chr((i & 0xFF000000) >> 24)


class FrameRanges(object):
    '''A monotonically increasing and disjoint series of frames.'''

    def __init__(self, ranges):
        '''Constructs a frame range series from a list of (first, last) tuples,
        which must be disjoint and monotonically increasing.
        '''
        self._idx = 0
        self._ranges = []
        self._started = False

        end = -1
        for first, last in ranges:
            if first <= end:
                raise ValueError("Expected first:%d > last:%d" % (first, end))
            self._ranges.append(FrameRange(first, last))
            end = last

    def __iter__(self):
        return self

    def __next__(self):
        '''Returns the next frame number.

        Raises:
            StopIteration: if there are no more frames to process
        '''
        self._started = True
        try:
            frame = next(self._ranges[self._idx])
        except StopIteration:
            self._idx += 1
            return next(self)
        except IndexError:
            raise StopIteration
        return frame

    @property
    def frame(self):
        '''The current frame number, or -1 if no frames have been read.'''
        if self._started:
            return self._ranges[self._idx].idx
        return -1

    @property
    def frame_range(self):
        '''The (first, last) values for the current range, or (-1, -1) if no
        frames have been read.
        '''
        if self._started:
            return self._ranges[self._idx].first, self._ranges[self._idx].last
        return -1, -1

    @property
    def is_new_frame_range(self):
        '''Whether the current frame is the first in a new range.'''
        if self._started:
            return self._ranges[self._idx].is_first_frame
        return False

    def to_list(self):
        '''Return a list of frames in the frame ranges.'''
        frames = []
        for r in self._ranges:
            frames += r.to_list()
        return frames

    def to_str(self):
        '''Return a string representation of the frame ranges.'''
        return ",".join([r.to_str() for r in self._ranges])

    @classmethod
    def from_str(cls, frames):
        '''Constructs a FrameRanges object from a string like "1-5,10-15".'''
        ranges = []
        for r in frames.split(','):
            v = list(map(int, r.split('-')))
            ranges.append((v[0], v[-1]))
        return cls(ranges)


class FrameRange(object):
    '''An iterator over a range of frames.'''

    def __init__(self, first, last):
        '''Constructs a frame range with the given first and last values,
        inclusive.
        '''
        if last < first:
            raise ValueError("Expected first:%d <= last:%d" % (first, last))
        self.first = first
        self.last = last
        self.idx = -1

    def __iter__(self):
        return self

    @property
    def is_first_frame(self):
        '''Whether the current frame is first in the range.'''
        return self.idx == self.first

    def __next__(self):
        '''Returns the next frame number.

        Raises:
            StopIteration: if there are no more frames in the range
        '''
        if self.idx < 0:
            self.idx = self.first
        elif self.idx < self.last:
            self.idx += 1
        else:
            raise StopIteration
        return self.idx

    def to_list(self):
        '''Return a list of frames in the range.'''
        return list(range(self.first, self.last + 1))

    def to_str(self):
        '''Return a string representation of the range.'''
        return "%d-%d" % (self.first, self.last)

    @classmethod
    def from_str(cls, frames):
        '''Constructs a FrameRange object from a string like "1-5".'''
        v = list(map(int, frames.split('-')))
        return cls(v[0], v[-1])
