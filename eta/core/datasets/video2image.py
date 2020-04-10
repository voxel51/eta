"""
Convert LabeledVideoDataset to LabeledImageDataset

Copyright 2017-2019 Voxel51, Inc.
voxel51.com

Matthew Lightman, matthew@voxel51.com
Jason Corso, jason@voxel51.com
Ben Kane, ben@voxel51.com
Tyler Ganter, tyler@voxel51.com
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
import os

import numpy as np

import eta.core.image as etai

from .labeled_datasets import LabeledImageDataset


logger = logging.getLogger(__name__)


# Functions involving LabeledDatasets


def sample_videos_to_images(
    video_dataset,
    image_dataset_path,
    stride=None,
    num_images=None,
    frame_filter=lambda labels: True,
    image_extension=".jpg",
    description=None,
):
    """Creates a `LabeledImageDataset` by extracting frames and their
    corresponding labels from a `LabeledVideoDataset`.

    Args:
        video_dataset: a `LabeledVideoDataset` instance from which to
            extract frames as images
        image_dataset_path: the path to the `manifest.json` file for
            the image dataset that will be written. The containing
            directory must either not exist or be empty
        stride: optional frequency with which to sample frames from
            videos
        num_images: optional total number of frames to sample from
            videos. If `stride` is not specified then it will be
            calculated based on `num_images`. If `stride` is
            specified, frames will be sampled at this stride until
            a total of `num_images` are obtained.
        frame_filter: function that takes an
            `eta.core.video.VideoFrameLabels` instance as input and
            returns False if the frame should not be included in the
            sample
        image_extension: optional extension for image files in new
            dataset (defaults to ".jpg")
        description: optional description for the manifest of the
            new image dataset

    Returns:
        image_dataset: `LabeledImageDataset` instance that points to
            the new image dataset
    """
    if stride is None and num_images is None:
        stride = 1

    _validate_stride_and_num_images(stride, num_images)

    if num_images is not None and stride is None:
        stride = _compute_stride(video_dataset, num_images, frame_filter)

    logger.info("Sampling video frames with stride %d", stride)

    image_dataset = LabeledImageDataset.create_empty_dataset(
        image_dataset_path, description=description
    )

    frame_iterator = _iter_filtered_video_frames(
        video_dataset, frame_filter, stride
    )
    for img_number, (frame_img, frame_labels, base_filename) in enumerate(
        frame_iterator, 1
    ):
        image_filename = "%s%s" % (base_filename, image_extension)
        labels_filename = "%s.json" % base_filename

        image_labels = etai.ImageLabels(
            filename=image_filename,
            attrs=frame_labels.attrs,
            objects=frame_labels.objects,
        )
        image_dataset.add_data(
            frame_img, image_labels, image_filename, labels_filename
        )

        if num_images is not None and img_number >= num_images:
            break

    if not image_dataset:
        logger.info(
            "All frames were filtered out in sample_videos_to_images(). "
            "Writing an empty image dataset to '%s'.",
            image_dataset_path,
        )

    image_dataset.write_manifest(image_dataset_path)

    return image_dataset


def _validate_stride_and_num_images(stride, num_images):
    if stride is not None and stride < 1:
        raise ValueError("stride must be >= 1, but got %d" % stride)

    if num_images is not None and num_images < 1:
        raise ValueError("num_images must be >= 1, but got %d" % num_images)


def _compute_stride(video_dataset, num_images, frame_filter):
    total_frames_retained = 0
    for video_labels in video_dataset.iter_labels():
        for frame_number in video_labels:
            frame_labels = video_labels[frame_number]
            if frame_filter(frame_labels):
                total_frames_retained += 1

    logger.info(
        "Found %d total frames after applying the filter",
        total_frames_retained,
    )

    # Handle corner cases
    if total_frames_retained < 2:
        return 1
    if num_images < 2:
        return total_frames_retained

    return _compute_stride_from_total_frames(total_frames_retained, num_images)


def _compute_stride_from_total_frames(total_frames, num_desired):
    if num_desired == 1:
        return total_frames

    stride_guess = (total_frames - 1) / (num_desired - 1)
    stride_guess = max(stride_guess, 1)
    stride_int_guesses = [np.floor(stride_guess), np.ceil(stride_guess)]
    actual_num_images = [
        total_frames / stride for stride in stride_int_guesses
    ]
    differences = [
        np.abs(actual - num_desired) for actual in actual_num_images
    ]
    return int(
        min(zip(stride_int_guesses, differences), key=lambda t: t[1])[0]
    )


def _iter_filtered_video_frames(video_dataset, frame_filter, stride):
    filtered_frame_index = -1
    for video_reader, video_path, video_labels in zip(
        video_dataset.iter_data(),
        video_dataset.iter_data_paths(),
        video_dataset.iter_labels(),
    ):
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        with video_reader:
            for frame_img in video_reader:
                frame_num = video_reader.frame_number
                base_filename = "%s-%d" % (video_name, frame_num)

                frame_labels = video_labels[frame_num]
                if not frame_filter(frame_labels):
                    continue
                filtered_frame_index += 1

                if filtered_frame_index % stride:
                    continue

                yield frame_img, frame_labels, base_filename
