'''
Core interfaces, data structures, and methods for working with datasets.

Copyright 2017-2019 Voxel51, Inc.
voxel51.com

Matthew Lightman, matthew@voxel51.com
Brian Moore, brian@voxel51.com
Jason Corso, jason@voxel51.com
Ben Kane, ben@voxel51.com
Kevin Qi, kevin@voxel51.com
Tyler Ganter, tyler@voxel51.com
Ravali Pinnaka, ravali@voxel51.com
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
import six
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import copy
import logging
import os
import random
from collections import Counter, defaultdict

import numpy as np

from eta.core.data import BaseDataRecord
import eta.core.image as etai
from eta.core.serial import Serializable
import eta.core.utils as etau
import eta.core.video as etav


logger = logging.getLogger(__name__)


# General split methods


def round_robin_split(iterable, split_fractions=None):
    '''Traverses the iterable in order and assigns items to samples in order,
    until a given sample has reached its desired size.

    If a random split is required, this function is not recommended unless your
    items are already randomly ordered.

    Args:
        iterable: any finite iterable
        split_fractions: an optional list of split fractions, which should sum
            to 1. By default, [0.5, 0.5] is used

    Returns:
        sample_lists: a list of lists, of the same length as `split_fractions`.
            Each sub-list contains items from the original iterable.
    '''
    split_fractions = _validate_split_fractions(split_fractions)

    # Initial estimate of size of each sample
    item_list = list(iterable)
    sample_sizes = [int(frac * len(item_list)) for frac in split_fractions]

    # `n` is the total number of items that will be divided into samples.
    # `n` may be less than len(item_list) if sum(split_fractions) < 1.
    n = int(np.round(len(item_list) * sum(split_fractions)))

    if n == 0:
        return [[] for _ in sample_sizes]

    # Calculate exact size of each sample, making sure the sum of the
    # samples sizes is equal to `n`
    remainder = n - sum(sample_sizes)
    num_to_add = int(remainder / len(sample_sizes))
    for idx, _ in enumerate(sample_sizes):
        sample_sizes[idx] += num_to_add
    remainder = n - sum(sample_sizes)
    for idx, _ in enumerate(sample_sizes):
        if idx < remainder:
            sample_sizes[idx] += 1

    assert sum(sample_sizes) == n, (sum(sample_sizes), n)

    # Iterate over items and add them to the appropriate sample
    sample_lists = [[] for _ in sample_sizes]
    sample_full = [sample_size == 0 for sample_size in sample_sizes]
    current_sample_idx = min(
        idx for idx, sample_size in enumerate(sample_sizes)
        if sample_size > 0)
    for item in item_list:
        sample_lists[current_sample_idx].append(item)
        curr_sample_size = len(sample_lists[current_sample_idx])
        if curr_sample_size >= sample_sizes[current_sample_idx]:
            sample_full[current_sample_idx] = True

        if all(sample_full):
            break

        current_sample_idx = _find_next_available_idx(
            current_sample_idx, sample_full)

    return sample_lists


def random_split_exact(iterable, split_fractions=None):
    '''Randomly splits items into multiple sample lists according to the given
    split fractions.

    The number of items in each sample list will be given exactly by the
    specified fractions.

    Args:
        iterable: any finite iterable
        split_fractions: an optional list of split fractions, which should sum
            to 1. By default, [0.5, 0.5] is used

    Returns:
        sample_lists: a list of lists, of the same length as `split_fractions`.
            Each sub-list contains items from the original iterable.
    '''
    split_fractions = _validate_split_fractions(split_fractions)

    shuffled = list(iterable)
    random.shuffle(shuffled)

    return _split_in_order(shuffled, split_fractions)


def random_split_approx(iterable, split_fractions=None):
    '''Randomly splits items into multiple sample lists according to the given
    split fractions.

    Each item is assigned to a sample list with probability equal to the
    corresponding split fraction.

    Args:
        iterable: any finite iterable
        split_fractions: an optional list of split fractions, which should sum
            to 1. By default, [0.5, 0.5] is used

    Returns:
        sample_lists: a list of lists, of the same length as `split_fractions`.
            Each sub-list contains items from the original iterable.
    '''
    split_fractions = _validate_split_fractions(split_fractions)

    sample_lists = [[] for _ in split_fractions]

    cum_frac = np.cumsum(split_fractions)
    for item in iterable:
        idx = np.searchsorted(cum_frac, random.random())
        if idx < len(sample_lists):
            sample_lists[idx].append(item)

    return sample_lists


def split_in_order(iterable, split_fractions=None):
    '''Splits items into multiple sample lists according to the given split
    fractions.

    The items are partitioned into samples in order according to their
    position in the input sample. If a random split is required, this function
    is not recommended unless your items are already randomly ordered.

    Args:
        iterable: any finite iterable
        split_fractions: an optional list of split fractions, which should sum
            to 1. By default, [0.5, 0.5] is used

    Returns:
        sample_lists: a list of lists, of the same length as `split_fractions`.
            Each sub-list contains items from the original iterable.
    '''
    split_fractions = _validate_split_fractions(split_fractions)

    return _split_in_order(list(iterable), split_fractions)


def _split_in_order(item_list, split_fractions):
    n = len(item_list)
    cum_frac = np.cumsum(split_fractions)
    cum_size = [int(np.round(frac * n)) for frac in cum_frac]
    sample_bounds = [0] + cum_size

    sample_lists = []
    for begin, end in zip(sample_bounds, sample_bounds[1:]):
        sample_lists.append(item_list[begin:end])

    return sample_lists


def _validate_split_fractions(split_fractions):
    if split_fractions is None:
        split_fractions = [0.5, 0.5]

    negative = [frac for frac in split_fractions if frac < 0]
    if negative:
        raise ValueError(
            "Split fractions must be non-negative, but got the following "
            "negative values: %s" % str(negative))

    if sum(split_fractions) > 1.0:
        raise ValueError(
            "Sum of split fractions must be <= 1.0, but got sum(%s) = %f" %
            (split_fractions, sum(split_fractions)))

    return split_fractions


def _find_next_available_idx(idx, unavailable_indicators):
    for next_idx in range(idx + 1, len(unavailable_indicators)):
        if not unavailable_indicators[next_idx]:
            return next_idx

    for next_idx in range(idx + 1):
        if not unavailable_indicators[next_idx]:
            return next_idx

    return None


SPLIT_FUNCTIONS = {
    "round_robin": round_robin_split,
    "random_exact": random_split_exact,
    "random_approx": random_split_approx,
    "in_order": split_in_order
}


# Functions involving LabeledDatasets


def sample_videos_to_images(
        video_dataset, image_dataset_path, stride=None, num_images=None,
        frame_filter=lambda labels: True, image_extension=".jpg",
        description=None):
    '''Creates a `LabeledImageDataset` by extracting frames and their
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
            videos. Only one of `stride` and `num_images` can be
            specified at the same time.
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
    '''
    if stride is None and num_images is None:
        stride = 1

    _validate_stride_and_num_images(stride, num_images)

    if num_images is not None:
        stride = _compute_stride(video_dataset, num_images, frame_filter)
        logger.info("Sampling video frames with stride %d", stride)

    image_dataset = LabeledImageDataset.create_empty_dataset(
        image_dataset_path, description=description)

    frame_iterator = _iter_filtered_video_frames(
        video_dataset, frame_filter, stride)
    for frame_img, frame_labels, base_filename in frame_iterator:
        image_filename = "%s%s" % (base_filename, image_extension)
        labels_filename = "%s.json" % base_filename

        image_labels = etai.ImageLabels(
            filename=image_filename,
            attrs=frame_labels.attrs,
            objects=frame_labels.objects)
        image_dataset.add_data(
            frame_img, image_labels, image_filename,
            labels_filename)

    if not image_dataset:
        logger.info(
            "All frames were filtered out in sample_videos_to_images(). "
            "Writing an empty image dataset to '%s'.",
            image_dataset_path)

    image_dataset.write_manifest(image_dataset_path)

    return image_dataset


def _validate_stride_and_num_images(stride, num_images):
    if stride is not None and num_images is not None:
        raise ValueError(
            "Only one of `stride` and `num_images` can be "
            "specified, but got stride = %s, num_images = %s" %
            (stride, num_images))

    if stride is not None and stride < 1:
        raise ValueError(
            "stride must be >= 1, but got %d" % stride)

    if num_images is not None and num_images < 1:
        raise ValueError(
            "num_images must be >= 1, but got %d" % num_images)


def _compute_stride(video_dataset, num_images, frame_filter):
    total_frames_retained = 0
    for video_labels in video_dataset.iter_labels():
        for frame_number in video_labels:
            frame_labels = video_labels[frame_number]
            if frame_filter(frame_labels):
                total_frames_retained += 1

    logger.info("Found %d total frames after applying the filter",
                total_frames_retained)

    # Handle corner cases
    if total_frames_retained < 2:
        return 1
    if num_images < 2:
        return total_frames_retained

    return _compute_stride_from_total_frames(
        total_frames_retained, num_images)


def _compute_stride_from_total_frames(total_frames, num_desired):
    if num_desired == 1:
        return total_frames

    stride_guess = (total_frames - 1) / (num_desired - 1)
    stride_guess = max(stride_guess, 1)
    stride_int_guesses = [np.floor(stride_guess), np.ceil(stride_guess)]
    actual_num_images = [
        total_frames / stride for stride in stride_int_guesses]
    differences = [
        np.abs(actual - num_desired) for actual in actual_num_images]
    return int(min(
        zip(stride_int_guesses, differences), key=lambda t: t[1])[0])


def _iter_filtered_video_frames(video_dataset, frame_filter, stride):
    filtered_frame_index = -1
    for video_reader, video_path, video_labels in zip(
            video_dataset.iter_data(), video_dataset.iter_data_paths(),
            video_dataset.iter_labels()):
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


# Core LabeledDataset infrastructure


class LabeledDataset(object):
    '''Base class for labeled datasets.

    Labeled datasets are stored on disk in the following format:

    ```
    /path/to/dataset/
        manifest.json
        data/
            image1.png (or) video1.mp4
            ...
        labels/
            image1.json (or) video1.json
            ...
    ```

    Class invariants:
    - `self.dataset_index` contains paths to data and labels files that exist
        on disk (assuming this is the case for the manifest.json file that is
        read initially)
    - each data file appears only once is `self.dataset_index`

    Attributes:
        dataset_index: a `LabeledDatasetIndex` containing the paths of data
            and labels files in the dataset
        data_dir: the top level directory for the dataset, which would contain
            manifest.json files
    '''

    _DATA_SUBDIR = "data"
    _LABELS_SUBDIR = "labels"

    def __init__(self, dataset_path):
        '''Creates a LabeledDataset instance.

        Args:
            dataset_path: the path to the `manifest.json` file for the dataset

        Raises:
            LabeledDatasetError: if the class reading the dataset is not a
                subclass of the dataset class recorded in the manifest
        '''
        self.dataset_index = LabeledDatasetIndex.from_json(dataset_path)
        if not isinstance(self, etau.get_class(self.dataset_index.type)):
            raise LabeledDatasetError(
                "Tried to read dataset of type '%s', from location '%s', "
                "but manifest is of type '%s'" % (
                    etau.get_class_name(self), dataset_path,
                    self.dataset_index.type))
        self.dataset_path = dataset_path
        self.data_dir = os.path.dirname(dataset_path)

        self._build_index_map()

    def __iter__(self):
        '''Iterates over the samples in the dataset.

        Returns:
            iterator: iterator over (data, labels) pairs, where data is an
                object returned by self._read_data() and labels is an object
                returned by self._read_labels() from the respective paths
                of a data and corresponding labels file
        '''
        return zip(self.iter_data(), self.iter_labels())

    def __len__(self):
        '''Returns the number of data elements in the dataset'''
        return len(self.dataset_index)

    def iter_data(self):
        '''Iterates over the data in the dataset.

        Returns:
            iterator: iterator over objects returned by self._read_data()
                from the paths to data files
        '''
        for data_path in self.iter_data_paths():
            yield self._read_data(data_path)

    def iter_data_paths(self):
        '''Iterates over the paths to data files in the dataset.

        Returns:
            iterator: iterator over paths to data files
        '''
        for record in self.dataset_index:
            yield os.path.join(self.data_dir, record.data)

    def iter_labels(self):
        '''Iterates over the labels in the dataset.

        Returns:
            iterator: iterator over objects returned by self._read_labels()
                from the paths to labels files
        '''
        for labels_path in self.iter_labels_paths():
            yield self._read_labels(labels_path)

    def iter_labels_paths(self):
        '''Iterates over the paths labels files in the dataset.

        Returns:
            iterator: iterator over paths to labels files
        '''
        for record in self.dataset_index:
            yield os.path.join(self.data_dir, record.labels)

    def iter_paths(self):
        '''Iterates over the data and labels paths tuple in the dataset.

        Returns:
            iterator: iterator over paths to labels files
        '''
        paths = zip(self.iter_data_paths(), self.iter_labels_paths())
        for data_path, labels_path in paths:
            yield (data_path, labels_path)

    def set_description(self, description):
        '''Set the description string of this dataset.

        Args:
            description: the new description string

        Returns:
            self
        '''
        self.dataset_index.description = description

        return self

    def write_manifest(self, filename, description=None):
        '''Writes the manifest to a new file inside the base dataset directory.

        This can be used after the dataset index has been manipulated to save
        a new view of the data in a different manifest file.

        Args:
            filename: the name of a new manifest file to be written in
                self.data_dir
            description: optional description for the new manifest. If not
                specified, the existing description is retained.
        '''
        if description is not None:
            self.set_description(description)

        out_path = os.path.join(self.data_dir, filename)
        self.dataset_index.write_json(out_path)

    def update_manifest(self):
        '''Overwrites existing manifest file of this dataset with any updates
        to the index.
        '''
        self.dataset_index.write_json(self.dataset_path)

    def sample(self, k):
        '''Randomly downsamples the dataset to k elements.

        Args:
            k: the number of data elements in the dataset after sampling

        Returns:
            self
        '''
        self.dataset_index.sample(k)
        self._build_index_map()

        return self

    def shuffle(self):
        '''Randomly shuffles the order of the data.

        Returns:
            self
        '''
        self.dataset_index.shuffle()

        return self

    def split(self, split_fractions=None, descriptions=None,
              split_method="random_exact"):
        '''Splits the dataset into multiple datasets containing disjoint
        subsets of the original dataset.

        Args:
            split_fractions: an optional list of split fractions, which
                should sum to 1, that specifies how to split the dataset.
                By default, [0.5, 0.5] is used.
            descriptions: an optional list of descriptions for the output
                datasets. The list should be the same length as
                `split_fractions`. If not specified, the description of
                the original dataset is used for all of the output
                datasets.
            split_method: string describing the method with which to split
                the data

        Returns:
            dataset_list: list of `LabeledDataset`s of the same length as
                `split_fractions`
        '''
        dataset_indices = self.dataset_index.split(
            split_fractions=split_fractions,
            descriptions=descriptions,
            split_method=split_method)

        dataset_copy = copy.deepcopy(self)
        dataset_list = []
        for dataset_index in dataset_indices:
            dataset_copy.dataset_index = dataset_index
            dataset_copy._build_index_map()
            dataset_list.append(copy.deepcopy(dataset_copy))

        return dataset_list

    def add_file(self, data_path, labels_path, move_files=False,
                 error_on_duplicates=False):
        '''Adds a single data file and its labels file to this dataset.

        Args:
            data_path: path to data file to be added
            labels_path: path to corresponding labels file to be added
            move_files: whether to move the files from their original
                location into the dataset directory. If False, files
                are copied into the dataset directory.
            error_on_duplicates: whether to raise an error if the file
                at `data_path` has the same filename as an existing
                data file in the dataset. If this is set to `False`, the
                previous mapping of the data filename to a labels file
                will be deleted.

        Returns:
            self

        Raises:
            ValueError: if the filename of `data_path` is the same as a
                data file already present in the dataset and
                `error_on_duplicates` is True
        '''
        if error_on_duplicates:
            self._validate_new_data_file(data_path)

        data_subdir = os.path.join(self.data_dir, self._DATA_SUBDIR)
        labels_subdir = os.path.join(self.data_dir, self._LABELS_SUBDIR)
        if os.path.dirname(data_path) != data_subdir:
            if move_files:
                etau.move_file(data_path, data_subdir)
            else:
                etau.copy_file(data_path, data_subdir)
        if os.path.dirname(labels_path) != labels_subdir:
            if move_files:
                etau.move_file(labels_path, labels_subdir)
            else:
                etau.copy_file(labels_path, labels_subdir)

        new_data_file = os.path.basename(data_path)
        new_labels_file = os.path.basename(labels_path)
        # First remove any other records with the same data filename
        self.dataset_index.cull_with_function(
            lambda record: os.path.basename(record.data) != new_data_file)
        self.dataset_index.append(
            LabeledDataRecord(
                os.path.join(self._DATA_SUBDIR, new_data_file),
                os.path.join(self._LABELS_SUBDIR, new_labels_file)
            )
        )

        self._data_to_labels_map[new_data_file] = new_labels_file

        return self

    def add_data(self, data, labels, data_filename, labels_filename,
                 error_on_duplicates=False):
        '''Creates and adds a single data file and its labels file to this
        dataset, using the input python data structure.

        Args:
            data: input data in a format that can be passed to
                self._write_data()
            labels: input labels in a format that can be passed to
                self._write_labels()
            data_filename: filename for the data in the dataset
            labels_filename: filename for the labels in the dataset
            error_on_duplicates: whether to raise an error if a data file
                with the name `data_filename` already exists in the dataset.
                If this is set to `False`, the previous mapping of
                `data_filename` to a labels file will be deleted.

        Returns:
            self
        '''
        if error_on_duplicates:
            self._validate_new_data_file(data_filename)

        data_path = os.path.join(
            self.data_dir, self._DATA_SUBDIR, data_filename)
        labels_path = os.path.join(
            self.data_dir, self._LABELS_SUBDIR, labels_filename)
        self._write_data(data, data_path)
        self._write_labels(labels, labels_path)

        # First remove any other records with the same data filename
        self.dataset_index.cull_with_function(
            lambda record: os.path.basename(record.data) != data_filename)
        self.dataset_index.append(
            LabeledDataRecord(
                os.path.join(self._DATA_SUBDIR, data_filename),
                os.path.join(self._LABELS_SUBDIR, labels_filename)
            )
        )

        self._data_to_labels_map[data_filename] = labels_filename

        return self

    def copy(self, dataset_path):
        '''Copies the dataset to another directory.

        If the dataset index has been manipulated, this will be reflected
        in the copy.

        Args:
            dataset_path: the path to the `manifest.json` file for the
                copy of the dataset that will be written. The containing
                directory must either not exist or be empty.

        Returns:
            dataset_copy: `LabeledDataset` instance that points to the new
                containing directory
        '''
        self._ensure_empty_dataset_dir(dataset_path)

        new_data_dir = os.path.dirname(dataset_path)
        new_data_subdir = os.path.join(new_data_dir, self._DATA_SUBDIR)
        new_labels_subdir = os.path.join(new_data_dir, self._LABELS_SUBDIR)

        for data_path, labels_path in zip(
                self.iter_data_paths(), self.iter_labels_paths()):
            etau.copy_file(data_path, new_data_subdir)
            etau.copy_file(labels_path, new_labels_subdir)
        self.dataset_index.write_json(dataset_path)

        type = etau.get_class_name(self)
        cls = etau.get_class(type)
        return cls(dataset_path)

    def merge(self, labeled_dataset_or_path, merged_dataset_path,
              in_place=False, description=None):
        '''Union of two labeled datasets.

        Args:
            labeled_dataset_or_path: an `LabeledDataset` instance or path
                to a `manifest.json`, that is of the same type as `self`
            merged_dataset_path: path to `manifest.json` for the merged
                dataset. If `in_place` is False, the containing directory
                must either not exist or be empty. If `in_place` is True,
                either the containing directory must be equal to
                `self.data_dir`, or `merged_dataset_path` is just a filename
                of a new `manifest.json` to write in `self.data_dir`.
            in_place: whether or not to write the merged dataset to a new
                directory. If not, the data from `labeled_dataset_or_path`
                will be added into `self.data_dir`.
            description: optional description for the manifest of the merged
                dataset. If not specified, the existing description is used.

        Returns:
            merged_dataset: a `LabeledDataset` instance pointing to the
                merged dataset. If `in_place` is True, this will just be
                `self`.
        '''
        labeled_dataset = self._parse_dataset(labeled_dataset_or_path)

        data_filenames_to_merge = self._get_filenames_for_merge(
            labeled_dataset)

        output_data_dir = os.path.dirname(merged_dataset_path)
        if not output_data_dir:
            output_data_dir = self.data_dir
            merged_dataset_path = os.path.join(
                output_data_dir, merged_dataset_path)

        if in_place and output_data_dir != self.data_dir:
            raise ValueError(
                "If merging datasets in place, merged_dataset_path should be "
                "within original base directory '%s', but got '%s'" %
                (self.data_dir, output_data_dir))

        if in_place:
            merged_dataset = self
        else:
            merged_dataset = self.copy(merged_dataset_path)

        # Copy files one-by-one from `labeled_dataset`
        for data_path, labels_path in zip(
                labeled_dataset.iter_data_paths(),
                labeled_dataset.iter_labels_paths()):
            if os.path.basename(data_path) in data_filenames_to_merge:
                merged_dataset.add_file(data_path, labels_path)

        if description is not None:
            merged_dataset.set_description(description)

        merged_dataset.write_manifest(merged_dataset_path)
        return merged_dataset

    def apply_to_data(self, func):
        '''Apply the given function to each data element and overwrite the
        data file with the output.

        Args:
            func: function that takes in a data element in the format
                returned by `self._read_data()` and outputs transformed
                data in the same format

        Returns:
            self
        '''
        for data, path in zip(self.iter_data(), self.iter_data_paths()):
            self._write_data(func(data), path)

        return self

    def apply_to_data_paths(self, func):
        '''Call the given function on the path of each data file, with the
        path as both the input and output argument.

        Args:
            func: function that takes in two arguments, an input path and
                an output path. It will be called with the same value for
                both arguments, so that the file is overwritten.

        Returns:
            self
        '''
        for path in self.iter_data_paths():
            func(path, path)

        return self

    def apply_to_labels(self, func):
        '''Apply the given function to each of the labels, and overwrite the
        labels file with the output.

        Args:
            func: function that takes in a labels object in the format
                returned by `self._read_labels()` and outputs transformed
                labels in the same format

        Returns:
            self
        '''
        for labels, path in zip(
                self.iter_labels(), self.iter_labels_paths()):
            self._write_labels(func(labels), path)

        return self

    def apply_to_labels_paths(self, func):
        '''Call the given function on the path of each labels file, with
        the path as both the input and output argument.

        Args:
            func: function that takes in two arguments, an input path and
                an output path. It will be called with the same value for
                both arguments, so that the file is overwritten.

        Returns:
            self
        '''
        for path in self.iter_labels_paths():
            func(path, path)

        return self

    @classmethod
    def create_empty_dataset(cls, dataset_path, description=None):
        '''Creates a new empty labeled dataset.

        Args:
            dataset_path: the path to the `manifest.json` file for the
                dataset to be created
            description: optional description for the manifest of the
                new dataset

        Returns:
            empty_dataset: `LabeledDataset` instance pointing to the
                new empty dataset
        '''
        cls._ensure_empty_dataset_dir(dataset_path)
        dataset_index = LabeledDatasetIndex(
            etau.get_class_name(cls), description=description)
        dataset_index.write_json(dataset_path)
        dataset = cls(dataset_path)

        try:
            cls.is_valid_dataset(dataset_path)
        except NotImplementedError:
            raise TypeError(
                "create_empty_dataset() can only be used with a "
                "`LabeledDataset` subclass that implements "
                "is_valid_dataset()")

        return dataset

    @classmethod
    def is_valid_dataset(cls, dataset_path):
        '''Determines whether the data at the given path is a valid
        LabeledDataset.

        Args:
            dataset_path: the path to the `manifest.json` file for the dataset

        Returns:
            True/False
        '''
        try:
            cls.validate_dataset(dataset_path)
        except LabeledDatasetError:
            return False

        return True

    @classmethod
    def validate_dataset(cls, dataset_path):
        '''Determines whether the data at the given path is a valid
        LabeledDataset.

        Args:
            dataset_path: the path to the `manifest.json` file for the dataset

        Raises:
            LabeledDatasetError: if the dataset at `dataset_path` is not a
                valid LabeledDataset
        '''
        raise NotImplementedError(
            "subclasses must implement validate_dataset()")

    def _build_index_map(self):
        '''Build data --> labels mapping, to ensure this mapping is unique,
        and remains unique.

        We do allow the same labels file to map to multiple data files. This
        may be desirable if many data files have the exact same labels.

        Raises:
            LabeledDatasetError: if the mapping is not unique
        '''
        self._data_to_labels_map = {}
        for record in self.dataset_index:
            data_file = os.path.basename(record.data)
            labels_file = os.path.basename(record.labels)
            if data_file in self._data_to_labels_map:
                raise LabeledDatasetError(
                    "Data file '%s' maps to multiple labels files" %
                    data_file)
            self._data_to_labels_map[data_file] = labels_file

    def _validate_new_data_file(self, data_path):
        '''Checks whether a data file would be a duplicate of an existing
        data file in the dataset.

        Args:
            data_path: path to or filename of the new data file

        Raises:
            ValueError: if the filename of `data_path` is the same as a
                data file already present in the dataset
        '''
        data_file = os.path.basename(data_path)
        if data_file in self._data_to_labels_map:
            raise ValueError(
                "Data file '%s' already present in dataset" % data_file)

    def _read_data(self, path):
        '''Reads data from a data file at the given path.

        Subclasses must implement this based on the particular data format for
        the subclass.

        Args:
            path: path to a data file in the dataset
        '''
        raise NotImplementedError("subclasses must implement _read_data()")

    def _read_labels(self, path):
        '''Reads a labels object from a labels JSON file at the given path.

        Subclasses must implement this based on the particular labels format
        for the subclass.

        Args:
            path: path to a labels file in the dataset
        '''
        raise NotImplementedError("subclasses must implement _read_labels()")

    def _write_data(self, data, path):
        '''Writes data to a data file at the given path.

        Subclasses must implement this based on the particular data format for
        the subclass.  The method should accept input `data` of the same type
        as output by `self._read_data()`.

        Args:
            data: a data element to be written to a file
            path: path to write the data
        '''
        raise NotImplementedError("subclasses must implement _write_data()")

    def _write_labels(self, labels, path):
        '''Writes a labels object to a labels JSON file at the given path.

        Subclasses must implement this based on the particular labels format
        for the subclass.  The method should accept input `labels` of the same
        type as output by `self._read_labels()`.

        Args:
            labels: a labels object to be written to a file
            path: path to write the labels JSON file
        '''
        raise NotImplementedError("subclasses must implement _write_labels()")

    def _parse_dataset(self, labeled_dataset_or_path):
        type = etau.get_class_name(self)
        cls = etau.get_class(type)
        if isinstance(labeled_dataset_or_path, six.string_types):
            labeled_dataset = cls(labeled_dataset_or_path)
        else:
            labeled_dataset = labeled_dataset_or_path

        if not isinstance(labeled_dataset, cls):
            raise TypeError(
                "'%s' is not an instance of '%s'" %
                (etau.get_class_name(labeled_dataset), type))

        return labeled_dataset

    def _get_filenames_for_merge(self, labeled_dataset):
        data_filenames_this = {
            os.path.basename(path) for path in self.iter_data_paths()}
        labels_filenames_this = {
            os.path.basename(path) for path in self.iter_labels_paths()}

        data_filenames_other = {
            os.path.basename(path) for path
            in labeled_dataset.iter_data_paths()}
        data_filenames_to_merge = data_filenames_other - data_filenames_this
        labels_filenames_to_merge = {
            os.path.basename(labels_path) for data_path, labels_path in zip(
                labeled_dataset.iter_data_paths(),
                labeled_dataset.iter_labels_paths())
            if os.path.basename(data_path) in data_filenames_to_merge}

        labels_in_both = labels_filenames_to_merge & labels_filenames_this
        if labels_in_both:
            raise ValueError(
                "Found different data filenames with the same corresponding "
                "labels filename. E.g. %s" % str(list(labels_in_both)[:5]))

        return data_filenames_to_merge

    @classmethod
    def _ensure_empty_dataset_dir(cls, dataset_path):
        etau.ensure_basedir(dataset_path)
        data_dir = os.path.dirname(dataset_path)

        existing_files = os.listdir(data_dir)
        if existing_files:
            raise ValueError(
                "Cannot create a new dataset in a non-empty directory. "
                "Found the following files in directory '%s': %s" %
                (data_dir, existing_files))

        data_subdir = os.path.join(data_dir, cls._DATA_SUBDIR)
        labels_subdir = os.path.join(data_dir, cls._LABELS_SUBDIR)
        etau.ensure_dir(data_subdir)
        etau.ensure_dir(labels_subdir)

    def builder(self):
        '''Creates a LabeledDatasetBuilder instance for this dataset for
        transformations to be run.

        Returns:
            LabeledDatasetBuilder
        '''
        builder = etau.get_class(self._BUILDER_CLS_FIELD)()
        for paths in self.iter_paths():
            builder.add_record(builder.record_cls(*paths))
        return builder


class LabeledVideoDataset(LabeledDataset):
    '''Core class for interacting with a labeled dataset of videos.

    Labeled video datasets are stored on disk in the following format:

    ```
    /path/to/video/dataset/
        manifest.json
        data/
            video1.mp4
            ...
        labels/
            video1.json
            ...
    ```

    where each labels file is stored in `eta.core.video.VideoLabels` format,
    and the `manifest.json` file is stored in `LabeledDatasetIndex` format.

    Labeled video datasets are referenced in code by their `dataset_path`,
    which points to the `manifest.json` file for the dataset.
    '''

    _BUILDER_CLS_FIELD = "eta.core.datasets.LabeledVideoDatasetBuilder"

    @classmethod
    def validate_dataset(cls, dataset_path):
        '''Determines whether the data at the given path is a valid
        LabeledVideoDataset.

        This function checks whether each video and labels path exists and has
        a valid extension, but makes no attempt to read the files.

        Args:
            dataset_path: the path to the `manifest.json` file for the dataset

        Raises:
            LabeledDatasetError: if the dataset at `dataset_path` is not a
                valid LabeledVideoDataset
        '''
        video_dataset = cls(dataset_path)

        for video_path in video_dataset.iter_data_paths():
            if not etav.is_supported_video_file(video_path):
                raise LabeledDatasetError(
                    "Unsupported video format: %s" % video_path)
            if not os.path.isfile(video_path):
                raise LabeledDatasetError("File not found: %s" % video_path)

        for labels_path in video_dataset.iter_labels_paths():
            if not os.path.splitext(labels_path)[1] == ".json":
                raise LabeledDatasetError(
                    "Unsupported labels format: %s" % labels_path)
            if not os.path.isfile(labels_path):
                raise LabeledDatasetError("File not found: %s" % labels_path)

    def _read_data(self, path):
        return etav.FFmpegVideoReader(path)

    def _read_labels(self, path):
        return etav.VideoLabels.from_json(path)

    def _write_data(self, data, path):
        with etav.FFmpegVideoWriter(
                path, data.frame_rate, data.frame_size) as writer:
            for img in data:
                writer.write(img)

    def _write_labels(self, labels, path):
        labels.write_json(path)


class LabeledImageDataset(LabeledDataset):
    '''Core class for interacting with a labeled dataset of images.

    Labeled image datasets are stored on disk in the following format:

    ```
    /path/to/image/dataset/
        manifest.json
        data/
            image1.png
            ...
        labels/
            image1.json
            ...
    ```

    where each labels file is stored in `eta.core.image.ImageLabels` format,
    and the `manifest.json` file is stored in `LabeledDatasetIndex` format.

    Labeled image datasets are referenced in code by their `dataset_path`,
    which points to the `manifest.json` file for the dataset.
    '''

    _BUILDER_CLS_FIELD = "eta.core.datasets.LabeledImageDatasetBuilder"

    @classmethod
    def validate_dataset(cls, dataset_path):
        '''Determines whether the data at the given path is a valid
        LabeledImageDataset.

        This function checks whether each image and labels path exists and has
        a valid extension, but makes no attempt to read the files.

        Args:
            dataset_path: the path to the `manifest.json` file for the dataset

        Raises:
            LabeledDatasetError: if the dataset at `dataset_path` is not a
                valid LabeledImageDataset
        '''
        image_dataset = cls(dataset_path)

        for img_path in image_dataset.iter_data_paths():
            if not etai.is_supported_image(img_path):
                raise LabeledDatasetError(
                    "Unsupported image format: %s" % img_path)
            if not os.path.isfile(img_path):
                raise LabeledDatasetError("File not found: %s" % img_path)

        for labels_path in image_dataset.iter_labels_paths():
            if not os.path.splitext(labels_path)[1] == ".json":
                raise LabeledDatasetError(
                    "Unsupported labels format: %s" % labels_path)
            if not os.path.isfile(labels_path):
                raise LabeledDatasetError("File not found: %s" % labels_path)

    def _read_data(self, path):
        return etai.read(path)

    def _read_labels(self, path):
        return etai.ImageLabels.from_json(path)

    def _write_data(self, data, path):
        etai.write(data, path)

    def _write_labels(self, labels, path):
        labels.write_json(path)


class LabeledDatasetIndex(Serializable):
    '''A class that encapsulates the manifest of a `LabeledDataset`.

    Manifest is stored on disk in the following format:

    ```
        manifest.json
        {
            "description": "",
            "type": "eta.core.datasets.LabeledDataset",
            ...
            "index": [
            {
                "data": "data/video1.mp4",
                "labels": "labels/video1.json"
            },
                ...
            ]
        }
    ```

    Attributes:
        type: the fully qualified class name of the `LabeledDataset` subclass
            that encapsulates the dataset
        index: a list of `LabeledDataRecord`s
        description: an optional description of the dataset
    '''

    def __init__(self, type, index=None, description=None):
        '''Initializes the LabeledDatasetIndex.

        Args:
            type: the fully qualified class name of the `LabeledDataset`
                subclass that encapsulates the dataset
            index: a list of `LabeledDataRecord`s. By default and empty list is
                created.
            description: an optional description of the dataset
        '''
        self.type = type
        self.index = index or []
        self.description = description or ""

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return len(self.index)

    def append(self, labeled_data_record):
        '''Appends an entry to the index.

        Args:
            labeled_data_record: a `LabeledDataRecord` instance
        '''
        self.index.append(labeled_data_record)

    def cull_with_function(self, func):
        '''Removes `LabeledDataRecord`s from the index using the provided
        function.

        Args:
            func: a function that takes in a `LabeledDataRecord` and
                returns a boolean. Only records for which the function
                evaluates to True will be retained in `self.index`.
        '''
        self.index = [record for record in self.index if func(record)]

    def sample(self, k):
        '''Randomly downsamples the index to k elements.

        Args:
            k: the number of entries in the index after sampling
        '''
        self.index = random.sample(self.index, k)

    def shuffle(self):
        '''Randomly shuffles the index.
        '''
        random.shuffle(self.index)

    def split(self, split_fractions=None, descriptions=None,
              split_method="random_exact"):
        '''Splits the `LabeledDatasetIndex` into multiple
        `LabeledDatasetIndex` instances, containing disjoint subsets of
        the original index.

        Args:
            split_fractions: an optional list of split fractions, which
                should sum to 1, that specifies how to split the index.
                By default, [0.5, 0.5] is used.
            descriptions: an optional list of descriptions for the output
                indices. The list should be the same length as
                `split_fractions`. If not specified, the description of
                the original index is used for all of the output
                indices.
            split_method: string describing the method with which to split
                the index

        Returns:
            dataset_indices: list of `LabeledDatasetIndex` instances of
                the same length as `split_fractions`
        '''
        if split_fractions is None:
            split_fractions = [0.5, 0.5]

        if descriptions is None:
            descriptions = [self.description for _ in split_fractions]

        if len(descriptions) != len(split_fractions):
            raise ValueError(
                "split_fractions and descriptions lists should be the "
                "same length, but got len(split_fractions) = %d, "
                "len(descriptions) = %d" %
                (len(split_fractions), len(descriptions)))

        split_func = SPLIT_FUNCTIONS[split_method]
        split_indices = split_func(self.index, split_fractions)

        return [
            LabeledDatasetIndex(self.type, split_index, description)
            for split_index, description in zip(
                    split_indices, descriptions)]

    @classmethod
    def from_dict(cls, d):
        '''Constructs a LabeledDatasetIndex object from a JSON dictionary.'''
        type = d["type"]
        index = d.get("index", None)
        if index is not None:
            index = [LabeledDataRecord.from_dict(rec) for rec in index]
        description = d.get("description", None)

        return cls(type, index=index, description=description)


class LabeledDataRecord(BaseDataRecord):
    '''A record containing a data file and an associated labels file.

    Attributes:
        data: the path to the data file
        labels: the path to the labels file
    '''

    def __init__(self, data, labels):
        '''Creates a LabeledDataRecord instance.

        Args:
            data: the path to the data file
            labels: the path to the labels file
        '''
        self.data = data
        self.labels = labels
        super(LabeledDataRecord, self).__init__()

    @classmethod
    def required(cls):
        return ["data", "labels"]


class LabeledDatasetError(Exception):
    '''Exception raised when there is an error reading a LabeledDataset'''
    pass


class LabeledDatasetBuilder(object):
    '''This object builds a LabeledDataset with with transformations applied,
    e.g. Sampler, Balancer.

    Transformations are run in the order they are added.
    '''

    _DATASET_CLS_FIELD = None
    _BUILDER_DATASET_CLS_FIELD = None

    def __init__(self):
        '''Initialize the LabeledDatasetBuilder.'''
        self._transformers = []
        self._dataset = etau.get_class(self._BUILDER_DATASET_CLS_FIELD)()

    def add_record(self, record):
        '''Add a record. LabeledImageDatasetBuilders take BuilderImageRecords
        and LabeledVideoDatasetBuilders take BuilderVideoRecords.

        Args:
            record (BuilderImageRecord or BuilderVideoRecord)

        Returns:
            None
        '''
        self._dataset.add(record)

    def add_transform(self, transform):
        '''Add a DatasetTransformer.

        Args:
            transform (DatasetTransformer)

        Returns:
            None
        '''
        self._transformers.append(transform)

    @property
    def record_cls(self):
        '''Record class getter.'''
        return self._dataset.record_cls

    def build(self, path, description=None, pretty_print=False):
        '''Build the new LabeledDataset after all records and transformations
        have been added.

        Args:
            path (str): path to write the new dataset (manifest.json)
            description (str): optional dataset description
            pretty_print (bool): pretty print flag for json labels

        Returns:
            LabeledDataset
        '''
        for transformer in self._transformers:
            transformer.transform(self._dataset)

        dataset_cls = etau.get_class(self._DATASET_CLS_FIELD)
        dataset = dataset_cls.create_empty_dataset(path, description)

        with etau.TempDir() as dir_path:
            for idx, record in enumerate(self._dataset):
                result = record.build(dir_path, str(idx),
                                      pretty_print=pretty_print)
                dataset.add_file(*result, move_files=True)
        dataset.update_manifest()
        return dataset


class LabeledImageDatasetBuilder(LabeledDatasetBuilder):
    '''LabeledDatasetBuilder for images.'''

    _DATASET_CLS_FIELD = "eta.core.datasets.LabeledImageDataset"
    _BUILDER_DATASET_CLS_FIELD = "eta.core.datasets.BuilderImageDataset"


class LabeledVideoDatasetBuilder(LabeledDatasetBuilder):
    '''LabeledDatasetBuilder for videos.'''

    _DATASET_CLS_FIELD = "eta.core.datasets.LabeledVideoDataset"
    _BUILDER_DATASET_CLS_FIELD = "eta.core.datasets.BuilderVideoDataset"


class BuilderDataRecord(BaseDataRecord):
    '''This class is responsible for tracking all of the metadata about a data
    record required for dataset operations on a BuilderDataset.
    '''

    def __init__(self, data_path, labels_path):
        '''Initialize the BuilderDataRecord. The label and data paths cannot and
        should not be modified after initializion.

        Args:
            data_path (str): path to data file
            labels_path (str): path to labels json
        '''
        self._data_path = data_path
        self._labels_path = labels_path
        self._labels_cls = None
        self._labels_obj = None

    def get_labels(self):
        '''Labels getter.

        Returns:
            ImageLabels or VideoLabels
        '''
        if self._labels_obj is not None:
            return self._labels_obj
        self._labels_obj = self._labels_cls.from_json(self.labels_path)
        return self._labels_obj

    def set_labels(self, labels):
        '''Labels setter.

        Args:
            labels (ImageLabels or VideoLabels)

        Returns:
            None
        '''
        self._labels_obj = labels

    @property
    def data_path(self):
        '''Data path getter.'''
        return self._data_path

    @property
    def labels_path(self):
        '''Labels path getter.'''
        return self._labels_path

    def build(self, dir_path, filename, pretty_print=False):
        '''Write the transformed labels and data files to dir_path. The
        subclasses BuilderVideoRecord and BuilderDataRecord are responsible for
        writing the data file.

        Args:
            dir_path (str): path to write the files
            filename (str): filename prefix that data and labels share
            pretty_print (bool): pretty_print json flag for labels

        Returns:
            tuple (data_path, labels_path): the paths to the written files
        '''
        labels_path = os.path.join(dir_path, filename + ".json")
        labels = self.get_labels()

        data_ext = os.path.splitext(self.data_path)[1]
        data_path = os.path.join(dir_path, filename + data_ext)

        labels.filename = filename + data_ext
        labels.write_json(labels_path, pretty_print=pretty_print)
        return data_path, labels_path

    def copy(self):
        '''Safely copy a record. Only copy should be used when creating new
        records in DatasetTransformers.

        Returns:
            BuilderImageRecord or BuilderVideoRecord
        '''
        return copy.deepcopy(self)


class BuilderImageRecord(BuilderDataRecord):
    '''BuilderDataRecord for images.'''

    def __init__(self, image_path, labels_path):
        '''Initialize a BuilderVideoRecord with the data_path and labels_path.

        Args:
            data_path (str): path to video
            labels_path (str): path to labels
        '''
        super(BuilderImageRecord, self).__init__(image_path, labels_path)
        self._labels_cls = etai.ImageLabels

    def build(self, dir_path, filename, pretty_print=False):
        '''Build the BuilderImageRecord.

        Args:
            dir_path (str): path to write the files
            filename (str): filename prefix that data and labels share
            pretty_print (bool): pretty_print json flag for labels

        Returns:
            tuple (data_path, labels_path): the paths to the written files
        '''
        args = (dir_path, filename, pretty_print)
        data_path, labels_path = super(BuilderImageRecord, self).build(*args)

        etau.copy_file(self.data_path, data_path)
        return data_path, labels_path


class BuilderVideoRecord(BuilderDataRecord):
    '''BuilderDataRecord for video.'''

    def __init__(self, data_path, labels_path, clip_start_frame=1,
                 clip_end_frame=None, duration=None,
                 total_frame_count=None):
        '''Initialize a BuilderVideoRecord with data_path, labels_path, and
        optional metadata about video. Without the optional arguments their
        values will be loaded from the video metadata and the start and end
        frames will default to covering the entire video.

        Args:
            data_path (str): path to video
            labels_path (str): path to labels
            clip_start_frame (int): start frame of the clip
            clip_end_frame (int): end frame of the clip
            duration (float): duration (seconds) of the VIDEO (NOT THE CLIP)
            total_frame_count (int): frame count of the VIDEO (NOT THE CLIP)
        '''
        super(BuilderVideoRecord, self).__init__(data_path, labels_path)
        self.clip_start_frame = clip_start_frame
        self._metadata = None
        if None in [clip_end_frame, duration, total_frame_count]:
            self._init_from_video_metadata()
        else:
            self.clip_end_frame = clip_end_frame
            self.duration = duration
            self.total_frame_count = total_frame_count
        self._labels_cls = etav.VideoLabels

    def _extract_video_labels(self):
        start_frame, end_frame = (self.clip_start_frame, self.clip_end_frame)
        segment = self._labels_cls()
        labels = self.get_labels()
        self.set_labels(segment)
        if not labels:
            return
        for frame_id in range(start_frame, end_frame + 1):
            frame = labels[frame_id]
            frame.frame_number = frame.frame_number - start_frame + 1
            if frame.objects:
                segment.add_objects(frame.objects, frame.frame_number)
            if frame.attrs:
                segment.add_frame_attributes(frame.attrs, frame.frame_number)

    def _init_from_video_metadata(self):
        metadata = etav.VideoMetadata.build_for(self.data_path)
        self.total_frame_count = metadata.total_frame_count
        self.duration = metadata.duration
        self.clip_end_frame = metadata.total_frame_count

    def build(self, dir_path, filename, pretty_print=False):
        '''Build the BuilderVideoRecord.

        Args:
            dir_path (str): path to write the files
            filename (str): filename prefix that data and labels share
            pretty_print (bool): pretty_print json flag for labels

        Returns:
            tuple (data_path, labels_path): the paths to the written files
        '''
        self._extract_video_labels()
        args = (dir_path, filename, pretty_print)
        data_path, labels_path = super(BuilderVideoRecord, self).build(*args)
        start_frame, end_frame = (self.clip_start_frame, self.clip_end_frame)
        if start_frame == 1 and end_frame == self.total_frame_count:
            etau.copy_file(self.data_path, data_path)
        else:
            args = (
                self.data_path,
                etav.FrameRanges([(start_frame, end_frame)])
            )
            with etav.VideoProcessor(*args, out_video_path=data_path) as p:
                for img in p:
                    p.write(img)
        return data_path, labels_path


class BuilderDataset(etad.DataRecords):
    '''A BuilderDataset is managed by a LabeledDatasetBuilder.
    DatasetTransformers operate on BuilderDatasets.
    '''

    def __init__(self, record_cls):
        super(BuilderDataset, self).__init__(record_cls)


class BuilderImageDataset(BuilderDataset):
    '''A BuilderDataset for images.'''

    def __init__(self, record_cls=BuilderImageRecord):
        super(BuilderImageDataset, self).__init__(record_cls)


class BuilderVideoDataset(BuilderDataset):
    '''A BuilderDataset for videos.'''

    def __init__(self, record_cls=BuilderVideoRecord):
        super(BuilderVideoDataset, self).__init__(record_cls)


class DatasetTransformer(object):
    '''Classes that subclass DatasetTransformer operate on BuilderDatasets
    (BuilderImageDataset or BuilderVideoDataset). Only transform() will be
    called outside the instances of a DatasetTransformer.
    '''

    def transform(self, src):
        ''' Transform a TransformableDataset

        Args:
            src (BuilderImageDataset or BuilderVideoDataset)

        Returns:
            None
        '''
        raise NotImplementedError("implementation required")


class Sampler(DatasetTransformer):
    '''Randomly sample the number of records in the dataset to some number k.
    '''

    def __init__(self, k):
        '''Initialize the Samples with k; the number of samples to take.

        Args:
            k (int)
        '''
        self.k = k

    def transform(self, src):
        '''Sample from the existing records.

        Args:
            src (BuilderImageDataset or BuilderVideoDataset

        Returns:
            None
        '''
        try:
            src.records = random.sample(src.records, self.k)
        except ValueError as err:
            raise DatasetTransformerError(err.message)


class Balancer(DatasetTransformer):
    '''Balance the the dataset's values of a categorical attribute by removing
    records.

    For example:
        Given a dataset with 10 green cars, 20 blue cars and 15 red cars,
        remove records with blue and red cars until there are the same number
        of each color.

        In this example 'color' is the `attribute_name` and 'car' is the
        `object_label`.
    '''

    def __init__(self, attribute_name, object_label=None,
                 target_quantile=0.25, negative_power=5.0):
        '''Creates a Balancer instance.

        Args:
            attribute_name (str): the name of the attribute to balance by
            object_label (str): the name of the object label that the
                attribute_name must be nested under. If this is None, it is
                assumed that the attributes are Image/Frame level attrs.
            target_quantile (float): value between [0, 1] to specify what the
                target count per attribute value will be.
                0.5 - will result in the true median
                0   - the minimum value
                It is recommended to set this somewhere between [0, 0.5]. The
                smaller this value is, the closer all values can be balanced,
                at the risk that if some values have particularly low number of
                samples, they dataset will be excessively trimmed.
            negative_power (float): value between [1, ~LARGE~] that weights the
                negative values (where the count of a value is less than the
                target) when computing the score for a set of indices to remove.
                1 - will weight them the same as positive values
                2 - will square the values
                ...
                Check Balancer._solution_score for more details.
        '''
        self.attr_name = attribute_name
        self.object_label = object_label
        self.target_quantile = target_quantile
        self.negative_power = negative_power

    def transform(self, src):
        '''Modify the BuilderDataset records by removing records until the
        target attribute is ~roughly~ balanced for each value.

        Args:
            src (BuilderDataset): the dataset builder
        '''

        # STEP 1: Get attribute value(s) for every record
        helper_list = self._to_helper_list(src.records)
        if not len(helper_list):
            return

        # STEP 2: determine target number to remove of each attribute value
        target_remove = self._get_target_remove(helper_list)

        # STEP 3: find the records to keep
        keep_idxs = self._get_keep_idxs(target_remove, helper_list)

        # STEP 4: modify the list of records
        old_records = src.records
        src.clear()
        for ki in keep_idxs:
            src.add(old_records[helper_list[ki][0]])

    def _to_helper_list(self, records):
        '''Recompile the records to a list of counts of each attribute value.

        Args:
            records: list of BuilderDataRecord's

        Returns:
            a list of tuples with two entries:

                [
                    (record_id, list_of_values),
                    (record_id, list_of_values),
                    (record_id, list_of_values),
                    ...
                ]

            record_id: integer ID of the corresponding old_record
            list_of_values: list of attribute values for the attribute to be
                            balanced, one per unique object, if using objects.
                            For example: ['red', 'red', 'green'] would imply
                            three objects with the 'color' attribute in this
                            record.
        '''

        if not len(records):
            return []

        elif isinstance(records[0], BuilderImageRecord):
            if self.object_label:
                return self._to_helper_list_image_objects(records)
            else:
                return self._to_helper_list_image(records)

        elif isinstance(records[0], BuilderVideoRecord):
            if self.object_label:
                return self._to_helper_list_video_objects(records)
            else:
                return self._to_helper_list_video(records)

        raise DatasetTransformerError(
            "Unknown record type: {}".format(type(records[0])))

    def _to_helper_list_image(self, records):
        '''Balancer._to_helper_list for image attributes'''
        helper_list = []

        for i, record in enumerate(records):
            labels = record.get_labels()

            for attr in labels.attrs:
                if attr.name == self.attr_name:
                    helper_list.append((i, [attr.value]))
                    break

        return helper_list

    def _to_helper_list_image_objects(self, records):
        '''Balancer._to_helper_list for object attributes in images'''
        helper_list = []

        for i, record in enumerate(records):
            labels = record.get_labels()
            helper = (i, [])

            for detected_object in labels.objects:
                if detected_object.label != self.object_label:
                    continue

                for attr in detected_object.attrs:
                    if attr.name == self.attr_name:
                        helper[1].append(attr.value)
                        break

            if len(helper[1]):
                helper_list.append(helper)

        return helper_list

    def _to_helper_list_video(self, records):
        '''Balancer._to_helper_list for video attributes'''
        helper_list = []

        for i, record in enumerate(records):
            labels = record.get_labels()
            helper = (i, set())

            for frame in labels.frames.values():
                for attr in frame.attrs:
                    if attr.name == self.attr_name:
                        helper[1].add(attr.value)
                        break

            if len(helper[1]):
                helper = (helper[0], list(helper[1]))
                helper_list.append(helper)

        return helper_list

    def _to_helper_list_video_objects(self, records):
        '''Balancer._to_helper_list for object attributes in videos'''
        helper_list = []

        for i, record in enumerate(records):
            labels = record.get_labels()
            NO_ID = 'NO_ID'
            helper_dict = defaultdict(set)

            for frame_no, frame in labels.frames.items():
                if (frame_no < record.clip_start_frame
                        or frame_no >= record.clip_end_frame):
                    continue

                for detected_object in frame.objects:
                    if detected_object.label != self.object_label:
                        continue

                    for attr in detected_object.attrs:
                        if attr.name == self.attr_name:
                            obj_idx = (
                                detected_object.index
                                if detected_object.index is not None
                                else NO_ID
                            )

                            helper_dict[obj_idx].add(attr.value)

                            break

            # At this point, the keys of helper dict are unique
            # object indices for objects of type self.object_label.
            # The values are unique attribute values for self.attr_name.

            if len(helper_dict):
                helper = (i, [])
                for s in helper_dict.values():
                    helper[1].extend(s)
                helper_list.append(helper)

        return helper_list

    def _get_target_remove(self, helper_list):
        '''Get the target number attributes to remove for each attribute value.

        Args:
            helper_list: return value from Balancer._to_helper_list(...)

        Returns:
            a dictionary with:
                keys: the values of the attribute to balance
                values: the target count to remove of this attribute value
        '''
        counts = Counter()
        for idx, attr_values in helper_list:
            counts.update(attr_values)

        # sort ascending by count
        counts_list = counts.most_common()[::-1]

        count_vals = [v for k, v in counts_list]
        target_count = int(np.quantile(count_vals, self.target_quantile))

        target_remove = {k: v - target_count for k, v in counts_list}
        # @TODO leave the negatives in? (comment this line out)
        # target_remove = {k: max(v, 0) for k, v in target_remove.items()}

        return target_remove

    def _get_keep_idxs(self, target_remove, helper_list):
        '''Algorithm fun! This function chooses the set of records to keep (and
        remove).

        There's still plenty of potential for testing and improvement here.

        Args:
            target_remove: dictionary returned from Balancer._get_target_remove
            helper_list: return value of Balancer._to_helper_list

        Returns:
            a list of integer indices to keep
        '''
        keep_idxs = list(range(len(helper_list)))
        remove_idxs = []
        score = self._get_score(target_remove, helper_list, remove_idxs)

        # # ALGO1 - random
        # import random
        # random.seed(1)
        # for _ in range(10000):
        #     i = random.randrange(len(keep_idxs))
        #     idx = keep_idxs[i]
        #     score2 = self._get_score(
        #         target_remove,
        #         helper_list,
        #         remove_idxs + [idx]
        #     )
        #     if score2 < score:
        #         score = score2
        #         remove_idxs.append(keep_idxs.pop(i))

        # ALGO2 - greedy
        while len(keep_idxs) > 0:
            best_i = 0
            best_score = self._get_score(
                target_remove,
                helper_list,
                remove_idxs + [keep_idxs[best_i]]
            )
            for i in range(1, len(keep_idxs)):
                cur_score = self._get_score(
                    target_remove,
                    helper_list,
                    remove_idxs + [keep_idxs[i]]
                )
                if cur_score < best_score:
                    best_score = cur_score
                    best_i = i

            if best_score >= score:
                break
            score = best_score
            remove_idxs.append(keep_idxs.pop(best_i))

        return keep_idxs

    def _get_score(self, target_remove, helper_list, remove_idxs):
        '''Compute the performance 'score' for a given set of indices to remove.

        Args:
            target_remove: dictionary returned from Balancer._get_target_remove
            helper_list: return value of Balancer._to_helper_list
            remove_idxs: a list of integer indices to remove

        Returns:
            the score for the given remove_idxs
        '''
        temp = target_remove.copy()
        for idx in remove_idxs:
            for value in helper_list[idx][1]:
                temp[value] = temp[value] - 1

        # v = list(temp.values())
        # v.sort()
        # print(v)

        return self._solution_score(temp.values())

    def _solution_score(self, vector):
        '''Compute the score for a vector (smaller is better). This is a custom
        scoring function that sorta computes the L1 norm for positive values
        and the L<X> norm for negative values where <X> is self.negative_power.

        Larger self.negative_power puts more weight on not reducing the count
        of any attribute values that are already below the target.

        Args:
            vector: Vector_Of_Counts - Target_Count

        Returns:
            (float) a score value, (which is only meaningful in a relative
                    sense). Smaller -> Better!
        '''
        v_pos = np.maximum(vector, 0)
        v_neg = np.abs(np.minimum(vector, 0))
        vector2 = v_pos + (v_neg ** self.negative_power)
        return np.sum(vector2)


class SchemaFilter(DatasetTransformer):
    '''Filter all labels in the dataset by the provided schema. If the schema is
    None, no filtering is done.
    '''

    def __init__(self, schema):
        '''Initialize the SchemaFilter with a schema.

        Args:
            schema (VideoLabelsSchema orImageLabelsSchema)
        '''
        self.schema = schema

    def transform(self, src):
        '''Filter all records in src. If the schema is None, no filtering is
        done.

        Args:
            src (BuilderImageDataset or BuilderVideoDataset)

        Returns:
            None
        '''
        if self.schema is None:
            return
        for record in src:
            labels = record.get_labels()
            labels.filter_by_schema(self.schema)
            record.set_labels(labels)


class Clipper(DatasetTransformer):
    '''Clip longer videos into shorter ones, and sample at some stride step.'''

    def __init__(self, clip_len, stride_len, min_clip_len):
        '''Creates a Clipper instance. min_clip_len determines whether
        remainders are included or not.

        Args:
            clip_len (int): number of frames per clip, must be > 0
            stride_len (int): stride (step size), must be > 0
            min_clip_len (int): minimum number of frames allowed, must be > 0
                                and greater than clip_len
        '''
        self.clip_len = int(clip_len)
        self.stride_len = int(stride_len)
        self.min_clip_len = int(min_clip_len)
        bad_args = self.clip_len < 1 or self.stride_len < 1
        bad_args = bad_args or self.min_clip_len < 1
        bad_args = bad_args or self.min_clip_len > self.clip_len
        if bad_args:
            raise DatasetTransformerError("Bad args provided to Clipper")

    def transform(self, src):
        '''Create the new record list made of clipped records from the old
        records list.

        Args:
            src (BuilderVideoDataset)

        Returns:
            None
        '''
        if not isinstance(src, BuilderVideoDataset):
            raise DatasetTransformerError(
                "Clipper transform can only operate on BuilderVideoDatasets")
        old_records = src.records
        src.clear()
        for record in old_records:
            start_frame = record.clip_start_frame
            while start_frame <= record.clip_end_frame:
                end_frame = start_frame + self.clip_len - 1
                if end_frame > record.clip_end_frame:
                    end_frame = record.clip_end_frame
                    clip_duration = int(end_frame - start_frame + 1)
                    if clip_duration < self.min_clip_len:
                        break
                self._add_clipping(start_frame, end_frame, record, src.records)
                start_frame += self.stride_len

    def _add_clipping(self, start_frame, end_frame, old_record, records):
        new_record = old_record.copy()
        new_record.clip_start_frame = start_frame
        new_record.clip_end_frame = end_frame
        records.append(new_record)


class EmptyLabels(DatasetTransformer):
    '''Assign empty labels to all records.'''

    def transform(self, src):
        '''Assign empty labels to all records.

        Args:
            src (BuilderDataRecord)

        Returns:
            None
        '''
        if not src:
            return

        labels_cls = src.records[0].get_labels().__class__

        for record in src:
            record.set_labels(labels_cls())


class DatasetTransformerError(Exception):
    '''Exception raised when there is an error in a DatasetTransformer'''
    pass
