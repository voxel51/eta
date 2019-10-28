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
from future.utils import iteritems, itervalues
import six
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict
import copy
import glob
import logging
import os
import random
import re
import shutil

import numpy as np

import eta.core.annotations as etaa
from eta.core.data import BaseDataRecord, DataRecords
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
    '''
    if stride is None and num_images is None:
        stride = 1

    _validate_stride_and_num_images(stride, num_images)

    if num_images is not None and stride is None:
        stride = _compute_stride(video_dataset, num_images, frame_filter)

    logger.info("Sampling video frames with stride %d", stride)

    image_dataset = LabeledImageDataset.create_empty_dataset(
        image_dataset_path, description=description)

    frame_iterator = _iter_filtered_video_frames(
        video_dataset, frame_filter, stride)
    for img_number, (frame_img, frame_labels, base_filename) in enumerate(
            frame_iterator, 1):
        image_filename = "%s%s" % (base_filename, image_extension)
        labels_filename = "%s.json" % base_filename

        image_labels = etai.ImageLabels(
            filename=image_filename,
            attrs=frame_labels.attrs,
            objects=frame_labels.objects)
        image_dataset.add_data(
            frame_img, image_labels, image_filename,
            labels_filename)

        if num_images is not None and img_number >= num_images:
            break

    if not image_dataset:
        logger.info(
            "All frames were filtered out in sample_videos_to_images(). "
            "Writing an empty image dataset to '%s'.",
            image_dataset_path)

    image_dataset.write_manifest(image_dataset_path)

    return image_dataset


def get_manifest_path_from_dir(dirpath):
    '''Infers the filename of the manifest within the given
    `LabeledDataset` directory, and returns the path to that file.

    Args:
        dirpath: path to a `LabeledDataset` directory

    Returns:
        path to the manifest inside the directory `dirpath`
    '''
    candidate_manifests = {
        os.path.basename(path)
        for path in glob.glob(os.path.join(dirpath, "*.json"))
    }

    if not candidate_manifests:
        raise ValueError(
            "Directory '%s' contains no JSON files to use as a "
            "manifest" % dirpath
        )

    if "manifest.json" in candidate_manifests:
        return os.path.join(dirpath, "manifest.json")

    starts_with_manifest = [
        filename for filename in candidate_manifests
        if filename.startswith("manifest")
    ]
    if starts_with_manifest:
        return os.path.join(dirpath, starts_with_manifest[0])

    return os.path.join(dirpath, candidate_manifests.pop())


def load_dataset(manifest_path):
    '''Loads a `LabeledDataset` instance from the manifest JSON at
    the given path.

    The manifest JSON is assumed to sit inside a `LabeledDataset`
    directory structure, as outlined in the `LabeledDataset`
    documentation. This function will read the specific subclass
    of `LabeledDataset` that should be used to load the dataset
    from the manifest, and return an instance of that subclass.

    Args:
        manifest_path: path to a `manifest.json` within a
            `LabeledDataset` directory

    Returns:
        an instance of a `LabeledDataset` subclass, (e.g.
            `LabeledImageDataset`, `LabeledVideoDataset`)
    '''
    index = LabeledDatasetIndex.from_json(manifest_path)
    dataset_type = index.type
    dataset_cls = etau.get_class(dataset_type)

    return dataset_cls(manifest_path)


def _validate_stride_and_num_images(stride, num_images):
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

    Note that any method that doesn't take a manifest path as input will only
    change the internal state in `self.dataset_index`, and will not write to
    any manifest JSON files on disk. (Example methods are `self.sample()` and
    `self.add_file()`.) Thus it may desirable to use the
    `self.write_manifest()` method to write the internal state to a manifest
    JSON file on disk, at some point after using these methods.

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
        self.data_dir = os.path.dirname(dataset_path)

        self._build_index_map()

    def __iter__(self):
        '''Iterates over the samples in the dataset.

        Returns:
            iterator: iterator over (data, labels) pairs, where data is an
                object returned by self._read_data() and labels is an object
                returned by self._read_labels() from the respective paths
                of a data file and corresponding labels file
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
        '''Iterates over the paths to labels files in the dataset.

        Returns:
            iterator: iterator over paths to labels files
        '''
        for record in self.dataset_index:
            yield os.path.join(self.data_dir, record.labels)

    def iter_paths(self):
        '''Iterates over the data and labels paths tuple in the dataset.

        Returns:
            iterator: iterator over (path to data, path to labels file)
                tuples
        '''
        return zip(self.iter_data_paths(), self.iter_labels_paths())

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

    def remove_data_files(self, data_filenames):
        '''Removes the given data files from `self.dataset_index`.

        Note that this method doesn't delete the files from disk, it just
        removes them from the index. To remove files from disk that are
        not present in the index, use the `prune()` method.

        Args:
            data_filenames: list of filenames of data to remove

        Returns:
            self
        '''
        removals = set(data_filenames)
        self.dataset_index.cull_with_function(
            lambda record: os.path.basename(record.data) not in removals)

        self._build_index_map()

        return self

    def copy(self, dataset_path, symlink_data=False):
        '''Copies the dataset to another directory.

        If the dataset index has been manipulated, this will be reflected
        in the copy.

        Args:
            dataset_path: the path to the `manifest.json` file for the
                copy of the dataset that will be written. The containing
                directory must either not exist or be empty.
            symlink_data: whether or not to symlink the data directory
                instead of copying over all of the files. Note that with
                this option, the entire labels directory will be copied,
                so the new `LabeledDataset` directory may contain data
                and labels that are not in the manifest, (depending on
                the current state of `self.dataset_index`).

        Returns:
            dataset_copy: `LabeledDataset` instance that points to the new
                containing directory
        '''
        self._ensure_empty_dataset_dir(dataset_path)

        new_data_dir = os.path.dirname(dataset_path)
        new_data_subdir = os.path.join(new_data_dir, self._DATA_SUBDIR)
        new_labels_subdir = os.path.join(new_data_dir, self._LABELS_SUBDIR)

        if symlink_data:
            os.rmdir(new_data_subdir)
            os.rmdir(new_labels_subdir)
            os.symlink(
                os.path.abspath(os.path.realpath(
                    os.path.join(self.data_dir, self._DATA_SUBDIR))),
                new_data_subdir
            )
            shutil.copytree(
                os.path.join(self.data_dir, self._LABELS_SUBDIR),
                new_labels_subdir
            )
        else:
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

        merged_dataset.write_manifest(
            os.path.basename(merged_dataset_path))
        return merged_dataset

    def deduplicate(self):
        '''Removes duplicate data files from the index.

        If sets of files are found with the same content, one file in each
        is set is chosen arbitarily to be kept, and the rest are removed.
        Note that no files are deleted; this method only removes entries
        from the index.

        Returns:
            self
        '''
        duplicate_data_paths = etau.find_duplicate_files(
            list(self.iter_data_paths()))

        if not duplicate_data_paths:
            return self

        data_paths_remove = set()
        for group in duplicate_data_paths:
            for path in group[1:]:
                data_paths_remove.add(path)

        self.dataset_index.cull_with_function(
            lambda record: os.path.join(
                self.data_dir, record.data) not in data_paths_remove)

        self._build_index_map()

        return self

    def prune(self):
        '''Deletes data and label files that are not in the index.

        Note that actual files will be deleted if they are not present in
        `self.dataset_index`, for which the current state can be different
        than when it was read from a manifest JSON file.

        Returns:
            self
        '''
        data_filenames = set()
        labels_filenames = set()
        for data_path, labels_path in self.iter_paths():
            data_filenames.add(os.path.basename(data_path))
            labels_filenames.add(os.path.basename(labels_path))

        data_subdir = os.path.join(self.data_dir, self._DATA_SUBDIR)
        for filename in etau.list_files(data_subdir):
            if filename not in data_filenames:
                os.remove(os.path.join(data_subdir, filename))

        labels_subdir = os.path.join(self.data_dir, self._LABELS_SUBDIR)
        for filename in etau.list_files(labels_subdir):
            if filename not in labels_filenames:
                os.remove(os.path.join(labels_subdir, filename))

        return self

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

    def builder(self):
        '''Creates a LabeledDatasetBuilder instance for this dataset for
        transformations to be run.

        Returns:
            LabeledDatasetBuilder
        '''
        builder = self.builder_cls()
        for paths in self.iter_paths():
            builder.add_record(builder.record_cls(*paths))
        return builder

    def get_labels_set(self, ensure_filenames=True):
        '''Creates a SetLabels type object from this dataset's labels,
        (e.g. etai.ImageSetLabels, etav.VideoSetLabels).

        Args:
            ensure_filenames: whether to add the filename into the individual
                labels if it is not present

        Returns:
            etai.ImageSetLabels, etav.VideoSetLabels, etc.
        '''
        raise NotImplementedError(
            "subclasses must implement get_labels_set()")

    def write_annotated_data(self, output_dir_path, annotation_config=None):
        '''Annotates the data with its labels, and outputs the
        annotated data to the specified directory.

        Args:
            output_dir_path: the path to the directory into which
                the annotated data will be written
            annotation_config: an optional etaa.AnnotationConfig specifying
                how to render the annotations. If omitted, the default config
                is used.
        '''
        raise NotImplementedError(
            "subclasses must implement write_annotated_data()")

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

    @property
    def builder_cls(self):
        '''Getter for the associated LabeledDatasetBuilder class.

        Returns:
            builder_cls: the associated LabeledDatasetBuilder class
        '''
        return etau.get_class(etau.get_class_name(self) + "Builder")


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

    def get_labels_set(self, ensure_filenames=True):
        '''Creates an etav.VideoSetLabels type object from this dataset's
        labels.

        Args:
            ensure_filenames: whether to add the filename into the individual
                labels if it is not present

        Returns:
            etav.VideoSetLabels
        '''
        if ensure_filenames:
            video_set_labels = etav.VideoSetLabels()
            for data_path, video_labels in zip(
                    self.iter_data_paths(), self.iter_labels()):
                if video_labels.filename is None:
                    filename = os.path.basename(data_path)
                    video_labels.filename = filename
                else:
                    filename = video_labels.filename
                video_set_labels[filename] = video_labels

            return video_set_labels

        return etav.VideoSetLabels(videos=list(self.iter_labels()))

    def write_annotated_data(self, output_dir_path, annotation_config=None):
        '''Annotates the data with its labels, and outputs the
        annotated data to the specified directory.

        Args:
            output_dir_path: the path to the directory into which
                the annotated data will be written
            annotation_config: an optional etaa.AnnotationConfig specifying
                how to render the annotations. If omitted, the default config
                is used.
        '''
        for video_path, video_labels in zip(
                self.iter_data_paths(), self.iter_labels()):
            output_path = os.path.join(
                output_dir_path, os.path.basename(video_path))
            etaa.annotate_video(
                video_path, video_labels, output_path,
                annotation_config=annotation_config)

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

    def get_labels_set(self, ensure_filenames=True):
        '''Creates an etai.ImageSetLabels type object from this dataset's
        labels.

        Args:
            ensure_filenames: whether to add the filename into the individual
                labels if it is not present

        Returns:
            etai.ImageSetLabels
        '''
        if ensure_filenames:
            image_set_labels = etai.ImageSetLabels()
            for data_path, image_labels in zip(
                    self.iter_data_paths(), self.iter_labels()):
                if image_labels.filename is None:
                    filename = os.path.basename(data_path)
                    image_labels.filename = filename
                else:
                    filename = image_labels.filename
                image_set_labels[filename] = image_labels

            return image_set_labels

        return etai.ImageSetLabels(images=list(self.iter_labels()))

    def write_annotated_data(self, output_dir_path, annotation_config=None):
        '''Annotates the data with its labels, and outputs the
        annotated data to the specified directory.

        Args:
            output_dir_path: the path to the directory into which
                the annotated data will be written
            annotation_config: an optional etaa.AnnotationConfig specifying
                how to render the annotations. If omitted, the default config
                is used.
        '''
        for img, image_path, image_labels in zip(
                self.iter_data(), self.iter_data_paths(), self.iter_labels()):
            img_annotated = etaa.annotate_image(
                img, image_labels, annotation_config=annotation_config)
            output_path = os.path.join(
                output_dir_path, os.path.basename(image_path))
            self._write_data(img_annotated, output_path)

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
    '''This object builds a LabeledDataset with transformations applied,
    e.g. Sampler, Balancer.

    Transformations are run in the order they are added.
    '''

    def __init__(self):
        '''Initialize the LabeledDatasetBuilder.'''
        self._transformers = []
        self._dataset = self.builder_dataset_cls()

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
    def builder_dataset(self):
        '''The underlying BuilderDataset instance'''
        return self._dataset

    @property
    def builder_dataset_cls(self):
        '''Associated BuilderDataset class getter.'''
        cls_breakup = etau.get_class_name(self).split(".")
        cls = cls_breakup[-1]
        cls = re.sub("^Labeled", "Builder", re.sub("Builder$", "", cls))
        cls_breakup[-1] = cls
        full_cls_path = ".".join(cls_breakup)
        return etau.get_class(full_cls_path)

    @property
    def dataset_cls(self):
        '''Associated LabeledDataset class getter.'''
        cls = etau.get_class_name(self)
        cls = re.sub("Builder$", "", cls).split(".")[-1]
        return etau.get_class(cls, "eta.core.datasets")

    @property
    def record_cls(self):
        '''Record class getter.'''
        return self._dataset.record_cls

    def build(self, path, description=None, pretty_print=False,
              tmp_dir_base=None):
        '''Build the new LabeledDataset after all records and transformations
        have been added.

        Args:
            path (str): path to write the new dataset (manifest.json)
            description (str): optional dataset description
            pretty_print (bool): pretty print flag for json labels
            tmp_dir_base (str): optional directory in which to make temp dirs

        Returns:
            LabeledDataset
        '''
        logger.info("Applying transformations to dataset")

        for transformer in self._transformers:
            transformer.transform(self._dataset)

        logger.info(
            "Building dataset with %d elements" % len(self.builder_dataset)
        )

        dataset = self.dataset_cls.create_empty_dataset(path, description)

        with etau.TempDir(dir=tmp_dir_base) as dir_path:
            for idx, record in enumerate(self._dataset):
                result = record.build(dir_path, str(idx),
                                      pretty_print=pretty_print)
                dataset.add_file(*result, move_files=True)
        dataset.write_manifest(os.path.basename(path))
        return dataset


class LabeledImageDatasetBuilder(LabeledDatasetBuilder):
    '''LabeledDatasetBuilder for images.'''


class LabeledVideoDatasetBuilder(LabeledDatasetBuilder):
    '''LabeledDatasetBuilder for videos.'''


class BuilderDataRecord(BaseDataRecord):
    '''This class is responsible for tracking all of the metadata about a data
    record required for dataset operations on a BuilderDataset.
    '''

    _LABELS_EXT = ".json"

    def __init__(self, data_path, labels_path):
        '''Initialize the BuilderDataRecord. The label and data paths cannot
        and should not be modified after initialization.

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
        self._build_labels()

        labels_path = os.path.join(dir_path, filename + self._LABELS_EXT)
        labels = self.get_labels()

        data_ext = os.path.splitext(self.data_path)[1]
        data_path = os.path.join(dir_path, filename + data_ext)

        labels.filename = filename + data_ext
        labels.write_json(labels_path, pretty_print=pretty_print)

        self._build_data(data_path)
        return data_path, labels_path

    def copy(self):
        '''Safely copy a record. Only copy should be used when creating new
        records in DatasetTransformers.

        Returns:
            BuilderImageRecord or BuilderVideoRecord
        '''
        return copy.deepcopy(self)

    def attributes(self):
        '''Overrides Serializable.attributes() to provide a custom list of
        attributes to be serialized.

        Returns:
            a list of class attributes to be serialized
        '''
        return super(BuilderDataRecord, self).attributes() + [
            "data_path",
            "labels_path"
        ]

    @classmethod
    def required(cls):
        '''Returns a list of attributes that are required by all instances of
        the data record.
        '''
        return super(BuilderDataRecord, cls).required() + [
            "data_path",
            "labels_path"
        ]

    def _build_labels(self):
        raise NotImplementedError(
            "subclasses must implement _build_labels()")

    def _build_data(self, data_path):
        raise NotImplementedError(
            "subclasses must implement _build_data()")


class BuilderImageRecord(BuilderDataRecord):
    '''BuilderDataRecord for images.'''

    def __init__(self, data_path, labels_path):
        '''Initialize a BuilderImageRecord with the data_path and labels_path.

        Args:
            data_path (str): path to image
            labels_path (str): path to labels
        '''
        super(BuilderImageRecord, self).__init__(data_path, labels_path)
        self._labels_cls = etai.ImageLabels

    def _build_labels(self):
        return

    def _build_data(self, data_path):
        etau.copy_file(self.data_path, data_path)


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
            self._init_from_video_metadata(
                clip_end_frame, duration, total_frame_count)
        else:
            self.clip_end_frame = clip_end_frame
            self.duration = duration
            self.total_frame_count = total_frame_count
        self._labels_cls = etav.VideoLabels

    @classmethod
    def optional(cls):
        '''Returns a list of attributes that are optionally included in the
        data record if they are present in the data dictionary.
        '''
        return super(BuilderDataRecord, cls).required() + [
            "clip_start_frame",
            "clip_end_frame",
            "duration",
            "total_frame_count"
        ]

    def _build_labels(self):
        start_frame, end_frame = (self.clip_start_frame, self.clip_end_frame)
        segment = self._labels_cls()
        labels = self.get_labels()
        self.set_labels(segment)
        if not labels:
            return
        for frame_id in range(start_frame, end_frame + 1):
            frame = labels[frame_id]
            new_frame_number = frame.frame_number - start_frame + 1
            if frame.objects:
                segment.add_objects(frame.objects, new_frame_number)
            if frame.attrs:
                segment.add_frame_attributes(frame.attrs, new_frame_number)

    def _init_from_video_metadata(
            self, clip_end_frame, duration, total_frame_count):
        metadata = etav.VideoMetadata.build_for(self.data_path)
        self.total_frame_count = (
            total_frame_count or metadata.total_frame_count)
        self.duration = duration or metadata.duration
        self.clip_end_frame = clip_end_frame or metadata.total_frame_count

    def _build_data(self, data_path):
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


class BuilderDataset(DataRecords):
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
        ''' Transform a BuilderDataset

        Args:
            src (BuilderDataset)

        Returns:
            None
        '''
        raise NotImplementedError("subclasses must implement transform()")


class Sampler(DatasetTransformer):
    '''Randomly sample the number of records in the dataset to some number k.

    If the number of records is less than k, then all records are kept, but
    the order is randomized.
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
            src (BuilderImageDataset or BuilderVideoDataset)

        Returns:
            None
        '''
        src.records = random.sample(
            src.records, min(self.k, len(src.records))
        )


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

    _NUM_RANDOM_ITER = 10000
    _BUILDER_RECORD_TO_SCHEMA = [
        (BuilderImageRecord, etai.ImageLabelsSchema),
        (BuilderVideoRecord, etav.VideoLabelsSchema)
    ]

    def __init__(
            self, attribute_name=None, object_label=None, labels_schema=None,
            target_quantile=0.25, negative_power=5, target_count=None,
            target_hard_min=False, algorithm="greedy"):
        '''Creates a Balancer instance.

        Args:
            attribute_name (str): the name of the attribute to balance by
            object_label (str): the name of the object label that the
                attribute_name must be nested under. If this is None, it is
                assumed that the attributes are Image/Frame level attrs.
            labels_schema (etai.ImageLabelsSchema or etav.VideoLabelsSchema):
                a schema that indicates which attributes, object labels, etc.
                should be used for balancing. This can be specified as an
                alternative to `attribute_name` and `object_label`. Note that
                labels are not altered; this schema just picks out the
                attributes that are used for balancing.
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
                target) when computing the score for a set of indices to
                remove.
                1 - will weight them the same as positive values
                2 - will square the values
                ...
                Check Balancer._solution_score for more details.
            target_count (int): override target count for each attribute value.
                If this is provided, target_quantile is ignored.
            target_hard_min (bool): whether or not to require that each
                attribute value have at least the target count after balancing
            algorithm (str): name of the balancing search algorithm. Currently
                available are: ["random", "greedy", "simple"]
        '''
        self.attr_name = attribute_name
        self.object_label = object_label
        self.labels_schema = labels_schema
        self.target_quantile = target_quantile
        self.negative_power = negative_power
        self.target_count = target_count
        self.target_hard_min = target_hard_min
        self.algorithm = algorithm

        self._validate()

    def transform(self, src):
        '''Modify the BuilderDataset records by removing records until the
        target attribute is ~roughly~ balanced for each value.

        Args:
            src (BuilderDataset): the dataset builder
        '''
        logger.info("Balancing dataset")

        # STEP 1: Get attribute value(s) for every record
        logger.info("Calculating occurrence matrix...")
        occurrence_matrix, attribute_values, record_idxs = \
            self._get_occurrence_matrix(src.records)
        if not attribute_values:
            return

        # STEP 2: determine target number to remove of each attribute value
        logger.info("Determining target counts...")
        counts = np.sum(occurrence_matrix, axis=1).astype(np.dtype('int'))
        target_count = self._get_target_count(counts)

        # STEP 3: find the records to keep
        logger.info("Calculating which records to keep...")
        keep_idxs = self._get_keep_idxs(
            occurrence_matrix, counts, target_count)

        # STEP 4: modify the list of records
        logger.info("Filtering records...")
        old_records = src.records
        src.clear()
        for ki in keep_idxs:
            src.add(old_records[record_idxs[ki]])

        logger.info("Balancing of dataset complete")

    def _validate(self):
        specified = {
            "attribute_name": self.attr_name is not None,
            "object_label": self.object_label is not None,
            "labels_schema": self.labels_schema is not None
        }

        # The following two patterns of null/non-null arguments are acceptable

        if specified["attribute_name"] and not specified["labels_schema"]:
            return

        if (not specified["attribute_name"] and not specified["object_label"]
            and specified["labels_schema"]):
            return

        # Anything else is unacceptable. Raise a ValueError with the
        # appropriate message.

        if not any(itervalues(specified)):
            raise ValueError("Must specify attribute_name or labels_schema")

        if specified["attribute_name"] and specified["labels_schema"]:
            raise ValueError(
                "Specify only one of attribute_name and labels_schema")

        if not specified["attribute_name"] and specified["object_label"]:
            raise ValueError(
                "Cannot specify object_label without specifying "
                "attribute_name")

        raise AssertionError("Internal logic error")

    def _get_occurrence_matrix(self, records):
        '''Compute occurrence of each attribute value for each class

        Args:
            records: list of BuilderDataRecord's

        Returns:
            A - NxM occurrence matrix, counting the number of instances of each
                attribute value in a record, where:
                    N - length of `values`
                    M - number of records that contain the attribute to balance
            values - list of N strings; one for each unique attribute value
            record_idxs - a list of M integers; each being the index into
                          `records` for the corresponding column in A

            A[i, j] = the number of instances of values[i] in
                      records[record_idxs[j]]
        '''
        helper_list = self._to_helper_list(records)
        record_idxs = [idx for idx, _ in helper_list]

        A = np.zeros((0, len(helper_list)), dtype=np.dtype('uint32'))
        values = []
        for j, (_, attr_values) in enumerate(helper_list):
            for attr_value in attr_values:
                try:
                    i = values.index(attr_value)
                    A[i, j] += 1
                except ValueError:
                    values.append(attr_value)
                    A = np.vstack([
                        A,
                        np.zeros(len(helper_list), dtype=np.dtype('uint32'))
                    ])
                    i = values.index(attr_value)
                    A[i, j] += 1

        return A, values, record_idxs

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

        if self.attr_name is not None:
            return self._to_helper_list_attr_name(records)

        return self._to_helper_list_schema(records)

    def _to_helper_list_attr_name(self, records):
        '''Balancer._to_helper_list when `self.attr_name` is specified.'''
        if isinstance(records[0], BuilderImageRecord):
            if self.object_label:
                return self._to_helper_list_image_objects(records)
            return self._to_helper_list_image(records)

        if isinstance(records[0], BuilderVideoRecord):
            if self.object_label:
                return self._to_helper_list_video_objects(records)
            return self._to_helper_list_video(records)

        raise DatasetTransformerError(
            "Unknown record type: {}".format(
                etau.get_class_name(records[0])
            )
        )

    def _to_helper_list_schema(self, records):
        '''Balancer._to_helper_list when `self.labels_schema` is specified.'''
        self._validate_schema(records)

        if isinstance(records[0], BuilderImageRecord):
            return self._to_helper_list_image_schema(records)

        return self._to_helper_list_video_schema(records)

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

            for frame_no in labels:
                if (frame_no < record.clip_start_frame or
                    frame_no >= record.clip_end_frame):
                    continue

                frame = labels[frame_no]
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

            for frame_no in labels:
                if (frame_no < record.clip_start_frame
                        or frame_no >= record.clip_end_frame):
                    continue

                frame = labels[frame_no]
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

    def _validate_schema(self, records):
        '''Checks that `self.labels_schema` and `records` are compatible.

        Args:
            records: list of BuilderDataRecords to be balanced
        '''
        for build_rec_cls, schema_cls in self._BUILDER_RECORD_TO_SCHEMA:
            if isinstance(records[0], build_rec_cls) and not isinstance(
                    self.labels_schema, schema_cls):
                raise TypeError(
                    "Expected self.labels_schema to be an instance of '%s' "
                    "since builder records are instances of '%s'" % (
                        etau.get_class_name(schema_cls),
                        etau.get_class_name(build_rec_cls)
                    )
                )

            if isinstance(records[0], build_rec_cls):
                break
        else:
            raise DatasetTransformerError(
                "Unknown record type: '%s'" % etau.get_class_name(records[0])
            )

    def _to_helper_list_image_schema(self, records):
        '''Balancer._to_helper_list when an etai.ImageLabelsSchema is given'''
        helper_list = []

        for i, record in enumerate(records):
            labels = record.get_labels()
            helper = (i, [])

            for attr in labels.attrs:
                if self.labels_schema.is_valid_image_attribute(attr):
                    helper[1].append(
                        ("image_attribute", attr.name, attr.value)
                    )

            for detected_object in labels.objects:
                if not self.labels_schema.is_valid_object_label(
                        detected_object.label):
                    continue

                for attr in detected_object.attrs:
                    if self.labels_schema.is_valid_object_attribute(
                            detected_object.label, attr):
                        helper[1].append(
                            ("object_attribute", detected_object.label,
                             attr.name, attr.value)
                        )

            if len(helper[1]):
                helper_list.append(helper)

        return helper_list

    def _to_helper_list_video_schema(self, records):
        '''Balancer._to_helper_list when an etav.VideoLabelsSchema is given'''
        helper_list = []

        for i, record in enumerate(records):
            labels = record.get_labels()
            helper = (i, [])
            helper_dict = defaultdict(set)

            for attr in labels.attrs:
                if self.labels_schema.is_valid_video_attribute(attr):
                    helper[1].append(
                        ("video_attribute", attr.name, attr.value)
                    )

            for frame_no in labels:
                if (frame_no < record.clip_start_frame
                        or frame_no >= record.clip_end_frame):
                    continue

                frame = labels[frame_no]
                for attr in frame.attrs:
                    if self.labels_schema.is_valid_frame_attribute(attr):
                        helper[1].append(
                            ("frame_attribute", attr.name, attr.value)
                        )

                for obj in frame.objects:
                    if not self.labels_schema.is_valid_object_label(
                            obj.label):
                        continue

                    for attr in obj.attrs:
                        if self.labels_schema.is_valid_object_attribute(
                                obj.label, attr):
                            helper_dict[(obj.label, obj.index)].add(
                                (attr.name, attr.value)
                            )

            for (label, _), attr_set in iteritems(helper_dict):
                for name, value in attr_set:
                    helper[1].append(
                        ("object_attribute", label, name, value)
                    )

            if len(helper[1]):
                helper_list.append(helper)

        return helper_list

    def _get_target_count(self, counts):
        '''Compute the target count that we would like to balance each
        value to.

        Args:
            counts (vector): the original counts for each value

        Returns: Integer target value
        '''
        if self.target_count:
            return self.target_count
        return int(np.quantile(counts, self.target_quantile))

    def _get_keep_idxs(self, A, counts, target_count):
        '''Algorithm fun! This function chooses the set of records to keep (and
        remove).

        There's still plenty of potential for testing and improvement here.


        This problem can be posed as:
            minimize:
                ||np.dot(A, x) - b||
            subject to:
                x[i] is an element of [0, 1]

        and different algorithms may be substituted in.

        Args:
            A: the occurrence matrix computed in
                Balancer._get_occurrence_matrix
            counts (vector): the original counts for each value
            target_count (int): the target value

        Returns:
            a list of integer indices to keep
        '''
        b = counts - target_count

        if self.algorithm == "random":
            x = self._random(A, b)
        elif self.algorithm == "greedy":
            x = self._greedy(A, b)
        elif self.algorithm == "simple":
            x = self._simple(A, b)
        else:
            raise ValueError(
                "Unknown balancing algorithm '{}'".format(self.algorithm))

        if self.target_hard_min:
            x = self._add_to_meet_minimum_count(x, A, target_count)

        return np.where(x == 0)[0]

    def _random(self, A, b):
        '''A random search algorithm for finding the indices to omit.

        Args:
            A: the occurrence matrix
            b: the target vector to match

        Returns:
            x: the solution vector. x[j]=1 -> omit the j'th record
        '''
        best_x = np.zeros(A.shape[1], dtype=np.dtype("int"))
        best_score = self._solution_score(b - np.dot(A, best_x))

        random.seed(1)
        for _ in range(self._NUM_RANDOM_ITER):
            i = random.choice(np.where(best_x == 0)[0])
            cur_x = best_x.copy()
            cur_x[i] = 1
            cur_score = self._solution_score(b - np.dot(A, cur_x))
            if cur_score < best_score:
                best_score = cur_score
                best_x = cur_x

        return best_x

    def _greedy(self, A, b):
        '''A greedy search algorithm for finding the indices to omit.

        Args:
            A: the occurrence matrix
            b: the target vector to match

        Returns:
            x: the solution vector. x[j]=1 -> omit the j'th record
        '''
        best_x = np.zeros(A.shape[1], dtype=np.dtype('int'))
        best_score = self._solution_score(b - np.dot(A, best_x))
        w = np.where(best_x == 0)[0]

        while len(w):
            x_matrix = np.zeros((len(best_x), len(w)), dtype=np.dtype('int'))
            for idx, val in enumerate(w):
                x_matrix[:, idx] = best_x
                x_matrix[val, idx] = 1
            Y = np.expand_dims(b, axis=1) - np.dot(A, x_matrix)

            cur_scores = self._solution_score(Y.T)

            i = np.argmin(cur_scores)

            if cur_scores[i] >= best_score:
                break

            best_score = cur_scores[i]
            best_x = x_matrix[:, i]
            w = np.where(best_x == 0)[0]

        return best_x

    def _simple(self, A, b):
        '''This algorithm for finding the indices to omit just goes through
        each class and adds records minimally such that the class has count
        equal to the target.

        Args:
            A: the occurrence matrix
            b: the target vector to match

        Returns:
            x: the solution vector. x[j]=1 -> omit the j'th record
        '''
        x = np.ones(A.shape[1], dtype="int")
        counts = np.dot(A, x)
        dropped_counts = counts.copy()

        # Go through attributes from lowest to highest count and try to
        # minimally add records to get the target for that attribute
        for attr_idx in np.argsort(counts):
            # We can only decrease the dropped counts for this attribute
            # by changing some 1's to 0's in `x`. Therefore, if
            # `dropped_counts[attr_idx]` is already less than or equal
            # to the target, then do nothing.
            attr_target = b[attr_idx]
            if dropped_counts[attr_idx] <= attr_target:
                continue

            # We can change some set of 1's in `x` to 0's, but not vice
            # versa.  Create an array of the counts for this attribute,
            # for records for which `x` contains a 1.
            dropped_counts_for_attr = A[attr_idx, :] * x

            # Right now, `dropped_counts_for_attr.sum()` would give a
            # a number equal to `dropped_counts[attr_idx]`. We want to
            # find which elements of `dropped_counts_for_attr` can be
            # removed so that `dropped_counts_for_attr.sum()` is equal
            # to `attr_target`. Sort `dropped_counts_for_attr` in
            # descending order and drop elements as long as that won't
            # make the sum go below attr_target.
            new_dropped_counts = dropped_counts[attr_idx]
            indices_to_remove = []
            for record_idx in np.flip(np.argsort(dropped_counts_for_attr)):
                if new_dropped_counts <= attr_target:
                    break

                count = dropped_counts_for_attr[record_idx]
                if count == 0 or new_dropped_counts - count < attr_target:
                    # Don't want to explicitly remove records with 0 counts
                    # for this attribute since it won't help us reach the
                    # object of `attr_target` for this attribute, and may
                    # affect other attributes.
                    continue

                # Remove this count
                indices_to_remove.append(record_idx)
                new_dropped_counts -= count

            # Update `x` and `dropped_counts`
            x[indices_to_remove] = 0
            dropped_counts = np.dot(A, x)

        return x

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
        try:
            return np.sum(vector2, axis=1)
        except np.AxisError:
            return np.sum(vector2)

    @staticmethod
    def _add_to_meet_minimum_count(x, A, target_count):
        '''Add more indices to `keep_idxs` so that the count for every
        attribute value is at least equal to `target_count`.

        If for some attribute values, there are fewer than `target_count`
        instances in the whole dataset, every record containing those
        attribute values will be added.

        Args:
            x: an array of shape (M,) containing 1's or 0's, indicating which
                records are being omitted
            A: the NxM occurrence matrix generated by
                `self._get_occurrence_matrix()`
            target_count: an integer giving the desired count for each value

        Returns:
            x_out: an array of shape (M,) containing 1's or 0's, indicating
                records to omit. All entries that were 0 in the input `x`
                will also be 0 in `x_out`, and some entries that were 1 in
                `x` may be 0 in `x_out` as well, such that the total count
                for each value is at least `target_count`.
        '''
        x_out = x.copy()
        current_counts = np.dot(A, 1 - x_out).astype("int")
        omitted_idxs = np.where(x_out == 1)[0]

        total_counts = np.dot(A, np.ones(x_out.shape)).astype("int")
        # If the total count in the entire dataset for a value is less than
        # `target_count`, then the target for that value should be its total
        # count
        target_vec = np.minimum(total_counts, target_count)

        random.shuffle(omitted_idxs)
        for candidate_idx in omitted_idxs:
            if not (current_counts < target_vec).any():
                # All counts meet the minimum
                break

            additional_counts = A[:, candidate_idx]
            # If keeping this index would increase the counts for any
            # value that doesn't meet the minimum, then keep this index
            if (additional_counts[current_counts < target_vec] > 0).any():
                x_out[candidate_idx] = 0
                current_counts += additional_counts

        return x_out


class SchemaFilter(DatasetTransformer):
    '''Filter all labels in the dataset by the provided schema. If the schema
    is None, no filtering is done.
    '''

    def __init__(self, schema):
        '''Initialize the SchemaFilter with a schema.

        Args:
            schema (VideoLabelsSchema or ImageLabelsSchema)
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
                                and less than clip_len
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

    @staticmethod
    def _add_clipping(start_frame, end_frame, old_record, records):
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


class Merger(DatasetTransformer):
    '''Merges another dataset into the existing dataset.'''

    def __init__(self, dataset_builder):
        '''Creates a Merger instance.

        Args:
            dataset_builder: a LabeledDatasetBuilder instance for the
                dataset to be merged with the existing one
        '''
        self._builder_dataset_to_merge = dataset_builder.builder_dataset

    def transform(self, src):
        '''Merges `self._builder_dataset_to_merge` into `src`.

        Args:
            src (BuilderDataset)

        Returns:
            None
        '''
        if self._builder_dataset_to_merge.record_cls != src.record_cls:
            raise DatasetTransformerError(
                "BuilderDatasets have different record_cls types: "
                "src.record_cls = %s, to_merge.record_cls = %s" % (
                    etau.get_class_name(src.record_cls),
                    etau.get_class_name(
                        self._builder_dataset_to_merge.record_cls)
                )
            )

        src.add_container(self._builder_dataset_to_merge)


class FilterByFilename(DatasetTransformer):
    '''Filters data from a dataset using a filename blacklist.'''

    def __init__(self, filename_blacklist):
        '''Creates a FilterByFilename instance.

        Args:
            filename_blacklist: a list of data filenames to filter out
        '''
        self._files_to_remove = set(filename_blacklist)

    def transform(self, src):
        '''Removes data with filenames that match the blacklist.

        Args:
            src (BuilderDataset)

        Returns:
            None
        '''
        src.cull_with_function(
            "data_path",
            lambda path: os.path.basename(
                path) not in self._files_to_remove
        )


class FilterByPath(DatasetTransformer):
    '''Filters data from a dataset using a full path blacklist.'''

    def __init__(self, full_path_blacklist):
        '''Creates a FilterByPath instance.

        Args:
            full_path_blacklist: a list of full paths to data files
                to filter out
        '''
        self._paths_to_remove = {
            os.path.abspath(path) for path in full_path_blacklist}

    def transform(self, src):
        '''Removes data with full paths that match the blacklist.

        Args:
            src (BuilderDataset)

        Returns:
            None
        '''
        src.cull_with_function(
            "data_path",
            lambda path: os.path.abspath(
                path) not in self._paths_to_remove
        )


class DatasetTransformerError(Exception):
    '''Exception raised when there is an error in a DatasetTransformer'''
    pass
