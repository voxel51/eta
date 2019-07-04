'''
Core interfaces, data structures, and methods for working with datasets.

Copyright 2017-2019 Voxel51, Inc.
voxel51.com

Matthew Lightman, matthew@voxel51.com
Brian Moore, brian@voxel51.com
Jason Corso, jason@voxel51.com
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

import itertools
import logging
import os
import random

import numpy as np

from eta.core.data import BaseDataRecord
import eta.core.data as etad
import eta.core.image as etai
from eta.core.serial import Serializable
import eta.core.utils as etau
import eta.core.video as etav


logger = logging.getLogger(__name__)


def odds_and_evens(data_records):
    '''Splits a DataRecords into two DataRecords, one from odd indexed records
    and the other from even indexed records.

    Args:
        data_records: a DataRecords instance

    Returns:
        a list of two DataRecords
    '''
    record_cls = data_records.record_cls
    records_list = [etad.DataRecords(record_cls), etad.DataRecords(record_cls)]
    for idx, record in enumerate(data_records):
        records_list[idx % 2].add(record)

    return records_list


def random_split_exact(data_records, split_fractions=None):
    '''Randomly splits a DataRecords into multiple DataRecords according to the
    given split fractions.

    The number of records in each sample will be given exactly by the specified
    fractions.

    Args:
        data_records: a DataRecords instance
        split_fractions: an optional list of split fractions, which should sum
            to 1. By default, [0.5, 0.5] is used

    Returns:
        a list of DataRecords of same length as `split_fractions`
    '''
    if split_fractions is None:
        split_fractions = [0.5, 0.5]

    shuffled = list(data_records)
    random.shuffle(shuffled)

    record_cls = data_records.record_cls

    sample_lists = _split_in_order(shuffled, split_fractions)

    return [
        etad.DataRecords(record_cls, records=samp_list)
        for samp_list in sample_lists]


def random_split_approx(data_records, split_fractions=None):
    '''Randomly splits a DataRecords into multiple DataRecords according to the
    given split fractions.

    Each record is assigned to a sample with probability equal to the
    corresponding split fraction.

    Args:
        data_records: a DataRecords instance
        split_fractions: an optional list of split fractions, which should sum
            to 1. By default, [0.5, 0.5] is used

    Returns:
        a list of DataRecords of same length as `split_fractions`
    '''
    if split_fractions is None:
        split_fractions = [0.5, 0.5]

    record_cls = data_records.record_cls
    records_list = [etad.DataRecords(record_cls) for _ in split_fractions]

    cum_frac = np.cumsum(split_fractions)
    for record in data_records:
        idx = np.searchsorted(cum_frac, random.random())
        if idx < len(records_list):
            records_list[idx].add(record)

    return records_list


def split_in_order(data_records, split_fractions=None):
    '''Splits a DataRecords into multiple DataRecords according to the
    given split fractions.

    The records are partitioned into samples in order according to their
    position in the input sample. This is not recommended unless your records
    are already randomly ordered.

    Args:
        data_records: a DataRecords instance
        split_fractions: an optional list of split fractions, which should sum
            to 1. By default, [0.5, 0.5] is used

    Returns:
        a list of DataRecords of same length as `split_fractions`
    '''
    if split_fractions is None:
        split_fractions = [0.5, 0.5]

    records_lists = _split_in_order(list(data_records), split_fractions)

    record_cls = data_records.record_cls
    return [
        etad.DataRecords(record_cls, records=records)
        for records in records_lists]


def _split_in_order(records_list, split_fractions):
    n = len(records_list)
    cum_frac = np.cumsum(split_fractions)
    cum_size = [int(np.round(frac * n)) for frac in cum_frac]
    sample_bounds = [0] + cum_size

    records_list = []
    for begin, end in zip(sample_bounds, sample_bounds[1:]):
        records_list.append(records_list[begin:end])

    return records_list


class LabeledDataset(object):
    '''Base class for labeled datasets.'''

    def __init__(self, dataset_path):
        '''Creates a LabeledDataset instance.

        Args:
            dataset_path: the path to the `manifest.json` file for the dataset
        '''
        self.dataset_index = LabeledDatasetIndex.from_json(dataset_path)
        if not isinstance(self, etau.get_class(self.dataset_index.type)):
            raise LabeledDatasetError(
                "Tried to read dataset of type '%s', but manifest is of "
                "type '%s'" % (
                    etau.get_class_name(self), self.dataset_index.type))
        self.data_dir = os.path.dirname(dataset_path)

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
        '''Iterates over the paths lables files in the dataset.

        Returns:
            iterator: iterator over paths to labels files
        '''
        for record in self.dataset_index:
            yield os.path.join(self.data_dir, record.labels)

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

        return self

    def shuffle(self):
        '''Randomly shuffles the order of the data.

        Returns:
            self
        '''
        self.dataset_index.shuffle()

        return self

    def add_file(self, data_path, labels_path):
        '''Adds a single data file and its labels file to this dataset.

        Args:
            data_path: path to data file to be added
            labels_path: path to corresponding labels file to be added

        Returns:
            self
        '''
        data_subdir = os.path.join(self.data_dir, "data")
        labels_subdir = os.path.join(self.data_dir, "labels")
        if os.path.dirname(data_path) != data_subdir:
            etau.copy_file(data_path, data_subdir)
        if os.path.dirname(labels_path) != labels_subdir:
            etau.copy_file(labels_path, labels_subdir)
        self.dataset_index.append(
            LabeledDataRecord(
                os.path.join("data", os.path.basename(data_path)),
                os.path.join("labels", os.path.basename(labels_path))
            )
        )

        return self

    def add_data(self, data, labels, data_filename, labels_filename):
        '''Creates and adds a single data file and its labels file to this
        dataset, using the input python data structure.

        Args:
            data: input data in a format that can be passed to
                self._write_data()
            labels: input labels in a format that can be passed to
                self._write_labels()
            data_filename: filename for the data in the dataset
            labels_filename: filename for the labels in the dataset

        Returns:
            self
        '''
        data_path = os.path.join(self.data_dir, "data", data_filename)
        labels_path = os.path.join(self.data_dir, "labels", labels_filename)
        self._write_data(data, data_path)
        self._write_labels(labels, labels_path)

        self.dataset_index.append(
            LabeledDataRecord(
                os.path.join("data", data_filename),
                os.path.join("labels", labels_filename)
            )
        )

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
        new_data_subdir = os.path.join(new_data_dir, "data")
        new_labels_subdir = os.path.join(new_data_dir, "labels")

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

    @staticmethod
    def is_valid_dataset(dataset_path):
        '''Determines whether the data at the given path is a valid
        LabeledDataset.

        Args:
            dataset_path: the path to the `manifest.json` file for the dataset

        Returns:
            True/False
        '''
        raise NotImplementedError(
            "subclasses must implement is_valid_dataset()")

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

    @staticmethod
    def _ensure_empty_dataset_dir(dataset_path):
        etau.ensure_basedir(dataset_path)
        data_dir = os.path.dirname(dataset_path)

        existing_files = os.listdir(data_dir)
        if existing_files:
            raise ValueError(
                "Cannot create a new dataset in a non-empty directory. "
                "Found the following files in directory '%s': %s" %
                (data_dir, existing_files))

        data_subdir = os.path.join(data_dir, "data")
        labels_subdir = os.path.join(data_dir, "labels")
        etau.ensure_dir(data_subdir)
        etau.ensure_dir(labels_subdir)


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

    def to_image_set(self, image_dataset_path, stride=1,
                     image_extension=".jpg", description=None):
        '''Writes the data to a `LabeledImageDataset` by extracting frames
        and their corresponding labels.

        @todo add logging between reading of full individual videos

        Args:
            image_dataset_path: the path to the `manifest.json` file for
                the image dataset that will be written. The containing
                directory must either not exist or be empty
            stride: optional frequency with which to sample frames from
                videos
            image_extension: optional extension for image files in new
                dataset (defaults to ".jpg")
            description: optional description for the manifest of the
                new image dataset

        Returns:
            image_dataset: `LabeledImageDataset` instance that points to
                the new image dataset
        '''
        image_dataset = LabeledImageDataset.create_empty_dataset(
            image_dataset_path, description=description)

        for video_reader, video_path, video_labels in zip(
                self.iter_data(), self.iter_data_paths(),
                self.iter_labels()):
            video_filename = os.path.basename(video_path)
            video_name = os.path.splitext(video_filename)[0]
            with video_reader:
                for frame_img in itertools.islice(
                        video_reader, 0, None, stride):
                    frame_num = video_reader.frame_number
                    base_filename = "%s-%d" % (video_name, frame_num)
                    image_filename = "%s%s" % (base_filename, image_extension)
                    labels_filename = "%s.json" % base_filename

                    frame_labels = video_labels[frame_num]
                    image_labels = etai.ImageLabels(
                        filename=image_filename,
                        attrs=frame_labels.attrs,
                        objects=frame_labels.objects)
                    image_dataset.add_data(
                        frame_img, image_labels, image_filename,
                        labels_filename)

        image_dataset.write_manifest(image_dataset_path)

        return image_dataset

    @staticmethod
    def is_valid_dataset(dataset_path):
        '''Determines whether the data at the given path is a valid
        LabeledVideoDataset.

        This function checks whether each video and labels path exists and has
        a valid extension, but makes no attempt to read the files.

        Args:
            dataset_path: the path to the `manifest.json` file for the dataset

        Returns:
            True/False
        '''
        try:
            video_dataset = LabeledVideoDataset(dataset_path)
        except LabeledDatasetError as e:
            logger.info(e)
            return False

        for video_path in video_dataset.iter_data_paths():
            if not etav.is_supported_video_file(video_path):
                logger.info("Unsupported video format: %s", video_path)
                return False
            if not os.path.isfile(video_path):
                logger.info("File not found: %s", video_path)
                return False

        for labels_path in video_dataset.iter_labels_paths():
            if not os.path.splitext(labels_path)[1] == ".json":
                logger.info("Unsupported labels format: %s", labels_path)
                return False
            if not os.path.isfile(labels_path):
                logger.info("File not found: %s", labels_path)
                return False

        return True

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

    @staticmethod
    def is_valid_dataset(dataset_path):
        '''Determines whether the data at the given path is a valid
        LabeledImageDataset.

        This function checks whether each image and labels path exists and has
        a valid extension, but makes no attempt to read the files.

        Args:
            dataset_path: the path to the `manifest.json` file for the dataset

        Returns:
            True/False
        '''
        try:
            image_dataset = LabeledImageDataset(dataset_path)
        except LabeledDatasetError as e:
            logger.info(e)
            return False

        for img_path in image_dataset.iter_data_paths():
            if not etai.is_supported_image(img_path):
                logger.info("Unsupported image format: %s", img_path)
                return False
            if not os.path.isfile(img_path):
                logger.info("File not found: %s", img_path)
                return False

        for labels_path in image_dataset.iter_labels_paths():
            if not os.path.splitext(labels_path)[1] == ".json":
                logger.info("Unsupported labels format: %s", labels_path)
                return False
            if not os.path.isfile(labels_path):
                logger.info("File not found: %s", labels_path)
                return False

        return True

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