'''
Core interfaces, data structures, and methods for working with datasets

Copyright 2017-2019 Voxel51, Inc.
voxel51.com

Matthew Lightman, matthew@voxel51.com
Jason Corso, jason@voxel51.com
Ben Kane, ben@voxel51.com
Tyler Ganter, tyler@voxel51.com
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
import glob
import logging
import os
import random

import numpy as np

import eta.core.annotations as etaa
from eta.core.data import BaseDataRecord
import eta.core.image as etai
from eta.core.serial import Serializable
import eta.core.utils as etau
import eta.core.video as etav

from .split_methods import SPLIT_FUNCTIONS
from .utils import COPY, FILE_METHODS, _FILE_METHODS_MAP


logger = logging.getLogger(__name__)


# Functions involving LabeledDatasets


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
    '''

    _DATA_SUBDIR = "data"
    _LABELS_SUBDIR = "labels"

    def __init__(self, manifest_path):
        '''Creates a LabeledDataset instance.

        Args:
            manifest_path: the path to the `manifest.json` file for the dataset

        Raises:
            LabeledDatasetError: if the class reading the dataset is not a
                subclass of the dataset class recorded in the manifest
        '''
        self._dataset_index = LabeledDatasetIndex.from_json(manifest_path)
        if not isinstance(self, etau.get_class(self.dataset_index.type)):
            raise LabeledDatasetError(
                "Tried to read dataset of type '%s', from location '%s', "
                "but manifest is of type '%s'" % (
                    etau.get_class_name(self), manifest_path,
                    self.dataset_index.type))
        self._manifest_path = manifest_path

        self._build_index_map()

    @property
    def dataset_dir(self):
        '''The top level directory for the dataset, which would contain
        manifest.json files.
        '''
        return os.path.dirname(self._manifest_path)

    @property
    def data_dir(self):
        '''Deprecated! Use `dataset_dir` instead.'''
        class_name = etau.get_class_name(self)
        logger.warning("%s.data_dir is deprecated. Use %s.dataset_dir"
                       " instead.", class_name, class_name)
        return self.dataset_dir

    @property
    def dataset_index(self):
        '''A `LabeledDatasetIndex` object containing the paths of data and
        labels files in the dataset.
        '''
        return self._dataset_index

    @dataset_index.setter
    def dataset_index(self, dataset_index):
        '''Sets the `LabeledDatasetIndex` of the `LabeledDataset`

        Args:
            dataset_index: a `LabeledDatasetIndex`
        '''
        if not isinstance(dataset_index, LabeledDatasetIndex):
            raise ValueError(
                "expected type %s but got type %s" %
                (type(LabeledDatasetIndex), type(dataset_index)))

        self._dataset_index = dataset_index

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

    def __getitem__(self, key):
        '''Returns a LabeledDataRecord from `self.dataset_index` with the
        given key.

        Args:
            key: an integer index into `self.dataset_index`
        '''
        return self.dataset_index[key]

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
            yield os.path.join(self.dataset_dir, record.data)

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
            yield os.path.join(self.dataset_dir, record.labels)

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
                self.dataset_dir
            description: optional description for the new manifest. If not
                specified, the existing description is retained.
        '''
        if description is not None:
            self.set_description(description)

        out_path = os.path.join(self.dataset_dir, filename)
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

    def has_data_with_name(self, data_path):
        '''Checks whether a data file already exists in the dataset with the
        provided filename.

        Args:
            data_path: path to or filename of a data file

        Returns:
            (bool): True if the filename of `data_path` is the same as a
                data file already present in the dataset
        '''
        data_file = os.path.basename(data_path)
        return data_file in self._data_to_labels_map

    @staticmethod
    def _parse_file_methods(file_method):
        if isinstance(file_method, tuple) and len(file_method) == 2:
            data_method, labels_method = file_method
        else:
            data_method = labels_method = file_method

        if (data_method not in FILE_METHODS
                or labels_method not in FILE_METHODS):
            raise ValueError("invalid file_method: %s" % str(file_method))

        return _FILE_METHODS_MAP[data_method], _FILE_METHODS_MAP[labels_method]

    def add_file(self, data_path, labels_path, new_data_filename=None,
                 new_labels_filename=None, file_method=COPY,
                 error_on_duplicates=False):
        '''Adds a single data file and its labels file to this dataset.

        Args:
            data_path: path to data file to be added
            labels_path: path to corresponding labels file to be added
            new_data_filename: optional filename for the data file to be
                renamed to
            new_labels_filename: optional filename for the labels file to be
                renamed to
            file_method: how to add the files to the dataset. One of "copy",
                "link", "move", or "symlink". A tuple, e.g. `("move", "copy")`,
                may be used as well to move data files and copy labels files,
                for example. The default is "copy"
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
                `error_on_duplicates` is True, or if `file_method` is not valid
        '''
        if new_data_filename is None:
            new_data_filename = os.path.basename(data_path)

        if new_labels_filename is None:
            new_labels_filename = os.path.basename(labels_path)

        if error_on_duplicates and self.has_data_with_name(new_data_filename):
            raise ValueError("Data file '%s' already present in dataset"
                             % new_data_filename)

        data_method, labels_method = self._parse_file_methods(file_method)

        new_data_path = os.path.join(
            self.dataset_dir, self._DATA_SUBDIR, new_data_filename)
        if data_path != new_data_path:
            data_method(data_path, new_data_path)

        new_labels_path = os.path.join(
            self.dataset_dir, self._LABELS_SUBDIR, new_labels_filename)
        if labels_path != new_labels_path:
            labels_method(labels_path, new_labels_path)

        # Update the filename attribute in the labels JSON if necessary
        if new_data_filename != os.path.basename(data_path):
            labels_ = self._read_labels(new_labels_path)
            labels_.filename = new_data_filename
            self._write_labels(labels_, new_labels_path)

        # First remove any other records with the same data filename
        self.dataset_index.cull_with_function(
            lambda record: os.path.basename(record.data) != new_data_filename)
        self.dataset_index.append(
            LabeledDataRecord(
                os.path.join(self._DATA_SUBDIR, new_data_filename),
                os.path.join(self._LABELS_SUBDIR, new_labels_filename)
            )
        )

        self._data_to_labels_map[new_data_filename] = new_labels_filename

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
        if error_on_duplicates and self.has_data_with_name(data_filename):
            raise ValueError("Data file '%s' already present in dataset"
                             % os.path.basename(data_filename))

        data_path = os.path.join(
            self.dataset_dir, self._DATA_SUBDIR, data_filename)
        labels_path = os.path.join(
            self.dataset_dir, self._LABELS_SUBDIR, labels_filename)

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

    def copy(self, dataset_path, file_method=COPY):
        '''Copies the dataset to another directory.

        If the dataset index has been manipulated, this will be reflected
        in the copy.

        Args:
            dataset_path: the path to the `manifest.json` file for the
                copy of the dataset that will be written. The containing
                directory must either not exist or be empty.
            file_method: how to add the files to the dataset. One of "copy",
                "link", "move", or "symlink". A tuple, e.g. `("move", "copy")`,
                may be used as well to move data files and copy labels files,
                for example. The default is "copy"

        Returns:
            dataset_copy: `LabeledDataset` instance that points to the new
                containing directory
        '''
        self._ensure_empty_dataset_dir(dataset_path)

        new_data_dir = os.path.dirname(dataset_path)

        new_data_subdir = os.path.join(new_data_dir, self._DATA_SUBDIR)

        new_labels_subdir = os.path.join(new_data_dir, self._LABELS_SUBDIR)

        data_method, labels_method = self._parse_file_methods(file_method)

        for data_path, labels_path in zip(
                self.iter_data_paths(), self.iter_labels_paths()):
            new_data_file = os.path.basename(data_path)
            new_data_path = os.path.join(new_data_subdir, new_data_file)
            data_method(data_path, new_data_path)

            new_labels_file = os.path.basename(labels_path)
            new_labels_path = os.path.join(new_labels_subdir, new_labels_file)
            labels_method(labels_path, new_labels_path)

        self.dataset_index.write_json(dataset_path)

        class_name = etau.get_class_name(self)
        cls = etau.get_class(class_name)
        return cls(dataset_path)

    def merge(self, labeled_dataset_or_path, merged_dataset_path,
              in_place=False, description=None, file_method=COPY):
        '''Union of two labeled datasets.

        Args:
            labeled_dataset_or_path: an `LabeledDataset` instance or path
                to a `manifest.json`, that is of the same type as `self`
            merged_dataset_path: path to `manifest.json` for the merged
                dataset. If `in_place` is False, the containing directory
                must either not exist or be empty. If `in_place` is True,
                either the containing directory must be equal to
                `self.dataset_dir`, or `merged_dataset_path` is just a filename
                of a new `manifest.json` to write in `self.dataset_dir`.
            in_place: whether or not to write the merged dataset to a new
                directory. If not, the data from `labeled_dataset_or_path`
                will be added into `self.dataset_dir`.
            description: optional description for the manifest of the merged
                dataset. If not specified, the existing description is used.
            file_method: how to add the files to the dataset. One of "copy",
                "link", "move", or "symlink". A tuple, e.g. `("move", "copy")`,
                may be used as well to move data files and copy labels files,
                for example. The default is "copy"

        Returns:
            merged_dataset: a `LabeledDataset` instance pointing to the
                merged dataset. If `in_place` is True, this will just be
                `self`.
        '''
        labeled_dataset = self._parse_dataset(labeled_dataset_or_path)

        data_filenames_to_merge = self._get_filenames_for_merge(
            labeled_dataset)

        output_dataset_dir = os.path.dirname(merged_dataset_path)
        if not output_dataset_dir:
            output_dataset_dir = self.dataset_dir
            merged_dataset_path = os.path.join(
                output_dataset_dir, merged_dataset_path)

        if in_place and output_dataset_dir != self.dataset_dir:
            raise ValueError(
                "If merging datasets in place, merged_dataset_path should be "
                "within original base directory '%s', but got '%s'" %
                (self.dataset_dir, output_dataset_dir))

        if in_place:
            merged_dataset = self
        else:
            merged_dataset = self.copy(merged_dataset_path)

        # Copy files one-by-one from `labeled_dataset`
        for data_path, labels_path in zip(
                labeled_dataset.iter_data_paths(),
                labeled_dataset.iter_labels_paths()):
            if os.path.basename(data_path) in data_filenames_to_merge:
                merged_dataset.add_file(
                    data_path, labels_path, file_method=file_method)

        if description is not None:
            merged_dataset.set_description(description)

        merged_dataset.write_manifest(
            os.path.basename(merged_dataset_path))
        return merged_dataset

    def deduplicate(self):
        '''Removes duplicate data files from the index.

        If sets of files are found with the same content, one file in each
        is set is chosen arbitrarily to be kept, and the rest are removed.
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
                self.dataset_dir, record.data) not in data_paths_remove)

        self._build_index_map()

        return self

    def prune(self):
        '''Deletes data and labels files that are not in the index.

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

        data_subdir = os.path.join(self.dataset_dir, self._DATA_SUBDIR)
        for filename in etau.list_files(data_subdir):
            if filename not in data_filenames:
                etau.delete_file(os.path.join(data_subdir, filename))

        labels_subdir = os.path.join(self.dataset_dir, self._LABELS_SUBDIR)
        for filename in etau.list_files(labels_subdir):
            if filename not in labels_filenames:
                etau.delete_file(os.path.join(labels_subdir, filename))

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
            a LabeledDatasetBuilder
        '''
        builder = self.builder_cls()
        for paths in self.iter_paths():
            builder.add_record(builder.record_cls(*paths))
        return builder

    def get_labels_set(self, ensure_filenames=True):
        '''Creates a SetLabels type object from this dataset's labels.

        Args:
            ensure_filenames: whether to add the filename into the individual
                labels if it is not present

        Returns:
            an ImageSetLabels or VideoSetLabels
        '''
        raise NotImplementedError(
            "subclasses must implement get_labels_set()")

    def get_active_schema(self):
        '''Returns a LabelsSchema-type object describing the active schema of
        the dataset.

        Returns:
            an ImageLabelsSchema or VideoLabelsSchema
        '''
        raise NotImplementedError(
            "subclasses must implement get_active_schema()")

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

    def add_metadata_to_labels(self, overwrite=False):
        '''Adds metadata about each data file into its labels file.

        Args:
            overwrite: whether to overwrite metadata already present in the
                labels file
        '''
        for data_path, labels, labels_path in zip(
                self.iter_data_paths(), self.iter_labels(),
                self.iter_labels_paths()):
            if not hasattr(labels, "metadata"):
                raise TypeError(
                    "'%s' has no 'metadata' attribute" % etau.get_class_name(
                        labels)
                )

            if overwrite or labels.metadata is None:
                labels.metadata = self._build_metadata(data_path)
                labels.write_json(labels_path)

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

    def _read_data(self, path):
        '''Reads data from a data file at the given path.

        Subclasses must implement this based on the particular data format for
        the subclass.

        Args:
            path: path to a data file in the dataset

        Returns:
            a data object in the particular format for the subclass
        '''
        raise NotImplementedError("subclasses must implement _read_data()")

    def _read_labels(self, path):
        '''Reads a labels object from a labels JSON file at the given path.

        Subclasses must implement this based on the particular labels format
        for the subclass.

        Args:
            path: path to a labels file in the dataset

        Returns:
            a labels object in the particular format for the subclass
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

    def _build_metadata(self, path):
        '''Reads metadata from a data file at the given path and builds an
        instance of the metadata class associated with the data type.

        Subclasses must implement this based on the particular metadata format
        for the subclass.

        Args:
            path: path to a data file in the dataset

        Returns:
            an instance of the metadata class associated with the data format
                for the subclass
        '''
        raise NotImplementedError(
            "subclasses must implement _build_metadata()")

    def _parse_dataset(self, labeled_dataset_or_path):
        cls_name = etau.get_class_name(self)
        cls = etau.get_class(cls_name)
        if isinstance(labeled_dataset_or_path, six.string_types):
            labeled_dataset = cls(labeled_dataset_or_path)
        else:
            labeled_dataset = labeled_dataset_or_path

        if not isinstance(labeled_dataset, cls):
            raise TypeError(
                "'%s' is not an instance of '%s'" %
                (etau.get_class_name(labeled_dataset), cls_name))

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
        '''Creates a VideoSetLabels containing this dataset's labels.

        Args:
            ensure_filenames: whether to add the filename into the individual
                labels if it is not present

        Returns:
            a VideoSetLabels instance
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

    def get_active_schema(self):
        '''Returns a VideoLabelsSchema describing the active schema of the
        dataset.

        Returns:
            a VideoLabelsSchema
        '''
        schema = etav.VideoLabelsSchema()
        for video_labels in self.iter_labels():
            schema.merge_schema(
                etav.VideoLabelsSchema.build_active_schema(video_labels))

        return schema

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

    def compute_average_video_duration(self):
        '''Computes the average duration over all videos in the dataset.

        Returns:
             the average duration in seconds
        '''
        video_durations = [etav.VideoMetadata.build_for(data_path).duration
                           for data_path in self.iter_data_paths()]

        return np.mean(video_durations)

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

    def _build_metadata(self, path):
        return etav.VideoMetadata.build_for(path)


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
        '''Creates an ImageSetLabels instance containing this dataset's labels.

        Args:
            ensure_filenames: whether to add the filename into the individual
                labels if it is not present

        Returns:
            an ImageSetLabels instance
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

    def get_active_schema(self):
        '''Returns the ImageLabelsSchema describing the active schema of the
        dataset.

        Returns:
            an ImageLabelsSchema
        '''
        schema = etai.ImageLabelsSchema()
        for image_labels in self.iter_labels():
            schema.merge_schema(
                etai.ImageLabelsSchema.build_active_schema(image_labels))

        return schema

    def write_annotated_data(self, output_dir_path, annotation_config=None):
        '''Annotates the data with its labels, and outputs the
        annotated data to the specified directory.

        Args:
            output_dir_path: the path to the directory into which
                the annotated data will be written
            annotation_config: an optional etaa.AnnotationConfig specifying
                how to render the annotations. If omitted, the default config
                is used
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

    def _build_metadata(self, path):
        return etai.ImageMetadata.build_for(path)


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
                created
            description: an optional description of the dataset
        '''
        self.type = type
        self.index = index or []
        self.description = description or ""

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return self.index[key]

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
        '''Randomly shuffles the index.'''
        random.shuffle(self.index)

    def split(self, split_fractions=None, descriptions=None,
              split_method="random_exact"):
        '''Splits the `LabeledDatasetIndex` into multiple `LabeledDatasetIndex`
        instances, containing disjoint subsets of the original index.

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
    def from_dict(cls, d, *args, **kwargs):
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
