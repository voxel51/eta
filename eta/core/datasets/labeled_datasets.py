"""
Core interfaces, data structures, and methods for working with datasets.

Copyright 2017-2020 Voxel51, Inc.
voxel51.com

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

import copy
import logging
import os
import random

import eta.core.annotations as etaa
from eta.core.data import BaseDataRecord
import eta.core.image as etai
from eta.core.serial import Serializable
import eta.core.utils as etau
import eta.core.video as etav

from .split_methods import SplitMethods
from .utils import FileMethods


logger = logging.getLogger(__name__)


def get_manifest_path_from_dir(dataset_dir):
    """Infers the filename of the manifest within the given LabeledDataset
    directory, and returns the path to that file.

    Args:
        dataset_dir: path to a LabeledDataset directory

    Returns:
        path to the manifest inside the directory `dirpath`
    """
    candidate_manifests = {
        os.path.basename(path)
        for path in etau.get_glob_matches(os.path.join(dataset_dir, "*.json"))
    }

    if not candidate_manifests:
        raise ValueError(
            "Directory '%s' contains no JSON files to use as a manifest"
            % dataset_dir
        )

    if "manifest.json" in candidate_manifests:
        return os.path.join(dataset_dir, "manifest.json")

    manifest_path = os.path.join(dataset_dir, candidate_manifests.pop())

    if len(candidate_manifests) > 1:
        logger.warning(
            "Multiple JSON files found in '%s'; using manifest '%s'",
            dataset_dir,
            manifest_path,
        )

    return manifest_path


def load_dataset(manifest_path_or_dataset_dir):
    """Loads the LabeledDataset at the given location, which can either be the
    path to the manifest JSON or the parent directory.

    If a directory is provided, the manifest JSON file is located by calling
    `get_manifest_path_from_dir()`.

    This function will return the appropriate LabeledDataset subclass specified
    by the manifest JSON.

    Args:
        manifest_path_or_dataset_dir: the path to a manifest JSON file or a
            dataset directory containing one

    Returns:
        a LabeledDataset
    """
    if os.path.isdir(manifest_path_or_dataset_dir):
        manifest_path = get_manifest_path_from_dir(
            manifest_path_or_dataset_dir
        )
    else:
        manifest_path = manifest_path_or_dataset_dir

    try:
        index = LabeledDatasetIndex.from_json(manifest_path)
    except:
        logger.warning(
            "Failed to load LabeledDatasetIndex from '%s' with the following "
            "error:",
            manifest_path,
        )
        raise

    dataset_cls = etau.get_class(index.type)
    return dataset_cls(manifest_path, manifest=index)


class LabeledDatasetIndex(Serializable):
    """Class that encapsulates the index of a LabeledDataset.

    The index is stored on disk in the following format::

        {
            "type": "eta.core.datasets.LabeledDataset",
            "description": "",
            "index": [
                {
                    "data": "data/video1.mp4",
                    "labels": "labels/video1.json"
                },
                ...
            ]
        }

    Attributes:
        type: the fully-qualified name of the LabeledDataset subclass that
            encapsulates the dataset
        index: a list of `LabeledDataRecord`s
        description: an optional description of the dataset
    """

    def __init__(self, type_or_cls, index=None, description=None):
        """Creates a LabeledDatasetIndex.

        Args:
            type: the LabeledDataset subclass or its fully-qualified class name
            index: a list of `LabeledDataRecord`s. By default and empty list is
                created
            description: an optional description of the dataset
        """
        if etau.is_str(type_or_cls):
            type_ = type_or_cls
        else:
            type_ = etau.get_class_name(type_or_cls)

        self.type = type_
        self.description = description or ""
        self.index = index or []

    def __getitem__(self, key):
        return self.index[key]

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        return iter(self.index)

    def append(self, record):
        """Appends an entry to the index.

        Args:
            record: a LabeledDataRecord
        """
        self.index.append(record)

    def cull_with_function(self, func):
        """Removes `LabeledDataRecord`s from the index using the provided
        function.

        Args:
            func: a function that takes in a LabeledDataRecord and returns
                True/False. Only records for which the function evaluates to
                True will be retained in the index
        """
        self.index = [record for record in self.index if func(record)]

    def sample(self, k):
        """Randomly downsamples the index to k elements.

        Args:
            k: the number of entries in the index after sampling
        """
        self.index = random.sample(self.index, k)

    def shuffle(self):
        """Randomly shuffles the index."""
        random.shuffle(self.index)

    def split(
        self,
        split_fractions=None,
        descriptions=None,
        split_method=SplitMethods.RANDOM_EXACT,
    ):
        """Splits the LabeledDatasetIndex into multiple LabeledDatasetIndex
        instances, containing disjoint subsets of the original index.

        Args:
            split_fractions: an optional list of split fractions, which should
                sum to 1, that specifies how to split the index. By default,
                [0.5, 0.5] is used
            descriptions: an optional list of descriptions for the output
                indices. The list should be the same length as
                `split_fractions`. If not specified, the description of the
                original index is used for all of the output indices
            split_method: an `eta.core.datasets.SplitMethods` value specifying
                the method to use to split the index. The default value is
                `SplitMethods.RANDOM_EXACT`

        Returns:
            a list of LabeledDatasetIndex instances of the same length as
                `split_fractions`
        """
        if split_fractions is None:
            split_fractions = [0.5, 0.5]

        if descriptions is None:
            descriptions = [self.description for _ in split_fractions]

        if len(descriptions) != len(split_fractions):
            raise ValueError(
                "split_fractions and descriptions lists should be the "
                "same length, but got len(split_fractions) = %d, "
                "len(descriptions) = %d"
                % (len(split_fractions), len(descriptions))
            )

        split_fcn = SplitMethods.get_function(split_method)
        split_indices = split_fcn(self.index, split_fractions)

        return [
            LabeledDatasetIndex(self.type, split_index, description)
            for split_index, description in zip(split_indices, descriptions)
        ]

    def attributes(self):
        """Returns a list of class attributes to be serialized.

        Returns:
            a list of attributes
        """
        return ["type", "description", "index"]

    @classmethod
    def from_dict(cls, d):
        """Constructs a LabeledDatasetIndex from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a LabeledDatasetIndex
        """
        index = d.get("index", None)
        if index is not None:
            index = [LabeledDataRecord.from_dict(r) for r in index]

        description = d.get("description", None)
        return cls(d["type"], index=index, description=description)


class LabeledDataRecord(BaseDataRecord):
    """A record containing a data file and an associated labels file.

    Attributes:
        data: the path to the data file
        labels: the path to the labels file
    """

    def __init__(self, data, labels):
        """Creates a LabeledDataRecord instance.

        Args:
            data: the path to the data file
            labels: the path to the labels file
        """
        self.data = data
        self.labels = labels
        super(LabeledDataRecord, self).__init__()

    @classmethod
    def required(cls):
        return ["data", "labels"]


class LabeledDataset(object):
    """Base class for labeled datasets, which encapsulate raw data samples and
    their associated labels.

    Labeled datasets are stored on disk in the following format::

        /path/to/dataset/
            manifest.json
            data/
                <data1>.<ext>
                ...
            labels/
                <labels1>.<ext>
                ...

    Here, `manifest.json` is a LabeledDatasetIndex that specifies the contents
    of the dataset. Note that this file may have a different name, if desired,
    as labeled datasets are referenced in code by their `manifest_path`, which
    points to their serialized LabeledDatasetIndex on disk (whatever the name).
    This flexibility allows you to, for example, maintain multiple manifests in
    a dataset that provide different views into the dataset.

    `LabeledDataset`s maintain the following class invariants:
        - `dataset_index` contains paths to data and labels that exist on disk
        - each data file appears only once in `dataset_index`

    Note that any method that doesn't take a manifest path as input will only
    change the internal state in `dataset_index`, and will not write to any
    manifest JSON files on disk. Example methods of this kind are `sample()`
    and `add_file()`. In order to serialize this state to disk, you must call
    `write_manifest()`.
    """

    _DATA_SUBDIR = "data"
    _LABELS_SUBDIR = "labels"

    def __init__(self, manifest_path, manifest=None):
        """Creates a LabeledDataset instance.

        Args:
            manifest_path: the path to the LabeledDatasetIndex for the dataset
            manifest: the LabeledDatasetIndex instance itself, if it has
                already been loaded. If not provided, the index is loaded from
                `manifest_path`

        Raises:
            LabeledDatasetError: if `type(self)` is not a subclass of the
                `type` recorded in the manifest
        """
        if manifest is not None:
            dataset_index = manifest
        else:
            dataset_index = LabeledDatasetIndex.from_json(manifest_path)

        if not isinstance(self, etau.get_class(dataset_index.type)):
            raise LabeledDatasetError(
                "Tried to read dataset of type '%s' from location '%s', but "
                "manifest is of type '%s'"
                % (
                    etau.get_class_name(self),
                    manifest_path,
                    dataset_index.type,
                )
            )

        self._manifest_path = os.path.abspath(manifest_path)
        self._dataset_dir = os.path.dirname(self._manifest_path)
        self._data_dir = os.path.join(self._dataset_dir, self._DATA_SUBDIR)
        self._labels_dir = os.path.join(self._dataset_dir, self._LABELS_SUBDIR)

        self._dataset_index = None
        self._data_to_labels_map = None
        self.set_dataset_index(dataset_index)

    def __getitem__(self, idx):
        """Gets the (data, labels) pair for the sample with the given index.

        Args:
            idx: the index

        Returns:
            a (data, labels) pair
        """
        data = self.get_data(idx)
        labels = self.get_labels(idx)
        return (data, labels)

    def __len__(self):
        """The number of samples in the dataset."""
        return len(self.dataset_index)

    def __iter__(self):
        """Returns an iterator over the samples in the dataset.

        Returns:
            an iterator over (data, labels) pairs, where data are objects
                returned by `read_data()` and labels are objects returned by
                `read_labels()`
        """
        return zip(self.iter_data(), self.iter_labels())

    @property
    def dataset_dir(self):
        """The absolute path to the top-level directory of the dataset."""
        return self._dataset_dir

    @property
    def manifest_path(self):
        """The absolute path to the LabeledDatasetIndex for the dataset."""
        return self._manifest_path

    @property
    def dataset_index(self):
        """Gets the LabeledDatasetIndex for the dataset."""
        return self._dataset_index

    @property
    def data_dirname(self):
        """The name of the directory in which data samples are stored within
        the dataset.
        """
        return self._DATA_SUBDIR

    @property
    def labels_dirname(self):
        """The name of the directory in which labels are stored within the
        dataset.
        """
        return self._LABELS_SUBDIR

    @property
    def data_dir(self):
        """The absolute path to the data directory of the dataset."""
        return self._data_dir

    @property
    def labels_dir(self):
        """The absolute path to the labels directory of the dataset."""
        return self._labels_dir

    def get_paths(self, idx):
        """Gets the data path and labels path for the sample with the given
        index.

        Args:
            idx: the index

        Returns:
            a (data path, labels path) tuple
        """
        data_path = self.get_data_path(idx)
        labels_path = self.get_labels_path(idx)
        return data_path, labels_path

    def get_data(self, idx):
        """Gets the data for the sample with the given index.

        Args:
            idx: the index

        Returns:
            the data read via `read_data()`
        """
        data_path = self.get_data_path(idx)
        return self.read_data(data_path)

    def get_data_path(self, idx):
        """Gets the data path for the sample with the given index.

        Args:
            idx: the index

        Returns:
            the data path
        """
        return os.path.join(self.dataset_dir, self.dataset_index[idx].data)

    def get_labels(self, idx):
        """Gets the labels for the sample with the given index.

        Args:
            idx: the index

        Returns:
            the labels read via `read_labels()`
        """
        labels_path = self.get_labels_path(idx)
        return self.read_labels(labels_path)

    def get_labels_path(self, idx):
        """Gets the data path for the sample with the given index.

        Args:
            idx: the index

        Returns:
            the data path
        """
        return os.path.join(self.dataset_dir, self.dataset_index[idx].labels)

    def iter(self):
        """Returns an iterator over the data and labels in the dataset.

        Returns:
            an iterator over (data, labels) tuples
        """
        for data_path, labels_path in self.iter_paths():
            data = self.read_data(data_path)
            labels = self.read_labels(labels_path)
            yield data, labels

    def iter_paths(self):
        """Returns an iterator over the data and labels paths in the dataset.

        Returns:
            an iterator over (data path, labels path) tuples
        """
        for record in self.dataset_index:
            data_path = os.path.join(self.dataset_dir, record.data)
            labels_path = os.path.join(self.dataset_dir, record.labels)
            yield data_path, labels_path

    def iter_data(self):
        """Returns an iterator over the data in the dataset.

        Returns:
            an iterator over objects returned by `read_data()`
        """
        for data_path in self.iter_data_paths():
            yield self.read_data(data_path)

    def iter_data_paths(self):
        """Returns an iterator over the paths to data files in the dataset.

        Returns:
            an iterator over paths to data files
        """
        for record in self.dataset_index:
            yield os.path.join(self.dataset_dir, record.data)

    def iter_labels(self):
        """Returns an iterator over the labels in the dataset.

        Returns:
            an iterator over objects returned by `read_labels()`
        """
        for labels_path in self.iter_labels_paths():
            yield self.read_labels(labels_path)

    def iter_labels_paths(self):
        """Returns an iterator over the paths to labels files in the dataset.

        Returns:
            an iterator over paths to labels files
        """
        for record in self.dataset_index:
            yield os.path.join(self.dataset_dir, record.labels)

    def set_dataset_index(self, dataset_index):
        """Sets the LabeledDatasetIndex for the dataset.

        Args:
            dataset_index: a LabeledDatasetIndex
        """
        self._dataset_index = dataset_index
        self._build_index_map()

    def set_description(self, description):
        """Sets the description of the dataset.

        Args:
            description: the new description

        Returns:
            self
        """
        self.dataset_index.description = description
        return self

    def write_manifest(self, filename=None):
        """Writes the current manifest to disk inside `dataset_dir`.

        Use this method to serialize the current state of a dataset after
        performing some manipulations on it.

        Args:
            filename: an optional name for the manifest to be written inside
                `dataset_dir`. By default, the name of the current manifest
                from `manifest_path` is used
        """
        if filename is None:
            manifest_path = self.manifest_path
        else:
            manifest_path = os.path.join(self.dataset_dir, filename)

        self.dataset_index.write_json(manifest_path, pretty_print=True)

    @staticmethod
    def from_manifest(manifest_path):
        """Creates a LabeledDataset from the given LabeledDatasetIndex.

        Args:
            manifest_path: the path to a LabeledDatasetIndex

        Returns:
            a LabeledDataset
        """
        return load_dataset(manifest_path)

    @staticmethod
    def from_dir(dataset_dir):
        """Creates a LabeledDataset from the given directory, which must
        contain a LabeledDatasetIndex.

        This method uses `get_manifest_path_from_dir(dataset_dir)` to locate
        the manifest in the directory.

        Args:
            dataset_dir: the LabeledDataset directory

        Returns:
            a LabeledDataset
        """
        return load_dataset(dataset_dir)

    def sample(self, k):
        """Randomly downsamples the dataset to k samples.

        Args:
            k: the number of data elements in the dataset after sampling

        Returns:
            self
        """
        self.dataset_index.sample(k)
        self._build_index_map()
        return self

    def shuffle(self):
        """Randomly shuffles the order of the samples in the dataset.

        Returns:
            self
        """
        self.dataset_index.shuffle()
        return self

    def split(
        self,
        split_fractions=None,
        descriptions=None,
        split_method="random_exact",
    ):
        """Splits the dataset into multiple datasets containing disjoint
        subsets of the original dataset.

        Args:
            split_fractions: an optional list of split fractions, which should
                sum to 1, that specifies how to split the dataset. By default,
                [0.5, 0.5] is used
            descriptions: an optional list of descriptions for the output
                datasets. The list should be the same length as
                `split_fractions`. If not specified, the description of the
                original dataset is used for all of the output datasets
            split_method: string describing the method with which to split the
                data

        Returns:
            a list of `LabeledDataset`s of same length as `split_fractions`
        """
        split_indexes = self.dataset_index.split(
            split_fractions=split_fractions,
            descriptions=descriptions,
            split_method=split_method,
        )

        datasets = []
        for split_index in split_indexes:
            split_dataset = copy.deepcopy(self)
            split_dataset.set_dataset_index(split_index)
            datasets.append(split_dataset)

        return datasets

    def has_data_with_name(self, data_path):
        """Checks whether a data file already exists in the dataset with the
        provided filename.

        Args:
            data_path: path to or filename of a data file

        Returns:
            True/False
        """
        data_file = os.path.basename(data_path)
        return data_file in self._data_to_labels_map

    def add_file(
        self,
        data_path,
        labels_path,
        new_data_filename=None,
        new_labels_filename=None,
        file_method=FileMethods.COPY,
        error_on_duplicates=False,
    ):
        """Adds a single data file and its labels file to this dataset.

        Args:
            data_path: path to data file to be added
            labels_path: path to corresponding labels file to be added
            new_data_filename: optional filename for the data file to be
                renamed to
            new_labels_filename: optional filename for the labels file to be
                renamed to
            file_method: an `eta.core.datasets.FileMethods` value specifying
                how to add the files to the dataset. This value can be a
                two-element tuple specifying separate file methods for data and
                labels, respectively. The default is `FileMethods.COPY`
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
        """
        if new_data_filename is None:
            new_data_filename = os.path.basename(data_path)

        if new_labels_filename is None:
            new_labels_filename = os.path.basename(labels_path)

        if error_on_duplicates and self.has_data_with_name(new_data_filename):
            raise ValueError(
                "Data file '%s' already present in dataset" % new_data_filename
            )

        data_method, labels_method = self._parse_file_method(file_method)

        new_data_path = os.path.join(self.data_dir, new_data_filename)
        if data_path != new_data_path:
            data_method(data_path, new_data_path)

        new_labels_path = os.path.join(self.labels_dir, new_labels_filename)
        if labels_path != new_labels_path:
            labels_method(labels_path, new_labels_path)

        # Update the filename attribute in the labels JSON if necessary
        if new_data_filename != os.path.basename(data_path):
            labels = self.read_labels(new_labels_path)
            labels.filename = new_data_filename
            self.write_labels(labels, new_labels_path)

        # First remove any other records with the same data filename
        self.dataset_index.cull_with_function(
            lambda record: os.path.basename(record.data) != new_data_filename
        )
        self.dataset_index.append(
            LabeledDataRecord(
                os.path.join(self.data_dirname, new_data_filename),
                os.path.join(self.labels_dirname, new_labels_filename),
            )
        )

        self._data_to_labels_map[new_data_filename] = new_labels_filename
        return self

    def add_data(
        self,
        data,
        labels,
        data_filename,
        labels_filename,
        error_on_duplicates=False,
    ):
        """Creates and adds a single data file and its labels file to this
        dataset, using the input python data structure.

        Args:
            data: input data in a format that can be passed to `write_data()`
            labels: input labels in a format that can be passed to
                `write_labels()`
            data_filename: filename for the data in the dataset
            labels_filename: filename for the labels in the dataset
            error_on_duplicates: whether to raise an error if a data file with
                the name `data_filename` already exists in the dataset. If this
                is set to `False`, the previous mapping of `data_filename` to a
                labels file will be deleted

        Returns:
            self
        """
        if error_on_duplicates and self.has_data_with_name(data_filename):
            raise ValueError(
                "Data file '%s' already present in dataset"
                % os.path.basename(data_filename)
            )

        data_path = os.path.join(self.data_dir, data_filename)
        self.write_data(data, data_path)

        labels_path = os.path.join(self.labels_dir, labels_filename)
        self.write_labels(labels, labels_path)

        # First remove any other records with the same data filename
        self.dataset_index.cull_with_function(
            lambda record: os.path.basename(record.data) != data_filename
        )
        self.dataset_index.append(
            LabeledDataRecord(
                os.path.join(self.data_dirname, data_filename),
                os.path.join(self.labels_dirname, labels_filename),
            )
        )

        self._data_to_labels_map[data_filename] = labels_filename
        return self

    def remove_data_files(self, data_filenames):
        """Removes the given data files from `self.dataset_index`.

        Note that this method doesn't delete the files from disk, it just
        removes them from the index. To remove files from disk that are not
        present in the index, use the `prune()` method.

        Args:
            data_filenames: list of filenames of data to remove

        Returns:
            self
        """
        removals = set(data_filenames)
        self.dataset_index.cull_with_function(
            lambda record: os.path.basename(record.data) not in removals
        )
        self._build_index_map()
        return self

    def copy(
        self, manifest_path, file_method=FileMethods.COPY, overwrite=False
    ):
        """Copies the dataset to another directory.

        If the dataset index has been manipulated, this will be reflected in
        the copy.

        Args:
            manifest_path: the path to write the LabeledDatasetIndex for the
                new dataset copy. The parent directory must either not exist or
                be empty
            file_method: an `eta.core.datasets.FileMethods` value specifying
                how to add the files to the dataset. This value can be a
                two-element tuple specifying separate file methods for data and
                labels, respectively. The default is `FileMethods.COPY`
            overwrite: whether to delete an existing dataset in the base
                directory of `manifest_path`, if necessary. By default, this is
                False

        Returns:
            a LabeledDataset instance that points to the new directory

        Raises:
            ValueError: if `overwrite == False` and the base directory of the
                specified `manifest_path` already exists
        """
        self._ensure_empty_dataset(manifest_path, overwrite=overwrite)
        self.dataset_index.write_json(manifest_path, pretty_print=True)

        dataset = self.__class__(manifest_path)

        data_method, labels_method = self._parse_file_method(file_method)
        for data_path, labels_path in zip(
            self.iter_data_paths(), self.iter_labels_paths()
        ):
            new_data_path = os.path.join(
                dataset.data_dir, os.path.basename(data_path)
            )
            data_method(data_path, new_data_path)

            new_labels_path = os.path.join(
                dataset.labels_dir, os.path.basename(labels_path)
            )
            labels_method(labels_path, new_labels_path)

        return dataset

    def merge(
        self,
        labeled_dataset_or_path,
        merged_manifest_path,
        in_place=False,
        description=None,
        file_method=FileMethods.COPY,
    ):
        """Creates a new dataset which contains the merged contents of this
        dataset and the given dataset.

        The datasets must be of the same type.

        Args:
            labeled_dataset_or_path: a LabeledDataset instance or path to its
                LabeledDatasetIndex on disk
            merged_manifest_path: the path tow rite the LabeledDatasetIndex of
                the merged dataset. If `in_place` is False, the parent
                directory of this path must either not exist or be empty. If
                `in_place` is True, either the parent directory must be equal
                to `dataset_dir` or `merged_manifest_path` should be the
                filename of a new manifest to write in the `dataset_dir` of
                this dataset
            in_place: whether to merge the new samples into this dataset (True)
                or write a new merged dataset (False). The default is False
            description: optional description for the merged dataset
            file_method: an `eta.core.datasets.FileMethods` value specifying
                how to add the files to the dataset. This value can be a
                two-element tuple specifying separate file methods for data and
                labels, respectively. The default is `FileMethods.COPY`

        Returns:
            a LabeledDataset instance pointing to the merged dataset. If
                `in_place == True`, this will just be `self`
        """
        labeled_dataset = self._parse_dataset(labeled_dataset_or_path)

        data_filenames_to_merge = self._get_filenames_for_merge(
            labeled_dataset
        )

        output_dataset_dir = os.path.dirname(merged_manifest_path)
        if not output_dataset_dir:
            output_dataset_dir = self.dataset_dir
            merged_manifest_path = os.path.join(
                output_dataset_dir, merged_manifest_path
            )

        if in_place and output_dataset_dir != self.dataset_dir:
            raise ValueError(
                "When merging datasets in place, the provided "
                "`merged_manifest_path` must be within the same directory as "
                "this dataset '%s'; found '%s'"
                % (self.dataset_dir, output_dataset_dir)
            )

        if in_place:
            merged_dataset = self
        else:
            merged_dataset = self.copy(merged_manifest_path)

        for data_path, labels_path in zip(
            labeled_dataset.iter_data_paths(),
            labeled_dataset.iter_labels_paths(),
        ):
            if os.path.basename(data_path) in data_filenames_to_merge:
                merged_dataset.add_file(
                    data_path, labels_path, file_method=file_method
                )

        if description is not None:
            merged_dataset.set_description(description)

        merged_dataset.write_manifest(os.path.basename(merged_manifest_path))
        return merged_dataset

    def deduplicate(self):
        """Removes duplicate data files from the index.

        If sets of files are found with the same content, one file in each is
        set is chosen arbitrarily to be kept, and the rest are removed.

        Note that no files are deleted; this method only removes entries from
        the index.

        Returns:
            self
        """
        duplicate_data_paths = etau.find_duplicate_files(
            list(self.iter_data_paths())
        )

        if not duplicate_data_paths:
            return self

        data_paths_remove = set()
        for group in duplicate_data_paths:
            for path in group[1:]:
                data_paths_remove.add(path)

        self.dataset_index.cull_with_function(
            lambda record: os.path.join(self.dataset_dir, record.data)
            not in data_paths_remove
        )

        self._build_index_map()
        return self

    def prune(self):
        """Deletes data and labels files that are not in the index.

        Note that actual files will be deleted if they are not present in
        `dataset_index`, for which the current state can be different than when
        it was read from a manifest JSON file.

        Returns:
            self
        """
        data_filenames = set()
        labels_filenames = set()
        for data_path, labels_path in self.iter_paths():
            data_filenames.add(os.path.basename(data_path))
            labels_filenames.add(os.path.basename(labels_path))

        for filename in etau.list_files(self.data_dir):
            if filename not in data_filenames:
                etau.delete_file(os.path.join(self.data_dir, filename))

        for filename in etau.list_files(self.labels_dir):
            if filename not in labels_filenames:
                etau.delete_file(os.path.join(self.labels_dir, filename))

        return self

    def apply_to_data(self, func):
        """Applies the given function to each data element and overwrites the
        data files with the outputs.

        Args:
            func: function that takes in a data element in the format returned
                by `read_data()` and outputs transformed data in the same
                format

        Returns:
            self
        """
        for data, path in zip(self.iter_data(), self.iter_data_paths()):
            self.write_data(func(data), path)

        return self

    def apply_to_data_paths(self, func):
        """Calls the given function on the path of each data file, with the
        path as both the input and output argument.

        Args:
            func: function that takes in two arguments, an input path and
                an output path. It will be called with the same value for
                both arguments, so that the file is overwritten

        Returns:
            self
        """
        for path in self.iter_data_paths():
            func(path, path)

        return self

    def apply_to_labels(self, func):
        """Applies the given function to each of the labels, and overwrites the
        labels file with the outputs.

        Args:
            func: function that takes in a labels object in the format returned
                by `read_labels()` and outputs transformed labels in the same
                format

        Returns:
            self
        """
        for labels, path in zip(self.iter_labels(), self.iter_labels_paths()):
            self.write_labels(func(labels), path)

        return self

    def apply_to_labels_paths(self, func):
        """Calls the given function on the path of each labels file, with the
        path as both the input and output argument.

        Args:
            func: function that takes in two arguments, an input path and an
                output path. It will be called with the same value for both
                arguments, so that the file is overwritten

        Returns:
            self
        """
        for path in self.iter_labels_paths():
            func(path, path)

        return self

    def builder(self):
        """Creates a LabeledDatasetBuilder instance for this dataset.

        Returns:
            a LabeledDatasetBuilder
        """
        builder = self.builder_cls()
        for paths in self.iter_paths():
            builder.add_record(builder.record_cls(*paths))
        return builder

    def get_labels_set(self, ensure_filenames=True):
        """Creates a SetLabels-type object from this dataset's labels.

        Args:
            ensure_filenames: whether to add the filename into the individual
                labels if it is not present

        Returns:
            an ImageSetLabels or VideoSetLabels
        """
        raise NotImplementedError("subclasses must implement get_labels_set()")

    def get_active_schema(self):
        """Returns a LabelsSchema-type object describing the active schema of
        the dataset.

        Returns:
            an ImageLabelsSchema or VideoLabelsSchema
        """
        raise NotImplementedError(
            "subclasses must implement get_active_schema()"
        )

    def write_annotated_data(self, output_dir, annotation_config=None):
        """Annotates the data with its labels, and outputs the annotated data
        to the specified directory.

        Args:
            output_dir: the path to the directory into which the annotated data
                will be written
            annotation_config: an optional
                `eta.core.annotations.AnnotationConfig` specifying how to
                render the annotations
        """
        raise NotImplementedError(
            "subclasses must implement write_annotated_data()"
        )

    def add_metadata_to_labels(self, overwrite=False):
        """Adds metadata about each data file into its labels file.

        Args:
            overwrite: whether to overwrite metadata already present in the
                labels file
        """
        for data_path, labels, labels_path in zip(
            self.iter_data_paths(),
            self.iter_labels(),
            self.iter_labels_paths(),
        ):
            if not hasattr(labels, "metadata"):
                raise TypeError(
                    "'%s' has no 'metadata' attribute"
                    % etau.get_class_name(labels)
                )

            if overwrite or labels.metadata is None:
                labels.metadata = self._build_metadata(data_path)
                labels.write_json(labels_path)

    @classmethod
    def create_empty_dataset(
        cls, manifest_path_or_dataset_dir, description=None, overwrite=False
    ):
        """Creates an empty LabeledDataset with the specified manifest path or
        in the given directory.

        Args:
            manifest_path_or_dataset_dir: the path to write the
                LabeledDatasetIndex of the dataset, or a directory in which to
                create the dataset
            description: an optional description for the dataset
            overwrite: whether to delete an existing dataset in the base
                directory of `manifest_path`, if necessary. By default, this is
                False

        Returns:
            a LabeledDataset instance pointing to the empty dataset

        Raises:
            ValueError: if `overwrite == False` and the base directory of the
                specified `manifest_path` already exists
        """
        if not os.path.splitext(manifest_path_or_dataset_dir)[1]:
            # Found a directory
            manifest_path = os.path.join(
                manifest_path_or_dataset_dir, "manifest.json"
            )
        else:
            # Found a manifest path
            manifest_path = manifest_path_or_dataset_dir

        cls._ensure_empty_dataset(manifest_path, overwrite=overwrite)
        dataset_index = LabeledDatasetIndex(
            etau.get_class_name(cls), description=description
        )
        dataset_index.write_json(manifest_path, pretty_print=True)
        dataset = cls(manifest_path)

        cls.validate_dataset(manifest_path)

        return dataset

    @classmethod
    def is_valid_dataset(cls, manifest_path):
        """Determines whether the dataset with the given manifest is a valid
        LabeledDataset.

        Args:
            manifest_path: the path to the LabeledDatasetIndex for the dataset

        Returns:
            True/False
        """
        try:
            cls.validate_dataset(manifest_path)
        except LabeledDatasetError:
            return False

        return True

    @classmethod
    def validate_dataset(cls, manifest_path):
        """Validates that the dataset with the given manifest is a valid
        LabeledDataset.

        Args:
            manifest_path: the path to the LabeledDatasetIndex for the dataset

        Raises:
            LabeledDatasetError: if the dataset is not valid
        """
        raise NotImplementedError(
            "subclasses must implement validate_dataset()"
        )

    def read_data(self, path):
        """Reads data from a data file at the given path.

        Subclasses must implement this based on the particular data format for
        the subclass.

        Args:
            path: path to a data file in the dataset

        Returns:
            a data object
        """
        raise NotImplementedError("subclasses must implement read_data()")

    def read_labels(self, path):
        """Reads a labels object from a labels JSON file at the given path.

        Subclasses must implement this based on the particular labels format
        for the subclass.

        Args:
            path: path to a labels file in the dataset

        Returns:
            a labels object
        """
        raise NotImplementedError("subclasses must implement read_labels()")

    def write_data(self, data, path):
        """Writes data to a data file at the given path.

        Subclasses must implement this based on the particular data format for
        the subclass.  The method should accept input `data` of the same type
        as output by `self.read_data()`.

        Args:
            data: a data element to be written to a file
            path: path to write the data
        """
        raise NotImplementedError("subclasses must implement write_data()")

    def write_labels(self, labels, path):
        """Writes a labels object to a labels JSON file at the given path.

        Subclasses must implement this based on the particular labels format
        for the subclass.  The method should accept input `labels` of the same
        type as output by `self.read_labels()`.

        Args:
            labels: a labels object to be written to a file
            path: path to write the labels JSON file
        """
        raise NotImplementedError("subclasses must implement write_labels()")

    def _build_metadata(self, path):
        """Reads metadata from a data file at the given path and builds an
        instance of the metadata class associated with the data type.

        Subclasses must implement this based on the particular metadata format
        for the subclass.

        Args:
            path: path to a data file in the dataset

        Returns:
            an instance of the metadata class associated with the data format
                for the subclass
        """
        raise NotImplementedError(
            "subclasses must implement _build_metadata()"
        )

    @staticmethod
    def _parse_file_method(file_method):
        if isinstance(file_method, tuple) and len(file_method) == 2:
            data_method, labels_method = file_method
        else:
            data_method = file_method
            labels_method = file_method

        data_fcn = FileMethods.get_function(data_method)
        labels_fcn = FileMethods.get_function(labels_method)
        return data_fcn, labels_fcn

    def _build_index_map(self):
        self._data_to_labels_map = {}
        for record in self.dataset_index:
            data_file = os.path.basename(record.data)
            labels_file = os.path.basename(record.labels)
            if data_file in self._data_to_labels_map:
                raise LabeledDatasetError(
                    "Data file '%s' maps to multiple labels files" % data_file
                )

            self._data_to_labels_map[data_file] = labels_file

    def _parse_dataset(self, labeled_dataset_or_path):
        cls_name = etau.get_class_name(self)
        cls = etau.get_class(cls_name)
        if etau.is_str(labeled_dataset_or_path):
            labeled_dataset = cls(labeled_dataset_or_path)
        else:
            labeled_dataset = labeled_dataset_or_path

        if not isinstance(labeled_dataset, cls):
            raise TypeError(
                "'%s' is not an instance of '%s'"
                % (etau.get_class_name(labeled_dataset), cls_name)
            )

        return labeled_dataset

    def _get_filenames_for_merge(self, labeled_dataset):
        data_filenames_this = {
            os.path.basename(path) for path in self.iter_data_paths()
        }
        labels_filenames_this = {
            os.path.basename(path) for path in self.iter_labels_paths()
        }

        data_filenames_other = {
            os.path.basename(path)
            for path in labeled_dataset.iter_data_paths()
        }
        data_filenames_to_merge = data_filenames_other - data_filenames_this
        labels_filenames_to_merge = {
            os.path.basename(labels_path)
            for data_path, labels_path in zip(
                labeled_dataset.iter_data_paths(),
                labeled_dataset.iter_labels_paths(),
            )
            if os.path.basename(data_path) in data_filenames_to_merge
        }

        labels_in_both = labels_filenames_to_merge & labels_filenames_this
        if labels_in_both:
            examples = ",".join(list(labels_in_both)[:5])
            raise ValueError(
                "Found different data filenames with the same corresponding "
                "labels filename: %s..." % examples
            )

        return data_filenames_to_merge

    @classmethod
    def _ensure_empty_dataset(
        cls, manifest_path, overwrite=False, warn_if_not_empty=True
    ):
        dataset_dir = os.path.dirname(manifest_path)

        if os.path.isdir(dataset_dir):
            if overwrite:
                logger.warning(
                    "Deleting existing dataset directory '%s'", dataset_dir
                )
                etau.delete_dir(dataset_dir)
            elif os.listdir(dataset_dir):
                if warn_if_not_empty:
                    logger.warning(
                        "Dataset directory '%s' is not empty", dataset_dir
                    )
                else:
                    raise ValueError(
                        "Dataset directory '%s' must be empty" % dataset_dir
                    )

        etau.ensure_dir(dataset_dir)
        etau.ensure_dir(os.path.join(dataset_dir, cls._DATA_SUBDIR))
        etau.ensure_dir(os.path.join(dataset_dir, cls._LABELS_SUBDIR))

    @property
    def builder_cls(self):
        """Getter for the associated LabeledDatasetBuilder class.

        Returns:
            builder_cls: the associated LabeledDatasetBuilder class
        """
        return etau.get_class(etau.get_class_name(self) + "Builder")


class LabeledDatasetError(Exception):
    """Exception raised when an error occurs with a LabeledDataset."""

    pass


class LabeledVideoDataset(LabeledDataset):
    """Core class for interacting with a labeled dataset of videos.

    Labeled video datasets are stored on disk in the following format::

        /path/to/video/dataset/
            manifest.json
            data/
                video1.mp4
                ...
            labels/
                video1.json
                ...

    where each labels file is stored in `eta.core.video.VideoLabels` format.
    """

    def get_labels_set(self, ensure_filenames=True):
        """Creates a VideoSetLabels containing this dataset's labels.

        Args:
            ensure_filenames: whether to add the filename into the individual
                labels if it is not present

        Returns:
            a VideoSetLabels instance
        """
        if ensure_filenames:
            video_set_labels = etav.VideoSetLabels()
            for data_path, video_labels in zip(
                self.iter_data_paths(), self.iter_labels()
            ):
                if video_labels.filename is None:
                    filename = os.path.basename(data_path)
                    video_labels.filename = filename
                else:
                    filename = video_labels.filename
                video_set_labels[filename] = video_labels

            return video_set_labels

        return etav.VideoSetLabels(videos=list(self.iter_labels()))

    def get_active_schema(self):
        """Returns a VideoLabelsSchema describing the active schema of the
        dataset.

        Returns:
            a VideoLabelsSchema
        """
        schema = etav.VideoLabelsSchema()
        for video_labels in self.iter_labels():
            schema.merge_schema(
                etav.VideoLabelsSchema.build_active_schema(video_labels)
            )

        return schema

    def write_annotated_data(self, output_dir, annotation_config=None):
        """Annotates the data with its labels, and outputs the annotated data
        to the specified directory.

        Args:
            output_dir: the path to the directory into which the annotated data
                will be written
            annotation_config: an optional
                `eta.core.annotations.AnnotationConfig` specifying how to
                render the annotations
        """
        for video_path, video_labels in zip(
            self.iter_data_paths(), self.iter_labels()
        ):
            output_path = os.path.join(
                output_dir, os.path.basename(video_path)
            )
            etaa.annotate_video(
                video_path,
                video_labels,
                output_path,
                annotation_config=annotation_config,
            )

    @classmethod
    def validate_dataset(cls, manifest_path):
        """Validates that the dataset with the given manifest is a valid
        LabeledVideoDataset.

        Args:
            manifest_path: the path to the LabeledDatasetIndex for the dataset

        Raises:
            LabeledDatasetError: if the dataset is not valid
        """
        video_dataset = cls(manifest_path)

        for video_path in video_dataset.iter_data_paths():
            if not os.path.isfile(video_path):
                raise LabeledDatasetError("File not found: %s" % video_path)

            if not etav.is_supported_video_file(video_path):
                raise LabeledDatasetError(
                    "Unsupported video format: %s" % video_path
                )

        for labels_path in video_dataset.iter_labels_paths():
            if not os.path.isfile(labels_path):
                raise LabeledDatasetError("File not found: %s" % labels_path)

            if not os.path.splitext(labels_path)[1] == ".json":
                raise LabeledDatasetError(
                    "Unsupported labels format: %s" % labels_path
                )

    def read_data(self, path):
        """Returns a video reader for the given path.

        Args:
            path: the path to the video to read

        Returns:
            an `eta.core.video.FFmpegVideoReader`
        """
        return etav.FFmpegVideoReader(path)

    def read_labels(self, path):
        """Reads the VideoLabels from the given path.

        Args:
            path: the path to the VideoLabels to read

        Returns:
            an `eta.core.video.VideoLabels`
        """
        return etav.VideoLabels.from_json(path)

    def write_data(self, video_reader, path):
        """Writes the video to the given path.

        Args:
            video_reader: an `eta.core.video.FFmpegVideoReader`
            path: the path to write the video
        """
        writer = etav.FFmpegVideoWriter(
            path, video_reader.frame_rate, video_reader.frame_size,
        )
        with writer:
            for img in video_reader:
                writer.write(img)

    def write_labels(self, labels, path):
        """Writes the labels to the given path.

        Args:
            labels: an `eta.core.video.VideoLabels`
            path: the path to write the labels
        """
        labels.write_json(path)

    def _build_metadata(self, path):
        return etav.VideoMetadata.build_for(path)


class LabeledImageDataset(LabeledDataset):
    """Core class for interacting with a labeled dataset of images.

    Labeled image datasets are stored on disk in the following format::

        /path/to/image/dataset/
            manifest.json
            data/
                image1.png
                ...
            labels/
                image1.json
                ...

    where each labels file is stored in `eta.core.image.ImageLabels` format.
    """

    def get_labels_set(self, ensure_filenames=True):
        """Creates an ImageSetLabels instance containing this dataset's labels.

        Args:
            ensure_filenames: whether to add the filename into the individual
                labels if it is not present

        Returns:
            an ImageSetLabels instance
        """
        if ensure_filenames:
            image_set_labels = etai.ImageSetLabels()
            for data_path, image_labels in zip(
                self.iter_data_paths(), self.iter_labels()
            ):
                if image_labels.filename is None:
                    filename = os.path.basename(data_path)
                    image_labels.filename = filename
                else:
                    filename = image_labels.filename
                image_set_labels[filename] = image_labels

            return image_set_labels

        return etai.ImageSetLabels(images=list(self.iter_labels()))

    def get_active_schema(self):
        """Returns the ImageLabelsSchema describing the active schema of the
        dataset.

        Returns:
            an ImageLabelsSchema
        """
        schema = etai.ImageLabelsSchema()
        for image_labels in self.iter_labels():
            schema.merge_schema(
                etai.ImageLabelsSchema.build_active_schema(image_labels)
            )

        return schema

    def write_annotated_data(self, output_dir, annotation_config=None):
        """Annotates the data with its labels, and outputs the annotated data
        to the specified directory.

        Args:
            output_dir: the path to the directory into which the annotated data
                will be written
            annotation_config: an optional
                `eta.core.annotations.AnnotationConfig` specifying how to
                render the annotations
        """
        for img, image_path, image_labels in zip(
            self.iter_data(), self.iter_data_paths(), self.iter_labels()
        ):
            img_annotated = etaa.annotate_image(
                img, image_labels, annotation_config=annotation_config
            )
            output_path = os.path.join(
                output_dir, os.path.basename(image_path)
            )
            self.write_data(img_annotated, output_path)

    @classmethod
    def validate_dataset(cls, manifest_path):
        """Validates that the dataset with the given manifest is a valid
        LabeledImageDataset.

        Args:
            manifest_path: the path to the LabeledDatasetIndex for the dataset

        Raises:
            LabeledDatasetError: if the dataset is not valid
        """
        image_dataset = cls(manifest_path)

        for img_path in image_dataset.iter_data_paths():
            if not os.path.isfile(img_path):
                raise LabeledDatasetError("File not found: %s" % img_path)

            if not etai.is_supported_image(img_path):
                raise LabeledDatasetError(
                    "Unsupported image format: %s" % img_path
                )

        for labels_path in image_dataset.iter_labels_paths():
            if not os.path.isfile(labels_path):
                raise LabeledDatasetError("File not found: %s" % labels_path)

            if not os.path.splitext(labels_path)[1] == ".json":
                raise LabeledDatasetError(
                    "Unsupported labels format: %s" % labels_path
                )

    def read_data(self, image_path):
        """Reads the image from the given path.

        Args:
            image_path: the path to the image to read

        Returns:
            the image
        """
        return etai.read(image_path)

    def read_labels(self, path):
        """Reads the ImageLabels from the given path.

        Args:
            path: the path to the ImageLabels to read

        Returns:
            an `eta.core.image.ImageLabels`
        """
        return etai.ImageLabels.from_json(path)

    def write_data(self, img, path):
        """Writes the image to the given path.

        Args:
            img: an image
            path: the path to write the image
        """
        etai.write(img, path)

    def write_labels(self, labels, path):
        """Writes the labels to the given path.

        Args:
            labels: an `eta.core.image.ImageLabels`
            path: the path to write the labels
        """
        labels.write_json(path)

    def _build_metadata(self, path):
        return etai.ImageMetadata.build_for(path)
