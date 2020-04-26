"""
Core definition of `LabeledDatasetBuilder`s, which serve the purpose of
managing and applying a series of `DatasetTransformer`s to `LabeledDataset`s.

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
import re

from eta.core.data import BaseDataRecord, DataRecords
import eta.core.frameutils as etaf
import eta.core.image as etai
import eta.core.utils as etau
import eta.core.video as etav

from .utils import FileMethods, append_index_if_necessary


logger = logging.getLogger(__name__)


class LabeledDatasetBuilder(object):
    """Class that enables the construction and application of a series of
    `DatasetTransformer`s to a `LabeledDataset`.

    Transformations are run in the order they are added.
    """

    def __init__(self):
        """Creates a LabeledDatasetBuilder instance."""
        self._transformers = []
        self._dataset = self.builder_dataset_cls()

    @property
    def builder_dataset(self):
        """The BuilderDataset instance managed by this builder."""
        return self._dataset

    @property
    def builder_dataset_cls(self):
        """The BuilderDataset class used by this builder."""
        cls_breakup = etau.get_class_name(self).split(".")
        cls = cls_breakup[-1]
        cls = re.sub("^Labeled", "Builder", re.sub("Builder$", "", cls))
        cls_breakup[-1] = cls
        return etau.get_class(".".join(cls_breakup))

    @property
    def dataset_cls(self):
        """The LabeledDataset class used by this builder."""
        cls = etau.get_class_name(self)
        cls = re.sub("Builder$", "", cls).split(".")[-1]
        return etau.get_class(cls, "eta.core.datasets")

    @property
    def record_cls(self):
        """The record class of the BuilderDataset used by this builder."""
        return self._dataset.record_cls

    def add_record(self, record):
        """Adds a record to the dataset managed by the builder.

        `LabeledImageDatasetBuilder`s take `BuilderImageRecord`s, and
        `LabeledVideoDatasetBuilder`s take `BuilderVideoRecord`s.

        Args:
            record: a BuilderImageRecord or BuilderVideoRecord
        """
        self._dataset.add(record)

    def add_transform(self, transform):
        """Appends a DatasetTransformer to the builder.

        Args:
            transform: a DatasetTransformer
        """
        self._transformers.append(transform)

    def build(
        self,
        manifest_path,
        description=None,
        pretty_print=False,
        create_empty=False,
        data_method=FileMethods.COPY,
    ):
        """Builds the new LabeledDataset after all records managed by this
        builder have been added and all transformations have been applied.

        Args:
            manifest_path: the path to write the `manifest.json` for the new
                dataset
            description: an optional description for the new dataset
            pretty_print: whether to render the JSON in human readable format
                with newlines and indentations. By default, this is False
            create_empty: whether to write empty datasets to disk. By default,
                this is False
            data_method: an `eta.core.datasets.FileMethods` value specifying
                how to add the data files to the dataset, when applicable. If
                clipping is required, this option is ignored, for example.
                The default value is `FileMethods.COPY`

        Returns:
            a LabeledDataset
        """
        data_method = FileMethods.get_function(data_method)

        if self._transformers:
            logger.info("Applying transformations to dataset")
            for transformer in self._transformers:
                transformer.transform(self._dataset)

        if not create_empty and not self.builder_dataset:
            logger.info("Dataset is empty; skipping write-out")
            return None

        logger.info(
            "Building dataset with %d elements", len(self.builder_dataset)
        )

        dataset = self.dataset_cls.create_empty_dataset(
            manifest_path, description=description
        )

        warned_duplicate_names = False
        for record in self._dataset:
            data_filename = os.path.basename(record.new_data_path)
            data_path = os.path.join(dataset.data_dir, data_filename)

            labels_filename = os.path.basename(record.new_labels_path)
            labels_path = os.path.join(dataset.labels_dir, labels_filename)

            old_data_path = data_path
            data_path, labels_path = append_index_if_necessary(
                dataset, data_path, labels_path
            )
            if data_path != old_data_path and not warned_duplicate_names:
                logger.warning(
                    "Duplicate data filenames found in dataset being built. "
                    "Appending indices to names as necessary"
                )
                warned_duplicate_names = True

            record.build(
                data_path, labels_path, data_method, pretty_print=pretty_print
            )

            # The `file_method` is irrelevant because the files were already
            # placed directly into the dataset directory by `record.build()`
            dataset.add_file(data_path, labels_path)

        dataset.write_manifest()

        return dataset


class LabeledImageDatasetBuilder(LabeledDatasetBuilder):
    """LabeledDatasetBuilder for images."""

    pass


class LabeledVideoDatasetBuilder(LabeledDatasetBuilder):
    """LabeledDatasetBuilder for videos."""

    pass


class BuilderDataRecord(BaseDataRecord):
    """Class that is responsible for tracking all of the metadata about a data
    record required for dataset operations on a BuilderDataset.

    The `data_path` and `labels_path` of a record cannot be modified after
    initialization.
    """

    def __init__(self, data_path, labels_path):
        """Creates a BuilderDataRecord instance.

        Args:
            data_path: path to data file
            labels_path: path to labels JSON
        """
        super(BuilderDataRecord, self).__init__()
        self._data_path = data_path
        self._labels_path = labels_path
        self._new_data_path = None
        self._new_labels_path = None
        self._labels_cls = None
        self._labels_obj = None

    @property
    def data_path(self):
        """The data path."""
        return self._data_path

    @property
    def labels_path(self):
        """The labels path."""
        return self._labels_path

    @property
    def new_data_path(self):
        """The data path to be written to."""
        if self._new_data_path is not None:
            return self._new_data_path

        return self._data_path

    @property
    def new_labels_path(self):
        """The labels path to be written to."""
        if self._new_labels_path is not None:
            return self._new_labels_path

        return self._labels_path

    @new_data_path.setter
    def new_data_path(self, value):
        self._new_data_path = value

    @new_labels_path.setter
    def new_labels_path(self, value):
        self._new_labels_path = value

    def get_labels(self):
        """Get the labels in this record.

        Returns:
            an ImageLabels or VideoLabels
        """
        if self._labels_obj is not None:
            return self._labels_obj

        self._labels_obj = self._labels_cls.from_json(self.labels_path)
        return self._labels_obj

    def set_labels(self, labels):
        """Sets the labels for this record.

        Args:
            labels: ImageLabels or VideoLabels
        """
        self._labels_obj = labels

    def build(self, data_path, labels_path, data_method, pretty_print=False):
        """Writes the transformed labels and data files to the specified paths.

        Args:
            data_path: path to which to write the data file
            labels_path: path to which to write the labels file
            data_method: a function from `eta.core.datasets.FileMethods`
                specifying how to add the data sample
            pretty_print: whether to pretty print JSON. By default, this is
                False
        """
        self._build_data(data_path, data_method)

        self._build_labels()
        labels = self.get_labels()
        labels.filename = os.path.basename(data_path)
        labels.write_json(labels_path, pretty_print=pretty_print)

    def copy(self):
        """Safely copy a record. Only copy should be used when creating new
        records in DatasetTransformers.

        Returns:
            BuilderImageRecord or BuilderVideoRecord
        """
        return copy.deepcopy(self)

    def prepend_to_name(self, prefix):
        """Prepends a prefix to the data and label filenames respectively.

        Args:
            prefix: the prefix
        """
        self._new_data_path = prefix + "_" + os.path.basename(self.data_path)
        self._new_labels_path = (
            prefix + "_" + os.path.basename(self.labels_path)
        )

    def attributes(self):
        """Returns the list of attributes to be serialized.

        Returns:
            a list of class attributes to be serialized
        """
        attrs_ = super(BuilderDataRecord, self).attributes()
        return attrs_ + ["data_path", "labels_path"]

    @classmethod
    def required(cls):
        """Returns a list of attributes that are required by all instances of
        the data record.
        """
        _required = super(BuilderDataRecord, cls).required()
        return _required + ["data_path", "labels_path"]

    def _build_data(self, data_path, data_method):
        """Internal implementation of building the data sample represented by
        this builder.

        Subclasses must implement this method.

        Args:
            data_path: the path to which to write the built data sample
            data_method: a function from `eta.core.datasets.FileMethods`
                specifying how to add the data sample
        """
        raise NotImplementedError("subclasses must implement _build_data()")

    def _build_labels(self):
        """Internal implementation of building the labels represented by this
        builder.

        Subclasses must implement this method.
        """
        raise NotImplementedError("subclasses must implement _build_labels()")


class BuilderImageRecord(BuilderDataRecord):
    """BuilderDataRecord for images."""

    def __init__(self, data_path, labels_path):
        """Creates a BuilderImageRecord instance.

        Args:
            data_path: path to image
            labels_path: path to labels
        """
        super(BuilderImageRecord, self).__init__(data_path, labels_path)
        self._labels_cls = etai.ImageLabels

    def _build_labels(self):
        return

    def _build_data(self, data_path, data_method):
        data_method(self.data_path, data_path)


class BuilderVideoRecord(BuilderDataRecord):
    """BuilderDataRecord for video."""

    def __init__(
        self,
        data_path,
        labels_path,
        clip_start_frame=1,
        clip_end_frame=None,
        duration=None,
        total_frame_count=None,
    ):
        """Creates a BuilderVideoRecord instacne.

        Args:
            data_path: path to video
            labels_path: path to labels
            clip_start_frame: start frame of the clip. By default, the first
                frame is used
            clip_end_frame: end frame of the clip. By default, the last frame
                of the video is used
            duration: duration (in seconds) of the full video. By default, this
                value is loaded dynamically via `VideoMetadata`
            total_frame_count: number of frames in full video. By default, this
                value is loaded dynamically via `VideoMetadata`
        """
        super(BuilderVideoRecord, self).__init__(data_path, labels_path)
        self.clip_start_frame = clip_start_frame
        self.clip_end_frame = clip_end_frame
        self.duration = duration
        self.total_frame_count = total_frame_count

        self._labels_cls = etav.VideoLabels
        self._initialize()

    @classmethod
    def optional(cls):
        """Returns a list of attributes that are optionally included in the
        data record if they are present in the data dictionary.
        """
        return super(BuilderVideoRecord, cls).required() + [
            "clip_start_frame",
            "clip_end_frame",
            "duration",
            "total_frame_count",
        ]

    def _build_data(self, data_path, data_method):
        start_frame = self.clip_start_frame
        end_frame = self.clip_end_frame
        if (start_frame == 1) and (end_frame == self.total_frame_count):
            data_method(self.data_path, data_path)
            return

        frames = etaf.FrameRanges.build_simple(start_frame, end_frame)
        processor = etav.VideoProcessor(
            self.data_path, frames=frames, out_video_path=data_path
        )
        with processor as p:
            for img in p:
                p.write(img)

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

    def _initialize(self):
        metadata = None

        if self.total_frame_count is None:
            metadata = metadata or self._load_metadata()
            self.total_frame_count = metadata.total_frame_count

        if self.duration is None:
            metadata = metadata or self._load_metadata()
            self.duration = metadata.duration

        if self.clip_end_frame is None:
            metadata = metadata or self._load_metadata()
            self.clip_end_frame = metadata.total_frame_count

    def _load_metadata(self):
        return etav.VideoMetadata.build_for(self.data_path)


class BuilderDataset(DataRecords):
    """Base class for records that are managed by `LabeledDatasetBuilder`s and
    operated on by `DatasetTransformer`s in order to build new
    `LabeledDataset`s.
    """

    pass


class BuilderImageDataset(BuilderDataset):
    """A BuilderDataset for images."""

    def __init__(self, record_cls=BuilderImageRecord):
        super(BuilderImageDataset, self).__init__(record_cls)


class BuilderVideoDataset(BuilderDataset):
    """A BuilderDataset for videos."""

    def __init__(self, record_cls=BuilderVideoRecord):
        super(BuilderVideoDataset, self).__init__(record_cls)
