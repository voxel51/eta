'''
LabeledDataset builders, which serve the purpose of managing a series of
dataset transformations and building a new LabeledDataset

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
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import copy
import logging
import os
import re

from eta.core.data import BaseDataRecord, DataRecords
import eta.core.frames as etaf
import eta.core.image as etai
import eta.core.utils as etau
import eta.core.video as etav

from .utils import COPY, FILE_METHODS, _FILE_METHODS_MAP, \
    _append_index_if_necessary


logger = logging.getLogger(__name__)


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
            record: a BuilderImageRecord or BuilderVideoRecord
        '''
        self._dataset.add(record)

    def add_transform(self, transform):
        '''Add a DatasetTransformer.

        Args:
            transform: a DatasetTransformer
        '''
        self._transformers.append(transform)

    @property
    def builder_dataset(self):
        '''The underlying BuilderDataset instance'''
        return self._dataset

    @property
    def builder_dataset_cls(self):
        '''The associated BuilderDataset class.'''
        cls_breakup = etau.get_class_name(self).split(".")
        cls = cls_breakup[-1]
        cls = re.sub("^Labeled", "Builder", re.sub("Builder$", "", cls))
        cls_breakup[-1] = cls
        full_cls_path = ".".join(cls_breakup)
        return etau.get_class(full_cls_path)

    @property
    def dataset_cls(self):
        '''The associated LabeledDataset class.'''
        cls = etau.get_class_name(self)
        cls = re.sub("Builder$", "", cls).split(".")[-1]
        return etau.get_class(cls, "eta.core.datasets")

    @property
    def record_cls(self):
        '''The record class.'''
        return self._dataset.record_cls

    def build(self, path, description=None, pretty_print=False,
              create_empty=False, data_method=COPY):
        '''Build the new LabeledDataset after all records and transformations
        have been added.

        Args:
            path: path to write the new dataset (manifest.json)
            description: optional dataset description
            pretty_print: whether to pretty print JSON labels. By default, this
                is False
            create_empty: whether to write empty datasets to disk. By default,
                this is False
            data_method: how to add the data files to the dataset, when
                applicable. If clipping is required, this option is ignored,
                for example. One of "copy", "link", "move", or "symlink".
                Labels files are written from their class instances and do not
                apply.

        Returns:
            a LabeledDataset
        '''
        if data_method not in FILE_METHODS:
            raise ValueError("invalid file_method: %s" % str(data_method))

        logger.info("Applying transformations to dataset")

        data_method = _FILE_METHODS_MAP[data_method]

        for transformer in self._transformers:
            transformer.transform(self._dataset)

        if not create_empty and not len(self.builder_dataset):
            logger.info("Built dataset is empty. Skipping write out.")
            return None

        logger.info(
            "Building dataset with %d elements", len(self.builder_dataset))

        dataset = self.dataset_cls.create_empty_dataset(path, description)
        data_subdir = os.path.join(dataset.dataset_dir, dataset._DATA_SUBDIR)
        labels_subdir = os.path.join(
            dataset.dataset_dir, dataset._LABELS_SUBDIR)

        did_warn_duplicate_name = False
        for record in self._dataset:
            data_filename = os.path.basename(record.new_data_path)
            labels_filename = os.path.basename(record.new_labels_path)
            data_path = os.path.join(data_subdir, data_filename)
            labels_path = os.path.join(labels_subdir, labels_filename)

            old_data_path = data_path
            data_path, labels_path = _append_index_if_necessary(
                dataset, data_path, labels_path)
            if data_path != old_data_path and not did_warn_duplicate_name:
                logger.warning(
                    "Duplicate data filenames found in dataset being built. "
                    "Appending indices to names as necessary")
                did_warn_duplicate_name = True

            record.build(
                data_path,
                labels_path,
                pretty_print=pretty_print,
                data_method=data_method
            )

            # The `file_method` is irrelevant because the files were already
            # placed directly into the dataset directory by `record.build()`.
            dataset.add_file(data_path, labels_path)

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

    def __init__(self, data_path, labels_path):
        '''Initialize the BuilderDataRecord. The label and data paths cannot
        and should not be modified after initialization.

        Args:
            data_path: path to data file
            labels_path: path to labels json
        '''
        super(BuilderDataRecord, self).__init__()
        self._data_path = data_path
        self._labels_path = labels_path
        self._new_data_path = None
        self._new_labels_path = None
        self._labels_cls = None
        self._labels_obj = None

    def get_labels(self):
        '''Get the labels in this record..

        Returns:
            an ImageLabels or VideoLabels
        '''
        if self._labels_obj is not None:
            return self._labels_obj
        self._labels_obj = self._labels_cls.from_json(self.labels_path)
        return self._labels_obj

    def set_labels(self, labels):
        '''Set the labels for this record.

        Args:
            labels: ImageLabels or VideoLabels
        '''
        self._labels_obj = labels

    @property
    def data_path(self):
        '''The data path.'''
        return self._data_path

    @property
    def labels_path(self):
        '''The labels path.'''
        return self._labels_path

    @property
    def new_data_path(self):
        '''The data path to be written to.'''
        if self._new_data_path is not None:
            return self._new_data_path
        return self._data_path

    @property
    def new_labels_path(self):
        '''The labels path to be written to.'''
        if self._new_labels_path is not None:
            return self._new_labels_path
        return self._labels_path

    @new_data_path.setter
    def new_data_path(self, value):
        self._new_data_path = value

    @new_labels_path.setter
    def new_labels_path(self, value):
        self._new_labels_path = value

    def build(self, data_path, labels_path, pretty_print=False,
              data_method=COPY):
        '''Write the transformed labels and data files to dir_path. The
        subclasses BuilderVideoRecord and BuilderDataRecord are responsible for
        writing the data file.

        Args:
            data_path: path to write the data file to
            labels_path: path to write the labels file to
            pretty_print: whether to pretty print JSON. By default, this is
                False
            data_method: how to create the data file, when applicable. The
                default is copy
        '''
        self._build_labels()
        labels = self.get_labels()
        labels.filename = os.path.basename(data_path)
        labels.write_json(labels_path, pretty_print=pretty_print)

        self._build_data(data_path, data_method)

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

    def prepend_to_name(self, prefix):
        '''Prepends a prefix to the data and label filenames respectively.'''
        self._new_data_path = prefix + '_' + os.path.basename(self.data_path)
        self._new_labels_path = prefix + '_' + os.path.basename(
            self.labels_path)

    def _build_labels(self):
        raise NotImplementedError(
            "subclasses must implement _build_labels()")

    def _build_data(self, data_path, data_method):
        raise NotImplementedError(
            "subclasses must implement _build_data()")


class BuilderImageRecord(BuilderDataRecord):
    '''BuilderDataRecord for images.'''

    def __init__(self, data_path, labels_path):
        '''Creates a BuilderImageRecord instance.

        Args:
            data_path: path to image
            labels_path: path to labels
        '''
        super(BuilderImageRecord, self).__init__(data_path, labels_path)
        self._labels_cls = etai.ImageLabels

    def _build_labels(self):
        return

    def _build_data(self, data_path, data_method):
        data_method(self.data_path, data_path)


class BuilderVideoRecord(BuilderDataRecord):
    '''BuilderDataRecord for video.'''

    def __init__(self, data_path, labels_path, clip_start_frame=1,
                 clip_end_frame=None, duration=None, total_frame_count=None):
        '''Initialize a BuilderVideoRecord with data_path, labels_path, and
        optional metadata about video. Without the optional arguments their
        values will be loaded from the video metadata and the start and end
        frames will default to covering the entire video.

        Args:
            data_path: path to video
            labels_path: path to labels
            clip_start_frame: start frame of the clip
            clip_end_frame: end frame of the clip
            duration: duration (in seconds) of the full video (not the clip)
            total_frame_count: number of frames in full video (not the clip)
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
        return super(BuilderVideoRecord, cls).required() + [
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

    def _build_data(self, data_path, data_method):
        start_frame, end_frame = (self.clip_start_frame, self.clip_end_frame)
        if start_frame == 1 and end_frame == self.total_frame_count:
            data_method(self.data_path, data_path)
        else:
            args = (
                self.data_path,
                etaf.FrameRanges.build_simple(start_frame, end_frame)
            )
            with etav.VideoProcessor(*args, out_video_path=data_path) as p:
                for img in p:
                    p.write(img)


class BuilderDataset(DataRecords):
    '''A BuilderDataset is managed by a LabeledDatasetBuilder.
    DatasetTransformers operate on BuilderDatasets.
    '''


class BuilderImageDataset(BuilderDataset):
    '''A BuilderDataset for images.'''

    def __init__(self, record_cls=BuilderImageRecord):
        super(BuilderImageDataset, self).__init__(record_cls)


class BuilderVideoDataset(BuilderDataset):
    '''A BuilderDataset for videos.'''

    def __init__(self, record_cls=BuilderVideoRecord):
        super(BuilderVideoDataset, self).__init__(record_cls)
