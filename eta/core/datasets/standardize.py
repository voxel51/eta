'''
Tools for standardizing datasets by automated means

Copyright 2017-2020 Voxel51, Inc.
voxel51.com

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

import logging
import os

import eta.core.image as etai
import eta.core.utils as etau
import eta.core.video as etav

from .labeled_datasets import LabeledDatasetError, LabeledImageDataset, \
    LabeledVideoDataset


logger = logging.getLogger(__name__)


def ensure_labels_filename_property(dataset, audit_only=True):
    '''Audit labels.filename's for each record in a dataset and optionally
    populate this field.

    Args:
        dataset: a `LabeledDataset` instance
        audit_only: If False, modifies the labels in place to populate the
            filename attribute

    Returns:
        a tuple of:
            missing_count: integer count of labels files without a
                labels.filename field
            mismatch_count: integer count of labels files with a labels.filename
                field inconsistent with the data record filename

    Raises:
        LabeledDatasetError if audit_only==False and a mismatching filename is
            found.
    '''
    logger.info("Checking labels.filename's for labeled dataset...")

    missing_count = 0
    mismatch_count = 0

    for idx, (data_path, labels_path) in enumerate(dataset.iter_paths()):
        if idx % 20 == 0:
            logger.info("%4d/%4d" % (idx, len(dataset)))

        data_filename = os.path.basename(data_path)
        labels = dataset.read_labels(labels_path)

        if labels.filename is None:
            missing_count += 1

            if not audit_only:
                labels.filename = data_filename
                dataset.write_labels(labels, labels_path)

        elif labels.filename != data_filename:
            mismatch_count += 1

            if not audit_only:
                raise LabeledDatasetError(
                    "Filename: '%s' in labels file does not match data"
                    " filename: '%s'." % (labels.filename, data_filename)
                )

    logger.info("Complete: %d missing filenames and %d mismatched filenames"
                % (missing_count, mismatch_count))

    return missing_count, mismatch_count


def check_dataset_syntax(dataset, target_schema, audit_only=True):
    '''Audit labels.filename's for each record in a dataset and optionally
    populate this field.

    Args:
        dataset: a `LabeledDataset` instance
        target_schema: an `ImageLabelsSchema` or `VideoLabelsSchema` matching
            the dataset type
        audit_only: If False, modifies the labels in place to fix syntax

    Returns:
        a tuple of:
            fixable_schema: schema of values that can be (or were) substituted
                with target_schema syntax
            unfixable_schema: schema of values that cannot be mapped to
                the target_schema
    '''
    logger.info("Checking consistent syntax for labeled dataset...")

    if isinstance(dataset, LabeledImageDataset):
        checker = etai.ImageLabelsSyntaxChecker(target_schema)
    elif isinstance(dataset, LabeledVideoDataset):
        checker = etav.VideoLabelsSyntaxChecker(target_schema)
    else:
        raise ValueError(
            "Unexpected input type: `%s`" % etau.get_class_name(dataset))

    modified_count = 0

    for idx, labels_path in enumerate(dataset.iter_labels_paths()):
        if idx % 20 == 0:
            logger.info("%4d/%4d" % (idx, len(dataset)))

        labels = dataset.read_labels(labels_path)

        was_modified = checker.check(labels)

        modified_count += int(was_modified)
        if not audit_only and was_modified:
            labels.write_json(labels_path)

    logger.info(
        "Complete: %d/%d files %supdated"
        % (modified_count, len(dataset), "can be " if audit_only else "")
    )

    return checker.fixable_schema, checker.unfixable_schema


def check_duplicate_attrs(dataset, video_attr_multi_value_names=None,
                          img_or_frame_attr_multi_value_names=None,
                          obj_attr_multi_value_names=None,
                          event_attr_multi_value_names=None):
    '''Check for duplicate attributes (and return boolean)

    Args:
        dataset: a `LabeledDataset` instance
        video_attr_multi_value_names: list of attr name strings that the
            video attrs is allowed to have multiple DIFFERENT values for
        img_or_frame_attr_multi_value_names: same for image/frame attrs
        obj_attr_multi_value_names: same for object attrs
        event_attr_multi_value_names: same for event attrs

    Returns:
        integer count of labels files containing duplicate attributes
    '''
    logger.info("Checking for duplicate attrs for labeled dataset...")

    dup_attrs_count = 0

    for idx, labels_path in enumerate(dataset.iter_labels_paths()):
        if idx % 20 == 0:
            logger.info("%4d/%4d" % (idx, len(dataset)))

        labels = dataset.read_labels(labels_path)

        if isinstance(labels, etai.ImageLabels):
            has_duplicates = labels.has_duplicate_attrs(
                img_or_frame_attr_multi_value_names,
                obj_attr_multi_value_names
            )
        elif isinstance(labels, etav.VideoLabels):
            has_duplicates = labels.has_duplicate_attrs(
                video_attr_multi_value_names,
                img_or_frame_attr_multi_value_names,
                obj_attr_multi_value_names,
                event_attr_multi_value_names
            )
        else:
            raise ValueError(
                "Unexpected labels type: '%s'" % etau.get_class_name(labels))

        dup_attrs_count += int(has_duplicates)

    logger.info(
        "Complete: %d/%d files have duplicate attributes"
        % (dup_attrs_count, len(dataset))
    )

    return dup_attrs_count