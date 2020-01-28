'''
Core interfaces, data structures, and methods for working with datasets

Copyright 2017-2019 Voxel51, Inc.
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

from .labeled_datasets import LabeledDatasetError


logger = logging.getLogger(__name__)


def check_labels_filename_property(dataset, audit_only=True):
    '''Audit labels.filename's for each record in a dataset and optionally
    populate this field.

    Args:
        dataset: a `LabeledDataset` instance
        audit_only: If False, modifies the labels in place to populate the
            filename attribute.

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



'''

duplicate equivalent attrs:
    input: labels
    output: schema of duplicate labels
    in place: remove one of the two duplicates

duplicate different attrs:
    input: labels
    output: pairs of attrs (schema with values joined by `:`?)
    in place: raise error


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


duplicate samples:
    inputs: list of files; dataset(s)
    outputs: list of list of duplicates


'''