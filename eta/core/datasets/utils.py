"""
Utilities for working with `LabeledDataset`s.

Copyright 2017-2020 Voxel51, Inc.
voxel51.com

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
import re

import eta.core.utils as etau


logger = logging.getLogger(__name__)


class FileMethods(etau.FunctionEnum):
    """Enum of supported methods for adding files to `LabeledDataset`s.

    By convention, all methods should follow the syntax `fcn(inpath, outpath)`.
    """

    COPY = "copy"
    LINK = "link"
    MOVE = "move"
    SYMLINK = "symlink"

    _FUNCTIONS_MAP = {
        "copy": etau.copy_file,
        "link": etau.link_file,
        "move": etau.move_file,
        "symlink": etau.symlink_file,
    }


def append_index_if_necessary(dataset, data_path, labels_path):
    """Appends an index to the data and labels names if the data filename
    already exists in the dataset.

    Args:
        dataset: a LabeledDataset
        data_path: a path where we want to add a data file to the dataset
        labels_path: a path where we want to add a labels file to the dataset

    Returns:
        new_data_path: a path for the data file which is not already present in
            the dataset
        new_labels_path: a path for the labels files which potentially has the
            same index appended to the name as for the data
    """
    if not dataset.has_data_with_name(data_path):
        return data_path, labels_path

    data_filename = os.path.basename(data_path)
    labels_filename = os.path.basename(labels_path)
    data_basename, data_ext = os.path.splitext(data_filename)
    labels_basename, labels_ext = os.path.splitext(labels_filename)

    filename_regex = re.compile("%s-([0-9]+)%s" % (data_basename, data_ext))
    existing_indices = []
    for existing_data_path in dataset.iter_data_paths():
        existing_data_filename = os.path.basename(existing_data_path)
        match = filename_regex.match(existing_data_filename)
        if match is not None:
            existing_indices.append(int(match.group(1)))

    if existing_indices:
        new_index = max(existing_indices) + 1
    else:
        new_index = 1

    new_data_path = os.path.join(
        os.path.dirname(data_path),
        "%s-%d%s" % (data_basename, new_index, data_ext),
    )

    new_labels_path = os.path.join(
        os.path.dirname(labels_path),
        "%s-%d%s" % (labels_basename, new_index, labels_ext),
    )

    return new_data_path, new_labels_path


def get_dataset_name(path):
    """Gets the name of the labeled dataset containing the given data or label
    file.

    The "name" of the dataset is the name of it's dataset directory folder.

    Example:
        `get_dataset_name("/datasets/special-dataset-1/labels/vid-1.json")`
        returns "special-dataset-1"

    Args:
        path: the path to a data sample or labels file in a dataset

    Returns:
        the name of the dataset
    """
    return os.path.basename(os.path.dirname(os.path.dirname(path)))
