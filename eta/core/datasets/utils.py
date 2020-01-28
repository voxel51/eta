'''
Utility functions supporting ETA datasets

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

import logging
import os
import re

import eta.core.utils as etau


logger = logging.getLogger(__name__)


# File method enums and helpers
COPY = "copy"
LINK = "link"
MOVE = "move"
SYMLINK = "symlink"
FILE_METHODS = {COPY, LINK, MOVE, SYMLINK}
_FILE_METHODS_MAP = {
    COPY: etau.copy_file,
    LINK: etau.link_file,
    MOVE: etau.move_file,
    SYMLINK: etau.symlink_file
}

def _append_index_if_necessary(dataset, data_path, labels_path):
    '''Appends an index to the data and labels names if the data filename
    already exists in the dataset.

    Args:
        dataset: a `LabeledDataset` instance
        data_path: a path where we want to add a data file to the dataset
        labels_path: a path where we want to add a labels file to the dataset

    Returns:
        new_data_path: a path for the data file which is not already present in
            the dataset
        new_labels_path: a path for the labels files which potentially has the
            same index appended to the name as for the data
    '''
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
    return (
        os.path.join(os.path.dirname(data_path),
                     "%s-%d%s" % (data_basename, new_index, data_ext)),
        os.path.join(os.path.dirname(labels_path),
                     "%s-%d%s" % (labels_basename, new_index, labels_ext))
    )


def _get_dataset_name(path):
    ''' Given a filepath to a specific data or label file in a labeled dataset,
    this will return the dataset name determined by the containing folder.

    E.g. => /datasets/special-dataset-1/labels/vid-1.json
    returns 'special-dataset-1'
    '''
    base = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return base
