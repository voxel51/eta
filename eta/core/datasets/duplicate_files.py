'''Tool for finding and serializing a list of duplicate files

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

import eta.core.serial as etas
import eta.core.utils as etau

from .labeled_datasets import load_dataset


logger = logging.getLogger(__name__)


class DuplicateFiles(etas.Serializable):
    '''Simple wrapper for eta.core.utils.find_duplicate_files that can take in
    a list of datasets and serializes the result.

    Finding duplicates can take a LONG time so this class serves to prevent
    this from needing to be done often.
    '''

    def __init__(self, duplicates):
        '''Create a DuplicateFiles instance

        Args:
            duplicates: a list of lists of duplicate files
        '''
        self.duplicates = duplicates

    @classmethod
    def generate(cls, file_path_list=None, dataset_path_list=None,
                 output_path=None):
        '''Generate a `DuplicateFiles` instance by computing from a list of
        file paths and/or datasets

        Args:
            file_path_list: a list of file_path strings
            dataset_path_list: a list of dataset path strings. All data files
                in the dataset are added to the file_path_list
            output_path: a highly encouraged path to IMMEDIATELY output the
                result. This way the result does not need to be computed
                multiple times!
         '''
        file_path_list = cls._add_datasets_to_file_path_list(
            file_path_list, dataset_path_list)

        if not output_path:
            logger.warning("Finding duplicate files can take a long time. You"
                           " may want to consider specifying a path to output"
                           " the result before running.")

        duplicates = etau.find_duplicate_files(file_path_list, verbose=True)

        dup_files = cls(duplicates)

        if output_path:
            dup_files.write_json(output_path)

        return dup_files

    def attributes(self):
        '''Returns the list of class attributes that will be serialized.

        Returns:
            a list of attribute names
        '''
        return ["duplicates"]

    @staticmethod
    def _add_datasets_to_file_path_list(file_path_list, dataset_path_list):
        file_path_list = file_path_list.copy() if file_path_list else []

        if dataset_path_list:
            for dataset_path in dataset_path_list:
                dataset = load_dataset(dataset_path)
                file_path_list += list(dataset.iter_data_paths())

        return file_path_list

    @classmethod
    def from_dict(cls, d, *args, **kwargs):
        '''Constructs a DuplicateFiles object from a JSON dictionary.'''
        duplicates = d["duplicates"]
        return cls(duplicates=duplicates)
