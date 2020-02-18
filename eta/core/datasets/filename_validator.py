'''
TODO

Copyright 2017-2020, Voxel51, Inc.
voxel51.com

Tyler Ganter, tyler@voxel51.com
'''
import logging
import os

import eta.core.serial as etas

from .labeled_datasets import LabeledDataset


logger = logging.getLogger(__name__)


class FileNameValidator(object):
    '''Populates the labels.filename'''

    @property
    def num_populated(self):
        return self._num_populated

    @property
    def num_skipped(self):
        return self._num_skipped

    @property
    def num_overridden(self):
        return self._num_overridden

    @property
    def report(self):
        return {
            "num_populated": self.num_populated,
            "num_skipped": self.num_skipped,
            "num_overridden": self.num_overridden
        }

    def __init__(self, mismatch_handle="raise"):
        '''

        Args:
            mismatch_handle: 'skip', 'override' or 'raise'
        '''
        self.clear_state()
        self._mismatch_handle = mismatch_handle

    def clear_state(self):
        self._num_populated = 0
        self._num_skipped = 0
        self._num_overridden = 0

    def transform(self, dataset: LabeledDataset, verbose=20):
        '''TODO
        Audit labels.filename's for each record in a dataset and optionally
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
        if verbose:
            logger.info("Checking labels.filename's for labeled dataset...")

        for idx, (data_path, labels_path) in enumerate(dataset.iter_paths()):
            if verbose and idx % verbose == 0:
                logger.info("%4d/%4d" % (idx, len(dataset)))

            data_filename = os.path.basename(data_path)
            labels = dataset.read_labels(labels_path)

            if labels.filename is None:
                self._num_populated += 1

                # populate the filename
                labels.filename = data_filename
                dataset.write_labels(labels, labels_path)

            elif labels.filename != data_filename:
                if self._mismatch_handle == "skip":
                    self._num_skipped += 1
                    continue

                elif self._mismatch_handle == "override":
                    self._num_overridden += 1
                    # populate the filename
                    labels.filename = data_filename
                    dataset.write_labels(labels, labels_path)

                else:
                    raise FileNameValidatorError(
                        "Filename: '%s' in labels file does not match data"
                        " filename: '%s'." % (labels.filename, data_filename)
                    )

        if verbose:
            logger.info("Complete: \n%s" % etas.json_to_str(self.report))


class FileNameValidatorError(Exception):
    '''Error raised when a FileNameValidator mismatch is found.'''
    pass
