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

from collections import defaultdict
import logging
import os

import eta.core.image as etai
import eta.core.serial as etas
import eta.core.utils as etau
import eta.core.video as etav

from .labeled_datasets import load_dataset, LabeledDatasetError, \
    LabeledImageDataset, LabeledVideoDataset


logger = logging.getLogger(__name__)


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
