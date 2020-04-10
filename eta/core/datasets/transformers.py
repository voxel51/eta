"""
Core definition of `DatasetTransformer`s, which define the interface and
implementations of applying transformations to `BuilderDataset`s.

Copyright 2017-2020 Voxel51, Inc.
voxel51.com

Matthew Lightman, matthew@voxel51.com
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
from future.utils import iteritems, itervalues

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict
import logging
import os
import random
import re

import numpy as np

import eta.core.image as etai
import eta.core.utils as etau
import eta.core.video as etav

from .builders import (
    BuilderImageRecord,
    BuilderVideoRecord,
    BuilderVideoDataset,
)
from .utils import get_dataset_name


logger = logging.getLogger(__name__)


class DatasetTransformer(object):
    """Interface for dataset transformers, which take `BuilderDataset`s as
    input and transform their samples according to a specified algorithm.

    Subclasses must implement the `transform()` method, which applies the
    transformation.
    """

    def transform(self, src):
        """Applies the dataset transform to the BuilderDataset.

        Args:
            src: a BuilderDataset
        """
        raise NotImplementedError("subclasses must implement transform()")


class Sampler(DatasetTransformer):
    """Dataset transformer that randomly samples a specified number of records
    from the dataset.

    If the number of records is less than the requested number, then all
    records are kept, but the order is randomized.
    """

    def __init__(self, k):
        """Creates a Sampler instance.

        Args:
            k: the number of samples to take
        """
        self.k = k

    def transform(self, src):
        """Samples from the existing records.

        Args:
            src: a BuilderImageDataset or BuilderVideoDataset
        """
        src.records = random.sample(src.records, min(self.k, len(src.records)))


class Balancer(DatasetTransformer):
    """Dataset transformer that balances the dataset's attributes and object
    labels by removing records as necessary.

    Currently only categorical attributes are supported.

    Example:
        Given a dataset with 10 green cars, 20 blue cars and 15 red cars,
        remove records with blue and red cars until there are the same number
        of each color.
    """

    _NUM_RANDOM_ITER = 10000
    _BUILDER_RECORD_TO_SCHEMA = [
        (BuilderImageRecord, etai.ImageLabelsSchema),
        (BuilderVideoRecord, etav.VideoLabelsSchema),
    ]

    def __init__(
        self,
        attribute_name=None,
        object_label=None,
        balance_by_object_label=False,
        labels_schema=None,
        target_quantile=0.25,
        negative_power=5,
        target_count=None,
        target_hard_min=False,
        algorithm="greedy",
    ):
        """Creates a Balancer instance.

        Args:
            attribute_name: the name of the attribute to balance by
            object_label: the name of the object label that the attribute_name
                must be nested under. If this is None, it is assumed that the
                attributes are Image/Frame level attrs
            balance_by_object_label: if True, the dataset is balanced by object
                label instead of attributes.
            labels_schema: an ImageLabelsSchema or VideoLabelsSchema that
                indicates which attributes, object labels, etc. should be used
                for balancing. This can be specified as an alternative to
                `attribute_name` and `object_label`. Note that labels are not
                altered; this schema just picks out the attributes that are
                used for balancing
            target_quantile: value between [0, 1] to specify what the target
                count per attribute value will be.
                    0.5: will result in the true median
                      0: the minimum value
                It is recommended to set this somewhere between [0, 0.5]. The
                smaller this value is, the closer all values can be balanced,
                at the risk that if some values have particularly low number of
                samples, they dataset will be excessively trimmed
            negative_power: value between [1, inf) that weights the negative
                values (where the count of a value is less than the target)
                when computing the score for a set of indices to remove.
                    1: will weight them the same as positive values
                    2: will square the values
                See `Balancer._solution_score` for more details
            target_count: override target count for each attribute value. If
                provided, target_quantile is ignored
            target_hard_min: whether or not to require that each attribute
                value have at least the target count after balancing
            algorithm: name of the balancing search algorithm. Supported values
                are ["random", "greedy", "simple"]
        """
        self.attr_name = attribute_name
        self.object_label = object_label
        self.balance_by_object_label = balance_by_object_label
        self.labels_schema = labels_schema
        self.target_quantile = target_quantile
        self.negative_power = negative_power
        self.target_count = target_count
        self.target_hard_min = target_hard_min
        self.algorithm = algorithm

        self._validate()

    def transform(self, src):
        """Modifues the BuilderDataset records by removing records until the
        target attribute is ~roughly~ balanced for each value.

        Args:
            src: a BuilderDataset
        """
        logger.info("Balancing dataset")

        # STEP 1: Get attribute value(s) for every record
        logger.info("Calculating occurrence matrix...")
        (
            occurrence_matrix,
            attribute_values,
            record_idxs,
        ) = self._get_occurrence_matrix(src.records)
        if not attribute_values:
            return

        # STEP 2: Determine target number to remove of each attribute value
        logger.info("Determining target counts...")
        counts = np.sum(occurrence_matrix, axis=1).astype(np.dtype("int"))
        target_count = self._get_target_count(counts)

        # STEP 3: Find the records to keep
        logger.info("Calculating which records to keep...")
        keep_idxs = self._get_keep_idxs(
            occurrence_matrix, counts, target_count
        )

        # STEP 4: Modify the list of records
        logger.info("Filtering records...")
        old_records = src.records
        src.clear()
        for ki in keep_idxs:
            src.add(old_records[record_idxs[ki]])

        logger.info("Balancing of dataset complete")

    def _validate(self):
        specified = {
            "attribute_name": self.attr_name is not None,
            "object_label": self.object_label is not None,
            "labels_schema": self.labels_schema is not None,
            "balance_by_object_label": self.balance_by_object_label,
        }

        acceptable_patterns = [
            # balance attribute
            {
                "attribute_name": True,
                "balance_by_object_label": False,
                "labels_schema": False,
            },
            # balance object label
            {
                "attribute_name": False,
                "object_label": False,
                "balance_by_object_label": True,
                "labels_schema": False,
            },
            # balance schema
            {
                "attribute_name": False,
                "object_label": False,
                "balance_by_object_label": False,
                "labels_schema": True,
            },
        ]

        if not any(
            all(specified[k] == v for k, v in iteritems(pattern))
            for pattern in acceptable_patterns
        ):
            raise ValueError(
                "Pattern of variables specified not allowed: %s\n"
                "Allowed patterns: %s" % (specified, acceptable_patterns)
            )

    def _get_occurrence_matrix(self, records):
        """Computes occurrence of each attribute value for each class.

        Args:
            records: list of `BuilderDataRecord`s

        Returns:
            A: an N x M occurrence matrix counting the number of instances of
                each attribute value in a record, where:
                    N: length of `values`
                    M: number of records that contain the attribute to balance
            values: list of N strings; one for each unique attribute value
            record_idxs: a list of M integers; each being the index into
                `records` for the corresponding column in A, where:
                    A[i, j] = the number of instances of values[i] in
                              records[record_idxs[j]]
        """
        helper_list = self._to_helper_list(records)
        record_idxs = [idx for idx, _ in helper_list]

        A = np.zeros((0, len(helper_list)), dtype=np.dtype("uint32"))
        values = []
        for j, (_, attr_values) in enumerate(helper_list):
            for attr_value in attr_values:
                try:
                    i = values.index(attr_value)
                    A[i, j] += 1
                except ValueError:
                    values.append(attr_value)
                    A = np.vstack(
                        [
                            A,
                            np.zeros(
                                len(helper_list), dtype=np.dtype("uint32")
                            ),
                        ]
                    )
                    i = values.index(attr_value)
                    A[i, j] += 1

        return A, values, record_idxs

    def _to_helper_list(self, records):
        """Recompile the records to a list of counts of each attribute value.

        Args:
            records: list of BuilderDataRecord's

        Returns:
            a list of (`record_id`, `values`) tuples, where `record_id` is the
            integer ID of the corresponding old record, and `values` is a list
            of attribute values for the attribute to be balanced (or, one entry
            per unique object, when using objects)
        """
        if not records:
            return []

        if self.attr_name is not None:
            return self._to_helper_list_attr_name(records)

        if self.balance_by_object_label:
            return self._to_helper_list_object_label(records)

        return self._to_helper_list_schema(records)

    def _to_helper_list_attr_name(self, records):
        """Balancer._to_helper_list when `self.attr_name` is specified."""
        if isinstance(records[0], BuilderImageRecord):
            if self.object_label:
                return self._to_helper_list_image_objects(records)
            return self._to_helper_list_image(records)

        if isinstance(records[0], BuilderVideoRecord):
            if self.object_label:
                return self._to_helper_list_video_objects(records)
            return self._to_helper_list_video(records)

        raise DatasetTransformerError(
            "Unknown record type: {}".format(etau.get_class_name(records[0]))
        )

    def _to_helper_list_object_label(self, records):
        """Balancer._to_helper_list when `self.object_label` is specified
        and `self'attr_name` is None
        """
        if isinstance(records[0], BuilderImageRecord):
            return self._to_helper_obj_label_list_image(records)

        if isinstance(records[0], BuilderVideoRecord):
            return self._to_helper_obj_label_list_video(records)

        raise DatasetTransformerError(
            "Unknown record type: {}".format(etau.get_class_name(records[0]))
        )

    def _to_helper_list_schema(self, records):
        """Balancer._to_helper_list when `self.labels_schema` is specified."""
        self._validate_schema(records)

        if isinstance(records[0], BuilderImageRecord):
            return self._to_helper_list_image_schema(records)

        return self._to_helper_list_video_schema(records)

    def _to_helper_list_image(self, records):
        """Balancer._to_helper_list for image attributes"""
        helper_list = []

        for i, record in enumerate(records):
            labels = record.get_labels()

            for attr in labels.attrs:
                if attr.name == self.attr_name:
                    helper_list.append((i, [attr.value]))
                    break

        return helper_list

    def _to_helper_list_image_objects(self, records):
        """Balancer._to_helper_list for object attributes in images"""
        helper_list = []

        for i, record in enumerate(records):
            labels = record.get_labels()  # ImageLabels
            helper = (i, [])

            for dobj in labels.objects:
                if dobj.label != self.object_label:
                    continue

                for attr in dobj.attrs:
                    if attr.name == self.attr_name:
                        helper[1].append(attr.value)
                        break

            if helper[1]:
                helper_list.append(helper)

        return helper_list

    @staticmethod
    def _to_helper_obj_label_list_image(records):
        """Balancer._to_helper_list for image object labels"""
        helper_list = []

        for i, record in enumerate(records):
            labels = record.get_labels()
            helper = (i, [])

            for dobj in labels.objects:
                helper[1].append(dobj.label)

            if helper[1]:
                helper_list.append(helper)

        return helper_list

    def _to_helper_list_video(self, records):
        """Balancer._to_helper_list for video frame attributes"""
        helper_list = []

        for i, record in enumerate(records):
            labels = record.get_labels()  # VideoLabels
            helper = (i, set())

            for frame_number in labels:
                if (
                    frame_number < record.clip_start_frame
                    or frame_number >= record.clip_end_frame
                ):
                    continue

                frame = labels[frame_number]
                for attr in frame.attrs:
                    if attr.name == self.attr_name:
                        helper[1].add(attr.value)
                        break

            if helper[1]:
                helper = (helper[0], list(helper[1]))
                helper_list.append(helper)

        return helper_list

    def _to_helper_list_video_objects(self, records):
        """Balancer._to_helper_list for object attributes in videos"""
        helper_list = []

        for i, record in enumerate(records):
            labels = record.get_labels()
            NO_ID = "NO_ID"
            helper_dict = defaultdict(set)

            for frame_number in labels:
                if (
                    frame_number < record.clip_start_frame
                    or frame_number >= record.clip_end_frame
                ):
                    continue

                frame = labels[frame_number]
                for dobj in frame.objects:
                    if dobj.label != self.object_label:
                        continue

                    for attr in dobj.attrs:
                        if attr.name == self.attr_name:
                            obj_idx = (
                                dobj.index if dobj.index is not None else NO_ID
                            )
                            helper_dict[obj_idx].add(attr.value)
                            break

            # At this point, the keys of helper dict are unique object indices
            # for objects of type self.object_label. The values are unique
            # attribute values for self.attr_name

            if helper_dict:
                helper = (i, [])
                for s in helper_dict.values():
                    helper[1].extend(s)

                helper_list.append(helper)

        return helper_list

    @staticmethod
    def _to_helper_obj_label_list_video(records):
        """Balancer._to_helper_list for object attributes in videos"""
        helper_list = []

        for i, record in enumerate(records):
            labels = record.get_labels()  # VideoLabels
            NO_ID = "NO_ID"
            helper_dict = defaultdict(set)

            for frame_number in labels:
                if (
                    frame_number < record.clip_start_frame
                    or frame_number >= record.clip_end_frame
                ):
                    continue

                frame = labels[frame_number]
                for dobj in frame.objects:
                    obj_idx = dobj.index if dobj.index is not None else NO_ID
                    helper_dict[obj_idx].add(dobj.label)

            #
            # At this point, the keys of helper dict are unique object indices
            # for objects of type self.object_label. The values are unique
            # attribute values for self.attr_name
            #

            if helper_dict:
                helper = (i, [])
                for s in helper_dict.values():
                    helper[1].extend(s)

                helper_list.append(helper)

        return helper_list

    def _validate_schema(self, records):
        """Checks that `self.labels_schema` and `records` are compatible.

        Args:
            records: list of BuilderDataRecords to be balanced
        """
        for build_rec_cls, schema_cls in self._BUILDER_RECORD_TO_SCHEMA:
            if isinstance(records[0], build_rec_cls) and not isinstance(
                self.labels_schema, schema_cls
            ):
                raise TypeError(
                    "Expected self.labels_schema to be an instance of '%s' "
                    "since builder records are instances of '%s'"
                    % (
                        etau.get_class_name(schema_cls),
                        etau.get_class_name(build_rec_cls),
                    )
                )

            if isinstance(records[0], build_rec_cls):
                break
        else:
            raise DatasetTransformerError(
                "Unknown record type: '%s'" % etau.get_class_name(records[0])
            )

    def _to_helper_list_image_schema(self, records):
        """Balancer._to_helper_list when an ImageLabelsSchema is given."""
        helper_list = []

        for i, record in enumerate(records):
            image_labels = record.get_labels()  # ImageLabels
            helper = (i, [])

            for attr in image_labels.attrs:
                if attr.constant:
                    # Constant attribute
                    if self.labels_schema.is_valid_constant_attribute(attr):
                        helper[1].append(
                            ("constant_attribute", attr.name, attr.value)
                        )
                else:
                    # Frame attribute
                    if self.labels_schema.is_valid_frame_attribute(attr):
                        helper[1].append(
                            ("frame_attribute", attr.name, attr.value)
                        )

            for dobj in image_labels.objects:
                # Object label
                if not self.labels_schema.is_valid_object_label(dobj.label):
                    continue

                for attr in dobj.attrs:
                    if attr.constant:
                        # Object attribute
                        if self.labels_schema.is_valid_object_attribute(
                            dobj.label, attr
                        ):
                            helper[1].append(
                                (
                                    "object_attribute",
                                    dobj.label,
                                    attr.name,
                                    attr.value,
                                )
                            )
                    else:
                        # Object frame attribute
                        if self.labels_schema.is_valid_object_frame_attribute(
                            dobj.label, attr
                        ):
                            helper[1].append(
                                (
                                    "object_frame_attribute",
                                    dobj.label,
                                    attr.name,
                                    attr.value,
                                )
                            )

            if helper[1]:
                helper_list.append(helper)

        return helper_list

    def _to_helper_list_video_schema(self, records):
        """Balancer._to_helper_list when an VideoLabelsSchema is given."""
        helper_list = []

        for i, record in enumerate(records):
            video_labels = record.get_labels()  # VideoLabels
            helper = (i, [])
            helper_dict = defaultdict(set)

            for attr in video_labels.attrs:
                # Video attribute
                if self.labels_schema.is_valid_video_attribute(attr):
                    helper[1].append(
                        ("video_attribute", attr.name, attr.value)
                    )

            for frame_number in video_labels:
                if (
                    frame_number < record.clip_start_frame
                    or frame_number >= record.clip_end_frame
                ):
                    continue

                frame = video_labels[frame_number]
                for attr in frame.attrs:
                    if attr.constant:
                        # Another way to store video attributes
                        if self.labels_schema.is_valid_video_attribute(attr):
                            helper[1].append(
                                ("video_attribute", attr.name, attr.value)
                            )
                    else:
                        # Frame attribute
                        if self.labels_schema.is_valid_frame_attribute(attr):
                            helper[1].append(
                                ("frame_attribute", attr.name, attr.value)
                            )

                for dobj in frame.objects:
                    # Object label
                    if not self.labels_schema.is_valid_object_label(
                        dobj.label
                    ):
                        continue

                    for attr in dobj.attrs:
                        if attr.constant:
                            # Object attribute
                            if self.labels_schema.is_valid_object_attribute(
                                dobj.label, attr
                            ):
                                helper_dict[(dobj.label, dobj.index)].add(
                                    (
                                        "object_attribute",
                                        dobj.label,
                                        attr.name,
                                        attr.value,
                                    )
                                )
                        else:
                            # Object frame attribute
                            if self.labels_schema.is_valid_object_frame_attribute(  # pylint: disable=line-too-long
                                dobj.label, attr
                            ):
                                helper_dict[(dobj.label, dobj.index)].add(
                                    (
                                        "object_frame_attribute",
                                        dobj.label,
                                        attr.name,
                                        attr.value,
                                    )
                                )

            for attr_set in itervalues(helper_dict):
                helper[1].extend(list(attr_set))

            if helper[1]:
                helper_list.append(helper)

        return helper_list

    def _get_target_count(self, counts):
        """Compute the target count that we'd like to balance each value to.

        Args:
            counts: a vector of original counts for each value

        Returns:
            the target value
        """
        if self.target_count:
            return self.target_count
        return int(np.quantile(counts, self.target_quantile))

    def _get_keep_idxs(self, A, counts, target_count):
        """This function chooses the set of records to keep (and remove).

        There's still plenty of potential for testing and improvement here.

        This problem can be posed as::

            minimize |Ax - b|
            subject to:
                x[i] is an element of [0, 1]

        and different algorithms may be substituted in.

        Args:
            A: the occurrence matrix from `Balancer._get_occurrence_matrix`
            counts: the vector of original counts for each value
            target_count: the target value

        Returns:
            a list of integer indices to keep
        """
        b = counts - target_count

        if self.algorithm == "random":
            x = self._random(A, b)
        elif self.algorithm == "greedy":
            x = self._greedy(A, b)
        elif self.algorithm == "simple":
            x = self._simple(A, b)
        else:
            raise ValueError(
                "Unknown balancing algorithm '{}'".format(self.algorithm)
            )

        if self.target_hard_min:
            x = self._add_to_meet_minimum_count(x, A, target_count)

        return np.where(x == 0)[0]

    def _random(self, A, b):
        """A random search algorithm for finding the indices to omit.

        Args:
            A: the occurrence matrix
            b: the target vector to match

        Returns:
            x: the solution vector, where `x[j] == 1` --> omit the j'th record
        """
        best_x = np.zeros(A.shape[1], dtype=np.dtype("int"))
        best_score = self._solution_score(b - np.dot(A, best_x))

        random.seed(1)
        for _ in range(self._NUM_RANDOM_ITER):
            i = random.choice(np.where(best_x == 0)[0])
            cur_x = best_x.copy()
            cur_x[i] = 1
            cur_score = self._solution_score(b - np.dot(A, cur_x))
            if cur_score < best_score:
                best_score = cur_score
                best_x = cur_x

        return best_x

    def _greedy(self, A, b):
        """A greedy search algorithm for finding the indices to omit.

        Args:
            A: the occurrence matrix
            b: the target vector to match

        Returns:
            x: the solution vector, where `x[j] == 1` --> omit the j'th record
        """
        best_x = np.zeros(A.shape[1], dtype=np.dtype("int"))
        best_score = self._solution_score(b - np.dot(A, best_x))
        w = np.where(best_x == 0)[0]

        while len(w) > 0:  # pylint: disable=len-as-condition
            x_matrix = np.zeros((len(best_x), len(w)), dtype=np.dtype("int"))
            for idx, val in enumerate(w):
                x_matrix[:, idx] = best_x
                x_matrix[val, idx] = 1
            Y = np.expand_dims(b, axis=1) - np.dot(A, x_matrix)

            cur_scores = self._solution_score(Y.T)

            i = np.argmin(cur_scores)

            if cur_scores[i] >= best_score:
                break

            best_score = cur_scores[i]
            best_x = x_matrix[:, i]
            w = np.where(best_x == 0)[0]

        return best_x

    @staticmethod
    def _simple(A, b):
        """This algorithm for finding the indices to omit just goes through
        each class and adds records minimally such that the class has count
        equal to the target.

        Args:
            A: the occurrence matrix
            b: the target vector to match

        Returns:
            x: the solution vector, where `x[j] == 1` --> omit the j'th record
        """
        x = np.ones(A.shape[1], dtype="int")
        counts = np.dot(A, x)
        dropped_counts = counts.copy()

        # Go through attributes from lowest to highest count and try to
        # minimally add records to get the target for that attribute
        for attr_idx in np.argsort(counts):
            # We can only decrease the dropped counts for this attribute
            # by changing some 1's to 0's in `x`. Therefore, if
            # `dropped_counts[attr_idx]` is already less than or equal
            # to the target, then do nothing.
            attr_target = b[attr_idx]
            if dropped_counts[attr_idx] <= attr_target:
                continue

            # We can change some set of 1's in `x` to 0's, but not vice
            # versa.  Create an array of the counts for this attribute,
            # for records for which `x` contains a 1.
            dropped_counts_for_attr = A[attr_idx, :] * x

            # Right now, `dropped_counts_for_attr.sum()` would give a
            # a number equal to `dropped_counts[attr_idx]`. We want to
            # find which elements of `dropped_counts_for_attr` can be
            # removed so that `dropped_counts_for_attr.sum()` is equal
            # to `attr_target`. Sort `dropped_counts_for_attr` in
            # descending order and drop elements as long as that won't
            # make the sum go below attr_target.
            new_dropped_counts = dropped_counts[attr_idx]
            indices_to_remove = []
            for record_idx in np.flip(np.argsort(dropped_counts_for_attr)):
                if new_dropped_counts <= attr_target:
                    break

                count = dropped_counts_for_attr[record_idx]
                if count == 0 or new_dropped_counts - count < attr_target:
                    # Don't want to explicitly remove records with 0 counts
                    # for this attribute since it won't help us reach the
                    # object of `attr_target` for this attribute, and may
                    # affect other attributes.
                    continue

                # Remove this count
                indices_to_remove.append(record_idx)
                new_dropped_counts -= count

            # Update `x` and `dropped_counts`
            x[indices_to_remove] = 0
            dropped_counts = np.dot(A, x)

        return x

    def _solution_score(self, vector):
        """Compute the score for a vector (smaller is better). This is a custom
        scoring function that sorta computes the L1 norm for positive values
        and the L<X> norm for negative values where <X> is self.negative_power.

        Larger self.negative_power puts more weight on not reducing the count
        of any attribute values that are already below the target.

        Args:
            vector: `vector of counts` - `target count`

        Returns:
            a relative score value, where smaller --> better
        """
        v_pos = np.maximum(vector, 0)
        v_neg = np.abs(np.minimum(vector, 0))
        vector2 = v_pos + (v_neg ** self.negative_power)
        try:
            return np.sum(vector2, axis=1)
        except np.AxisError:
            return np.sum(vector2)

    @staticmethod
    def _add_to_meet_minimum_count(x, A, target_count):
        """Add more indices to `keep_idxs` so that the count for every
        attribute value is at least equal to `target_count`.

        If for some attribute values, there are fewer than `target_count`
        instances in the whole dataset, every record containing those
        attribute values will be added.

        Args:
            x: an array of shape (M,) containing 1's or 0's, indicating which
                records are being omitted
            A: the N x M occurrence matrix generated by
                `self._get_occurrence_matrix()`
            target_count: an integer giving the desired count for each value

        Returns:
            x_out: an array of shape (M,) containing 1's or 0's, indicating
                records to omit. All entries that were 0 in the input `x` will
                also be 0 in `x_out`, and some entries that were 1 in `x` may
                be 0 in `x_out` as well, such that the total count for each
                value is at least `target_count`
        """
        x_out = x.copy()
        current_counts = np.dot(A, 1 - x_out).astype("int")
        omitted_idxs = np.where(x_out == 1)[0]

        total_counts = np.dot(A, np.ones(x_out.shape)).astype("int")
        # If the total count in the entire dataset for a value is less than
        # `target_count`, then the target for that value should be its total
        # count
        target_vec = np.minimum(total_counts, target_count)

        random.shuffle(omitted_idxs)
        for candidate_idx in omitted_idxs:
            if not (current_counts < target_vec).any():
                # All counts meet the minimum
                break

            additional_counts = A[:, candidate_idx]
            # If keeping this index would increase the counts for any
            # value that doesn't meet the minimum, then keep this index
            if (additional_counts[current_counts < target_vec] > 0).any():
                x_out[candidate_idx] = 0
                current_counts += additional_counts

        return x_out


class SchemaFilter(DatasetTransformer):
    """Dataset transformer that filter all labels in the dataset by the
    provided schema.
    """

    def __init__(
        self,
        schema=None,
        remove_objects_without_attrs=False,
        object_labels_to_filter=None,
        prune_empty=True,
    ):
        """Creates a SchemaFilter instance.

        Args:
            schema: a VideoLabelsSchema or ImageLabelsSchema. By default, no
                filtering will be performed
            remove_objects_without_attrs: whether to remove objects with no
                attributes, after filtering. Use the `object_labels_to_filter`
                argument to control which object labels are filtered. By
                default, this is False
            object_labels_to_filter: an optional list of DetectedObject label
                strings to which to restrict attention when filtering. If None,
                all objects are filtered
            prune_empty: whether to remove records from the dataset whose
                labels are empty after filtering. By default, this is True
        """
        self.schema = schema
        self.remove_objects_without_attrs = remove_objects_without_attrs
        self.object_labels_to_filter = object_labels_to_filter
        self.prune_empty = prune_empty

    def transform(self, src):
        """Filters all records in the given source dataset.

        If this transformer has no schema, no filtering is done.

        Args:
            src: a BuilderImageDataset or BuilderVideoDataset
        """
        if self.schema is None:
            return

        old_records = src.records
        src.clear()
        for record in old_records:
            labels = record.get_labels()

            # Filter by schema
            labels.filter_by_schema(self.schema)

            # Filter objects that don't have attributes
            if self.remove_objects_without_attrs:
                labels.remove_objects_without_attrs(
                    labels=self.object_labels_to_filter
                )

            # Add the filtered record to the new dataset
            if not self.prune_empty or not labels.is_empty:
                record.set_labels(labels)
                src.add(record)


class Clipper(DatasetTransformer):
    """Dataset transformer that extracts video clips from source videos
    according to the specified parameters.
    """

    def __init__(self, clip_len, stride_len, min_clip_len):
        """Creates a Clipper instance.

        Args:
            clip_len: number of frames per clip, must be > 0
            stride_len: stride (step size), must be > 0
            min_clip_len: minimum number of frames allowed, must be > 0 and
                less than clip_len
        """
        self.clip_len = int(clip_len)
        self.stride_len = int(stride_len)
        self.min_clip_len = int(min_clip_len)
        self._validate()

    def transform(self, src):
        """Creates the new record list made of clipped records from the old
        records list.

        Args:
            src: a BuilderVideoDataset
        """
        if not isinstance(src, BuilderVideoDataset):
            raise DatasetTransformerError(
                "`Clipper`s can only operate on `BuilderVideoDataset`s"
            )

        old_records = src.records
        src.clear()
        for record in old_records:
            start_frame = record.clip_start_frame
            while start_frame <= record.clip_end_frame:
                end_frame = start_frame + self.clip_len - 1
                if end_frame > record.clip_end_frame:
                    end_frame = record.clip_end_frame
                    clip_duration = int(end_frame - start_frame + 1)
                    if clip_duration < self.min_clip_len:
                        break

                self._add_clipping(start_frame, end_frame, record, src.records)
                start_frame += self.stride_len

    @staticmethod
    def _add_clipping(start_frame, end_frame, old_record, records):
        new_record = old_record.copy()
        new_record.clip_start_frame = start_frame
        new_record.clip_end_frame = end_frame
        records.append(new_record)

    def _validate(self):
        if (
            (self.clip_len < 1)
            or (self.stride_len < 1)
            or (self.min_clip_len < 1)
            or (self.min_clip_len > self.clip_len)
        ):
            raise DatasetTransformerError("Invalid Clipper args found")


class EmptyLabels(DatasetTransformer):
    """Dataset transformer that assigns empty labels to all records."""

    def transform(self, src):
        """Assigns empty labels to all records.

        Args:
            src: a BuilderDataRecord
        """
        if not src:
            return

        labels_cls = src.records[0].get_labels().__class__
        for record in src:
            record.set_labels(labels_cls())


class Merger(DatasetTransformer):
    """Dataset transformer that merges another dataset into the existing
    dataset.
    """

    def __init__(self, dataset_builder, prepend_dataset_name=True):
        """Creates a Merger instance.

        Args:
            dataset_builder: a LabeledDatasetBuilder instance for the dataset
                to be merged with the existing one
            prepend_dataset_name: This flag enables an option to prepend both
                the data and labels filepaths with the folder name containing
                the original files. E.g. /path/to/dataset001/data/001-123.mp4
                =>/new_directory/data/dataset001_001-123
        """
        self._builder_dataset_to_merge = dataset_builder.builder_dataset
        self.prepend_dataset_name = prepend_dataset_name

    def transform(self, src):
        """Merges the given BuilderDataset into this instance.

        Args:
            src: a BuilderDataset
        """
        if self._builder_dataset_to_merge.record_cls != src.record_cls:
            raise DatasetTransformerError(
                "BuilderDatasets have different record_cls types: "
                "src.record_cls = %s, to_merge.record_cls = %s"
                % (
                    etau.get_class_name(src.record_cls),
                    etau.get_class_name(
                        self._builder_dataset_to_merge.record_cls
                    ),
                )
            )

        if self.prepend_dataset_name:
            for record in self._builder_dataset_to_merge.records:
                base = get_dataset_name(record.data_path)
                record.prepend_to_name(prefix=base)

        src.add_container(self._builder_dataset_to_merge)


class PrependDatasetNameToRecords(DatasetTransformer):
    """Dataset transformer that prepends the dataset name followed by an
    underscore to all data and label files in the dataset.

    Example:
        `mydataset/data/vid.mp4` ==> `mydataset/data/mydataset_vid.mp4`
    """

    def transform(self, src):
        """Prepends the dataset name to all records in the dataset.

        Args:
            src: a BuilderDataset
        """
        for record in src.records:
            base = get_dataset_name(record.data_path)
            record.prepend_to_name(prefix=base)


class FilterByFilename(DatasetTransformer):
    """Dataset transformer that filters data from a dataset using a filename
    blacklist.
    """

    def __init__(self, filename_blacklist):
        """Creates a FilterByFilename instance.

        Args:
            filename_blacklist: a list of data filenames to filter out
        """
        self._files_to_remove = set(filename_blacklist)

    def transform(self, src):
        """Removes data with filenames that match the blacklist.

        Args:
            src: a BuilderDataset
        """
        src.cull_with_function(
            "data_path",
            lambda path: os.path.basename(path) not in self._files_to_remove,
        )


class FilterByFilenameRegex(DatasetTransformer):
    """Dataset transformer that filters data from a dataset using a regex
    blacklist for filenames.
    """

    def __init__(self, filename_regex_blacklist):
        """Creates a FilterByFilenameRegex instance.

        Args:
            filename_regex_blacklist: a list of data filename regexes to filter
                out
        """
        self._regex_blacklist = [
            re.compile(s) for s in filename_regex_blacklist
        ]

    def transform(self, src):
        """Removes data with filenames that match the regex blacklist.

        Args:
            src: a BuilderDataset
        """
        src.cull_with_function(
            "data_path",
            lambda path: not any(
                rgx.match(os.path.basename(path))
                for rgx in self._regex_blacklist
            ),
        )


class FilterByPath(DatasetTransformer):
    """Dataset transformer that filters data from a dataset using a full path
    blacklist.
    """

    def __init__(self, full_path_blacklist):
        """Creates a FilterByPath instance.

        Args:
            full_path_blacklist: a list of full paths to data files to filter
                out
        """
        self._paths_to_remove = {
            os.path.abspath(path) for path in full_path_blacklist
        }

    def transform(self, src):
        """Removes data with full paths that match the blacklist.

        Args:
            src: a BuilderDataset
        """
        src.cull_with_function(
            "data_path",
            lambda path: os.path.abspath(path) not in self._paths_to_remove,
        )


class DatasetTransformerError(Exception):
    """Exception raised when there is an error in a DatasetTransformer"""

    pass
