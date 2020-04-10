"""
Core methods for splitting iterables into subsets.

Copyright 2017-2020 Voxel51, Inc.
voxel51.com

Matthew Lightman, matthew@voxel51.com
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
import random

import numpy as np

import eta.core.utils as etau


logger = logging.getLogger(__name__)


def round_robin_split(iterable, split_fractions=None):
    """Traverses the iterable in order and assigns items to samples in order,
    until a given sample has reached its desired size.

    If a random split is required, this function is not recommended unless your
    items are already randomly ordered.

    Args:
        iterable: a finite iterable
        split_fractions: an optional list of split fractions, which should sum
            to 1. By default, [0.5, 0.5] is used

    Returns:
        sample_lists: a list of lists, of the same length as `split_fractions`.
            Each sub-list contains items from the original iterable
    """
    split_fractions = _validate_split_fractions(split_fractions)

    # Initial estimate of size of each sample
    item_list = list(iterable)
    sample_sizes = [int(frac * len(item_list)) for frac in split_fractions]

    # `n` is the total number of items that will be divided into samples.
    # `n` may be less than len(item_list) if sum(split_fractions) < 1.
    n = int(np.round(len(item_list) * sum(split_fractions)))

    if n == 0:
        return [[] for _ in sample_sizes]

    # Calculate exact size of each sample, making sure the sum of the samples'
    # sizes is equal to `n`
    remainder = n - sum(sample_sizes)
    num_to_add = int(remainder / len(sample_sizes))
    for idx, _ in enumerate(sample_sizes):
        sample_sizes[idx] += num_to_add
    remainder = n - sum(sample_sizes)
    for idx, _ in enumerate(sample_sizes):
        if idx < remainder:
            sample_sizes[idx] += 1

    assert sum(sample_sizes) == n, (sum(sample_sizes), n)

    # Iterate over items and add them to the appropriate sample
    sample_lists = [[] for _ in sample_sizes]
    sample_full = [sample_size == 0 for sample_size in sample_sizes]
    current_sample_idx = min(
        idx for idx, sample_size in enumerate(sample_sizes) if sample_size > 0
    )
    for item in item_list:
        sample_lists[current_sample_idx].append(item)
        curr_sample_size = len(sample_lists[current_sample_idx])
        if curr_sample_size >= sample_sizes[current_sample_idx]:
            sample_full[current_sample_idx] = True

        if all(sample_full):
            break

        current_sample_idx = _find_next_available_idx(
            current_sample_idx, sample_full
        )

    return sample_lists


def random_split_exact(iterable, split_fractions=None):
    """Randomly splits items into multiple sample lists according to the given
    split fractions.

    The number of items in each sample list will be given exactly by the
    specified fractions.

    Args:
        iterable: a finite iterable
        split_fractions: an optional list of split fractions, which should sum
            to 1. By default, [0.5, 0.5] is used

    Returns:
        sample_lists: a list of lists, of the same length as `split_fractions`.
            Each sub-list contains items from the original iterable
    """
    split_fractions = _validate_split_fractions(split_fractions)

    shuffled = list(iterable)
    random.shuffle(shuffled)

    return _split_in_order(shuffled, split_fractions)


def random_split_approx(iterable, split_fractions=None):
    """Randomly splits items into multiple sample lists according to the given
    split fractions.

    Each item is assigned to a sample list with probability equal to the
    corresponding split fraction.

    Args:
        iterable: a finite iterable
        split_fractions: an optional list of split fractions, which should sum
            to 1. By default, [0.5, 0.5] is used

    Returns:
        sample_lists: a list of lists, of the same length as `split_fractions`.
            Each sub-list contains items from the original iterable
    """
    split_fractions = _validate_split_fractions(split_fractions)

    sample_lists = [[] for _ in split_fractions]

    cum_frac = np.cumsum(split_fractions)
    for item in iterable:
        idx = np.searchsorted(cum_frac, random.random())
        if idx < len(sample_lists):
            sample_lists[idx].append(item)

    return sample_lists


def split_in_order(iterable, split_fractions=None):
    """Splits items into multiple sample lists according to the given split
    fractions.

    The items are partitioned into samples in order according to their position
    in the input sample. If a random split is required, this function is not
    recommended unless your items are already randomly ordered.

    Args:
        iterable: a finite iterable
        split_fractions: an optional list of split fractions, which should sum
            to 1. By default, [0.5, 0.5] is used

    Returns:
        sample_lists: a list of lists, of the same length as `split_fractions`.
            Each sub-list contains items from the original iterable
    """
    split_fractions = _validate_split_fractions(split_fractions)

    return _split_in_order(list(iterable), split_fractions)


def _split_in_order(item_list, split_fractions):
    n = len(item_list)
    cum_frac = np.cumsum(split_fractions)
    cum_size = [int(np.round(frac * n)) for frac in cum_frac]
    sample_bounds = [0] + cum_size

    sample_lists = []
    for begin, end in zip(sample_bounds, sample_bounds[1:]):
        sample_lists.append(item_list[begin:end])

    return sample_lists


def _validate_split_fractions(split_fractions=None):
    if not split_fractions:
        split_fractions = [0.5, 0.5]

    negative_fractions = [f for f in split_fractions if f < 0]
    if negative_fractions:
        raise ValueError(
            "Split fractions must be non-negative; found negative values: %s"
            % str(negative_fractions)
        )

    if sum(split_fractions) > 1.0:
        raise ValueError(
            "Sum of split fractions must be <= 1.0; found sum(%s) = %f"
            % (split_fractions, sum(split_fractions))
        )

    return split_fractions


def _find_next_available_idx(idx, unavailable_indicators):
    for next_idx in range(idx + 1, len(unavailable_indicators)):
        if not unavailable_indicators[next_idx]:
            return next_idx

    for next_idx in range(idx + 1):
        if not unavailable_indicators[next_idx]:
            return next_idx

    return None


class SplitMethods(etau.FunctionEnum):
    """Enum of supported methods for splitting iterables according to split
    fractions.

    By convention, all methods should follow the syntax
    `fcn(iterable, split_fractions=None) -> list`.
    """

    ROUND_ROBIN = "round_robin"
    RANDOM_EXACT = "random_exact"
    RANDOM_APPROX = "random_approx"
    IN_ORDER = "in_order"

    _FUNCTIONS_MAP = {
        "round_robin": round_robin_split,
        "random_exact": random_split_exact,
        "random_approx": random_split_approx,
        "in_order": split_in_order,
    }
