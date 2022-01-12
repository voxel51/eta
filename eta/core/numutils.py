"""
Core numeric and computational utilities.

Copyright 2017-2022, Voxel51, Inc.
voxel51.com
"""
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from future.utils import iteritems

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict
import operator

import numpy as np


def is_close(a, b, rel_tol=1e-09, abs_tol=0):
    """Determines whether two numbers are nearly equal.

    The maximum of the relative-based and absolute tolerances is used to test
    equality.

    This function is taken from `math.isclose` in Python 3 but is explicitly
    implemented here for Python 2 compatibility.

    Args:
        a: a number
        b: a number
        rel_tol: a relative tolerance to use when testing equality. By default,
            this is 1e-09
        abs_tol: an absolute tolerance to use when testing equality. By
            default, this is 0

    Returns:
        True/False whether the numbers are close
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def round_to_even(x):
    """Rounds to the nearest even number."""
    return int(round(x / 2.0) * 2)


def safe_divide(num, denom):
    """Divides the two numbers, avoiding ZeroDivisionError.

    Args:
        num: numerator
        denom: demoninator

    Returns:
        the quotient, or 0 if the demoninator is 0
    """
    try:
        return num / denom
    except ZeroDivisionError:
        return 0


class Accumulator(object):
    """A histogram-like class that supports counting arbitrary hashable
    objects.

    Both weighted and unweighted counts are supported.

    For classical histogram needs, `numpy.histogram` and related functions are
    likely more efficient. This class lets you accumulate entries of any type
    and is hence slower but more general.
    """

    def __init__(self):
        """Creates an Accumulator instance."""
        self._weights = defaultdict(float)
        self._counts = defaultdict(int)

    def add(self, thing, weight=None):
        """Add `thing` to the accumulator.

        Args:
            thing: anything hashable
            weight: an optional weight. By default, this is 1
        """
        if weight is None:
            weight = 1

        self._weights[thing] += weight
        self._counts[thing] += 1

    def add_all(self, things, weights=None):
        """Adds all `thing`s in the iterable to the accumulator.

        Args:
            things: an iterable of things
            weights: an optional iteratable of weights. By default, each thing
                is given a weight of 1
        """
        if weights:
            for thing, weight in zip(things, weights):
                self.add(thing, weight=weight)
        else:
            for thing in things:
                self.add(thing)

    def get_count(self, thing):
        """Gets the count of `thing`."""
        return self._counts[thing]

    def get_weight(self, thing):
        """Gets the weight of `thing`."""
        return self._weights[thing]

    def get_average_weight(self, thing):
        """Gets the average weight of `thing`."""
        count = self.get_count(thing)
        return self.get_weight(thing) / count if count else None

    def argmax(self, weighted=True):
        """Returns the `thing` with the highest count/weight.

        Args:
            weighted: whether to return the entry with the highest weight
                (True) or count (False). By default, this is True

        Returns:
            the `thing` with the highest count/weight
        """
        return self.max(weighted=weighted)[0]

    def max(self, weighted=True):
        """Returns the tuple of (thing, count/weight) for the `thing` with the
        highest count/weight.

        Args:
            weighted: whether to return the entry with the highest weight
                (True) or count (False). By default, this is True

        Returns:
            the (thing, count/weight) for the `thing` with the highest
                count/weight
        """
        vals = self._weights if weighted else self._counts
        return max(iteritems(vals), key=operator.itemgetter(1))


class GrowableArray(object):
    """A class for building a numpy array from streaming data."""

    def __init__(self, rowlen):
        """Creates a GrowableArray instance.

        Args:
            rowlen: the desired length of each row
        """
        self.rowlen = rowlen
        self._data = []

    def update(self, row):
        """Add row to array."""
        if len(row) != self.rowlen:
            raise GrowableArrayError(
                "Expected row length of %d, but found %d"
                % (self.rowlen, len(row))
            )

        for r in row:
            self._data.append(r)

    def finalize(self):
        """Return numpy array."""
        return np.reshape(
            self._data, newshape=(len(self._data) // self.rowlen, self.rowlen),
        )


class GrowableArrayError(Exception):
    """Exception raised when an invalid GrowableArray is encountered."""

    pass
