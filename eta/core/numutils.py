'''
Core numeric and computational utilities.

Copyright 2017, Voxel51, LLC
voxel51.com

Jason Corso, jjc@voxel51.com
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

import operator

import numpy as np


def is_close(a, b, rel_tol=1e-09, abs_tol=0):
    '''Determines whether two numbers are nearly equal.

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
    '''
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def round_to_even(x):
    '''Rounds to the nearest even number.'''
    return int(round(x / 2.0) * 2)


class Accumulator(object):
    '''Accumulates counts of entries, like a histogram.  Then provides
    functions for extracting properties over that.  Inputs can be anything
    hashable.

    For classical histogram needs, the numpy histogram functions are probably
    better.  This class lets you accumulate entries of any type and is hence
    slower but more general.
    '''

    def __init__(self):
        '''Initialize the accumulator.'''
        self.data = {}

    def __str__(self):
        '''Renders the accumulator.'''
        return self.data.__str__()

    def add(self, thing):
        '''Add `thing` to the accumulator.

        Args:
            thing: anything hashable

        Returns:
            Count for entry `thing` after added.
        '''
        if thing in self.data:
            self.data[thing] += 1
        else:
            self.data[thing] = 1
        return self.data[thing]

    def argmax(self):
        '''Return the entry with the highest count.'''
        return self.max()[0]

    def max(self):
        '''Return a tuple of (entry, count) for the entry with the highest
        count.
        '''
        return max(self.data.items(), key=operator.itemgetter(1))


class GrowableArray(object):
    '''A class for building a numpy array from streaming data.'''

    def __init__(self, rowlen):
        '''Creates a GrowableArray.

        Args:
            rowlen: the desired length of each row
        '''
        self.data = []
        self.rowlen = rowlen

    def update(self, row):
        '''Add row to array.'''
        if len(row) != self.rowlen:
            raise GrowableArrayError(
                "Expected row length of %d, but found %d" % (
                    self.rowlen, len(row)))

        for r in row:
            self.data.append(r)

    def finalize(self):
        '''Return numpy array.'''
        return np.reshape(
            self.data,
            newshape=(len(self.data) // self.rowlen, self.rowlen),
        )


class GrowableArrayError(Exception):
    '''Exception raised when an invalid GrowableArray is encountered.'''
    pass
