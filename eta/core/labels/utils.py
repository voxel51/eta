'''
TODO

Copyright 2017-2020, Voxel51, Inc.
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


logger = logging.getLogger(__name__)


# identifier when matching a value saying that anything should match this value
MATCH_ANY = "*"


def is_true(thing_to_test):
    '''Cast an arg from client to native boolean'''
    if type(thing_to_test) == bool:
        return thing_to_test
    elif type(thing_to_test) == int or type(thing_to_test) == float:
        return thing_to_test == 1
    elif type(thing_to_test) == str:
        return thing_to_test.lower() == 'true'
    else:
        # make a best guess? hopefully you should never get here
        return bool(thing_to_test)
