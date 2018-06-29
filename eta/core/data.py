'''
Core data structures for representing data and containers of data.

Copyright 2017-2018, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
Jason Corso, jason@voxel51.com
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


from eta.core.serial import Container


class DataContainer(Container):
    '''Base class for containers that store lists of data instances.

    This class should not be instantiated directly. Instead a subclass should
    be created for each type of data to be stored.

    By default, DataContainer subclasses embed their class names and
    underlying data instance class names in their JSON representations, so
    data containers can be read reflectively from disk.

    Examples:
        ```
        from eta.core.data import DataContainer
        from eta.core.geometry import LabeledPointContainer

        tags = LabeledPointContainer(...)
        tags.write_json("tags.json")

        tags2 = DataContainer.from_json("tags.json")
        print(tags2.__class__)  # LabeledPointContainer, not DataContainer
        ```

    Attributes:
        <data>: a list of data instances. The field name <data> is specified by
            the `_ELE_ATTR` member, and the class of the data instances is
            specified by the `_ELE_CLS` member
    '''

    @property
    def _data(self):
        '''Convenient access to the list of data instances stored independent
        of what the name of that attribute is.
        '''
        return self._elements
