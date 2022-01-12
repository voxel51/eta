# ETA Python 2/3 Compatibility Guide

The ETA codebase supports both Python 2 and 3. This document:

* explains how to update existing Python 2 code to make it Python 3 compatible

* lists common idioms for writing code that is Python 2/3 compatible


## Necessary packages

```shell
pip install future
```

## Updating existing Python 2 code

* See what changes are required for Python 3 compatibility:

```shell
# single file
futurize -ua <file>

# multiple files
futurize -ua *.py
```

* Apply changes:

```shell
# Perform safe fixes
futurize -ua --stage1 -w <file>

# Add any dependencies on future package
futurize -ua --stage2 -w <file>
```

* Manually mark any binary strings with `b''`. Using the `futurize -a` option
automatically interprets all strings as unicode unless explicility specified

* Run tests on Python 2 and 3!

#### Style conventions

* Replace all `__future__` and `builtins` imports added to the module by
 `futurize` with the following block:

```python
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
```

By convention, we add this block to *every* module, even if some of the imports
aren't specified used. These imports clearly state our intention to support
both Python 2 and 3.

* Unless the module specifically uses a standard library function that was
renamed, remove the following lines from the post-`futurize`-ed module:

```
from future import standard_library
standard_library.install_aliases()
```

This code looks ugly at the top of a module!


## Python 2/3 compatible idioms

See [these compatible idioms](
http://python-future.org/compatible_idioms.html#compatible-idioms) for more
details, but here are the basic idioms to follow:

#### Essentials

```python
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
```

* `absolute_import`: `import foo` is an absolute import (searches `sys.path`)
rather than a relative import (searches current directory)

* `division`: division returns floating point values

* `print_function`: `print` is a function: `print("hello")`

* `unicode_literals`: undecorated strings like `"text"` are unicode by default.
To create a binary string, use `b"binary"`

* `builtins`: use the Python 3 versions of the builtin functions
(`str`, `range`, `open`, etc)

* the pylint directives are included to prevent `pylint` from generating
superfluous warnings

#### Strings and bytes

In Python 2 you could use the `str` type for both text and binary data.
However, in Python 3 text and binary data distinct types that cannot blindly
be mixed together. Indeed:

```python
# Python 2
type("asdf")  # 'str'
type(u"asdf")  # 'unicode'
type(b"asdf")  # 'str'

# Python 3
type("asdf")  # 'str'
type(u"asdf")  # 'str'
type(b"asdf")  # 'bytes'
```

To write Python 2/3 compatible strings, we use `__future__` to force all
unspecified string literals to be unicode:

```python
from __future__ import unicode_literals

# text
t = "text"  # interpreted as u"text"

# binary
b = b"binary"
```

The text vs binary string distinction can be especially tricky when ETA code
interfaces with external libraries written in Python 2. Therefore, it is
recommended to explicitly convert *text strings* to Python 3 style using:

```python
from builtins import str

s = str(text_string_from_python2_code)
```

To check if an object is a Python 2/3 text string, you can use:

```python
import six

is_str = isinstance(string, six.string_types)
```

#### Iterable objects

```python
from future.utils import itervalues

for key in itervalues(heights):
    ...

# not these!
# for value in heights.itervalues():  # Python 2
# for value in heights.values():      # Python 2/3
```

```python
from future.utils import iteritems

for key, value in iteritems(heights):
    ...

# not these!
# for key, value in heights.iteritems():  # Python 2
# for key, value in heights.items():      # Python 2/3
```

#### Objects

```python
from builtins import object


class Iterator(object):
    def __next__(self):  # use Python 3 interface
        ...
```

#### Ranges

```python
from builtins import range

# don't explicitly construct list
for i in range(10 ** 8):
    ...

# do construct list
mylist = list(range(5))
```

#### Files

```python
from builtins import open

f = open(pathname, "rb")  # f.read() will return bytes
f = open(pathname, "rt")  # f.read() will return unicode
```

#### Exceptions

```python
raise ValueError("msg")

# not this!
# raise ValueError, "msg"
```

```python
try:
    ...
except ValueError as e:
    ...

# not this!
# except ValueError, e:
```


## References

https://pypi.python.org/pypi/future

https://docs.python.org/3/howto/pyporting.html

http://python-future.org/automatic_conversion.html

http://python-future.org/futurize_cheatsheet.html

http://python-future.org/unicode_literals.html

http://python-future.org/compatible_idioms.html#compatible-idioms


## Copyright

Copyright 2017-2022, Voxel51, Inc.<br>
voxel51.com
