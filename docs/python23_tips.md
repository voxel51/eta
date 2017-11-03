# Python 2/3 compatibility

This document:

* explains how to update existing Python 2 code to make it Python 3 compatible

* lists common idioms for writing code that is Python 2/3 compatible


## Necessary packages

```shell
pip install six
pip install future
```

## Updating existing Python 2 code

See what changes are required for Python 3 compatibility:

```shell
# single file
futurize <file>

# multiple files
futurize *.py
```

Apply changes:

```shell
futurize --stage1 -w <file>
futurize --stage2 -w <file>
```


## Python 2/3 compatible idioms

See [these compatible idioms](
http://python-future.org/compatible_idioms.html#compatible-idioms) for
explanation

#### Essentials

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
```

* `absolute_import`: `import foo` is an absolute import (searches `sys.path`)
rather than a relative import (searches current directory)

* `division`: division returns floating point values

* `print_function`: `print` is a function: `print("hello")`

* `unicode_literals`: undecorated strings like `"text"` are unicode by default.
To create a binary string, use `b"binary"`

#### Strings

```python
from __future__ import unicode_literals

# text strings
s = "text"

# binary strings
b = b"binary"
```

#### Iterable objects

```python
from six import itervalues

for key in itervalues(heights):
    ...

# not these!
# for value in heights.itervalues():  # Python 2
# for value in heights.values():  # Python 2/3

```

```python
from six import iteritems

for key, value in iteritems(heights):
    ...

# not these!
# for key, value in heights.iteritems():  # Python 2
# for key, value in heights.items():  # Python 2/3
```


#### Iterables

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
from io import open

f = open(pathname, "rb")   # f.read() will return bytes
f = open(pathname, "rt")   # f.read() will return unicode
```

#### Exceptions

```python
raise ValueError("msg")

# not this!
#raise ValueError, "msg"
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

https://pythonhosted.org/six

https://docs.python.org/3/howto/pyporting.html

http://python-future.org/automatic_conversion.html

http://python-future.org/futurize_cheatsheet.html

http://python-future.org/compatible_idioms.html#compatible-idioms
