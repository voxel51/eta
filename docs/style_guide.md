# ETA Style Guide

ETA is implemented in Python, and we require all contributed code to adhere to
our style. Our priority is *consistency*, so that developers can quickly ingest
and understand the entire ETA codebase. *When in doubt, follow the existing
style in the module you are contributing to.*

We mostly follow the [Google Python style](
https://github.com/google/styleguide/blob/gh-pages/pyguide.md), so please
review it before contributing.

Here are some highlights of our Python style:

- Maximum line length is **79 characters**, with the exception of long URLs
that cannot be split

- Indent your code with **4 spaces**. That is, **no tabs**!

- Leave two blank lines between top-level definitions, and one blank line
between class method definitions

- Unlike the official Google style, we always use single quotes `'''` for
    docstrings, and we prefer double quotes `"` for regular strings, although
    it is okay to use the single quote `'` on a string to avoid the need to
    escape double quotes within the string

- Imports should always be on separate lines at the top of the file, just after
any module comments and doc strings. Imports should be grouped by type with
one space between each group, with the groups sorted in order of most generic
to least generic
    * future import block for Python 2/3 compatibility
    * standard library imports
    * third-party imports
    * application-specific imports

For ETA-library imports, we import modules as `etax`, where `x` is the first
letter of the module imported. If necessary, we use `etaxy` to disambiguate
between two modules that start with the same letter. We also allow direct
importing of (a small number of) attributes into the local namespace at the
developer's discretion.

```python
import eta.core.image as etai
from eta.core.serial import Serializable
import eta.core.video as etav
```

Within each group, imports should be sorted alphabetically by full package
path, ignoring `from` and `import`:

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import os
import sys

import cv2
import numpy as np

import eta.core.image as etai
from eta.core.serial import Serializable
import eta.core.video as etav
```

Long imports should be implemented with hanging indentation:

```python
from eta.core.features import VideoFramesFeaturizer, \
                              VideoFramesFeaturizerConfig
```

- Names should follow the conventions

```python
module_name, package_name, ClassName, method_name, ExceptionName,
function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name,
function_parameter_name, local_var_name
```

- Use `@todo` to mark todo items in the source

- If a class inherits from no other base classes, explicitly inherit from
  `object`

- Follow standard typographic rules for spaces around punctuation except for
colons, which should only have one space rather than two.

```python
# YES!
spam(ham[1], {eggs: 2}, [])

def complex(real, imag=0.0):
    return magic(r=real, i=imag)

foo = 1000  # comment
long_name = 2  # comment that should not be aligned

dictionary = {
    'foo': 1,
    'long_name': 2,
}
```

```python
# NO!
spam( ham[ 1 ], { eggs: 2 }, [ ] )

def complex(real, imag = 0.0):
    return magic(r = real, i = imag)

foo       = 1000  # comment
long_name = 2     # comment that should not be aligned

dictionary = {
    'foo'      : 1,
    'long_name': 2,
}
```

- All non-trivial public module/class methods should have docstrings describing
their behavior, inputs, outputs, and exceptions (when appropriate)

```python
def parse_object(d, key, cls, default=None):
    '''Parses an object attribute.

    Args:
        d: a JSON dictionary
        key: the key to parse
        cls: the class of the d[key]
        default: default value if key is not present

    Returns:
        an instance of cls

    Raises:
        ConfigError: if no default value was provided and the key was
            not present in the dictionary.
  '''
  pass
```

- Indentation and spacing around punctuation follows the general pep8
guidelines, including [the note](
https://www.python.org/dev/peps/pep-0008/#indentation) that hanging indents may
be aligned to other than 4-spaces. The highlights of these guidelines are
below.

```python
# Yes!
# Aligned with opening delimiter.
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

# More indentation included to distinguish this from the rest.
def long_function_name(
        var_one, var_two, var_three,
        var_four):
    print(var_one)

# Hanging indents should add a level.
foo = long_function_name(
    var_one, var_two,
    var_three, var_four)

# Hanging indents *may* be indented to other than 4 spaces.
# Note that this is option in the pep8 spec and we follow it.
# Use your human judgement for when to leverage it.
foo = long_function_name(
  var_one, var_two,
  var_three, var_four)
```

```python
# No!

# Arguments on first line forbidden when not using vertical alignment.
foo = long_function_name(var_one, var_two,
    var_three, var_four)

# Further indentation required as indentation is not distinguishable.
def long_function_name(
    var_one, var_two, var_three,
    var_four):
    print(var_one)
```


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
