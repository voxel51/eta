# ETA Python Style Guide

ETA is implemented in Python, and we require all contributed code to adhere to
our style. Our priority is _consistency_, so that developers can quickly ingest
and understand the entire ETA codebase. _When in doubt, follow the existing
style in the module you are contributing to._

We use Black (deterministic) auto-formatting, and Pylint as pre-commit hooks.
Installing ETA with the `-d` (dev) flag automatically configures these hooks.
Much of the style guide is automatically handled by Black. See the
[linting guide](https://github.com/voxel51/eta/blob/develop/docs/linting_guide.md)
for more information.

Here are some highlights of our Python style:

-   Maximum line length is **79 characters**, with the exception of long URLs
    that cannot be split

-   Indent your code with **4 spaces**. That is, **no tabs**!

-   Leave two blank lines between top-level definitions, and one blank line
    between class method definitions

-   Imports should always be on separate lines at the top of the file, just
    after any module comments and doc strings. Imports should be grouped by
    type with one space between each group, with the groups sorted in order of
    most generic to least generic _ future import block for Python 2/3
    compatibility _ standard library imports _ third-party imports _
    application-specific imports

-   When encountering a pylint error during a commit that cannot be addressed
    for whatever reason, add an inline comment `# pylint: disable=rule` where
    `rule` is the rule in question. See the
    [linting guide](https://github.com/voxel51/eta/blob/develop/docs/linting_guide.md)
    for more information.

For ETA-library imports, we import modules as `etax`, where `x` is the first
letter of the module imported. If necessary, we use `etaxy` to disambiguate
between two modules that start with the same letter. We also allow direct
importing of (a small number of) attributes into the local namespace at the
developer's discretion.

```py
import eta.core.image as etai
from eta.core.serial import Serializable
import eta.core.video as etav
```

Within each group, imports should be sorted alphabetically by full package
path, ignoring `from` and `import`:

```py
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

-   Names should follow the conventions

```py
module_name, package_name, ClassName, method_name, ExceptionName,
function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name,
function_parameter_name, local_var_name
```

-   Use `@todo` to mark todo items in the source

-   If a class inherits from no other base classes, explicitly inherit from
    `object`

-   Follow standard typographic rules for spaces around punctuation except for
    colons, which should only have one space rather than two.

-   All non-trivial public module/class methods should have docstrings
    describing their behavior, inputs, outputs, and exceptions (when
    appropriate)

```py
def parse_object(d, key, cls, default=None):
    """Parses an object attribute.

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
    """
    pass
```

## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
