# ETA Core Developer's Guide

This document describes best practices for contributing to the core ETA
infrastructure. See `modules_dev_guide.md` to learn how to contribute modules
to ETA, which are more general and may even live outside of this codebase.


## Repository structure

We use the [GitFlow branching model](
https://datasift.github.io/gitflow/IntroducingGitFlow.html) for ETA.
Thus our repositories have protected `master` and `develop` branches, a
temporary *release branch* when we are ready to deploy a new release, and
multiple unprotected *feature branches* for work on new features. You can read
more about branching workflows in general [here](
https://git-scm.com/book/en/v2/Git-Branching-Branching-Workflows).

The `master` branch is the latest stable release of ETA. It is protected, and
it is only merged from a release branch. Each merge corresponds to a new ETA
release and is tagged with a version number. The one exception to this rule
are *hotfix branches*, which are directly merged into `master` if an emergency
bug fix is required.

The `develop` branch is the bleeding edge (mostly) stable version of ETA. It
is also protected and hence directly committing to it is not allowed.
Instead, when a feature is ready to be integrated, we open a pull request on
`develop`, which initiates a code chat (we prefer "code chat" to "code review",
since this should be a friendly endeavor!) where we discuss the changes and
ultimately merge them into `develop`.

A *release branch* is created from `develop` when we are ready to make a new
release. Only bugfixes (not new features) are committed to a release branch.
When the release is ready, the branch is merged into `master` (and back into
`develop`) and then deleted. The release is done!

*Feature branches* are where most of the development work is done. They are
unprotected, collaborative spaces to develop new features. When a new feature
is ready for deployment, a pull request is made to `develop`.


## Development workflow

Your typical workflow when contributing a new feature to ETA will be:

```shell
git checkout develop
git checkout -b <new_feature_branch>
git push -u origin <new_feature_branch>
# WORK
pylint <changed_files>
pep8 <changed_files>
# ADDRESS LINT OUTPUT
git status -s
git add <changed_files>
git commit -m "message describing your changes"
# MORE WORK, LINTING, AND COMMITS
# PULL REQUEST
# CODE CHAT AND DISCUSSION
# MORE WORK, LINTING, AND COMMITS
# PULL REQUEST APPROVED AND MERGED
git branch -d <new_feature_branch>
```

Note that it is best practice to commit *often* in small, logical chunks rather
than combining multiple changes into a single commit.


## Python 2 and 3 compatibility

ETA supports both Python 2 and 3 via the `future` package. Therefore, we
include the following imports at the top of each module:

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
```

By convention, we add this block to *every* module, even if some of the imports
aren't used. These imports clearly state our intention to support both Python 2
and 3.

See `python23_tips.md` for more tips on writing cross-compatible code.


## Style guide

ETA is implemented in Python, and we require all contributed code to adhere to
our style. Our priority is *consistency*, so that developers can quickly ingest
and understand the entire ETA codebase. When in doubt, follow the existing style
in the module you are contributing to.

Generally, we follow the [Google Python style](
https://google.github.io/styleguide/pyguide.html), so please review it before
contributing.

Here are some highlights of our Python style:

- Maximum line length is **80 characters**, with the exception of long URLs that
  cannot be split

- Indent your code with **4 spaces**. That is, **no tabs**!

- Leave two blank lines between top-level definitions, and one blank line
  between class method definitions

- Imports should always be on separate lines at the top of the file, just after
  any module comments and doc strings. Imports should be grouped by type with
  one space between each group, with the groups sorted in order of most generic
  to least generic
    * future import block for Python 2/3 compatibility
    * standard library imports
    * third-party imports
    * application-specific imports

  Within each group, imports should be sorted alphabetically by full package
  path

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

from eta.core import utils
from eta.core.serial import Serializable
import eta.core.video as vd
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

- Follow standard typographic rules for spaces around punctuation

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
