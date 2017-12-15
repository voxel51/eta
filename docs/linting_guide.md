# ETA Linting Guide

The ETA project uses the `pylint` and `pycodestyle` packages for linting. We
maintain customized configurations for these tools in the root directory of
the repository:

* `pylint` configuration: `eta/pylintrc`

* `pycodestyle` configuration: `[pycodestyle]` section of `eta/setup.cfg`


## Linting a file

```shell
pycodestyle <file>
pylint <file>
```


## Customizing pylint

To permanently disable a pylint message, add it to the `disable` field in
the `pylintrc` file:

```shell
[MESSAGES CONTROL]
disable=too-few-public-methods,too-many-arguments
```

To disable a pylint message for the rest of the current block (i.e.,
indentation level) in a module, add the comment:

```python
# pylint: disable=too-many-instance-attributes
```

To disable a pylint message for the current line:

```python
from builtins import *  # pylint disable=wildcard-import
```

To disable pylint errors temporarily in a module:

```python
# pragma pylint: disable=redefined-builtin
# pragma pylint: enable=wildcard-import
from builtins import *
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=wildcard-import
```

See the [pylint user guide](https://pylint.readthedocs.io/en/latest/) for more
information.


## Customizing pycodestyle

Add new entries to the `[pycodestyle]` section of the `setup.cfg` file:

```shell
[pycodestyle]
max-line-length=79
```

See the [pycodestyle user guide](
https://pycodestyle.readthedocs.io/en/latest/intro.html) for more information.
