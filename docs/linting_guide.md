# ETA Linting Guide

The ETA project uses the following tools for linting and code style:

-   [Black](https://github.com/psf/black)
-   [Prettier](https://prettier.io)
-   [Pylint](https://www.pylint.org)

We maintain customized configurations for these tools in the root directory of
the repository:

-   Black configuration: `[tool.black]` section of `pyproject.toml`
-   Prettier configuration: `.prettierrc`
-   Pylint configuration: `pylintrc`

## Pre-commit hooks

When you installed ETA, pre-commit hooks were automatically installed that run
Black, Pylint, and Prettier on any files you modified in your commit. If these
tools produced any errors or changes to your code, you will need to recommit to
accept the changes.

The `.pre-commit-config.yaml` file in the repository root contains the
definition of these hooks.

## Linting a file

To manually lint a file, run the following:

```shell
# Manually run linting configured in the pre-commit hook
pre-commit run --files <file>
```

The Pylint component of the pre-commit hook only checks for errors. To see the
full output, run:

```shell
pylint <file>
```

## Customizing Black

You don't customize Black, silly! From the docs:

> Pro-tip: If you’re asking yourself “Do I need to configure anything?” the
> answer is “No”. Black is all about sensible defaults.

## Customizing Prettier

We currently use Prettier only for Markdown files. See `.prettierrc` for the
configuration.

See https://prettier.io/docs/en/configuration.html for configuration
documentation.

## Customizing Pylint

To permanently disable a Pylint message, add it to the `disable` field in the
`pylintrc` file:

```shell
[MESSAGES CONTROL]
disable=too-few-public-methods,too-many-arguments
```

To disable a Pylint message for the rest of the current block (i.e.,
indentation level) in a module, add the comment:

```py
# pylint: disable=too-many-instance-attributes
```

To disable a Pylint message for the current line:

```py
from builtins import *  # pylint disable=wildcard-import
```

To disable Pylint errors temporarily in a module:

```py
# pragma pylint: disable=redefined-builtin
# pragma pylint: enable=wildcard-import
from builtins import *

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=wildcard-import
```

See the [Pylint user guide](https://pylint.readthedocs.io/en/latest/) for more
information.

## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
