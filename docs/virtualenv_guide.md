# ETA Virtual Environment Guide

This document describes how to install ETA in virtual environments.


## Creating virtual environments

Follow these instructions to create Python 2 and Python 3 virtual environments.

* If desired, install fresh python distributions from
https://www.python.org/downloads

```shell
# Mac only
brew install python
brew install python3
```

* Install the `virtualenv` package:

```shell
pip install virtualenv
```

* If desired, disable prompt modification when a virtual environment is active
by setting the `VIRTUAL_ENV_DISABLE_PROMPT` environment variable to a
non-empty value.

* Navigate to a directory in which to store your environments:

```shell
ENV_DIR="/path/to/env"
cd "${ENV_DIR}"
```

* Make a Python 2 environment, modifying the python executable path as needed:

```shell
virtualenv -p /usr/local/bin/python2 eta2
```

* Make a Python 3 environment, modifying the python executable path as needed:

```shell
virtualenv -p /usr/local/bin/python3 eta3
```

> Note: the above commands may fail if an Anaconda python distribution is used.

* Add some convenience functions to your `~/.bashrc` or `~/.bash_profile` for
activating and deactivating the environments:

```bash
# Python environments
export ENV_DIR="/path/to/env"  # modify this
eta2() { source "${ENV_DIR}/eta2/bin/activate"; }
eta3() { source "${ENV_DIR}/eta3/bin/activate"; }
exit() {
    case `command -v python` in
        ${ENV_DIR}/*) deactivate;;
        *) builtin exit;;
    esac
}
```


## Installing ETA in a virtual environment

Follow these instructions to install ETA in a virtual environment.

* View details of your default Python setup:

```shell
which python
which pip
pip freeze
```

* Activate the virtual environment. For example, to activate the `eta2`
environment:

```shell
eta2
```

* View details of your virtual environment:

```shell
which python
which pip
pip freeze
```

* Proceed with standard ETA installation:

```shell
cd /path/to/eta  # modify this

# Install external dependencies
bash install_externals.bash

# Install ETA
pip install -e .
```

* See what packages were installed in your virtual environment:

```shell
pip freeze
```

* Deactivate the virtual environment:

```shell
exit
```

* Verify that your default python setup hasn't changed:

```shell
which python
which pip
pip freeze
```


## More resources

`virtualenv` user guide:
https://virtualenv.pypa.io/en/stable/userguide

Cheat sheet:
http://stuarteberg.github.io/conda-docs/_downloads/conda-pip-virtualenv-translator.html
