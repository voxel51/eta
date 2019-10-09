# ETA Virtual Environment Guide

This document describes how to install ETA in a
[virtual environment](https://virtualenv.pypa.io/en/stable). Using a virtual
environment is highly recommended because it allows you to maintain a separate
Python working environment for ETA that operates independently of other
packages and Python applications on your machine.


## Creating virtual environments

Follow these instructions to create Python 2 and Python 3 virtual environments.

* If desired, install fresh python distributions (2.7 and 3.6) from
https://www.python.org/downloads

* Sometimes installing an older version of Python can be unclear. If you need a fresh install of Python 3.6, you can also run the following:
```shell
# Ubuntu
sudo apt-get update
sudo apt-get -y --no-install-recommends install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get -y --no-install-recommends install python3.6 python3.6-dev
sudo ln -s /usr/bin/python3.6 /usr/local/bin/python
curl https://bootstrap.pypa.io/get-pip.py | sudo python
sudo pip install --upgrade pip setuptools
sudo pip install virtualenv

# Mac
brew unlink python
brew install pkg-config gdbm openssl readline sqlite xz
brew install --ignore-dependencies https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb
```

* <b>Note:</b>
If you have previously had Python 3.6.5 installed through Homebrew on MacOS, running the following will switch your Python 3.7.x with Python 3.6.5. In some cases, this may be an easier solution.
```shell
# Mac
brew switch python 3.6.5_1
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

* Make a virtual environment, modifying the python executable path as needed:

```shell
# example for Python 2.X
virtualenv -p /usr/local/bin/python2 eta2

# example for Python 3.X
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

To install ETA in a virtual environment, simply activate the virtual
environment:

```shell
# Example of activating the Python 3.X environment created above
eta3
```

and then run the ETA install script from the ETA root directory:

```shell
bash install.bash
```

To verify that your ETA installation is independent of your global Python
environment, view the details of your virtual setup:

```shell
which python
which pip
pip freeze
```

then deactivate the virtual environment

```shell
# Example of deactivating an environment created in the previous section
exit
```

and compare to the details of your global setup:

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


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
