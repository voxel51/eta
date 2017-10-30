# ETA virtual environments

This document describes how to install ETA in Python 2 and Python 3 virtual
environments running independent copies of OpenCV.


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
function eta2 { source "${ENV_DIR}/eta2/bin/activate"; }
function eta3 { source "${ENV_DIR}/eta3/bin/activate"; }
function exit {
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

* Proceed with standasrd ETA installation:

```bash
cd /path/to/eta  # modify this
bash install_externals.bash
pip install -e .
```

* Install OpenCV in your virtual environment via one of
[these methods](#installing-opencv).

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


## Installing OpenCV

Use one of the following methods to install OpenCV in your virtual environment.

### Install a fresh copy of OpenCV

> This is the preferred method.

Follow these instructions to install a fresh copy of OpenCV that is accessible
only from the virtual environment.

* Configure the installation by setting the following environment variables
to their appropriate values:

```shell
# eta2
VIRTUAL_ENV="${ENV_DIR}/eta2"
PYTHON_EXECUTABLE="${VIRTUAL_ENV}/bin/python"
PYTHON_INCLUDE_DIR="${VIRTUAL_ENV}/include/python2.7"
PYTHON_LIBRARY="${VIRTUAL_ENV}/lib/python2.7"
PYTHON_PACKAGES_PATH="${VIRTUAL_ENV}/lib/python2.7/site-packages"
OPENCV_VERSION=2.4.13.3  # modify this
WITH_CUDA=OFF  # modify this
```

```shell
# eta3
VIRTUAL_ENV="${ENV_DIR}/eta3"
PYTHON_EXECUTABLE="${VIRTUAL_ENV}/bin/python"
PYTHON_INCLUDE_DIR="${VIRTUAL_ENV}/include/python3.6m"
PYTHON_LIBRARY="${VIRTUAL_ENV}/lib/python3.6"
PYTHON_PACKAGES_PATH="${VIRTUAL_ENV}/lib/python3.6/site-packages"
OPENCV_VERSION=3.3.0  # modify this
WITH_CUDA=OFF  # modify this
```

* Navigate to the virtual environment directory:

```shell
cd "${VIRTUAL_ENV}"
```

* Install OpenCV from source

```
# Download source
wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
unzip ${OPENCV_VERSION}.zip
rm -rf ${OPENCV_VERSION}.zip

# Make release directory
mkdir opencv-${OPENCV_VERSION}/release
cd opencv-${OPENCV_VERSION}/release

# Setup build
cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
    -D PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    -D PYTHON_INCLUDE_DIR="${PYTHON_INCLUDE_DIR}" \
    -D PYTHON_LIBRARY="${PYTHON_LIBRARY}" \
    -D PYTHON_PACKAGES_PATH="${PYTHON_PACKAGES_PATH}" \
    -D BUILD_PYTHON_SUPPORT=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D BUILD_EXAMPLES=ON \
    -D WITH_CUDA=${WITH_CUDA} ..

# Install
make -j8
make -j8 install

# Delete source (to save disk space)
cd ../..
rm -rf opencv-${OPENCV_VERSION}
```


### Symlink a global OpenCV installation

> This is a bit of a hack, but it seems to work.

You can symlink an existing global OpenCV installation into your virtual
environment. For example:

```shell
# eta2
GLOBAL="/usr/local/lib/python2.7/site-packages"
VIRTUAL="${ENV_DIR}/eta2/lib/python2.7/site-packages"
ln -s "${GLOBAL}/cv.py" "${VIRTUAL}/cv.py"
ln -s "${GLOBAL}/cv2.so" "${VIRTUAL}/cv2.so"

# eta3
GLOBAL="/usr/local/lib/python3.6/site-packages"
VIRTUAL="${ENV_DIR}/eta3/lib/python3.6/site-packages"
ln -s "${GLOBAL}/cv2.cpython-36m-darwin.so" "${VIRTUAL}/cv2.so"
```


## More resources

`virtualenv` user guide:
https://virtualenv.pypa.io/en/stable/userguide

Cheat sheet:
http://stuarteberg.github.io/conda-docs/_downloads/conda-pip-virtualenv-translator.html
