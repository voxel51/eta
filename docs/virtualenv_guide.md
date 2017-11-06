# ETA Virtual Environment Guide

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

# Install external dependencies in virtual environment
VIRTUAL_ENV="${ENV_DIR}/eta2"
OPENCV_VERSION=3.3.0
bash install_externals.bash -e "${VIRTUAL_ENV}" -v "${OPENCV_VERSION}"

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


## Manually installing OpenCV

You can use the `-e` flag of the `install_externals.bash` script to install
OpenCV in a virtual environment. However, we also provide two options here for
manually installing OpenCV in a virtual environment.

### Install a fresh copy of OpenCV in a virtual environment

> This is the preferred method.

Follow these instructions to install a fresh copy of OpenCV in a virtual
environment.

* Configure the installation by setting the following environment variables
to their appropriate values. For example:

```shell
# eta2
VIRTUAL_ENV="${ENV_DIR}/eta2"
OPENCV_VERSION=2.4.13.3
WITH_CUDA=OFF
```

```shell
# eta3
VIRTUAL_ENV="${ENV_DIR}/eta3"
OPENCV_VERSION=3.3.0
WITH_CUDA=OFF
```

* Download the OpenCV source:

```shell
# Navigate to virtual environment directory
cd "${VIRTUAL_ENV}"

# Download source
wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
unzip ${OPENCV_VERSION}.zip
rm -rf ${OPENCV_VERSION}.zip
```

* Install OpenCV from source:

```shell
# Make release directory
mkdir opencv-${OPENCV_VERSION}/release
cd opencv-${OPENCV_VERSION}/release

# Compute python-related paths
pydir() { ls -d "$1/python"* | head -1; }
PYTHON_EXECUTABLE="${VIRTUAL_ENV}/bin/python"
PYTHON_INCLUDE_DIR="$(pydir "${VIRTUAL_ENV}/include")"
PYTHON_LIBRARY="$(pydir "${VIRTUAL_ENV}/lib")"
PYTHON_PACKAGES_PATH="$(pydir "${VIRTUAL_ENV}/lib")/site-packages"

# Setup build
cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
    -D PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    -D PYTHON_INCLUDE_DIR="${PYTHON_INCLUDE_DIR}" \
    -D PYTHON_LIBRARY="${PYTHON_LIBRARY}" \
    -D PYTHON_PACKAGES_PATH="${PYTHON_PACKAGES_PATH}" \
    -D BUILD_PYTHON_SUPPORT=ON \
    -D WITH_CUDA="${WITH_CUDA}" ..

# Install
make -j8
make -j8 install

cd ../..
```


### Symlink a global OpenCV installation

> This is a bit of a hack, but it seems to work.

You can symlink an existing globally-installed OpenCV distribution on your
machine into a virtual environment. For example:

```shell
# eta2
GLOBAL="/usr/local/lib/python2.7/site-packages"
VIRTUAL="${ENV_DIR}/eta2/lib/python2.7/site-packages"
ln -s "${GLOBAL}/cv.py" "${VIRTUAL}/cv.py"
ln -s "${GLOBAL}/cv2.so" "${VIRTUAL}/cv2.so"
```

```shell
# eta3
GLOBAL="/usr/local/lib/python3.6/site-packages"
VIRTUAL="${ENV_DIR}/eta3/lib/python3.6/site-packages"
ln -s "${GLOBAL}/cv2.cpython-36m-darwin.so" "${VIRTUAL}/cv2.so"
```


## Uninstalling OpenCV

Follow these instructions to uninstall OpenCV.

* Navigate to the directory from which `cmake` was run to install OpenCV:

```shell
cd "${VIRTUAL_ENV}/opencv-${OPENCV_VERSION}/release"
```

* Run the uninstaller:

```shell
sudo make uninstall
```


## More resources

`virtualenv` user guide:
https://virtualenv.pypa.io/en/stable/userguide

Cheat sheet:
http://stuarteberg.github.io/conda-docs/_downloads/conda-pip-virtualenv-translator.html
