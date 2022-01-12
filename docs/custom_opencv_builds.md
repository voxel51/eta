# Custom OpenCV builds

The ETA codebase supports both OpenCV 2.4+ and OpenCV 3.0+. By default, ETA
installs a pre-built OpenCV package via `pip install opencv-python-headless`.
However, one can build a custom OpenCV installation if desired.


## Installing OpenCV

Configure the following environment variables:

```bash
# the OpenCV version to install
OPENCV_VERSION=3.3.0

# whether to install with GPU support
GCARD=ON/OFF

# set this to install OpenCV in a virtual environment
VIRTUAL_ENV=/path/to/env/dir

# set these to install OpenCV globally
GLOBAL_ENV=/usr/local
EXTDIR=/path/to/eta/external
```

and then execute the following bash snippet to download, build, and make a
custom OpenCV installation:

```bash
if [ ! -z "${VIRTUAL_ENV}" ]; then
    # Check for existing installation in virtual environment
    PKG_CONFIG_PATH="${VIRTUAL_ENV}/lib/pkgconfig"
else
    # Check for existing global installation
    PKG_CONFIG_PATH="${GLOBAL_ENV}/lib/pkgconfig"
fi
CURR_VER=$(pkg-config --modversion opencv)
if [ $? -eq 0 ]; then
    printf "OpenCV ${CURR_VER} already installed\n"

    if [ "${CURR_VER}" != "${OPENCV_VERSION}" ]; then
        printf "Found OpenCV ${CURR_VER}, but you requested ${OPENCV_VERSION}\n"
        printf "To uninstall ${OPENCV_VERSION}, navigate to the directory where\n"
        printf "OpenCV was built and run \"sudo make uninstall\"\n"
    fi
else
    printf "Installing OpenCV ${OPENCV_VERSION}\n"

    # Download source
    if [ ! -z "${VIRTUAL_ENV}" ]; then
        # Write source to virtual environment directory
        cd "${VIRTUAL_ENV}"
    else
        # Write source to eta/externals directory
        cd "${EXTDIR}"
    fi
    URL="https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip"
    printf "Downloading OpenCV source from ${URL}\n"
    wget -nv "${URL}"
    unzip "${OPENCV_VERSION}.zip"
    rm -rf "${OPENCV_VERSION}.zip"
    mkdir "opencv-${OPENCV_VERSION}/release"
    cd "opencv-${OPENCV_VERSION}/release"

    # Setup build
    if [ ! -z "${VIRTUAL_ENV}" ]; then
        # Install in a virtual environment
        # This function is needed because Python 2/3 have slightly different
        # naming conventions for these folders...
        pydir() { ls -d "${1}/python"* | head -1; }
        PYTHON_EXECUTABLE="${VIRTUAL_ENV}/bin/python"
        PYTHON_INCLUDE_DIR="$(pydir "${VIRTUAL_ENV}/include")"
        PYTHON_LIBRARY="$(pydir "${VIRTUAL_ENV}/lib")"
        PYTHON_PACKAGES_PATH="$(pydir "${VIRTUAL_ENV}/lib")/site-packages"

        cmake \
            -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
            -D PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
            -D PYTHON_INCLUDE_DIR="${PYTHON_INCLUDE_DIR}" \
            -D PYTHON_LIBRARY="${PYTHON_LIBRARY}" \
            -D PYTHON_PACKAGES_PATH="${PYTHON_PACKAGES_PATH}" \
            -D BUILD_PYTHON_SUPPORT=ON \
            -D WITH_CUDA="${GCARD}" ..

        make -j8
        make -j8 install
    else
        # Install globally
        cmake \
            -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX="${GLOBAL_ENV}" \
            -D BUILD_PYTHON_SUPPORT=ON \
            -D WITH_CUDA="${GCARD}" ..

        make -j8
        sudo make -j8 install
    fi
fi
```


## Symlinking an OpenCV installation

You can symlink an existing globally-installed OpenCV distribution on your
machine into a virtual environment. For example:

```shell
GLOBAL="/usr/local/lib/python2.7/site-packages"
VIRTUAL="/path/to/virtual/env/lib/python2.7/site-packages"
ln -s "${GLOBAL}/cv.py" "${VIRTUAL}/cv.py"
ln -s "${GLOBAL}/cv2.so" "${VIRTUAL}/cv2.so"
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


## Copyright

Copyright 2017-2022, Voxel51, Inc.<br>
voxel51.com
