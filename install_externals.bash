#!/bin/bash
# Install external dependencies
#
# @todo Installing OpenCV 2 in a virtual environment doesn't seem to work. The
#       cv2.so file can't find the dylibs, even though they exist...
#
# Copyright 2017, Voxel51, LLC
# voxel51.com
#
# Brian Moore, brian@voxel51.com
#


# Show usage information
usage() {
    echo "Usage:  bash $0 [-h] [-e dir] [-v ver] [-u] [-bp]

Getting help:
-h      Display this help message.

OpenCV options:
-e dir  Install OpenCV in the virtual environment defined in the given
        directory. By default, OpenCV is installed globally in /usr/local.
-v ver  A specific OpenCV version to install. The default is 3.3.0.

Package manager options:
-u      Update package manager before installing. The default is false.

Mac-only options:
-b      Use brew to install packages (mac only). The default is false.
-p      Use port to install packages (mac only). The default is true.
"
}


# Parse flags
SHOW_HELP=false
VIRTUAL_ENV=""
GLOBAL_ENV="/usr/local"
OPENCV_VERSION="3.3.0"
USE_MACPORTS=true
UPDATE_PACKAGES=false
while getopts "he:v:ubp" FLAG; do
    case "${FLAG}" in
        h) SHOW_HELP=true ;;
        e) VIRTUAL_ENV="${OPTARG}" ;;
        v) OPENCV_VERSION="${OPTARG}" ;;
        u) UPDATE_PACKAGES=true ;;
        b) USE_MACPORTS=false ;;
        p) USE_MACPORTS=true ;;
        *) usage ;;
    esac
done
[ ${SHOW_HELP} = true ] && usage && exit 0


CWD=$(pwd)

EXTDIR=external
EXTLOG="${CWD}/${EXTDIR}/install.log"
EXTERR="${CWD}/${EXTDIR}/install.err"

mkdir -p "${EXTDIR}"
rm -rf "${EXTLOG}"
rm -rf "${EXTERR}"

OS=$(uname -s)

set -o pipefail


# Run command and print stdout/stderr to terminal and (separate) logs
INFO () {
    ("$@" | tee -a "${EXTLOG}") 3>&1 1>&2 2>&3 | tee -a "${EXTERR}"
}


# Print message and log to stderr log
WARN () {
    printf "***** WARNING: ${1}\n" 2>&1 | tee -a "${EXTERR}"
}


# Print message and log to stdout log
MSG () {
    INFO printf "***** ${1}\n"
}


# Exit by printing message and locations of log files
EXIT () {
    MSG "${1}"
    MSG "Log file: ${EXTLOG}"
    MSG "Error file: ${EXTERR}"
    exit 0
}


# Run command, log stdout/stderr, and exit upon error
CRITICAL () {
    INFO "$@"
    if [ $? -ne 0 ]; then
        EXIT "INSTALLATION FAILED"
    fi
}


MSG "INSTALLATION STARTED"


# GPU flag
MSG "Checking system for GPU"
if [ "${OS}" == "Linux" ]; then
    CRITICAL lspci | grep -q "NVIDIA"
    if [ $? -eq 0 ]; then
        GCARD=ON
    else
        GCARD=OFF
    fi
elif [ "${OS}" == "Darwin" ]; then
    GCARD=OFF
fi
MSG "Setting GCARD=${GCARD}"


# Update package managers
if [ ${UPDATE_PACKAGES} = true ]; then
    if [ "${OS}" == "Linux" ]; then
        MSG "Installing build-essential"
        CRITICAL sudo apt-get -y install build-essential
    elif [ "${OS}" == "Darwin" ]; then
        if [ ${USE_MACPORTS} = true ]; then
            MSG "Updating MacPorts"
            CRITICAL sudo port selfupdate
        else
            MSG "Updating Homebrew"
            CRITICAL brew update
        fi
    fi
fi


# Install python requirements
MSG "Installing Python packages"
CRITICAL pip install -r requirements.txt


# Tensorflow is also a requirement, but it depends on the GPU, so we install
# that explicitly
MSG "Installing TensorFlow"
if [ "${GCARD}" == "ON" ]; then
    CRITICAL pip install tensorflow-gpu
else
    CRITICAL pip install tensorflow
fi


# ffmpeg
INFO command -v ffmpeg
if [ $? -eq 0 ]; then
    MSG "ffmpeg already installed"
else
    MSG "Installing ffmpeg"
    if [ "${OS}" == "Linux" ]; then
        CRITICAL sudo apt-get -y install ffmpeg
    elif [ "${OS}" == "Darwin" ]; then
        if [ ${USE_MACPORTS} = true ]; then
            CRITICAL sudo port install ffmpeg
        else
            CRITICAL brew install ffmpeg
        fi
    fi
fi


# imagemagick
INFO command -v convert
if [ $? -eq 0 ]; then
    MSG "imagemagick already installed"
else
    MSG "Installing imagemagick"
    if [ "${OS}" == "Linux" ]; then
        CRITICAL sudo apt-get -y install imagemagick
    elif [ "${OS}" == "Darwin" ]; then
        if [ ${USE_MACPORTS} = true ]; then
            CRITICAL sudo port install imagemagick
        else
            CRITICAL brew install imagemagick
        fi
    fi
fi


# OpenCV
if [ ! -z "${VIRTUAL_ENV}" ]; then
    # Check for existing installation in virtual environment
    PKG_CONFIG_PATH="${VIRTUAL_ENV}/lib/pkgconfig"
else
    # Check for existing global installation
    PKG_CONFIG_PATH="${GLOBAL_ENV}/lib/pkgconfig"
fi
CURR_VER=$(pkg-config --modversion opencv)
if [ $? -eq 0 ]; then
    MSG "OpenCV ${CURR_VER} already installed"

    if [ "${CURR_VER}" != "${OPENCV_VERSION}" ]; then
        WARN "Found OpenCV ${CURR_VER}, but you requested ${OPENCV_VERSION}"
        WARN "To uninstall ${OPENCV_VERSION}, navigate to the directory where"
        WARN "OpenCV was built and run \"sudo make uninstall\""
    fi
else
    MSG "Installing OpenCV ${OPENCV_VERSION}"

    # Download source
    if [ ! -z "${VIRTUAL_ENV}" ]; then
        # Write source to virtual environment directory
        cd "${VIRTUAL_ENV}"
    else
        # Write source to eta/externals directory
        cd "${EXTDIR}"
    fi
    URL="https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip"
    MSG "Downloading OpenCV source from ${URL}"
    CRITICAL wget -nv "${URL}"
    CRITICAL unzip "${OPENCV_VERSION}.zip"
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

        CRITICAL cmake \
            -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
            -D PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
            -D PYTHON_INCLUDE_DIR="${PYTHON_INCLUDE_DIR}" \
            -D PYTHON_LIBRARY="${PYTHON_LIBRARY}" \
            -D PYTHON_PACKAGES_PATH="${PYTHON_PACKAGES_PATH}" \
            -D BUILD_PYTHON_SUPPORT=ON \
            -D WITH_CUDA="${GCARD}" ..

        CRITICAL make -j8
        CRITICAL make -j8 install
    else
        # Install globally
        CRITICAL cmake \
            -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX="${GLOBAL_ENV}" \
            -D BUILD_PYTHON_SUPPORT=ON \
            -D WITH_CUDA="${GCARD}" ..

        CRITICAL make -j8
        CRITICAL sudo make -j8 install
    fi

    cd "${CWD}"
fi


EXIT "INSTALLATION COMPLETE"
