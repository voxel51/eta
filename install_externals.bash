#!/bin/bash
# Install external dependencies
#
# @todo add ability to install OpenCV in a virtual environment. Currently it
# is installed globally
#
# Copyright 2017, Voxel51, LLC
# voxel51.com
#
# Brian Moore, brian@voxel51.com
#


# Show usage information
usage() {
    echo "Usage:  bash $0 [-h] [-bp] [-u] [-v <opencv-version>]

Options:
     -h    Display this help message.
     -b    Use brew to install packages (mac only). The default is false.
     -p    Use port to install packages (mac only). The default is true.
     -u    Update package manager before installing. The default is false.
     -v    A specific OpenCV version to install. The default is 3.3.0."
}


# Parse flags
SHOW_HELP=false
USE_MACPORTS=true
UPDATE_PACKAGES=false
OPENCV_VERSION=3.3.0
while getopts "hbpuv:" FLAG; do
    case "${FLAG}" in
        h) SHOW_HELP=true ;;
        b) USE_MACPORTS=false ;;
        p) USE_MACPORTS=true ;;
        u) UPDATE_PACKAGES=true ;;
        v) OPENCV_VERSION="${OPTARG}" ;;
        *) usage ;;
    esac
done
[ ${SHOW_HELP} = true ] && usage && exit 0


CWD=`pwd`

EXTDIR=external
EXTLOG="${CWD}/${EXTDIR}/install.log"
EXTERR="${CWD}/${EXTDIR}/install.err"

mkdir -p ${EXTDIR}
rm -rf ${EXTLOG}
rm -rf ${EXTERR}

OS=`uname -s`

set -o pipefail


# Run command and print stdout/stderr to terminal and (separate) logs
INFO () {
    #"$@" >>${EXTLOG} 2>>${EXTERR}
    ("$@" | tee -a ${EXTLOG}) 3>&1 1>&2 2>&3 | tee -a ${EXTERR}
}


# Print message
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
if [ ${OS} == "Linux" ]; then
    CRITICAL lspci | grep -q "NVIDIA"
    if [ $? -eq 0 ]; then
        GCARD=ON
    else
        GCARD=OFF
    fi
elif [ ${OS} == "Darwin" ]; then
    GCARD=OFF
fi
MSG "Setting GCARD=${GCARD}"


# Update package managers
if [ ${UPDATE_PACKAGES} = true ]; then
    if [ ${OS} == "Linux" ]; then
        MSG "Installing build-essential"
        CRITICAL sudo apt-get -y install build-essential
    elif [ ${OS} == "Darwin" ]; then
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
if [ ${GCARD} == "ON" ]; then
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
    if [ ${OS} == "Linux" ]; then
        CRITICAL sudo apt-get -y install ffmpeg
    elif [ ${OS} == "Darwin" ]; then
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
    if [ ${OS} == "Linux" ]; then
        CRITICAL sudo apt-get -y install imagemagick
    elif [ ${OS} == "Darwin" ]; then
        if [ ${USE_MACPORTS} = true ]; then
            CRITICAL sudo port install imagemagick
        else
            CRITICAL brew install imagemagick
        fi
    fi
fi


# OpenCV
# @todo use `python -c "import cv2"` to check for OpenCV installation in the
# current python environment?
INFO pkg-config --cflags opencv
if [ $? -eq 0 ]; then
    MSG "OpenCV already installed"
else
    MSG "Installing OpenCV ${OPENCV_VERSION}"

    # Download source
    cd ${EXTDIR}
    wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
    CRITICAL unzip ${OPENCV_VERSION}.zip
    rm -rf ${OPENCV_VERSION}.zip
    mkdir opencv-${OPENCV_VERSION}/release
    cd opencv-${OPENCV_VERSION}/release

    # Setup build
    CRITICAL cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D BUILD_PYTHON_SUPPORT=ON \
        -D BUILD_EXAMPLES=ON \
        -D WITH_CUDA=${GCARD} ..

    # Make + install
    CRITICAL make -j8
    CRITICAL sudo make -j8 install

    cd "${CWD}"
fi


EXIT "INSTALLATION COMPLETE"
