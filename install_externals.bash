#!/bin/bash
# Install external dependencies
#
# Usage:
#   bash install_externals.bash [opencv-version]
#
# Copyright 2017, Voxel51, LLC
# voxel51.com
#

#OPENCV_VERSION=${1:-2.4.13.3}
OPENCV_VERSION=${1:-3.3.0}

CWD=`pwd`

EXTDIR=external
EXTLOG="${CWD}/${EXTDIR}/install.log"
EXTERR="${CWD}/${EXTDIR}/install.err"

mkdir -p ${EXTDIR}
rm -rf ${EXTLOG}
rm -rf ${EXTERR}

OS=`uname -s`


# Print message
ECHO () {
    printf "${1}\n"
    printf "${1}\n" >> "${EXTLOG}"
}


# Run command and log stdout/stderr
INFO () {
    "$@" >>${EXTLOG} 2>>${EXTERR}
}


# Run command, log stdout/stderr, and exit upon error
CRITICAL () {
    INFO "$@"
    if [ $? -ne 0 ]
    then
        ECHO "***** INSTALLATION FAILED WITH ERROR:"
        cat ${EXTERR}
        exit 1
    fi
}


ECHO "***** INSTALLATION STARTED"
ECHO "Log file: ${EXTLOG}"
ECHO "Error file: ${EXTERR}"


# GPU flag
ECHO "Checking system for GPU"
if [ ${OS} == "Linux" ]
then
    CRITICAL lspci | grep -q "NVIDIA"
    if [ $? -eq 0 ]
    then
      GCARD=ON
    else
      GCARD=OFF
    fi
elif [ ${OS} == "Darwin" ]
then
    GCARD=OFF
fi
ECHO "Setting GCARD=${GCARD}"


# Linux-specific items
if [ ${OS} == "Linux" ]
then
    CRITICAL sudo apt-get -y install build-essential
fi


# Install python requirements
ECHO "Installing Python packages"
CRITICAL pip install -r requirements.txt


# Tensorflow is also a requirement, but it depends on the GPU, so we install
# that explicitly
ECHO "Installing TensorFlow"
if [ ${GCARD} == "ON" ]
then
    CRITICAL pip install tensorflow-gpu
else
    CRITICAL pip install tensorflow
fi


# ffmpeg
INFO command -v ffmpeg
if [ $? -eq 0 ]
then
    ECHO "ffmpeg already installed"
else
    ECHO "Installing ffmpeg"
    if [ ${OS} == "Linux" ]
    then
        CRITICAL sudo apt-get -y install ffmpeg
    elif [ ${OS} == "Darwin" ]
    then
        CRITICAL brew install ffmpeg
    fi
fi


# imagemagick
INFO command -v convert
if [ $? -eq 0 ]
then
    ECHO "imagemagick already installed"
else
    ECHO "Installing imagemagick"
    if [ ${OS} == "Linux" ]
    then
        CRITICAL sudo apt-get -y install imagemagick
    elif [ ${OS} == "Darwin" ]
    then
        CRITICAL brew install imagemagick
    fi
fi


# OpenCV
INFO pkg-config --cflags opencv
if [ $? -eq 0 ]
then
    ECHO "OpenCV already installed"
else
    ECHO "Installing OpenCV ${OPENCV_VERSION}"

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


ECHO "***** INSTALLATION COMPLETE"
