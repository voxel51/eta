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


# Print to STDOUT and EXTLOG
log () {
    printf "${1}\n"
    printf "${1}\n" >> "${EXTLOG}"
}


log "***** INSTALLATION STARTED"

log "Log file: ${EXTLOG}"
log "Error file: ${EXTERR}"


# GPU flag
log "Checking system for GPU"
if [ ${OS} == "Linux" ]
then
    (lspci | grep -q "NVIDIA") >>${EXTLOG} 2>>${EXTERR} || exit 1
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
log "Setting GCARD=${GCARD}"


# Linux-specific items
if [ ${OS} == "Linux" ]
then
    sudo apt-get -y install build-essential >>${EXTLOG} 2>>${EXTERR} || exit 1
fi


# Install python requirements
log "Installing Python packages"
pip install -r requirements.txt >>${EXTLOG} 2>>${EXTERR} || exit 1


# Tensorflow is also a requirement, but it depends on the GPU, so we install
# that explicitly
log "Installing TensorFlow"
if [ ${GCARD} == "ON" ]
then
    pip install tensorflow-gpu >>${EXTLOG} 2>>${EXTERR} || exit 1
else
    pip install tensorflow >>${EXTLOG} 2>>${EXTERR} || exit 1
fi


# ffmpeg
(command -v ffmpeg) >>${EXTLOG} 2>>${EXTERR} || exit 1
if [ $? -eq 0 ]
then
    log "ffmpeg already installed"
else
    log "Installing ffmpeg"
    if [ ${OS} == "Linux" ]
    then
        sudo apt-get -y install ffmpeg >>${EXTLOG} 2>>${EXTERR} || exit 1
    elif [ ${OS} == "Darwin" ]
    then
        brew install ffmpeg >>${EXTLOG} 2>>${EXTERR} || exit 1
    fi
fi


# imagemagick
(command -v convert) >>${EXTLOG} 2>>${EXTERR} || exit 1
if [ $? -eq 0 ]
then
    log "imagemagick already installed"
else
    log "Installing imagemagick"
    if [ ${OS} == "Linux" ]
    then
        sudo apt-get -y install imagemagick >>${EXTLOG} 2>>${EXTERR} || exit 1
    elif [ ${OS} == "Darwin" ]
    then
        brew install imagemagick >>${EXTLOG} 2>>${EXTERR} || exit 1
    fi
fi


# OpenCV
(pkg-config --cflags opencv) >>${EXTLOG} 2>>${EXTERR} || exit 1
if [ $? -eq 0 ]
then
    log "OpenCV already installed"
else
    log "Installing OpenCV ${OPENCV_VERSION}"

    # Download source
    cd ${EXTDIR}
    wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
    unzip ${OPENCV_VERSION}.zip >>${EXTLOG} 2>>${EXTERR} || exit 1
    rm -rf ${OPENCV_VERSION}.zip
    mkdir opencv-${OPENCV_VERSION}/release
    cd opencv-${OPENCV_VERSION}/release

    # Setup build
    cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D BUILD_PYTHON_SUPPORT=ON \
        -D BUILD_EXAMPLES=ON \
        -D WITH_CUDA=${GCARD} .. >>${EXTLOG} 2>>${EXTERR} || exit 1

    # Make + install
    make -j8 >>${EXTLOG} 2>>${EXTERR} || exit 1
    sudo make -j8 install >>${EXTLOG} 2>>${EXTERR} || exit 1

    cd "${CWD}"
fi


log "***** INSTALLATION COMPLETE"
