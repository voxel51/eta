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
cd ${EXTDIR}

rm -rf ${EXTLOG}
rm -rf ${EXTERR}

OS=`uname -s`

echo "*** INSTALLATION STARTED ***"

# GPU flag
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
echo "*** Checking System for GPU.  Setting GPU: ${GCARD}"

if [ ${OS} == "Linux" ]
then
    sudo apt-get -y install build-essential >>${EXTLOG} 2>>${EXTERR} || exit 1
fi

# install python requirements first
#  note that tensorflow is also a requirement, but it depends on the GPU.
#  so, we check that explicitly
pip install -r requirements.txt

if [ ${GCARD} == "ON" ]
then
    pip install tensorflow-gpu
else
    pip install tensorflow
fi


# ffmpeg
(command -v ffmpeg) >>${EXTLOG} 2>>${EXTERR} || exit 1
if [ $? -eq 0 ]
then
    echo "ffmpeg already installed"
else
    echo "Installing ffmpeg"
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
    echo "imagemagick already installed"
else
    echo "Installing imagemagick"
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
    echo "OpenCV already installed"
else
    echo "Installing OpenCV ${OPENCV_VERSION}"

    # Download source
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

    cd ../..
fi


cd "${CWD}"
echo "*** INSTALLATION COMPLETE ***"
