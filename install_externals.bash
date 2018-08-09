#!/usr/bin/env bash
# Install external dependencies
#
# Copyright 2017-2018, Voxel51, LLC
# voxel51.com
#
# Brian Moore, brian@voxel51.com
#


# Show usage information
usage() {
    echo "Usage:  bash $0 [-h] [-bp]

Getting help:
-h      Display this help message.

Mac-only options:
-b      Use brew to install packages (mac only). The default is false.
-p      Use port to install packages (mac only). The default is true.
"
}


# Parse flags
SHOW_HELP=false
USE_MACPORTS=true
while getopts "he:v:bp" FLAG; do
    case "${FLAG}" in
        h) SHOW_HELP=true ;;
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
    lspci | grep -q "NVIDIA"
    if [ $? -eq 0 ]; then
        GCARD=ON
    else
        GCARD=OFF
    fi
elif [ "${OS}" == "Darwin" ]; then
    GCARD=OFF
fi
MSG "Setting GCARD=${GCARD}"


# Install base packages
MSG "Installing base machine packages"
if [ "${OS}" == "Linux" ]; then
    CRITICAL sudo apt-get update
    CRITICAL sudo apt-get -y install build-essential
    CRITICAL sudo apt-get -y install pkg-config
    CRITICAL sudo apt-get -y install python-pip
    CRITICAL sudo apt-get -y install python-dev
    CRITICAL sudo apt-get -y install cmake
    CRITICAL sudo apt-get -y install cmake-data
    CRITICAL sudo apt-get -y install unzip
elif [ "${OS}" == "Darwin" ]; then
    # Macs already have most goodies, so just update package managers
    if [ ${USE_MACPORTS} = true ]; then
        CRITICAL sudo port selfupdate
    else
        CRITICAL brew update
    fi
fi
CRITICAL pip install --upgrade pip
CRITICAL pip install --upgrade virtualenv


# Install python requirements
MSG "Installing Python packages"
CRITICAL pip install -r requirements.txt


# Tensorflow
MSG "Installing TensorFlow"
if [ "${GCARD}" == "ON" ]; then
    # Force TensorFlow 1.3 for use with CUDA 8.0
    CRITICAL pip install --upgrade tensorflow-gpu==1.3
else
    CRITICAL pip install --upgrade tensorflow
fi


# OpenCV
CRITICAL pip install --upgrade opencv-python


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


EXIT "INSTALLATION COMPLETE"
