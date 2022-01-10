# Dockerfile for building a base image with an ETA environment atop a
# Debian, Ubuntu, or related Linux distribution.
#
# ARGs:
#   BASE_IMAGE
#   TENSORFLOW_VERSION
#
# TensorFlow version notes:
#   - For CPU-only images, use tensorflow~=1.15
#   - For GPU-enabled images, use the TensorFlow version compatible with the
#     CUDA version you are using:
#       - CUDA 8: tensorflow-gpu~=1.4
#       - CUDA 9: tensorflow-gpu~=1.12
#       - CUDA 10: tensorflow-gpu~=1.15
#
# Copyright 2017-2022, Voxel51, Inc.
# voxel51.com
#

#
# The base image to build from, which must be a Debian, Ubuntu, or related
# Linux distribution
#

ARG BASE_IMAGE
FROM $BASE_IMAGE

#
# Install ETA
#
# Notes:
#   ETA supports Python 2.7.X or Python 3.6.X
#
# `ppa:deadsnakes/ppa` is used in order to install Python 3.6 on Ubuntu 16.04
# https://askubuntu.com/questions/865554/how-do-i-install-python-3-6-using-apt-get
#
# `https://bootstrap.pypa.io/get-pip.py` is used to install pip on Python 3.6
# https://askubuntu.com/questions/889535/how-to-install-pip-for-python-3-6-on-ubuntu-16-10
#
# numpy==1.16.0 is enforced as a last step because tensorflow requires this
# version to function properly, and some commands here seem to mess with the
# numpy version installed via the `requirements.txt` file
#

RUN apt-get update \
    && apt-get -y --no-install-recommends install software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get -y --no-install-recommends install \
        sudo \
        build-essential \
        pkg-config \
        ca-certificates \
        cmake \
        cmake-data \
        unzip \
        pciutils \
        git \
        curl \
        wget \
        python3.6 \
        python3.6-dev \
        libcupti-dev \
        ffmpeg \
        imagemagick \
    && ln -s /usr/bin/python3.6 /usr/local/bin/python \
    && curl https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /usr/src/app
COPY . eta/

ARG TENSORFLOW_VERSION
RUN pip --no-cache-dir  install --upgrade pip setuptools \
    && pip --no-cache-dir install -r eta/requirements/common.txt \
    && pip --no-cache-dir install -r eta/requirements/pipeline.txt \
    && pip --no-cache-dir install -r eta/requirements/storage.txt \
    && pip --no-cache-dir install --upgrade setuptools \
    && pip --no-cache-dir install -e eta/. \
    && pip --no-cache-dir install -I $TENSORFLOW_VERSION \
    && pip --no-cache-dir install --upgrade numpy==1.16.0 \
    && pip --no-cache-dir install -e eta/eta/tensorflow/darkflow/. \
    && pip --no-cache-dir install pycocotools \
    && pip --no-cache-dir install protobuf \
    && curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip \
    && unzip protoc-3.6.1-linux-x86_64.zip -d protoc3 \
    && rm -rf protoc-3.6.1-linux-x86_64.zip \
    && mv protoc3/bin/* /usr/local/bin \
    && mv protoc3/include/* /usr/local/include \
    && rm -rf protoc3 \
    && cd eta/eta/tensorflow/models \
    && protoc research/object_detection/protos/*.proto \
        --proto_path=research \
        --python_out=research \
    && rm -rf /var/lib/apt
