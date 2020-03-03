# Dockerfile for building a base image with an ETA environment atop a
# Linux-based OS.
#
# ARGs:
#   BASE_IMAGE
#   TENSORFLOW_VERSION
#   ETA_BRANCH_OR_COMMIT
#
# TensorFlow version notes:
#   - For CPU-only images, use tensorflow==1.12.0
#   - For GPU-enabled images, use the TensorFlow version compatible with the
#     CUDA version you are using:
#       - CUDA 8: tensorflow-gpu==1.4.0
#       - CUDA 9: tensorflow-gpu==1.12.0
#       - CUDA 10: tensorflow-gpu==1.14.0
#
# Copyright 2017-2020, Voxel51, Inc.
# voxel51.com
#
# Brian Moore, brian@voxel51.com
#

#
# The base image to build from, which must be Linux-based
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

ARG ETA_BRANCH_OR_COMMIT
WORKDIR /dev
RUN git clone https://github.com/voxel51/eta \
    && cd /dev/eta \
    && git checkout $ETA_BRANCH_OR_COMMIT \
    && git submodule init \
    && git submodule update \
    && cp config-example.json config.json \
    && rm .git/modules

ARG TENSORFLOW_VERSION
RUN pip --no-cache-dir  install --upgrade pip setuptools \
    && pip --no-cache-dir install -r /dev/eta/requirements.txt \
    && pip --no-cache-dir  install --upgrade setuptools \
    && pip --no-cache-dir install -e /dev/eta/. \
    && pip --no-cache-dir install -I $TENSORFLOW_VERSION \
    && pip --no-cache-dir install --upgrade numpy==1.16.0 \
    && make --directory=/dev/eta/cocoapi/PythonAPI/ \
    && rm -rf /dev/eta/tensorflow/models/research/pycocotools \
    && cp -r /dev/eta/cocoapi/PythonAPI/pycocotools /dev/eta/tensorflow/models/research/ \
    && curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip \
    && unzip protoc-3.6.1-linux-x86_64.zip -d protoc3 \
    && rm -rf protoc-3.6.1-linux-x86_64.zip \
    && mv protoc3/bin/* /usr/local/bin \
    && mv protoc3/include/* /usr/local/include \
    && rm -rf protoc3 \
    && cd /dev/eta/tensorflow/models \
    && protoc research/object_detection/protos/*.proto \
        --proto_path=research \
        --python_out=research \
    && rm -rf /var/lib/apt
