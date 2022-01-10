# Docker Build Guide

This document guides you through building a Docker image with a pre-installed
ETA environment.

## Dependencies

-   [Docker Community Edition](https://hub.docker.com/search/?type=edition&offering=community)
-   To build/run GPU-enabled images, you must install `nvidia-docker` by
    following [these instructions](https://github.com/NVIDIA/nvidia-docker)

## Setup

In order to build an ETA Docker image, you must have ETA cloned on your machine
with a valid ETA `config.json` file installed:

```shell
# Clone repository
git clone https://github.com/voxel51/eta
cd eta

# Clone submodules
git submodule init
git submodule update

# Install your ETA config (customize as necessary)
cp config-example.json eta/config.json
```

## Building an image

Follow the instructions below to build an ETA Docker image for your desired
environment.

These commands should be run from the root directory of your local ETA clone.

#### CPU example

The following snippet builds a CPU-enabled image on Ubuntu 16.04:

```shell
BASE_IMAGE=ubuntu:16.04
TENSORFLOW_VERSION=tensorflow==1.12.0
TAG=voxel51/eta:cpu-ubuntu-16.04

docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg TENSORFLOW_VERSION="${TENSORFLOW_VERSION}" \
    --tag "${TAG}" \
    .
```

#### GPU example

The following snippet builds a GPU-enabled image on Ubuntu 16.04 with CUDA 9.0:

```shell
BASE_IMAGE=nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
TENSORFLOW_VERSION=tensorflow-gpu==1.12.0
TAG=voxel51/eta:gpu-ubuntu-16.04-cuda-9.0

docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg TENSORFLOW_VERSION="${TENSORFLOW_VERSION}" \
    --tag "${TAG}" \
    .
```

## Using an image

After you have built an ETA image, you run the image with an interactive shell
as follows:

```shell
# For CPU images
docker run -it $TAG

# For GPU images
docker run -it --runtime=nvidia $TAG
```

## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
