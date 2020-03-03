# ETA Docker Images

This folder provides support for building
[Docker images](https://www.docker.com) with pre-installed ETA environments.


## Dependencies

- [Docker Community Edition](https://hub.docker.com/search/?type=edition&offering=community)
- To build/run GPU-enabled images, you must install `nvidia-docker` by
following [these instructions](https://github.com/NVIDIA/nvidia-docker)


## Building an image

The `build.bash` script in this directory builds a Docker image with ETA
installed. Its syntax is:

```
Usage:  bash build.bash [-h] [-b branch_or_commit] base_image tf_version tag

base_image             The base image to build from.
tf_version             The TensorFlow version to install, e.g., tensorflow-gpu==1.14.0.
tag                    A tag for the built image.
-b                     The ETA branch or commit to build. (default = develop).
-h                     Display this help message.
```

#### CPU example

The following snippet builds a CPU-enabled image on Ubuntu 16.04:

```shell
BASE_IMAGE="ubuntu:16.04"
TENSORFLOW_VERSION="tensorflow==1.12.0"
TAG="eta-cpu-ubuntu-16.04"

bash build.bash $BASE_IMAGE $TENSORFLOW_VERSION $TAG
```

#### GPU example

The following snippet builds a GPU-enabled image on Ubuntu 16.04 with CUDA 9.0:

```shell
BASE_IMAGE="nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04"
TENSORFLOW_VERSION="tensorflow-gpu==1.12.0"
TAG="eta-gpu-ubuntu-16.04-cuda-9.0"

bash build.bash $BASE_IMAGE $TENSORFLOW_VERSION $TAG
```


## Using an image

After you have built an ETA image, you can run the image with an interactive
shell as follows:

```shell
# For CPU images
docker run -it --entrypoint=/bin/bash $TAG

# For GPU images
docker run -it --runtime=nvidia --entrypoint=/bin/bash $TAG
```


## Copyright

Copyright 2017-2020, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
