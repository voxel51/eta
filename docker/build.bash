#!/usr/bin/env bash
# Builds a Docker image with ETA installed.
#
# Copyright 2017-2020, Voxel51, Inc.
# voxel51.com
#
# Brian Moore, brian@voxel51.com
#

# Usage information
usage() {
    echo "Usage:  bash $0 [-h] [-b branch_or_commit] base_image tf_version tag

base_image             The base image to build from.
tf_version             The TensorFlow version to install, e.g., tensorflow-gpu==1.14.0.
tag                    A tag for the built image.
-b                     The ETA branch or commit to build. (default = develop).
-h                     Display this help message.
"
}

# Parse flags
unset OPTIND
SHOW_HELP=false
ETA_BRANCH_OR_COMMIT=""
while getopts "hb:" FLAG; do
    case ${FLAG} in
        h) SHOW_HELP=true ;;
        b) ETA_BRANCH_OR_COMMIT="${OPTARG}" ;;
        *) SHOW_HELP=true ;;
    esac
done
[ ${SHOW_HELP} = true ] && usage && exit 0

# Parse positional arguments
shift $((OPTIND - 1))
BASE_IMAGE=$1
TENSORFLOW_VERSION=$2
TAG=$3

# Clone ETA
git clone https://github.com/voxel51/eta
cd eta
if [[ ! -z $ETA_BRANCH_OR_COMMIT ]]; then
    git checkout $ETA_BRANCH_OR_COMMIT
fi
git submodule init
git submodule update
cp config-example.json config.json
rm -rf .git/modules  # trim the fat (removes ~500MB)
cd ..

# Build image
docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg TENSORFLOW_VERSION="${TENSORFLOW_VERSION}" \
    --tag "${TAG}" \
    .

# Cleanup
rm -rf eta
