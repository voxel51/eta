#!/usr/bin/env bash
# Installs the `tensorflow/models` submodule.
#
# Copyright 2017-2020, Voxel51, Inc.
# voxel51.com
#

ETA_DIR="$(cd "$(dirname $(dirname "${BASH_SOURCE[0]}"))" && pwd)"

echo "Updating submodule tensorflow/models"
cd "${ETA_DIR}"
git submodule update --init tensorflow/models

MODELS_DIR="${ETA_DIR}/tensorflow/models"
cd "${MODELS_DIR}"

command -v protoc
if [ $? -eq 0 ]; then
    echo "Found protoc"
else
    echo "Installing protoc"
    if [ $(uname -s) == "Darwin" ]; then
        PROTOC_ZIP=protoc-3.6.1-osx-x86_64.zip
    else
        PROTOC_ZIP=protoc-3.6.1-linux-x86_64.zip
    fi

    curl -OL https://github.com/google/protobuf/releases/download/v3.6.1/${PROTOC_ZIP}
    sudo unzip -o ${PROTOC_ZIP} -d /usr/local bin/protoc
    sudo unzip -o ${PROTOC_ZIP} -d /usr/local include/*
    rm -f ${PROTOC_ZIP}
fi

echo "Compiling protocol buffers"
protoc research/object_detection/protos/*.proto \
    --proto_path=research \
    --python_out=research

echo "Installation complete"
