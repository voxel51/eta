#!/usr/bin/env bash
# Installs the `eta/tensorflow/models` submodule.
#
# Copyright 2017-2022, Voxel51, Inc.
# voxel51.com
#

SKIP_CLONE=false
while getopts "s" FLAG; do
    case "${FLAG}" in
        s) SKIP_CLONE=true ;;
    esac
done

if ! command -v git &> /dev/null; then
    echo "You must install 'git' in order to run this script"
    exit
fi

TENSORFLOW_DIR="$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)"
MODELS_DIR="${TENSORFLOW_DIR}/models"

if [ ${SKIP_CLONE} = false ]; then
    if [ -d "${MODELS_DIR}" ]; then
        echo "Deleting existing directory ${MODELS_DIR}"
        rm -rf "${MODELS_DIR}"
    fi

    echo "Cloning https://github.com/voxel51/models"
    git clone https://github.com/voxel51/models "${MODELS_DIR}"
fi

cd "${MODELS_DIR}"

echo "Installing protobuf"
pip install protobuf

if command -v protoc &> /dev/null; then
    echo "Found protoc"
else
    echo "Installing protoc"
    if [ $(uname -s) == "Darwin" ]; then
        PROTOC_ZIP=protoc-3.7.1-osx-x86_64.zip
    else
        PROTOC_ZIP=protoc-3.7.1-linux-x86_64.zip
    fi

    curl -OL https://github.com/google/protobuf/releases/download/v3.7.1/${PROTOC_ZIP}
    unzip -o ${PROTOC_ZIP} -d /usr/local bin/protoc
    unzip -o ${PROTOC_ZIP} -d /usr/local include/*
    rm -f ${PROTOC_ZIP}
fi

echo "Compiling protocol buffers"
protoc research/object_detection/protos/*.proto \
    --proto_path=research \
    --python_out=research

echo "Installing tf_slim"
pip install tf_slim

echo "Installation complete"
