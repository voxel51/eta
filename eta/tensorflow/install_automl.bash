#!/usr/bin/env bash
# Installs the `eta/tensorflow/automl` submodule.
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
AUTOML_DIR="${TENSORFLOW_DIR}/automl"

if [ ${SKIP_CLONE} = false ]; then
    if [ -d "${AUTOML_DIR}" ]; then
        echo "Deleting existing directory ${AUTOML_DIR}"
        rm -rf "${AUTOML_DIR}"
    fi

    echo "Cloning https://github.com/voxel51/automl"
    git clone https://github.com/voxel51/automl "${AUTOML_DIR}"
fi

echo "Installing pycocotools"
pip install pycocotools

echo "Installation complete"
