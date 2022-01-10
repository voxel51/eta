#!/usr/bin/env bash
# Installs the `eta/tensorflow/darkflow` submodule.
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
DARKFLOW_DIR="${TENSORFLOW_DIR}/darkflow"

if [ ${SKIP_CLONE} = false ]; then
    if [ -d "${DARKFLOW_DIR}" ]; then
        echo "Deleting existing directory ${DARKFLOW_DIR}"
        rm -rf "${DARKFLOW_DIR}"
    fi

    echo "Cloning https://github.com/voxel51/darkflow"
    git clone https://github.com/voxel51/darkflow "${DARKFLOW_DIR}"
fi

echo "Installing darkflow package"
pip install -e "${DARKFLOW_DIR}"

echo "Installation complete"
