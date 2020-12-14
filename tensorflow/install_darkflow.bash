#!/usr/bin/env bash
# Installs the `tensorflow/darkflow` submodule.
#
# Copyright 2017-2020, Voxel51, Inc.
# voxel51.com
#

ETA_DIR="$(cd "$(dirname $(dirname "${BASH_SOURCE[0]}"))" && pwd)"

echo "Updating submodule tensorflow/darkflow"
cd "${ETA_DIR}"
git submodule update --init tensorflow/darkflow

echo "Installing darkflow package"
DARKFLOW_DIR="${ETA_DIR}/tensorflow/darkflow"
pip install -e "${DARKFLOW_DIR}"

echo "Installation complete"
