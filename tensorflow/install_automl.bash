#!/usr/bin/env bash
# Installs the `tensorflow/automl` submodule.
#
# Copyright 2017-2020, Voxel51, Inc.
# voxel51.com
#

ETA_DIR="$(cd "$(dirname $(dirname "${BASH_SOURCE[0]}"))" && pwd)"

echo "Updating submodule tensorflow/automl"
cd "${ETA_DIR}"
git submodule update --init tensorflow/automl

echo "Installing pycocotools"
pip install pycocotools

echo "Installation complete"
