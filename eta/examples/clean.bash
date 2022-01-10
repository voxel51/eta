#!/usr/bin/env bash
# Cleanup all examples output.
#
# Usage:
#   bash clean.bash
#
# Copyright 2017-2022, Voxel51, Inc.
# voxel51.com
#

EXAMPLES_DIR=`dirname "$0"`

find "${EXAMPLES_DIR}" -type d -name out -exec rm -rf {} \;
find "${EXAMPLES_DIR}" -type f -name '*.md5' -exec rm -f {} \;
find "${EXAMPLES_DIR}" -type f -name '*.log' -exec rm -f {} \;
