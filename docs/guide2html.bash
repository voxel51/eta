#!/usr/bin/env bash
# Convert markdown dev guide to HTML.
#
# Installation:
#   pip install grip
#
# Usage:
#   bash guide2html.bash modules_dev_guide.md
#   bash guide2html.bash pipelines_dev_guide.md
#
# Copyright 2017-2022, Voxel51, Inc.
# voxel51.com
#

grip --export ${1}
