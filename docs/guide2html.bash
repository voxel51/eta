#!/usr/bin/env bash
# Convert markdown dev guide to html
#
# Installation:
#   pip install grip
#
# Usage:
#   bash guide2html.bash modules_dev_guide.md
#   bash guide2html.bash pipelines_dev_guide.md
#
# Voxel51, LLC, Copyright 2017-2018
# voxel51.com
#
# Brian Moore, brian@voxel51.com
#

grip --export ${1}
