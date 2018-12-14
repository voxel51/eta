#!/usr/bin/env bash
# Convert markdown dev guide to tex
#
# Note:
#   All content in the dev guide before the first second-level (##) or greater
#   heading is omitted
#
# Installation:
#   https://github.com/jgm/pandoc/releases
#
# Usage:
#   bash guide2tex.bash modules_dev_guide.md
#   bash guide2tex.bash pipelines_dev_guide.md
#
# Voxel51, Inc., Copyright 2017-2018
# voxel51.com
#
# Brian Moore, brian@voxel51.com
#

MDPATH=${1}
TEXPATH=${MDPATH/.md/.tex}
sed '/##/,$!d' $MDPATH | pandoc -o $TEXPATH
