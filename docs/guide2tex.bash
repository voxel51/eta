#!/usr/bin/env bash
# Convert markdown dev guide to tex
#
# Note:
#   All content in the dev guide before the first second-level (##) or greater
#   heading is omitted
#
# Usage:
#   bash guide2tex.bash modules_dev_guide.md
#
# Voxel51, LLC, Copyright 2017
# voxel51.com
#
# Brian Moore, brian@voxel51.com
# Jason Corso, jjc@voxel51.com
#

MDPATH=${1}
TEXPATH=${MDPATH/.md/.tex}
sed '/##/,$!d' $MDPATH | pandoc -o $TEXPATH
