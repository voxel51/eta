#!/usr/bin/env bash
# Generates documentation for ETA.
#
# Usage:
#   bash sphnix/generate_docs.bash
#
# Copyright 2017-2020, Voxel51, Inc.
# Brian Moore, brian@voxel51.com
#

echo "**** Generating documentation"

sphinx-apidoc -f -o sphinx/source eta/

cd sphinx
make html
cd ..

echo "**** Documentation complete"
printf "To view the docs, run:\n\nopen sphinx/build/html/index.html\n\n"
