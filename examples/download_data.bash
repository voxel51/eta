#!/usr/bin/env bash
# Download examples data
#
# Usage:
#   bash download_data.bash
#
# Copyright 2017, Voxel51, LLC
# voxel51.com
#
# Brian Moore, brian@voxel51.com
#

FILE=0B7phNvpRqNdpNEVpVjE2VXQxOWc
EXAMPLES=`dirname "$0"`
ZIP="${EXAMPLES}/data.zip"

python -c "from eta.core import web;\
web.download_google_drive_file('${FILE}', path='${ZIP}')"

unzip -o "${ZIP}" -d "${EXAMPLES}"
rm "${ZIP}"
