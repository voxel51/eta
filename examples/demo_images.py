#!/usr/bin/env python
'''
ETA image manipulation demo.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import os

import cv2

import eta.core.image as etai


def plot(img):
    cv2.imshow("*** Press any key to exit ***", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


here = os.path.dirname(__file__)
path1 = os.path.join(here, "data/water.jpg")
path2 = os.path.join(here, "data/logo.pdf")

img1 = etai.resize(etai.read(path1), height=540)
img2 = etai.rasterize(path2, width=704)

x0 = etai.Width("13.3%").render(img=img1)
y0 = etai.Height("10%").render(img=img1)
img3 = etai.overlay(img1, img2, x0=x0, y0=y0)

plot(img3)
