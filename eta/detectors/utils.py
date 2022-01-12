"""
Detectors utilities.

Copyright 2017-2022, Voxel51, Inc.
voxel51.com
"""
import os
import sys

import eta.constants as etac


def reset_path():
    """Reset any path/module changes that may have resulted from using
    detectors in this module.
    """
    try:
        sys.modules.pop("object_detection")
    except KeyError:
        pass

    try:
        sys.path.remove(etac.TF_RESEARCH_DIR)
    except ValueError:
        pass

    try:
        sys.path.remove(os.path.join(etac.TF_OBJECT_DETECTION_DIR, "utils"))
    except ValueError:
        pass

    try:
        sys.path.remove(etac.EFFICIENTDET_DIR)
    except ValueError:
        pass
