"""
Interface to the EfficientDet object detection model.

This module wraps the EfficientDet implementation at
https://github.com/voxel51/automl/tree/master/efficientdet.

Copyright 2017-2020 Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
"""
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

import logging
import os
import sys

import eta.constants as etac
from eta.core.config import Config
import eta.core.geometry as etag
import eta.core.learning as etal
import eta.core.objects as etao
import eta.core.tfutils as etat
import eta.core.utils as etau

from .utils import reset_path


def _setup():
    reset_path()

    sys.path.insert(1, etac.EFFICIENTDET_DIR)


_ensure_tf1 = lambda: etau.ensure_import("tensorflow>=1.14,<2")
tf = etau.lazy_import("tensorflow", callback=_ensure_tf1)

_ERROR_MSG = "You must run `eta install automl` in order to use this model"
efficientdet_arch = etau.lazy_import("efficientdet_arch", error_msg=_ERROR_MSG)
hparams_config = etau.lazy_import("hparams_config", error_msg=_ERROR_MSG)
inference = etau.lazy_import("inference", error_msg=_ERROR_MSG)


logger = logging.getLogger(__name__)


class EfficientDetConfig(Config, etal.HasPublishedModel):
    """EfficientDet configuration settings.

    Note that `labels_path` is passed through
    `eta.core.utils.fill_config_patterns` at load time, so it can contain
    patterns to be resolved.

    Attributes:
        model_name: the name of the published model to load. If this value is
            provided, `model_path` does not need to be
        model_path: the path to the archived checkpoint directory to use. If
            this value is provided, `model_name` does not need to be
        architecture_name: the name of the EfficientDet architecture for the
            model
        labels_path: the path to the labels map for the model
        confidence_thresh: a confidence threshold to apply to candidate
            detections
    """

    def __init__(self, d):
        d = self.init(d)

        self.architecture_name = self.parse_string(d, "architecture_name")
        self.labels_path = etau.fill_config_patterns(
            self.parse_string(d, "labels_path")
        )
        self.confidence_thresh = self.parse_number(
            d, "confidence_thresh", default=0
        )


class EfficientDet(etal.ObjectDetector, etat.UsesTFSession):
    """Interface to the EfficientDet detectors trained using the AutoML library
    at https://github.com/google/automl/tree/master/efficientdet.

    This class uses `eta.core.tfutils.UsesTFSession` to create TF sessions, so
    it automatically applies settings in your `eta.config.tf_config`.
    """

    def __init__(self, config):
        """Creates a EfficientDet instance.

        Args:
            config: a EfficientDetConfig instance
        """
        self.config = config
        etat.UsesTFSession.__init__(self)

        # Get path to model
        self.config.download_model_if_necessary()
        model_path = self.config.model_path

        # Extract archive, if necessary
        self._model_dir = etau.split_archive(model_path)[0]
        if not os.path.isdir(self._model_dir):
            logger.info("Extracting archive '%s'", model_path)
            etau.extract_archive(model_path)

        # Load class labels
        self._labels_map = etal.load_labels_map(self.config.labels_path)

        self._sess = None
        self._img = None
        self._detections = None
        self._preprocess = False

    def __enter__(self):
        sess, img, detections = self._load_model(self.config)
        self._sess = sess
        self._img = img
        self._detections = detections
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def ragged_batches(self):
        """True/False whether :meth:`transforms` may return images of different
        sizes and therefore passing ragged lists of images to
        :meth:`detect_all` is not allowed.
        """
        return True

    @property
    def transforms(self):
        """The preprocessing transformation that will be applied to each image
        before detection, or ``None`` if no preprocessing is performed.
        """
        return None

    @property
    def preprocess(self):
        """Whether to apply :meth:`transforms` during inference (True) or to
        assume that they have already been applied (False).
        """
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value):
        pass

    def detect(self, img):
        """Performs detection on the input image.

        Args:
            img: an image

        Returns:
            an `eta.core.objects.DetectedObjectContainer` describing the
                detections
        """
        return self._detect(img)

    def _detect(self, img):
        detections = self._evaluate(img)

        objects = etao.DetectedObjectContainer()
        for detection in detections:
            # detection = [image_id, x, y, width, height, score, class]
            box = detection[1:5]
            score = detection[5]
            class_id = int(detection[6])
            if (
                class_id in self._labels_map
                and score > self.config.confidence_thresh
            ):
                objects.add(
                    _to_detected_object(
                        box, score, class_id, img, self._labels_map
                    )
                )

        return objects

    def _evaluate(self, img):
        return self._sess.run(self._detections, feed_dict={self._img: img})

    def _load_model(self, config):
        tf.reset_default_graph()
        sess = self.make_tf_session()

        with etat.TFLoggingLevel(tf.logging.ERROR):
            with etau.CaptureStdout():
                img, detections = _load_efficientdet_model(
                    sess, config.architecture_name, self._model_dir
                )

        return sess, img, detections


def _load_efficientdet_model(sess, architecture_name, model_dir):
    """Loads an EfficientDet model from a checkpoint.

    Args:
        sess: a `tf.Session` to use
        architecture_name: the name of the EfficientDet architecture that
            you're loading
        model_dir: the TF models directory containing the checkpoint

    Returns:
        img_tensor: the input image `tf.Tensor`
        detections: the output detections `tf.Tensor`
    """
    _setup()

    # Get model parameters
    params = hparams_config.get_detection_config(architecture_name).as_dict()
    params.update(dict(is_training_bn=False, use_bfloat16=False))

    # Add preprocessing
    img_tensor = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    # pylint: disable=no-member
    image, scale = inference.image_preprocess(img_tensor, params["image_size"])
    images = tf.stack([image])
    scales = tf.stack([scale])

    # Build architecture from scratch
    class_outputs, box_outputs = efficientdet_arch.efficientdet(
        images, model_name=architecture_name, **params
    )
    sess.run(tf.global_variables_initializer())

    # Load checkpoint
    # pylint: disable=no-member
    inference.restore_ckpt(sess, model_dir, enable_ema=False)

    # Add postprocessing
    params.update(dict(batch_size=1))
    # pylint: disable=no-member
    detections_batch = inference.det_post_process(
        params, class_outputs, box_outputs, scales
    )
    detections = detections_batch[0]

    return img_tensor, detections


def _to_detected_object(box, score, class_id, img, labels_map):
    """Converts an EfficientDet detection to a DetectedObject.

    Args:
        box: [x, y, width, height]
        score: confidence score
        class_id: predicted class ID
        img: the image for which the detections were made
        labels_map: dictionary mapping class IDs to class names

    Returns:
        a DetectedObject
    """
    label = labels_map[class_id]

    bounding_box = etag.BoundingBox.from_abs_coords(
        box[0], box[1], box[0] + box[2], box[1] + box[3], img=img
    )

    return etao.DetectedObject(label, bounding_box, confidence=score)
