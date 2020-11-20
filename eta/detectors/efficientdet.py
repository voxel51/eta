"""
Interface to the EfficientDet object detection model.

This module assumes that the
`voxel51/automl <https://github.com/voxel51/automl>`_ repository has been
cloned on your machine.

In particulr, this model wraps the EfficientDet implementation at
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

import tensorflow as tf

import eta.constants as etac
from eta.core.config import Config, ConfigError
import eta.core.geometry as etag
import eta.core.learning as etal
import eta.core.models as etam
import eta.core.objects as etao
import eta.core.tfutils as etat
import eta.core.utils as etau


_ERROR_MSG = """

You must clone a GitHub repository in order to use an EfficientDet model:

mkdir -p '{0}'
git clone https://github.com/voxel51/automl '{0}'

""".format(
    etac.AUTOML_DIR
)

try:
    #
    # Prevents possible name clashes when
    # `{{eta}}/tensorflow/models/research/object_detection` has previously
    # been imported
    #
    # @todo find a better solution for this
    #
    sys.modules.pop("object_detection")
except KeyError:
    pass

sys.path.insert(1, etac.EFFICIENTDET_DIR)
efficientdet_arch = etau.lazy_import("efficientdet_arch", error_msg=_ERROR_MSG)
hparams_config = etau.lazy_import("hparams_config", error_msg=_ERROR_MSG)
inference = etau.lazy_import("inference", error_msg=_ERROR_MSG)


logger = logging.getLogger(__name__)


class EfficientDetConfig(Config, etal.HasDefaultDeploymentConfig):
    """EfficientDet configuration settings.

    Note that `labels_path` is passed through
    `eta.core.utils.fill_config_patterns` at load time, so it can contain
    patterns to be resolved.

    Note that this class implements the `HasDefaultDeploymentConfig` mixin, so
    if a published model is provided via the `model_name` attribute, then any
    omitted fields present in the default deployment config for the published
    model will be automatically populated.

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
        self.model_name = self.parse_string(d, "model_name", default=None)
        self.model_path = self.parse_string(d, "model_path", default=None)

        # Loads any default deployment parameters, if possible
        if self.model_name:
            d = self.load_default_deployment_params(d, self.model_name)

        self.architecture_name = self.parse_string(d, "architecture_name")
        self.labels_path = etau.fill_config_patterns(
            self.parse_string(d, "labels_path")
        )
        self.confidence_thresh = self.parse_number(
            d, "confidence_thresh", default=0
        )

        self._validate()

    def _validate(self):
        if not self.model_name and not self.model_path:
            raise ConfigError(
                "Either `model_name` or `model_path` must be provided"
            )


class EfficientDet(etal.ObjectDetector, etat.UsesTFSession):
    """Interface to the EfficientDet detectors trained using the AutoML library
    at https://github.com/google/automl/tree/master/efficientdet.

    This class uses `eta.core.tfutils.UsesTFSession` to create TF sessions, so
    it automatically applies settings in your `eta.config.tf_config`.

    Instances of this class must either use the context manager interface or
    manually call `close()` when finished to release memory.
    """

    def __init__(self, config):
        """Creates a EfficientDet instance.

        Args:
            config: a EfficientDetConfig instance
        """
        self.config = config
        etat.UsesTFSession.__init__(self)

        # Get path to model
        if self.config.model_path:
            model_path = self.config.model_path
        else:
            # Downloads the published model, if necessary
            model_path = etam.download_model(self.config.model_name)

        # Extract archive, if necessary
        model_dir = os.path.splitext(model_path)[0]
        if not os.path.isdir(model_dir):
            logger.info("Extracting archive '%s'", model_path)
            etau.extract_archive(model_path)

        # Load model
        logger.info("Loading model from '%s'", model_dir)
        self._sess = self.make_tf_session()
        self._img_tensor, self._detections = _load_efficientdet_model(
            self._sess, self.config.architecture_name, model_dir
        )

        # Load class labels
        logger.info("Loading class labels from '%s'", self.config.labels_path)
        self._labels_map = etal.load_labels_map(self.config.labels_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

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
        # Perform inference
        detections = self._evaluate(img)

        # Parse detections
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
        return self._sess.run(
            self._detections, feed_dict={self._img_tensor: img}
        )


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
