'''
Interface to the instance segmentation models from the TF object detection
library available at
https://github.com/tensorflow/models/tree/master/research/object_detection.

Copyright 2017-2019, Voxel51, Inc.
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
import sys

import numpy as np
import tensorflow as tf

import eta.constants as etac
from eta.core.config import Config, ConfigError
from eta.core.geometry import BoundingBox, RelativePoint
from eta.core.learning import ObjectDetector, HasDefaultDeploymentConfig
import eta.core.models as etam
from eta.core.objects import DetectedObject, DetectedObjectContainer
from eta.core.tfutils import UsesTFSession
import eta.core.utils as etau

sys.path.append(os.path.join(etac.TF_OBJECT_DETECTION_DIR, "utils"))
import label_map_util as gool


class TFModelsSegmenterConfig(Config, HasDefaultDeploymentConfig):
    '''TFModelsSegmenter configuration settings.

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
        model_path: the path to a TF SavedModel in `.pb` format to load. If
            this value is provided, `model_name` does not need to be
        labels_path: the path to the labels map for the model
        confidence_thresh: a confidence threshold to apply to candidate
            detections
        mask_thresh: the threshold to use when generating the instance masks
            for detections
    '''

    def __init__(self, d):
        self.model_name = self.parse_string(d, "model_name", default=None)
        self.model_path = self.parse_string(d, "model_path", default=None)

        # Loads any default deployment parameters, if possible
        if self.model_name:
            d = self.load_default_deployment_params(d, self.model_name)

        self.labels_path = etau.fill_config_patterns(
            self.parse_string(d, "labels_path"))
        self.confidence_thresh = self.parse_number(
            d, "confidence_thresh", default=None)
        self.mask_thresh = self.parse_number(d, "mask_thresh", default=0.5)

        self._validate()

    def _validate(self):
        if not self.model_name and not self.model_path:
            raise ConfigError(
                "Either `model_name` or `model_path` must be provided")


class TFModelsSegmenter(ObjectDetector, UsesTFSession):
    '''Interface to the instance segmentation models from the TF-Models
    detection library at
    https://github.com/tensorflow/models/tree/master/research/object_detection.

    This class uses `eta.core.tfutils.UsesTFSession` to create TF sessions, so
    it automatically applies settings in your `eta.config.tf_config`.

    Instances of this class must either use the context manager interface or
    manually call `close()` when finished to release memory.
    '''

    def __init__(self, config):
        '''Creates a TFModelsSegmenter instance.

        Args:
            config: a TFModelsSegmenterConfig instance
        '''
        self.config = config
        UsesTFSession.__init__(self)

        if self.config.model_name:
            # Downloads the published model, if necessary
            model_path = etam.download_model(self.config.model_name)
        else:
            model_path = self.config.model_path

        # Load model
        self._tf_graph = self._build_graph(model_path)
        self._sess = self.make_tf_session(graph=self._tf_graph)

        # Load labels
        label_map = gool.load_labelmap(self.config.labels_path)
        categories = gool.convert_label_map_to_categories(
            label_map, float("inf"), use_display_name=True)
        self._category_index = gool.create_category_index(categories)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def detect(self, img):
        '''Performs detection on the input image.

        Args:
            img: an image

        Returns:
            objects: An `eta.core.objects.DetectedObjectContainer` describing
                the detections
        '''
        img_exp = np.expand_dims(img, axis=0)
        image_tensor = self._tf_graph.get_tensor_by_name("image_tensor:0")
        boxes = self._tf_graph.get_tensor_by_name("detection_boxes:0")
        scores = self._tf_graph.get_tensor_by_name("detection_scores:0")
        classes = self._tf_graph.get_tensor_by_name("detection_classes:0")
        masks = self._tf_graph.get_tensor_by_name("detection_masks:0")

        boxes, scores, classes, masks = self._sess.run(
            [boxes, scores, classes, masks], feed_dict={image_tensor: img_exp})
        boxes, scores, classes, masks = map(
            np.squeeze, [boxes, scores, classes, masks])
        objects = [
            _to_detected_object(
                b, s, c, m, self._category_index, self.config.mask_thresh)
            for b, s, c, m in zip(boxes, scores, classes, masks)
            if c in self._category_index and (
                self.config.confidence_thresh is None or
                s > self.config.confidence_thresh)
        ]
        return DetectedObjectContainer(objects=objects)

    @staticmethod
    def _build_graph(model_path):
        tf_graph = tf.Graph()
        with tf_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
        return tf_graph


def _to_detected_object(
        box, score, class_id, mask_probs, label_map, mask_thresh):
    '''Converts a detection to a DetectedObject.

    Args:
        box (array): [ymin, xmin, ymax, xmax]
        score (float): confidence score
        class_id (int): predicted class ID
        mask_probs (array): a numpy array containing the mask probabilities
        label_map (dict): mapping from class IDs to names
        mask_thresh (float): the threshold to use when computing the instance
            mask for the detection

    Returns:
        a DetectedObject describing the detection
    '''
    return DetectedObject(
        label_map[class_id]["name"],
        BoundingBox(
            RelativePoint(box[1], box[0]),
            RelativePoint(box[3], box[2]),
        ),
        mask=(mask_probs >= mask_thresh),
        confidence=score,
    )