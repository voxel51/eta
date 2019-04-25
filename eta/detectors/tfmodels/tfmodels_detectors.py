'''
Interface to the TensorFlow Models object detection library available at
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
from eta.core.config import Config
from eta.core.geometry import BoundingBox, RelativePoint
from eta.core.learning import ObjectDetector
import eta.core.models as etam
from eta.core.objects import DetectedObject, DetectedObjectContainer
import eta.core.serial as etas
import eta.core.utils as etau

sys.path.insert(
    1, os.path.join(etac.TF_MODELS_DIR, "research/object_detection"))
import utils.label_map_util as gool


# Default TFModelsDetectorConfig
DEFAULT_CONFIG = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "default-config.json")


class TFModelsDetectorConfig(Config):
    '''TFModelsDetector configuration settings.

    Note that `labels_path` may contain the pattern `{{eta}}`, which will
    be replaced with the `/path/to/eta` when the config is loaded.
    '''

    def __init__(self, d):
        self.model_name = self.parse_string(d, "model_name")
        _labels_path = self.parse_string(d, "labels_path")
        self.labels_path = etau.fill_eta_pattern(_labels_path)

    @classmethod
    def load_default(cls):
        '''Loads the default TFModelsDetectorConfig.'''
        return cls.from_json(DEFAULT_CONFIG)


class TFModelsDetector(ObjectDetector):
    '''Interface to the `tensorflow/models` object detection library.

    https://github.com/tensorflow/models/tree/master/research/object_detection.
    '''

    def __init__(self, config=None):
        '''Constructs a TFModelsDetector instance.

        Args:
            config: an optional TFModelsDetectorConfig instance. If absent, the
                default TFModelsDetectorConfig will be used
        '''
        self.config = config or TFModelsDetectorConfig.load_default()

        # Only downloads the model if necessary
        model_path = etam.download_model(self.config.model_name)

        # Load model
        self._tf_graph = self._build_graph(model_path)
        self._sess = tf.Session(graph=self._tf_graph)

        # Load labels
        label_map = gool.load_labelmap(self.config.labels_path)
        categories = gool.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        self._category_index = gool.create_category_index(categories)

    @staticmethod
    def _build_graph(model_path):
        tf_graph = tf.Graph()
        with tf_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
        return tf_graph

    def detect(self, img):
        '''Performs object detection on the input image.

        Args:
            img: an image

        Returns:
            objects: A DetectedObjectContainer describing the detected objects
                in the image
        '''
        img_exp = np.expand_dims(img, axis=0)
        image_tensor = self._tf_graph.get_tensor_by_name("image_tensor:0")
        boxes = self._tf_graph.get_tensor_by_name("detection_boxes:0")
        scores = self._tf_graph.get_tensor_by_name("detection_scores:0")
        classes = self._tf_graph.get_tensor_by_name("detection_classes:0")
        boxes, scores, classes = self._sess.run(
            [boxes, scores, classes], feed_dict={image_tensor: img_exp})
        boxes, scores, classes = map(np.squeeze, [boxes, scores, classes])
        objects = [
            _to_detected_object(b, s, c, self._category_index)
            for b, s, c in zip(boxes, scores, classes)
        ]
        return DetectedObjectContainer(objects=objects)


def _to_detected_object(box, score, class_id, label_map):
    '''Converts a tensorflow/models prediction dictionary to a DetectedObject.

    Args:
        box (array): [ymin, xmin, ymax, xmax]
        score (float): confidence score
        class_id (int): predicted class ID
        label_map (dict): mapping from class IDs to names
    '''
    return DetectedObject(
        label_map[class_id]["name"],
        BoundingBox(
            RelativePoint(box[1], box[0]),
            RelativePoint(box[3], box[2]),
        ),
        confidence=score,
    )
