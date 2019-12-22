'''
Interface to the TF models object detection library available at
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

import logging
import os
import sys

import numpy as np
import tensorflow as tf

import eta.constants as etac
from eta.core.config import Config, ConfigError
from eta.core.geometry import BoundingBox, RelativePoint
import eta.core.learning as etal
import eta.core.models as etam
from eta.core.objects import DetectedObject, DetectedObjectContainer
import eta.core.tfutils as etat
import eta.core.utils as etau

sys.path.append(os.path.join(etac.TF_OBJECT_DETECTION_DIR, "utils"))
import label_map_util as gool


logger = logging.getLogger(__name__)


class TFModelsDetectorConfig(Config, etal.HasDefaultDeploymentConfig):
    '''TFModelsDetector configuration settings.

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
        model_path: the path to a frozen inference graph to load. If this value
            is provided, `model_name` does not need to be
        labels_path: the path to the labels map for the model
        confidence_thresh: a confidence threshold to apply to candidate
            detections
        input_name: the name of the `tf.Operation` to use as input. If omitted,
            the default value "image_tensor" is used
        boxes_name: the name of the `tf.Operation` to use to extract the box
            coordinates. If omitted, the default value "detection_boxes" is
            used
        scores_name: the name of the `tf.Operation` to use to extract the
            detection scores. If omitted, the default value "detection_scores"
            is used
        classes_name: the name of the `tf.Operation` to use to extract the
            class indices. If omitted, the default value "detection_classes"
            is used
        features_name: the name of the `tf.Operation` to use to extract
            features for detections. If omitted, the default value
            "detection_features" is used
        generate_features: whether to generate features for detections. By
            default, this is False
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
        self.input_name = self.parse_number(
            d, "input_name", default="image_tensor")
        self.boxes_name = self.parse_number(
            d, "boxes_name", default="detection_boxes")
        self.scores_name = self.parse_number(
            d, "scores_name", default="detection_scores")
        self.classes_name = self.parse_number(
            d, "classes_name", default="detection_classes")
        self.features_name = self.parse_string(
            d, "features_name", default="detection_features")
        self.generate_features = self.parse_bool(
            d, "generate_features", default=False)

        self._validate()

    def _validate(self):
        if not self.model_name and not self.model_path:
            raise ConfigError(
                "Either `model_name` or `model_path` must be provided")


class TFModelsDetector(
        etal.ObjectDetector, etal.ExposesFeatures, etat.UsesTFSession):
    '''Interface to the TF-Models object detection library at
    https://github.com/tensorflow/models/tree/master/research/object_detection.

    This class uses `eta.core.tfutils.UsesTFSession` to create TF sessions, so
    it automatically applies settings in your `eta.config.tf_config`.

    This class implements the `eta.core.learning.ExposesFeatures` mixin, so
    it can expose features for its detections, if appropriately configured.

    Unfortunately, none of the pre-trained models available in ETA support
    featurization because the frozen graphs do not contain the expected
    `detection_features` nodes. In July 2019, the `tensorflow/models`
    repository was upgraded (https://github.com/tensorflow/models/pull/7208)
    to support a `detection_features` node that generates features for all
    models. It was also discovered that pre-trained models from the TF Models
    Zoo must be re-exported in order for this node to be populated
    (https://stackoverflow.com/a/57536793). Unfortunately, as of December 2019,
    this re-export process only seemed to work for Faster R-CNN models.

    Instances of this class must either use the context manager interface or
    manually call `close()` when finished to release memory.
    '''

    def __init__(self, config):
        '''Creates a TFModelsDetector instance.

        Args:
            config: a TFModelsDetectorConfig instance
        '''
        self.config = config
        etat.UsesTFSession.__init__(self)

        # Get path to model
        if self.config.model_path:
            model_path = self.config.model_path
        else:
            # Downloads the published model, if necessary
            model_path = etam.download_model(self.config.model_name)

        # Load model
        logger.info("Loading graph from '%s'", model_path)
        self._graph = etat.load_graph(model_path)
        self._sess = self.make_tf_session(graph=self._graph)

        # Load labels
        label_map = gool.load_labelmap(self.config.labels_path)
        categories = gool.convert_label_map_to_categories(
            label_map, float("inf"), use_display_name=True)
        self._category_index = gool.create_category_index(categories)

        # Get operations
        self._input_op = self._graph.get_operation_by_name(
            self.config.input_name)
        self._boxes_op = self._graph.get_operation_by_name(
            self.config.boxes_name)
        self._scores_op = self._graph.get_operation_by_name(
            self.config.scores_name)
        self._classes_op = self._graph.get_operation_by_name(
            self.config.classes_name)
        if self.config.generate_features and self.config.features_name:
            self._features_op = self._graph.get_operation_by_name(
                self.config.features_name)
        else:
            self._features_op = None

        # Feature vectors for the last detection
        self._last_features = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def exposes_features(self):
        '''Whether this detector exposes features for its detections.'''
        return self._features_op is not None

    @property
    def features_dim(self):
        '''The dimension of the features extracted by this detector, or None
        if it cannot generate features.
        '''
        if not self.exposes_features:
            return None

        dim = self._features_op.outputs[0].get_shape().as_list()[-1]
        if dim is None:
            logger.warning(
                "Unable to statically get feature dimension; returning None")

        return dim

    def get_features(self):
        '''Gets the features generated by the detector from its last call to
        `detect()`.

        Returns:
            a list of feature vectors, or None if the detector has not (or
                cannot) generate features
        '''
        if not self.exposes_features:
            return None

        return self._last_features

    def detect(self, img):
        '''Performs detection on the input image.

        Args:
            img: an image

        Returns:
            an `eta.core.objects.DetectedObjectContainer` describing the
                detections
        '''
        output_ops = [self._boxes_op, self._scores_op, self._classes_op]

        if self.exposes_features:
            output_ops.append(self._features_op)
            boxes, scores, classes, features = self._evaluate(img, output_ops)
            self._last_features = _avg_pool_features(features)
        else:
            boxes, scores, classes = self._evaluate(img, output_ops)

        objects = [
            _to_detected_object(b, s, c, self._category_index)
            for b, s, c in zip(boxes, scores, classes)
            if c in self._category_index and (
                self.config.confidence_thresh is None or
                s > self.config.confidence_thresh)
        ]
        return DetectedObjectContainer(objects=objects)

    def _evaluate(self, img, ops):
        imgs = np.expand_dims(img, axis=0)
        in_tensor = self._input_op.outputs[0]
        out_tensors = [op.outputs[0] for op in ops]
        results = self._sess.run(out_tensors, feed_dict={in_tensor: imgs})
        return map(np.squeeze, results)


class TFModelsSegmenterConfig(Config, etal.HasDefaultDeploymentConfig):
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
        model_path: the path to a frozen inference graph to load. If this value
            is provided, `model_name` does not need to be
        labels_path: the path to the labels map for the model
        confidence_thresh: a confidence threshold to apply to candidate
            detections
        mask_thresh: the threshold to use when generating the instance masks
            for detections
        input_name: the name of the `tf.Operation` to use as input. If omitted,
            the default value "image_tensor" is used
        boxes_name: the name of the `tf.Operation` to use to extract the box
            coordinates. If omitted, the default value "detection_boxes" is
            used
        scores_name: the name of the `tf.Operation` to use to extract the
            detection scores. If omitted, the default value "detection_scores"
            is used
        classes_name: the name of the `tf.Operation` to use to extract the
            class indices. If omitted, the default value "detection_classes"
            is used
        masks_name: the name of the `tf.Operation` to use to extract the
            instance masks. If omitted, the default value "detection_masks"
            is used
        features_name: the name of the `tf.Operation` to use to extract
            features for detections. If omitted, the default value
            "detection_features" is used
        generate_features: whether to generate features for detections. By
            default, this is False
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
        self.input_name = self.parse_number(
            d, "input_name", default="image_tensor")
        self.boxes_name = self.parse_number(
            d, "boxes_name", default="detection_boxes")
        self.scores_name = self.parse_number(
            d, "scores_name", default="detection_scores")
        self.classes_name = self.parse_number(
            d, "classes_name", default="detection_classes")
        self.masks_name = self.parse_number(
            d, "masks_name", default="detection_masks")
        self.features_name = self.parse_string(
            d, "features_name", default="detection_features")
        self.generate_features = self.parse_bool(
            d, "generate_features", default=False)

        self._validate()

    def _validate(self):
        if not self.model_name and not self.model_path:
            raise ConfigError(
                "Either `model_name` or `model_path` must be provided")


class TFModelsSegmenter(
        etal.ObjectDetector, etal.ExposesFeatures, etat.UsesTFSession):
    '''Interface to the instance segmentation models from the TF-Models
    detection library at
    https://github.com/tensorflow/models/tree/master/research/object_detection.

    This class uses `eta.core.tfutils.UsesTFSession` to create TF sessions, so
    it automatically applies settings in your `eta.config.tf_config`.

    This class implements the `eta.core.learning.ExposesFeatures` mixin, so
    it can expose features for its detections, if appropriately configured.

    Unfortunately, none of the pre-trained models available in ETA support
    featurization because the frozen graphs do not contain the expected
    `detection_features` nodes. In July 2019, the `tensorflow/models`
    repository was upgraded (https://github.com/tensorflow/models/pull/7208)
    to support a `detection_features` node that generates features for all
    models. It was also discovered that pre-trained models from the TF Models
    Zoo must be re-exported in order for this node to be populated
    (https://stackoverflow.com/a/57536793). Unfortunately, as of December 2019,
    this re-export process only seemed to work for Faster R-CNN models.

    Instances of this class must either use the context manager interface or
    manually call `close()` when finished to release memory.
    '''

    def __init__(self, config):
        '''Creates a TFModelsSegmenter instance.

        Args:
            config: a TFModelsSegmenterConfig instance
        '''
        self.config = config
        etat.UsesTFSession.__init__(self)

        # Get path to model
        if self.config.model_path:
            model_path = self.config.model_path
        else:
            # Downloads the published model, if necessary
            model_path = etam.download_model(self.config.model_name)

        # Load model
        logger.info("Loading graph from '%s'", model_path)
        self._graph = etat.load_graph(model_path)
        self._sess = self.make_tf_session(graph=self._graph)

        # Load labels
        label_map = gool.load_labelmap(self.config.labels_path)
        categories = gool.convert_label_map_to_categories(
            label_map, float("inf"), use_display_name=True)
        self._category_index = gool.create_category_index(categories)

        # Get operations
        self._input_op = self._graph.get_operation_by_name(
            self.config.input_name)
        self._boxes_op = self._graph.get_operation_by_name(
            self.config.boxes_name)
        self._scores_op = self._graph.get_operation_by_name(
            self.config.scores_name)
        self._classes_op = self._graph.get_operation_by_name(
            self.config.classes_name)
        self._masks_op = self._graph.get_operation_by_name(
            self.config.masks_name)
        if self.config.generate_features and self.config.features_name:
            self._features_op = self._graph.get_operation_by_name(
                self.config.features_name)
        else:
            self._features_op = None

        # Feature vectors for the last detection
        self._last_features = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def exposes_features(self):
        '''Whether this detector exposes features for its detections.'''
        return self._features_op is not None

    @property
    def features_dim(self):
        '''The dimension of the features extracted by this detector, or None
        if it does not expose features.
        '''
        if not self.exposes_features:
            return None

        dim = self._features_op.outputs[0].get_shape().as_list()[-1]
        if dim is None:
            logger.warning(
                "Unable to statically get feature dimension; returning None")

        return dim

    def get_features(self):
        '''Gets the features generated by the detector from its last call to
        `detect()`.

        Returns:
            a `num_images x features_dim` array of features, or None if the
                detector has not (or does not) generate features
        '''
        if not self.exposes_features:
            return None

        return self._last_features

    def detect(self, img):
        '''Performs detection on the input image.

        Args:
            img: an image

        Returns:
            an `eta.core.objects.DetectedObjectContainer` describing the
                detections
        '''
        output_ops = [
            self._boxes_op, self._scores_op, self._classes_op, self._masks_op]

        if self.exposes_features:
            output_ops.append(self._features_op)
            boxes, scores, classes, masks, features = self._evaluate(
                img, output_ops)
            self._last_features = _avg_pool_features(features)
        else:
            boxes, scores, classes, masks = self._evaluate(img, output_ops)

        objects = [
            _to_detected_object(
                b, s, c, self._category_index, mask_probs=m,
                mask_thresh=self.config.mask_thresh)
            for b, s, c, m in zip(boxes, scores, classes, masks)
            if c in self._category_index and (
                self.config.confidence_thresh is None or
                s > self.config.confidence_thresh)
        ]
        return DetectedObjectContainer(objects=objects)

    def _evaluate(self, img, ops):
        imgs = np.expand_dims(img, axis=0)
        in_tensor = self._input_op.outputs[0]
        out_tensors = [op.outputs[0] for op in ops]
        results = self._sess.run(out_tensors, feed_dict={in_tensor: imgs})
        return list(map(np.squeeze, results))


def export_frozen_inference_graph(
        checkpoint_path, pipeline_config_path, output_dir):
    '''Exports the given TF-Models checkpoint as a frozen inference graph
    suitable for running inference.

    Args:
        checkpoint_path: path to the training checkpoint to export
        pipeline_config_path: path to the pipeline config file for the
        output_dir: the directory in which to write the frozen inference graph
    '''
    # Import here because they are sooooo slow
    sys.path.append(etac.TF_OBJECT_DETECTION_DIR)
    from google.protobuf import text_format
    from object_detection import exporter
    from object_detection.protos import pipeline_pb2

    # Load pipeline config
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, "r") as f:
        text_format.Merge(f.read(), pipeline_config)

    # Export inference graph
    exporter.export_inference_graph(
        "image_tensor", pipeline_config, checkpoint_path, output_dir,
        input_shape=None)


def _avg_pool_features(features):
    axes = tuple(range(1, features.ndim - 1))
    if axes:
        return features.mean(axis=axes, keepdims=False)

    return features


def _to_detected_object(
        box, score, class_id, label_map, mask_probs=None, mask_thresh=None):
    '''Converts a detection to a DetectedObject.

    Args:
        box: [ymin, xmin, ymax, xmax]
        score: confidence score
        class_id: predicted class ID
        label_map: dictionary mapping class IDs to names
        mask_probs: an optional numpy array containing the mask probabilities,
            if supported by the model
        mask_thresh: an optional threshold to use when computing the instance
            mask for the detection, if masks are supported by the model

    Returns:
        a DetectedObject describing the detection
    '''
    label = label_map[class_id]["name"]

    bounding_box = BoundingBox(
        RelativePoint(box[1], box[0]),
        RelativePoint(box[3], box[2]),
    )

    if mask_probs is not None:
        mask = mask_probs
        if mask_thresh is not None:
            mask = (mask >= mask_thresh)
    else:
        mask = None

    return DetectedObject(label, bounding_box, mask=mask, confidence=score)
