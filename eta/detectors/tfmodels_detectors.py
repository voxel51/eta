"""
Interface to the TF-Models Object Detection Library available at
https://github.com/voxel51/models/tree/master/research/object_detection.

Copyright 2017-2022, Voxel51, Inc.
voxel51.com
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

import numpy as np

import eta.constants as etac
from eta.core.config import Config
from eta.core.geometry import BoundingBox
import eta.core.learning as etal
from eta.core.objects import DetectedObject, DetectedObjectContainer
import eta.core.tfutils as etat
import eta.core.utils as etau

from .utils import reset_path


def _setup():
    reset_path()

    sys.path.insert(1, etac.TF_RESEARCH_DIR)
    sys.path.insert(1, os.path.join(etac.TF_OBJECT_DETECTION_DIR, "utils"))


_ensure_tf = lambda: etau.ensure_import("tensorflow")
tf = etau.lazy_import("tensorflow", callback=_ensure_tf)
tf1 = etat.import_tf1()

_ERROR_MSG = "You must run `eta install models` in order to use this model"
gool = etau.lazy_import("label_map_util", error_msg=_ERROR_MSG)


logger = logging.getLogger(__name__)


class TFModelsDetectorConfig(Config, etal.HasPublishedModel):
    """TFModelsDetector configuration settings.

    Note that `labels_path` is passed through
    `eta.core.utils.fill_config_patterns` at load time, so it can contain
    patterns to be resolved.

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
        class_probs_name: the name of the `tf.Operation` to use to extract
            class probabilities for detections. If omitted, the default value
            "detection_multiclass_scores" is used
        generate_features: whether to generate features for detections. By
            default, this is False
        generate_class_probs: whether to generate class probabilities for
            detections. By default, this is False
    """

    def __init__(self, d):
        d = self.init(d)

        self.labels_path = etau.fill_config_patterns(
            self.parse_string(d, "labels_path")
        )
        self.confidence_thresh = self.parse_number(
            d, "confidence_thresh", default=0
        )
        self.input_name = self.parse_number(
            d, "input_name", default="image_tensor"
        )
        self.boxes_name = self.parse_number(
            d, "boxes_name", default="detection_boxes"
        )
        self.scores_name = self.parse_number(
            d, "scores_name", default="detection_scores"
        )
        self.classes_name = self.parse_number(
            d, "classes_name", default="detection_classes"
        )
        self.features_name = self.parse_string(
            d, "features_name", default="detection_features"
        )
        self.class_probs_name = self.parse_string(
            d, "class_probs_name", default="detection_multiclass_scores"
        )
        self.generate_features = self.parse_bool(
            d, "generate_features", default=False
        )
        self.generate_class_probs = self.parse_bool(
            d, "generate_class_probs", default=False
        )


class TFModelsDetector(
    etal.ObjectDetector,
    etal.ExposesFeatures,
    etal.ExposesProbabilities,
    etat.UsesTFSession,
):
    """Interface to the TF 1.X models from the TF-Models object detection
    library at
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
    """

    def __init__(self, config):
        """Creates a TFModelsDetector instance.

        Args:
            config: a TFModelsDetectorConfig instance
        """
        self.config = config
        etat.UsesTFSession.__init__(self)

        # Get path to model
        self.config.download_model_if_necessary()
        model_path = self.config.model_path

        # Load model
        self._graph = etat.load_graph(model_path)
        self._sess = None

        # Load labels
        self._category_index, self._class_labels = _parse_labels_map(
            self.config.labels_path
        )
        self._num_classes = len(self._class_labels)

        # Get operations
        self._input_op = self._graph.get_operation_by_name(
            self.config.input_name
        )
        self._boxes_op = self._graph.get_operation_by_name(
            self.config.boxes_name
        )
        self._scores_op = self._graph.get_operation_by_name(
            self.config.scores_name
        )
        self._classes_op = self._graph.get_operation_by_name(
            self.config.classes_name
        )
        if self.config.generate_features and self.config.features_name:
            self._features_op = self._graph.get_operation_by_name(
                self.config.features_name
            )
        else:
            self._features_op = None
        if self.config.generate_class_probs and self.config.class_probs_name:
            self._class_probs_op = self._graph.get_operation_by_name(
                self.config.class_probs_name
            )
        else:
            self._class_probs_op = None

        self._last_features = None
        self._last_probs = None
        self._preprocess = False

    def __enter__(self):
        self._sess = self.make_tf_session(graph=self._graph)
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

    @property
    def exposes_features(self):
        """Whether this detector exposes features for its detections."""
        return self._features_op is not None

    @property
    def features_dim(self):
        """The dimension of the features extracted by this detector, or None
        if it cannot generate features.
        """
        if not self.exposes_features:
            return None

        dim = self._features_op.outputs[0].get_shape().as_list()[-1]
        if dim is None:
            logger.warning(
                "Unable to statically get feature dimension; returning None"
            )

        return dim

    @property
    def exposes_probabilities(self):
        """Whether this detector exposes probabilities for predictions."""
        return self._class_probs_op is not None

    @property
    def num_classes(self):
        """The number of classes for the model."""
        return self._num_classes

    @property
    def class_labels(self):
        """The list of class labels generated by the detector."""
        return self._class_labels

    def get_features(self):
        """Gets the features generated by the detector from its last detection.

        Returns:
            an array of features, or None if the detector has not (or does not)
                generate features
        """
        if not self.exposes_features:
            return None

        return self._last_features

    def get_probabilities(self):
        """Gets the class probabilities generated by the detector from its last
        detection.

        Returns:
            an array of class probabilities, or None if the detector has not
                (or does not) generate probabilities
        """
        if not self.exposes_probabilities:
            return None

        return self._last_probs

    def detect(self, img):
        """Performs detection on the input image.

        Args:
            img: an image

        Returns:
            an `eta.core.objects.DetectedObjectContainer` describing the
                detections
        """
        return self._detect_all([img])[0]

    def detect_all(self, imgs):
        """Performs detection on the given tensor of images.

        Args:
            imgs: a list (or n x h x w x 3 tensor) of images

        Returns:
            a list of `eta.core.objects.DetectedObjectContainer`s describing
                the detections
        """
        return self._detect_all(imgs)

    def _detect_all(self, imgs):
        output_ops = [self._boxes_op, self._scores_op, self._classes_op]

        if self.exposes_features and self.exposes_probabilities:
            output_ops.extend([self._features_op, self._class_probs_op])
            boxes, scores, classes, features, probs = self._evaluate(
                imgs, output_ops
            )
        elif self.exposes_features:
            output_ops.append(self._features_op)
            boxes, scores, classes, features = self._evaluate(imgs, output_ops)
            probs = None
        elif self.exposes_probabilities:
            output_ops.append(self._class_probs_op)
            boxes, scores, classes, probs = self._evaluate(imgs, output_ops)
            features = None
        else:
            boxes, scores, classes = self._evaluate(imgs, output_ops)
            features = None
            probs = None

        if features is not None:
            features = _avg_pool_features(features)

        # Parse detections
        max_num_objects = 0
        detections = []
        for i, (b, s, c) in enumerate(zip(boxes, scores, classes)):
            keep = []
            objects = DetectedObjectContainer()
            for j, (boxj, scorej, classj) in enumerate(zip(b, s, c)):
                # Filter detections, if necessary
                if (
                    classj in self._category_index
                    and scorej > self.config.confidence_thresh
                ):
                    # Construct DetectedObject for detection
                    keep.append(j)
                    obj = _to_detected_object(
                        boxj, scorej, classj, self._category_index
                    )
                    objects.add(obj)

            # Record detections
            detections.append(objects)

            # Collect valid detections at beginning of arrays
            num_objects = len(keep)
            max_num_objects = max(num_objects, max_num_objects)
            if self.exposes_features:
                features[i, :num_objects, :] = features[i, keep, :]
            if self.exposes_probabilities:
                probs[i, :num_objects, :] = probs[i, keep, :]

        # Trim unnecessary dimensions
        if self.exposes_features:
            features = features[:, :max_num_objects, :]
        if self.exposes_probabilities:
            probs = probs[:, :max_num_objects, :]

        # Save data, if necessary
        if self.exposes_features:
            self._last_features = features  # n x num_objects x features_dim
        if self.exposes_probabilities:
            self._last_probs = probs  # n x num_objects x num_classes

        return detections

    def _evaluate(self, imgs, ops):
        in_tensor = self._input_op.outputs[0]
        out_tensors = [op.outputs[0] for op in ops]
        return self._sess.run(out_tensors, feed_dict={in_tensor: imgs})


class TF2ModelsDetectorConfig(Config, etal.HasPublishedModel):
    """TF2ModelsDetector configuration settings.

    Note that `labels_path` is passed through
    `eta.core.utils.fill_config_patterns` at load time, so it can contain
    patterns to be resolved.

    Attributes:
        model_name: the name of the published model to load. If this value is
            provided, `model_path` does not need to be
        model_path: the path to the `saved_model` to load. If this value
            is provided, `model_name` does not need to be
        labels_path: the path to the labels map for the model
        confidence_thresh: a confidence threshold to apply to candidate
            detections
    """

    def __init__(self, d):
        d = self.init(d)

        self.labels_path = etau.fill_config_patterns(
            self.parse_string(d, "labels_path")
        )
        self.confidence_thresh = self.parse_number(
            d, "confidence_thresh", default=0
        )


class TF2ModelsDetector(etal.ObjectDetector):
    """Interface to the TF 2.X models from the TF-Models object detection
    library at
    https://github.com/tensorflow/models/tree/master/research/object_detection.
    """

    def __init__(self, config):
        """Creates a TF2ModelsDetector instance.

        Args:
            config: a TF2ModelsDetectorConfig instance
        """
        self.config = config

        # Get path to model
        self.config.download_model_if_necessary()
        model_path = self.config.model_path

        # Extract archive, if necessary
        self._model_dir = etau.split_archive(model_path)[0]
        if not os.path.isdir(self._model_dir):
            logger.info("Extracting archive '%s'", model_path)
            etau.extract_archive(model_path)

        # Load labels
        self._category_index, self._class_labels = _parse_labels_map(
            self.config.labels_path
        )
        self._num_classes = len(self._class_labels)

        self._detect_fn = None
        self._preprocess = False

    def __enter__(self):
        self._detect_fn = _load_tf2_detection_model(self._model_dir)
        return self

    def __exit__(self, *args):
        pass

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

    @property
    def num_classes(self):
        """The number of classes for the model."""
        return self._num_classes

    @property
    def class_labels(self):
        """The list of class labels generated by the detector."""
        return self._class_labels

    def detect(self, img):
        """Performs detection on the input image.

        Args:
            img: an image

        Returns:
            an `eta.core.objects.DetectedObjectContainer` describing the
                detections
        """
        return self._detect_all([img])[0]

    def detect_all(self, imgs):
        """Performs detection on the given tensor of images.

        Args:
            imgs: a list (or n x h x w x 3 tensor) of images

        Returns:
            a list of `eta.core.objects.DetectedObjectContainer`s describing
                the detections
        """
        return self._detect_all(imgs)

    def _detect_all(self, imgs):
        imgs = tf.convert_to_tensor(imgs, dtype=tf.uint8)
        boxes, scores, classes = self._evaluate(imgs)

        # Parse detections
        max_num_objects = 0
        detections = []
        for i, (b, s, c) in enumerate(zip(boxes, scores, classes)):
            keep = []
            objects = DetectedObjectContainer()
            for j, (boxj, scorej, classj) in enumerate(zip(b, s, c)):
                boxj = boxj.numpy()
                scorej = scorej.numpy()
                classj = classj.numpy()
                # Filter detections, if necessary
                if (
                    classj in self._category_index
                    and scorej > self.config.confidence_thresh
                ):
                    # Construct DetectedObject for detection
                    keep.append(j)
                    obj = _to_detected_object(
                        boxj, scorej, classj, self._category_index
                    )
                    objects.add(obj)

            # Record detections
            detections.append(objects)

            # Collect valid detections at beginning of arrays
            num_objects = len(keep)
            max_num_objects = max(num_objects, max_num_objects)

        return detections

    def _evaluate(self, imgs):
        return self._detect_fn(imgs)


class TFModelsInstanceSegmenterConfig(Config, etal.HasPublishedModel):
    """TFModelsInstanceSegmenter configuration settings.

    Note that `labels_path` is passed through
    `eta.core.utils.fill_config_patterns` at load time, so it can contain
    patterns to be resolved.

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
        class_probs_name: the name of the `tf.Operation` to use to extract
            class probabilities for detections. If omitted, the default value
            "detection_multiclass_scores" is used
        generate_features: whether to generate features for detections. By
            default, this is False
    """

    def __init__(self, d):
        d = self.init(d)

        self.labels_path = etau.fill_config_patterns(
            self.parse_string(d, "labels_path")
        )
        self.confidence_thresh = self.parse_number(
            d, "confidence_thresh", default=0
        )
        self.mask_thresh = self.parse_number(d, "mask_thresh", default=0.5)
        self.input_name = self.parse_number(
            d, "input_name", default="image_tensor"
        )
        self.boxes_name = self.parse_number(
            d, "boxes_name", default="detection_boxes"
        )
        self.scores_name = self.parse_number(
            d, "scores_name", default="detection_scores"
        )
        self.classes_name = self.parse_number(
            d, "classes_name", default="detection_classes"
        )
        self.masks_name = self.parse_number(
            d, "masks_name", default="detection_masks"
        )
        self.features_name = self.parse_string(
            d, "features_name", default="detection_features"
        )
        self.class_probs_name = self.parse_string(
            d, "class_probs_name", default="detection_multiclass_scores"
        )
        self.generate_features = self.parse_bool(
            d, "generate_features", default=False
        )
        self.generate_class_probs = self.parse_bool(
            d, "generate_class_probs", default=False
        )


class TFModelsInstanceSegmenter(
    etal.ObjectDetector,
    etal.ExposesFeatures,
    etal.ExposesProbabilities,
    etat.UsesTFSession,
):
    """Interface to the instance segmentation models from the TF-Models
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
    """

    def __init__(self, config):
        """Creates a TFModelsInstanceSegmenter instance.

        Args:
            config: a TFModelsInstanceSegmenterConfig instance
        """
        self.config = config
        etat.UsesTFSession.__init__(self)

        # Get path to model
        self.config.download_model_if_necessary()
        model_path = self.config.model_path

        # Load model
        self._graph = etat.load_graph(model_path)
        self._sess = None

        # Load labels
        self._category_index, self._class_labels = _parse_labels_map(
            self.config.labels_path
        )
        self._num_classes = len(self._class_labels)

        # Get operations
        self._input_op = self._graph.get_operation_by_name(
            self.config.input_name
        )
        self._boxes_op = self._graph.get_operation_by_name(
            self.config.boxes_name
        )
        self._scores_op = self._graph.get_operation_by_name(
            self.config.scores_name
        )
        self._classes_op = self._graph.get_operation_by_name(
            self.config.classes_name
        )
        self._masks_op = self._graph.get_operation_by_name(
            self.config.masks_name
        )
        if self.config.generate_features and self.config.features_name:
            self._features_op = self._graph.get_operation_by_name(
                self.config.features_name
            )
        else:
            self._features_op = None
        if self.config.generate_class_probs and self.config.class_probs_name:
            self._class_probs_op = self._graph.get_operation_by_name(
                self.config.class_probs_name
            )
        else:
            self._class_probs_op = None

        self._last_features = None
        self._last_probs = None
        self._preprocess = False

    def __enter__(self):
        self._sess = self.make_tf_session(graph=self._graph)
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

    @property
    def exposes_features(self):
        """Whether this detector exposes features for its detections."""
        return self._features_op is not None

    @property
    def features_dim(self):
        """The dimension of the features extracted by this detector, or None
        if it does not expose features.
        """
        if not self.exposes_features:
            return None

        dim = self._features_op.outputs[0].get_shape().as_list()[-1]
        if dim is None:
            logger.warning(
                "Unable to statically get feature dimension; returning None"
            )

        return dim

    @property
    def exposes_probabilities(self):
        """Whether this detector exposes probabilities for predictions."""
        return self._class_probs_op is not None

    @property
    def num_classes(self):
        """The number of classes for the model."""
        return self._num_classes

    @property
    def class_labels(self):
        """The list of class labels generated by the detector."""
        return self._class_labels

    def get_features(self):
        """Gets the features generated by the detector from its last detection.

        Returns:
            an array of features, or None if the detector has not (or does not)
                generate features
        """
        if not self.exposes_features:
            return None

        return self._last_features

    def get_probabilities(self):
        """Gets the class probabilities generated by the detector from its last
        detection.

        Returns:
            an array of class probabilities, or None if the detector has not
                (or does not) generate probabilities
        """
        if not self.exposes_probabilities:
            return None

        return self._last_probs

    def detect(self, img):
        """Performs detection on the input image.

        Args:
            img: an image

        Returns:
            an `eta.core.objects.DetectedObjectContainer` describing the
                detections
        """
        return self._detect_all([img])[0]

    def detect_all(self, imgs):
        """Performs detection on the given tensor of images.

        Args:
            imgs: a list (or n x h x w x 3 tensor) of images

        Returns:
            a list of `eta.core.objects.DetectedObjectContainer`s describing
                the detections
        """
        return self._detect_all(imgs)

    def _detect_all(self, imgs):
        output_ops = [
            self._boxes_op,
            self._scores_op,
            self._classes_op,
            self._masks_op,
        ]

        # Perform inference
        if self.exposes_features and self.exposes_probabilities:
            output_ops.extend([self._features_op, self._class_probs_op])
            boxes, scores, classes, masks, features, probs = self._evaluate(
                imgs, output_ops
            )
        elif self.exposes_features:
            output_ops.append(self._features_op)
            boxes, scores, classes, masks, features = self._evaluate(
                imgs, output_ops
            )
            probs = None
        elif self.exposes_probabilities:
            output_ops.append(self._class_probs_op)
            boxes, scores, classes, masks, probs = self._evaluate(
                imgs, output_ops
            )
            features = None
        else:
            boxes, scores, classes, masks = self._evaluate(imgs, output_ops)
            features = None
            probs = None

        if features is not None:
            features = _avg_pool_features(features)

        # Parse detections
        max_num_objects = 0
        detections = []
        for i, (b, s, c, m) in enumerate(zip(boxes, scores, classes, masks)):
            keep = []
            objects = DetectedObjectContainer()
            for j, (boxj, scorej, classj, maskj) in enumerate(zip(b, s, c, m)):
                # Filter detections, if necessary
                if (
                    classj in self._category_index
                    and scorej > self.config.confidence_thresh
                ):
                    # Construct DetectecObject for detection
                    keep.append(j)
                    obj = _to_detected_object(
                        boxj,
                        scorej,
                        classj,
                        self._category_index,
                        mask_probs=maskj,
                        mask_thresh=self.config.mask_thresh,
                    )
                    objects.add(obj)

            # Record detections
            detections.append(objects)

            # Collect valid detections at beginning of arrays
            num_objects = len(keep)
            max_num_objects = max(num_objects, max_num_objects)
            if self.exposes_features:
                features[i, :num_objects, :] = features[i, keep, :]
            if self.exposes_probabilities:
                probs[i, :num_objects, :] = probs[i, keep, :]

        # Trim unnecessary dimensions
        if self.exposes_features:
            features = features[:, :max_num_objects, :]
        if self.exposes_probabilities:
            probs = probs[:, :max_num_objects, :]

        # Save data, if necessary
        if self.exposes_features:
            self._last_features = features  # n x num_objects x features_dim
        if self.exposes_probabilities:
            self._last_probs = probs  # n x num_objects x num_classes

        return detections

    def _evaluate(self, imgs, ops):
        in_tensor = self._input_op.outputs[0]
        out_tensors = [op.outputs[0] for op in ops]
        return self._sess.run(out_tensors, feed_dict={in_tensor: imgs})


def export_frozen_inference_graph(
    checkpoint_path, pipeline_config_path, output_dir
):
    """Exports the given TF-Models checkpoint as a frozen inference graph
    suitable for running inference.

    Args:
        checkpoint_path: path to the training checkpoint to export
        pipeline_config_path: path to the pipeline config file for the graph
        output_dir: the directory in which to write the frozen inference graph
    """
    # Import here because they are sooooo slow
    sys.path.append(etac.TF_OBJECT_DETECTION_DIR)
    from google.protobuf import text_format
    from object_detection import exporter  # pylint: disable=import-error

    # pylint: disable=import-error
    from object_detection.protos import pipeline_pb2

    # Load pipeline config
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf1.gfile.GFile(pipeline_config_path, "r") as f:
        text_format.Merge(f.read(), pipeline_config)

    # Export inference graph
    exporter.export_inference_graph(
        "image_tensor",
        pipeline_config,
        checkpoint_path,
        output_dir,
        input_shape=None,
    )


def _avg_pool_features(features):
    axes = tuple(range(2, features.ndim - 1))
    if axes:
        return features.mean(axis=axes, keepdims=False)

    return features


def _load_tf2_detection_model(model_dir):
    """Loads the inference function for a detection model from the TF2 Model
    Zoo.

    References:

        -   https://github.com/abdelrahman-gaber/tf2-object-detection-api-tutorial
        -   https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb

    Args:
        model_dir: a directory containing the saved model in a `saved_model`
            subdirectory

    Returns:
        a `tf.function` that performs prediction on an image and returns a
        `(boxes, scores, clases)` tuple
    """
    with etat.TFLoggingLevel(tf1.logging.ERROR):
        with etau.CaptureStdout():
            detect_fn = tf.saved_model.load(
                os.path.join(model_dir, "saved_model")
            )

    def predict(image):
        detections = detect_fn(image)

        return (
            detections["detection_boxes"],
            detections["detection_scores"],
            detections["detection_classes"],
        )

    return predict


def _parse_labels_map(labels_path):
    """Loads a labels map via the `label_map_util` from the TF-Models library.

    Args:
        labels_path: path to a labels map in `.pbtxt` format

    Returns:
        (category_index, class_labels), where `category_index` is a dict
            mapping IDs to names, and `class_labels` is a list of class names
            for all IDs sequentially from `min(1, min(category_index))` to
            `max(category_index)`
    """
    _setup()

    labelmap = gool.load_labelmap(labels_path)
    category_index = _parse_labelmap_proto(labelmap)

    mini = min(1, min(category_index))
    maxi = max(category_index)
    class_labels = [
        category_index.get(i, "class %d" % i) for i in range(mini, maxi + 1)
    ]

    return category_index, class_labels


def _parse_labelmap_proto(labelmap):
    """Converts a labelmap proto into a category index.

    Adapted from `tensorflow/models/research/object_detection/utils/label_map_util.py`,
    which was necessary to properly include `id == 0` in the index, if present.

    Args:
        a StringIntLabelMapProto

    Returns:
        a dictionary mapping class IDs to class names
    """
    category_index = {}
    for item in labelmap.item:
        if item.HasField("display_name"):
            name = item.display_name
        else:
            name = item.name

        category_index[item.id] = name

    return category_index


def _to_detected_object(
    box, score, class_id, category_index, mask_probs=None, mask_thresh=None
):
    """Converts a TF-Models detection to a DetectedObject.

    Args:
        box: [ymin, xmin, ymax, xmax]
        score: confidence score
        class_id: predicted class ID
        category_index: dictionary mapping class IDs to class names
        mask_probs: an optional numpy array containing the mask probabilities,
            if supported by the model
        mask_thresh: an optional threshold to use when computing the instance
            mask for the detection, if masks are supported by the model

    Returns:
        a DetectedObject
    """
    label = category_index[class_id]

    bounding_box = BoundingBox.from_coords(box[1], box[0], box[3], box[2])

    if mask_probs is not None:
        mask = mask_probs
        if mask_thresh is not None:
            mask = mask >= mask_thresh
    else:
        mask = None

    return DetectedObject(label, bounding_box, mask=mask, confidence=score)
