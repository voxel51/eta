"""
Generic interface for performing inference on semantic segmentation models
stored as frozen TF graphs.

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

import numpy as np

from eta.core.config import Config
import eta.core.data as etad
import eta.core.image as etai
import eta.core.learning as etal
import eta.core.tfutils as etat
import eta.core.utils as etau


logger = logging.getLogger(__name__)


class TFSemanticSegmenterConfig(Config, etal.HasPublishedModel):
    """TFSemanticSegmenter configuration settings.

    Note that `labels_path` is passed through
    `eta.core.utils.fill_config_patterns` at load time, so it can contain
    patterns to be resolved.

    Attributes:
        model_name: the name of the published model to load. If this value is
            provided, `model_path` does not need to be
        model_path: the path to a frozen inference graph to load. If this value
            is provided, `model_name` does not need to be
        labels_path: the path to the labels map for the model
        resize_to_max_dim: resize input images so that their maximum dimension
            is equal to this value
        preprocessing_fcn: the fully-qualified name of a preprocessing function
            to use
        input_name: the name of the `tf.Operation` to which to feed the input
            image tensor
        output_name: the name of the `tf.Operation` from which to extract the
            output segmentation masks
        outputs_logits: whether the specified output node produces logits
            (True) or directly produces segmentation masks (False)
    """

    def __init__(self, d):
        d = self.init(d)

        _labels_path = self.parse_string(d, "labels_path", default=None)
        if _labels_path:
            _labels_path = etau.fill_config_patterns(_labels_path)
        self.labels_path = _labels_path

        self.resize_to_max_dim = self.parse_number(
            d, "resize_to_max_dim", default=None
        )
        self.preprocessing_fcn = self.parse_string(
            d, "preprocessing_fcn", default=None
        )
        self.input_name = self.parse_string(d, "input_name")
        self.output_name = self.parse_string(d, "output_name")
        self.outputs_logits = self.parse_bool(
            d, "outputs_logits", default=False
        )


class TFSemanticSegmenter(
    etal.ImageSemanticSegmenter, etal.ExposesMaskIndex, etat.UsesTFSession
):
    """Generic interface for semantic segmentation models stored as frozen TF
    graphs.

    This class uses `eta.core.tfutils.UsesTFSession` to create TF sessions, so
    it automatically applies settings in your `eta.config.tf_config`.
    """

    def __init__(self, config):
        """Creates a TFSemanticSegmenter instance.

        Args:
            config: a TFSemanticSegmenterConfig instance
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
        self._labels_map = None
        if self.config.labels_path:
            self._labels_map = etal.load_labels_map(self.config.labels_path)

        # Setup preprocessing
        self._transforms = self._make_preprocessing(config)
        self._preprocess = True

        # Get operations
        self._input_op = self._graph.get_operation_by_name(
            self.config.input_name
        )
        self._output_op = self._graph.get_operation_by_name(
            self.config.output_name
        )

    def __enter__(self):
        self._sess = self.make_tf_session(graph=self._graph)
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def ragged_batches(self):
        """True/False whether :meth:`transforms` may return images of different
        sizes and therefore passing ragged lists of images to
        :meth:`segment_all` is not allowed.
        """
        return True  # no guarantees on preprocessing output sizes

    @property
    def transforms(self):
        """The preprocessing transformation that will be applied to each image
        before segmentation, or ``None`` if no preprocessing is performed.
        """
        return self._transforms

    @property
    def preprocess(self):
        """Whether to apply :meth:`transforms` during inference (True) or to
        assume that they have already been applied (False).
        """
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value):
        self._preprocess = value

    @property
    def exposes_mask_index(self):
        """Whether this segmenter exposes a MaskIndex for its predictions."""
        return self._labels_map is not None

    def get_mask_index(self):
        """Returns the MaskIndex describing the semantics of this model's
        segmentations.

        Returns:
            A MaskIndex, or None if this model does not expose its mask index
        """
        if not self.exposes_mask_index:
            return None

        return etad.MaskIndex.from_labels_map(self._labels_map)

    def segment(self, img):
        """Performs segmentation on the input image.

        Args:
            img: an image

        Returns:
            an `eta.core.image.ImageLabels` instance containing the
                segmentation
        """
        return self._segment([img])[0]

    def segment_all(self, imgs):
        """Performs segmention on the given tensor of images.

        Args:
            imgs: a list (or n x h x w x 3 tensor) of images

        Returns:
            a list of `eta.core.image.ImageLabels` instances containing the
                segmentions
        """
        return self._segment(imgs)

    def _segment(self, imgs):
        if self.preprocess:
            imgs = self._preprocess_batch(imgs)

        masks = self._evaluate(imgs, [self._output_op])[0]

        if self.config.outputs_logits:
            masks = np.asarray(masks)
            masks = np.argmax(masks, axis=masks.ndim - 1)

        masks = np.asarray(masks, dtype=np.uint8)
        return [etai.ImageLabels(mask=mask) for mask in masks]

    def _preprocess_batch(self, imgs):
        if self._transforms is not None:
            imgs = [self._transforms(img) for img in imgs]

        return imgs

    def _evaluate(self, imgs, ops):
        in_tensor = self._input_op.outputs[0]
        out_tensors = [op.outputs[0] for op in ops]
        return self._sess.run(out_tensors, feed_dict={in_tensor: imgs})

    def _make_preprocessing(self, config):
        if self.config.resize_to_max_dim is not None:
            transform = lambda img: etai.resize_to_fit_max(
                img, config.resize_to_max_dim
            )
        else:
            transform = None

        if config.preprocessing_fcn:
            user_fcn = etau.get_function(config.preprocessing_fcn)
            if transform is not None:
                transform = lambda img: user_fcn(transform(img))
            else:
                transform = lambda img: user_fcn(img)

        return transform
