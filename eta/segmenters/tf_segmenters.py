'''
Generic interface for performing inference on semantic segmentation models
stored as frozen TF graphs.

Copyright 2017-2020, Voxel51, Inc.
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

from eta.core.config import Config, ConfigError
import eta.core.data as etad
import eta.core.image as etai
import eta.core.learning as etal
import eta.core.models as etam
import eta.core.tfutils as etat
import eta.core.utils as etau


logger = logging.getLogger(__name__)


class TFSemanticSegmenterConfig(Config, etal.HasDefaultDeploymentConfig):
    '''TFSemanticSegmenter configuration settings.

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
        resize_to_max_dim: resize input images so that their maximum dimension
            is equal to this value
        input_name: the name of the `tf.Operation` to which to feed the input
            image tensor
        output_name: the name of the `tf.Operation` from which to extract the
            output segmentation masks
    '''

    def __init__(self, d):
        self.model_name = self.parse_string(d, "model_name", default=None)
        self.model_path = self.parse_string(d, "model_path", default=None)

        # Loads any default deployment parameters, if possible
        if self.model_name:
            d = self.load_default_deployment_params(d, self.model_name)

        _labels_path = self.parse_string(d, "labels_path", default=None)
        if _labels_path:
            _labels_path = etau.fill_config_patterns(_labels_path)
        self.labels_path = _labels_path

        self.resize_to_max_dim = self.parse_number(d, "resize_to_max_dim")
        self.input_name = self.parse_string(d, "input_name")
        self.output_name = self.parse_string(d, "output_name")

        self._validate()

    def _validate(self):
        if not self.model_name and not self.model_path:
            raise ConfigError(
                "Either `model_name` or `model_path` must be provided")


class TFSemanticSegmenter(
        etal.ImageSemanticSegmenter, etal.ExposesMaskIndex,
        etat.UsesTFSession):
    '''Generic interface for semantic segmentation models stored as frozen TF
    graphs.

    This class uses `eta.core.tfutils.UsesTFSession` to create TF sessions, so
    it automatically applies settings in your `eta.config.tf_config`.

    Instances of this class must either use the context manager interface or
    manually call `close()` when finished to release memory.
    '''

    def __init__(self, config):
        '''Creates a TFSemanticSegmenter instance.

        Args:
            config: a TFSemanticSegmenterConfig instance
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
        self._labels_map = None
        if self.config.labels_path:
            self._labels_map = etal.load_labels_map(self.config.labels_path)

        # Get operations
        self._input_op = self._graph.get_operation_by_name(
            self.config.input_name)
        self._output_op = self._graph.get_operation_by_name(
            self.config.output_name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def exposes_mask_index(self):
        '''Whether this segmenter exposes a MaskIndex for its predictions.'''
        return self._labels_map is not None

    def get_mask_index(self):
        '''Returns the MaskIndex describing the semantics of this model's
        segmentations.

        Returns:
            A MaskIndex, or None if this model does not expose its mask index
        '''
        if not self.exposes_mask_index:
            return None

        return etad.MaskIndex.from_labels_map(self._labels_map)

    def segment(self, img):
        '''Performs segmentation on the input image.

        Args:
            img: an image

        Returns:
            an `eta.core.image.ImageLabels` instance containing the
                segmentation
        '''
        return self._segment([img])[0]

    def segment_all(self, imgs):
        '''Performs segmention on the given tensor of images.

        Args:
            imgs: a list (or n x h x w x 3 tensor) of images

        Returns:
            a list of `eta.core.image.ImageLabels` instances containing the
                segmentions
        '''
        return self._segment(imgs)

    def _segment(self, imgs):
        #
        # Preprocessing: resize so that maximum dimension is equal to the
        # specified value
        #
        imgs = [
            etai.resize_to_fit_max(img, self.config.resize_to_max_dim)
            for img in imgs]

        masks = self._evaluate(imgs, [self._output_op])
        return [etai.ImageLabels(mask=mask[0]) for mask in masks]

    def _evaluate(self, imgs, ops):
        in_tensor = self._input_op.outputs[0]
        out_tensors = [op.outputs[0] for op in ops]
        return self._sess.run(out_tensors, feed_dict={in_tensor: imgs})
