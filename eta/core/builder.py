'''
Core pipeline building system.

Copyright 2018, Voxel51, LLC
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
from future.utils import iteritems
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import


from eta.core.config import Config, Configurable
import eta.core.pipeline as etap


class PipelineRequestConfig(Config):
    '''Pipeline request configuration class.'''

    def __init__(self, d):
        self.pipeline = self.parse_string(d, "pipeline")
        self.inputs = self.parse_dict(d, "inputs", default={})
        self.parameters = self.parse_dict(d, "parameters", default={})


class PipelineRequest(Configurable):
    '''Pipeline request class.

    Attributes:
        name: the name of the pipeline to be run
        metadata: the PipelineMetadata instance for the pipeline
        inputs: a dictionary mapping input names to input paths
        parameters: a dictionary mapping <module>.<parameter> names to
            parameter values
    '''

    def __init__(self, config):
        '''Creates a new PipelineRequest instance.

        Args:
            config: a PipelineRequestConfig instance.

        Raises:
            PipelineRequestError: if the pipeline request was invalid
        '''
        self.validate(config)

        self.name = config.pipeline
        self.metadata = etap.load_metadata(config.pipeline)
        self.inputs = config.inputs
        self.parameters = config.parameters

        self._validate_inputs()
        self._validate_parameters()

    def _validate_inputs(self):
        # Validate inputs
        for iname, ival in iteritems(self.inputs):
            if not self.metadata.has_input(iname):
                raise PipelineRequestError(
                    "Pipeline '%s' has no input '%s'" % (self.name, iname))
            if not self.metadata.is_valid_input(iname, ival):
                raise PipelineRequestError((
                    "'%s' is not a valid value for input '%s' of pipeline "
                    "'%s'") % (ival, iname, self.name)
                )

        # Ensure that mandatory inputs were supplied
        for miname, miobj in iteritems(self.metadata.inputs):
            if miobj.is_mandatory and miname not in self.inputs:
                raise PipelineRequestError((
                    "Mandatory input '%s' of pipeline '%s' was not "
                    "specified") % (miname, self.name)
                )

    def _validate_parameters(self):
        # Validate parameters
        for pname, pval in iteritems(self.parameters):
            if not self.metadata.has_parameter(pname):
                raise PipelineRequestError(
                    "Pipeline '%s' has no parameter '%s'" % (self.name, pname))
            if not self.metadata.is_valid_parameter(pname, pval):
                raise PipelineRequestError((
                    "'%s' is not a valid value for parameter '%s' of pipeline "
                    "'%s'") % (pval, pname, self.name)
                )

        # Ensure that mandatory parmeters were supplied
        for mpname, mpobj in iteritems(self.metadata.parameters):
            if mpobj.is_mandatory and mpname not in self.parameters:
                raise PipelineRequestError((
                    "Mandatory parameter '%s' of pipeline '%s' was not "
                    "specified") % (mpname, self.name)
                )


class PipelineRequestError(Exception):
    pass
