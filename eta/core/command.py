'''
Core module that defines the functionality of the `eta` command-line tool.

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
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import argparse
import logging
import sys

import eta
import eta.core.builder as etab
import eta.core.metadata as etam
import eta.core.pipeline as etap
import eta.core.serial as etas
import eta.core.utils as etau


logger = logging.getLogger(__name__)


class Command(object):
    '''Interface for defining ETA commands.'''

    @staticmethod
    def setup(parser):
        '''Setup the command-line arguments for the command.

        Args:
            parser: an argparse.ArgumentParser instance
        '''
        raise NotImplementedError("subclass must implement setup()")

    @staticmethod
    def run(args):
        '''Execute the command on the given args.

        args:
            args: an argparse.Namespace instance containing the arguments
                for the command
        '''
        raise NotImplementedError("subclass must implement run()")


class RunPipeline(Command):
    '''Command-line tool for running ETA pipelines.

    Examples:
        # Run the pipeline defined by a PipelineConfig JSON file
        eta run '/path/to/pipeline.json'
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument("config", help="path to a PipelineConfig file")

    @staticmethod
    def run(args):
        logger.info("Running ETA pipeline '%s'", args.config)
        etap.run(args.config)


class BuildPipeline(Command):
    '''Command-line tool for building ETA pipelines.

    Examples:
        # Build pipeline from a PipelineBuildRequest JSON file
        eta build -r '/path/to/pipeline/request.json'

        # Build pipeline from a PipelineBuildRequest dictionary
        eta build -r '{...}'

        # Build the pipeline request interactively
        eta build \\
            -n video_formatter \\
            -i '{"video": "/path/to/video.mp4"}' \\
            -p '{"resize_videos.scale": 0.5}'
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument("-r", "--request",
            type=etas.load_json, help="path to a PipelineBuildRequest file")
        parser.add_argument("-n", "--name",
            help="pipeline name")
        parser.add_argument("-i", "--inputs",
            type=etas.load_json, metavar="'{\"key\": val, ...}'",
            help="pipeline input(s)")
        parser.add_argument("-o", "--outputs",
            type=etas.load_json, metavar="'{\"key\": val, ...}'",
            help="pipeline output(s)")
        parser.add_argument("-p", "--parameters",
            type=etas.load_json, metavar="'{\"key\": val, ...}'",
            help="pipeline parameter(s)")
        parser.add_argument("--run-now",
            action="store_true", help="run the pipeline after building")
        parser.add_argument("--cleanup",
            action="store_true", help="delete all generated files after "
            "running the pipeline")

    @staticmethod
    def run(args):
        # PipelineBuildRequest dictionary
        d = args.request or {"inputs": {}, "outputs": {}, "parameters": {}}

        # Set values interactively
        if args.name:
            d["pipeline"] = args.name
        if args.inputs:
            d["inputs"].update(args.inputs)
        if args.outputs:
            d["outputs"].update(args.outputs)
        if args.parameters:
            d["parameters"].update(args.parameters)

        # Build pipeline
        logger.info("Loading pipeline build request")
        request = etab.PipelineBuildRequest.from_dict(d)
        logger.info("Building pipeline '%s' from request", request.pipeline)
        builder = etab.PipelineBuilder(request)
        builder.build()

        if args.run_now:
            # Run pipeline
            logger.info("Running pipeline '%s'", request.pipeline)
            builder.run()

        if args.cleanup:
            # Cleanup pipeline files
            logger.info("Cleaning up pipeline-generated files")
            builder.cleanup()


class GenerateMetadata(Command):
    '''Command-line tool for generating metadata files for ETA modules.

    Examples:
        # Generate the metadata file for the given ETA module
        eta metadata '/path/to/eta_module.py'
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument("module", help="path to an ETA module .py file")

    @staticmethod
    def run(args):
        logger.info("Generating metadata for ETA module '%s'", args.module)
        etam.generate(args.module)


def _register_command(cmd, cls):
    '''Registers the Commannd subclass `cls` as a command with the name `cmd`
    in the subparsers.
    '''
    parser = subparsers.add_parser(cmd,
        help=cls.__doc__.splitlines()[0],
        description=cls.__doc__.rstrip(),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.set_defaults(run=cls.run)
    cls.setup(parser)


# Main setup
parser = argparse.ArgumentParser(description="ETA command-line tool.")
parser.add_argument("-v", "--version",
    action="version", version=eta.version, help="show version info")
subparsers = parser.add_subparsers(title="available commands")

# Command setup
_register_command("run", RunPipeline)
_register_command("build", BuildPipeline)
_register_command("metadata", GenerateMetadata)


def main():
    '''Execute the ETA tool with the current command-line args.'''
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.run(args)
