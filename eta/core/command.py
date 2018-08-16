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
from future.utils import iteritems, itervalues
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import argparse
from collections import defaultdict
import logging
import os
import sys

import eta
import eta.core.builder as etab
import eta.core.metadata as etame
import eta.core.models as etamode
import eta.core.module as etamodu
import eta.core.pipeline as etap
import eta.core.serial as etas


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


class BuildCommand(Command):
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
            -p '{"format_videos.scale": 0.5}'
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "-r", "--request", type=etas.load_json,
            help="path to a PipelineBuildRequest file")
        parser.add_argument("-n", "--name", help="pipeline name")
        parser.add_argument(
            "-i", "--inputs", type=etas.load_json,
            metavar="'{\"key\": val, ...}'", help="pipeline inputs")
        parser.add_argument(
            "-o", "--outputs", type=etas.load_json,
            metavar="'{\"key\": val, ...}'", help="pipeline outputs")
        parser.add_argument(
            "-p", "--parameters", type=etas.load_json,
            metavar="'{\"key\": val, ...}'", help="pipeline parameters")
        parser.add_argument(
            "-l", "--logging", type=etas.load_json,
            metavar="'{\"key\": val, ...}'", help="logging config settings")
        parser.add_argument(
            "--run-now", action="store_true",
            help="run the pipeline after building")
        parser.add_argument(
            "--cleanup", action="store_true",
            help="delete all generated files after running the pipeline")
        parser.add_argument(
            "--debug", action="store_true",
            help="set the pipeline logging level to DEBUG")

    @staticmethod
    def run(args):
        # Process args
        d = args.request or {
            "inputs": {},
            "outputs": {},
            "parameters": {},
            "logging_config": {}
        }
        d = defaultdict(dict, d)
        if args.name:
            d["pipeline"] = args.name
        if args.inputs:
            d["inputs"].update(args.inputs)
        if args.outputs:
            d["outputs"].update(args.outputs)
        if args.parameters:
            d["parameters"].update(args.parameters)
        if args.logging:
            d["logging_config"].update(args.logging)
        if args.debug:
            d["logging_config"]["stdout_level"] = "DEBUG"
            d["logging_config"]["file_level"] = "DEBUG"

        # Parse pipeline request
        logger.info("Parsing pipeline request")
        request = etab.PipelineBuildRequest.from_dict(d)

        # Build pipeline
        logger.info("Building pipeline '%s'", request.pipeline)
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


class RunCommand(Command):
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


class ModelsCommand(Command):
    '''Command-line tool for working with ETA models.

    Examples:
        # List all available published models
        eta models --list

        # Find a model
        eta models --find <model-name>

        # Download a model
        eta models --download <model-name>

        # Initialize a new models directory
        eta models --init <models-dir>

        # Flush the given model
        eta models --flush <model-name>

        # Flush all old models
        eta models --flush-old

        # Flush all models
        eta models --flush-all
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "-l", "--list", action="store_true",
            help="list all published models on the current search path")
        parser.add_argument(
            "-f", "--find", help="find the model with the given name")
        parser.add_argument(
            "-d", "--download", help="download the model with the given name")
        parser.add_argument(
            "-i", "--init", help="initialize the given models directory")
        parser.add_argument(
            "--flush", help="flush the model with the given name")
        parser.add_argument(
            "--flush-old", action="store_true", help="flush all old models")
        parser.add_argument(
            "--flush-all", action="store_true", help="flush all models")

    @staticmethod
    def run(args):
        if args.list:
            models = etamode.find_all_models()
            logger.info(_render_names_in_dirs_str(models))

        if args.find:
            model_path = etamode.find_model(args.find)
            logger.info(model_path)

        if args.download:
            etamode.download_model(args.download)

        if args.init:
            etamode.init_models_dir(args.init)

        if args.flush:
            etamode.flush_model(args.flush)

        if args.flush_old:
            etamode.flush_old_models()

        if args.flush_all:
            etamode.flush_all_models()


class ModulesCommand(Command):
    '''Command-line tool for working with ETA modules.

    Examples:
        # List all available ETA modules
        eta modules --list

        # Find the metadata file for a module
        eta modules --find <module-name>

        # Generate the block diagram for a module
        eta modules --diagram <module-name>

        # Generate the metadata file for a module
        eta modules --metadata '/path/to/eta_module.py'

        # Refresh all module metadata files
        eta modules --refresh-metadata
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "-l", "--list", action="store_true",
            help="list all ETA modules on the search path")
        parser.add_argument(
            "-f", "--find",
            help="find the metadata file for the module with the given name")
        parser.add_argument(
            "-d", "--diagram",
            help="generate the block diagram for the module with the given "
            "name")
        parser.add_argument(
            "-m", "--metadata",
            help="generate the metadata file for the given ETA module")
        parser.add_argument(
            "-r", "--refresh-metadata", action="store_true",
            help="refresh all module metadata files")

    @staticmethod
    def run(args):
        if args.list:
            modules = etamodu.find_all_metadata()
            logger.info(_render_names_in_dirs_str(modules))

        if args.find:
            metadata_path = etamodu.find_metadata(args.find)
            logger.info(metadata_path)

        if args.diagram:
            metadata = etamodu.load_metadata(args.diagram)
            metadata.render("./" + args.diagram + ".svg")

        if args.metadata:
            logger.info(
                "Generating metadata for ETA module '%s'", args.metadata)
            etame.generate(args.metadata)

        if args.refresh_metadata:
            for json_path in itervalues(etamodu.find_all_metadata()):
                py_path = os.path.splitext(json_path)[0] + ".py"
                logger.info("Generating metadata for ETA module '%s'", py_path)
                etame.generate(py_path)


class PipelinesCommand(Command):
    '''Command-line tool for working with ETA pipelines.

    Examples:
        # List all available ETA pipelines
        eta pipelines --list

        # Find the metadata file for a pipeline
        eta pipelines --find <pipeline-name>

        # Generate the block diagram for a pipeline
        eta pipelines --diagram <pipeline-name>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "-l", "--list", action="store_true",
            help="list all ETA pipelines on the current search path")
        parser.add_argument(
            "-f", "--find",
            help="find the metadata file for the pipeline with the given name")
        parser.add_argument(
            "-d", "--diagram",
            help="generate the block diagram for the module with the given "
            "name")

    @staticmethod
    def run(args):
        if args.list:
            pipelines = etap.find_all_metadata()
            logger.info(_render_names_in_dirs_str(pipelines))

        if args.find:
            metadata_path = etap.find_metadata(args.find)
            logger.info(metadata_path)

        if args.diagram:
            metadata = etap.load_metadata(args.diagram)
            metadata.render("./" + args.diagram + ".svg")


def _render_names_in_dirs_str(d):
    chunks = []
    mdict = _group_by_dir(d)
    for mdir in sorted(mdict):
        mstrs = ["  " + mname for mname in sorted(mdict[mdir])]
        chunks.append("[ %s ]\n" % mdir + "\n".join(mstrs))
    return "\n\n".join(chunks)


def _group_by_dir(d):
    dd = defaultdict(list)
    for name, path in iteritems(d):
        dd[os.path.dirname(path)].append(name)
    return dd


def _register_command(cmd, cls):
    parser = subparsers.add_parser(
        cmd, help=cls.__doc__.splitlines()[0],
        description=cls.__doc__.rstrip(),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.set_defaults(run=cls.run)
    cls.setup(parser)


# Main setup
parser = argparse.ArgumentParser(description="ETA command-line tool.")
parser.add_argument(
    "-v", "--version", action="version", version=eta.version,
    help="show version info")
subparsers = parser.add_subparsers(title="available commands")


# Command setup
_register_command("build", BuildCommand)
_register_command("run", RunCommand)
_register_command("models", ModelsCommand)
_register_command("modules", ModulesCommand)
_register_command("pipelines", PipelinesCommand)


def main():
    '''Executes the `eta` tool with the current command-line args.'''
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.run(args)
