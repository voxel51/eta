'''
Core module that defines the `eta` command-line interface (CLI).

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
import eta.core.logging as etal
import eta.core.metadata as etame
import eta.core.models as etamode
import eta.core.module as etamodu
import eta.core.pipeline as etap
import eta.core.serial as etas
import eta.core.utils as etau


logger = logging.getLogger(__name__)


class Command(object):
    '''Interface for defining commands.

    Command instances must implement the `setup()` method, and they should
    implement the `run()` method if they perform any functionality beyond
    defining subparsers.
    '''

    @staticmethod
    def setup(parser):
        '''Setup the command-line arguments for the command.

        Args:
            parser: an `argparse.ArgumentParser` instance
        '''
        raise NotImplementedError("subclass must implement setup()")

    @staticmethod
    def run(args):
        '''Execute the command on the given args.

        args:
            args: an `argparse.Namespace` instance containing the arguments
                for the command
        '''
        pass


class ETACommand(Command):
    '''ETA command-line interface.'''

    @staticmethod
    def setup(parser):
        subparsers = parser.add_subparsers(title="available commands")
        _register_command(subparsers, "build", BuildCommand)
        _register_command(subparsers, "run", RunCommand)
        _register_command(subparsers, "clean", CleanCommand)
        _register_command(subparsers, "models", ModelsCommand)
        _register_command(subparsers, "modules", ModulesCommand)
        _register_command(subparsers, "pipelines", PipelinesCommand)


class BuildCommand(Command):
    '''Tools for building pipelines.

    Examples:
        # Build pipeline from a pipeline build request
        eta build -r '/path/to/pipeline/request.json'

        # Build pipeline request interactively, run it, and cleanup after
        eta build \\
            -n video_formatter \\
            -i 'video="examples/data/water.mp4"' \\
            -o 'formatted_video="water-small.mp4"' \\
            -p 'format_videos.scale=0.5' \\
            --run-now --cleanup
    '''

    @staticmethod
    def setup(parser):
        request = parser.add_argument_group("request arguments")
        request.add_argument("-n", "--name", help="pipeline name")
        request.add_argument(
            "-r", "--request", type=etas.load_json,
            help="path to a PipelineBuildRequest file")
        request.add_argument(
            "-i", "--inputs", type=etas.load_json,
            metavar="'KEY=VAL,...'", help="pipeline inputs")
        request.add_argument(
            "-o", "--outputs", type=etas.load_json,
            metavar="'KEY=VAL,...'", help="pipeline outputs")
        request.add_argument(
            "-p", "--parameters", type=etas.load_json,
            metavar="'KEY=VAL,...'", help="pipeline parameters")
        request.add_argument(
            "-e", "--eta-config", type=etas.load_json,
            metavar="'KEY=VAL,...'", help="ETA config settings")
        request.add_argument(
            "-l", "--logging", type=etas.load_json,
            metavar="'KEY=VAL,...'", help="logging config settings")
        request.add_argument(
            "--patterns", type=etau.parse_kvps, metavar="'KEY=VAL,...'",
            help="patterns to replace in the build request")

        parser.add_argument(
            "--unoptimized", action="store_true",
            help="don't optimize pipeline when building")
        parser.add_argument(
            "--run-now", action="store_true",
            help="run pipeline after building")
        parser.add_argument(
            "--cleanup", action="store_true",
            help="delete all generated files after running the pipeline")
        parser.add_argument(
            "--debug", action="store_true",
            help="set pipeline logging level to DEBUG")

    @staticmethod
    def run(args):
        # Load pipeline request
        d = args.request or {
            "inputs": {},
            "outputs": {},
            "parameters": {},
            "eta_config": {},
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
        if args.eta_config:
            d["eta_config"].update(args.eta_config)
        if args.logging:
            d["logging_config"].update(args.logging)
        if args.debug:
            etal.set_logging_level(logging.DEBUG)
            d["logging_config"]["stdout_level"] = "DEBUG"
            d["logging_config"]["file_level"] = "DEBUG"

        # Replace any patterns
        if args.patterns:
            d = etas.load_json(
                etau.fill_patterns(etas.json_to_str(d), args.patterns))

        logger.info("Parsing pipeline request")
        request = etab.PipelineBuildRequest.from_dict(d)

        # Build pipeline
        logger.info("Building pipeline '%s'", request.pipeline)
        builder = etab.PipelineBuilder(request)
        optimized = not args.unoptimized
        builder.build(optimized=optimized)

        if args.run_now:
            _run_pipeline(builder.pipeline_config_path)
        else:
            logger.info(
                "\n***** To run this pipeline *****\neta run %s\n",
                builder.pipeline_config_path)

        if args.cleanup:
            logger.info("Cleaning up pipeline-generated files")
            builder.cleanup()


class RunCommand(Command):
    '''Tools for running pipelines and modules.

    Examples:
        # Run pipeline defined by a pipeline config
        eta run '/path/to/pipeline-config.json'

        # Run pipeline and force existing module outputs to be overwritten
        eta run --overwrite '/path/to/pipeline-config.json'

        # Run specified module with the given module config
        eta run --module <module-name> '/path/to/module-config.json'

        # Run last built pipeline
        eta run --last
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "config", nargs="?",
            help="path to PipelineConfig or ModuleConfig file")
        parser.add_argument(
            "-o", "--overwrite", action="store_true",
            help="force overwrite existing module outputs")
        parser.add_argument(
            "-m", "--module", help="run module with the given name")
        parser.add_argument(
            "-l", "--last", action="store_true",
            help="run last built pipeline")

    @staticmethod
    def run(args):
        if args.module:
            logger.info("Running module '%s'", args.module)
            etamodu.run(args.module, args.config)
            return

        if args.last:
            args.config = etab.find_last_built_pipeline()
            if not args.config:
                logger.info("No built pipelines found...")
                return

        _run_pipeline(args.config, force_overwrite=args.overwrite)


def _run_pipeline(config, force_overwrite=False):
    logger.info("Running pipeline '%s'", config)
    etap.run(config, force_overwrite=force_overwrite)

    logger.info("\n***** To re-run this pipeline *****\neta run %s\n", config)
    if etau.is_in_root_dir(config, eta.config.config_dir):
        logger.info(
            "\n***** To clean this pipeline *****\neta clean %s\n", config)


class CleanCommand(Command):
    '''Tools for cleaning up after pipelines.

    Examples:
        # Cleanup pipeline defined by a given pipeline config
        eta clean '/path/to/pipeline-config.json'

        # Cleanup last built pipeline
        eta clean --last

        # Cleanup all built pipelines
        eta clean --all
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "config", nargs="?", help="path to a PipelineConfig file")
        parser.add_argument(
            "-l", "--last", action="store_true",
            help="cleanup the last built pipeline")
        parser.add_argument(
            "-a", "--all", action="store_true",
            help="cleanup all built pipelines")

    @staticmethod
    def run(args):
        if args.config:
            etab.cleanup_pipeline(args.config)

        if args.last:
            config = etab.find_last_built_pipeline()
            if config:
                etab.cleanup_pipeline(config)
            else:
                logger.info("No built pipelines found...")

        if args.all:
            etab.cleanup_all_pipelines()


class ModelsCommand(Command):
    '''Tools for working with models.

    Examples:
        # List all available models
        eta models --list

        # Find model
        eta models --find <model-name>

        # Download model
        eta models --download <model-name>

        # Initialize new models directory
        eta models --init <models-dir>

        # Flush given model
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
            "-f", "--find", metavar="NAME",
            help="find the model with the given name")
        parser.add_argument(
            "-d", "--download", metavar="NAME",
            help="download the model with the given name")
        parser.add_argument(
            "-i", "--init", metavar="DIR",
            help="initialize the given models directory")
        parser.add_argument(
            "--flush", metavar="NAME",
            help="flush the model with the given name")
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
    '''Tools for working with modules.

    Examples:
        # List all available modules
        eta modules --list

        # Find metadata file for module
        eta modules --find <module-name>

        # Generate block diagram for module
        eta modules --diagram <module-name>

        # Generate metadata file for module
        eta modules --metadata '/path/to/eta_module.py'

        # Refresh all module metadata files
        eta modules --refresh-metadata
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "-l", "--list", action="store_true",
            help="list all modules on search path")
        parser.add_argument(
            "-f", "--find", metavar="NAME",
            help="find metadata file for module with the given name")
        parser.add_argument(
            "-d", "--diagram", metavar="NAME",
            help="generate block diagram for module with the given name")
        parser.add_argument(
            "-m", "--metadata", metavar="PATH",
            help="generate metadata file for the given module")
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
                "Generating metadata for module '%s'", args.metadata)
            etame.generate(args.metadata)

        if args.refresh_metadata:
            for json_path in itervalues(etamodu.find_all_metadata()):
                py_path = os.path.splitext(json_path)[0] + ".py"
                logger.info("Generating metadata for module '%s'", py_path)
                etame.generate(py_path)


class PipelinesCommand(Command):
    '''Tools for working with pipelines.

    Examples:
        # List all available pipelines
        eta pipelines --list

        # Find metadata file for pipeline
        eta pipelines --find <pipeline-name>

        # Generate block diagram for pipeline
        eta pipelines --diagram <pipeline-name>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "-l", "--list", action="store_true",
            help="list all ETA pipelines on the current search path")
        parser.add_argument(
            "-f", "--find", metavar="NAME",
            help="find metadata file for pipeline with the given name")
        parser.add_argument(
            "-d", "--diagram", metavar="NAME",
            help="generate block diagram for pipeline with the given name")

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


def _register_main_command(command, version=None):
    parser = argparse.ArgumentParser(description=command.__doc__.rstrip())
    if version:
        parser.add_argument(
            "-v", "--version", action="version", version=version,
            help="show version info")

    parser.set_defaults(run=command.run)
    command.setup(parser)
    return parser


def _register_command(parent, name, command):
    parser = parent.add_parser(
        name, help=command.__doc__.splitlines()[0],
        description=command.__doc__.rstrip(),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.set_defaults(run=command.run)
    command.setup(parser)
    return parser


def main():
    '''Executes the `eta` tool with the given command-line args.'''
    parser = _register_main_command(ETACommand, version=eta.version)

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    args.run(args)
