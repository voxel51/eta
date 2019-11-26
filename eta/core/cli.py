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
import re
import sys

from tabulate import tabulate
from tzlocal import get_localzone

import eta
import eta.core.builder as etab
import eta.constants as etac
import eta.core.logging as etal
import eta.core.metadata as etame
import eta.core.models as etamode
import eta.core.module as etamodu
import eta.core.pipeline as etap
from eta.core.serial import load_json, json_to_str
import eta.core.storage as etas
import eta.core.utils as etau
import eta.core.web as etaw


logger = logging.getLogger(__name__)


MAX_NAME_COLUMN_WIDTH = None
TABLE_FORMAT = "simple"


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
        _register_command(subparsers, "constants", ConstantsCommand)
        _register_command(subparsers, "auth", AuthCommand)
        _register_command(subparsers, "s3", S3Command)
        _register_command(subparsers, "gcs", GCSCommand)
        _register_command(subparsers, "gdrive", GoogleDriveStorageCommand)
        _register_command(subparsers, "http", HTTPStorageCommand)
        _register_command(subparsers, "sftp", SFTPStorageCommand)


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
            "-r", "--request", type=load_json,
            help="path to a PipelineBuildRequest file")
        request.add_argument(
            "-i", "--inputs", type=load_json,
            metavar="'KEY=VAL,...'", help="pipeline inputs")
        request.add_argument(
            "-o", "--outputs", type=load_json,
            metavar="'KEY=VAL,...'", help="pipeline outputs")
        request.add_argument(
            "-p", "--parameters", type=load_json,
            metavar="'KEY=VAL,...'", help="pipeline parameters")
        request.add_argument(
            "-e", "--eta-config", type=load_json,
            metavar="'KEY=VAL,...'", help="ETA config settings")
        request.add_argument(
            "-l", "--logging", type=load_json,
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
            d = load_json(etau.fill_patterns(json_to_str(d), args.patterns))

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
            "config", nargs="?", metavar="PATH",
            help="path to a PipelineConfig file")
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


class ConstantsCommand(Command):
    '''Print constants from `eta.constants`.

    Examples:
        # Print a constant defined in `eta.constants`
        eta constants <CONSTANT>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "constant", metavar="CONSTANT", help="the constant to print")

    @staticmethod
    def run(args):
        logger.info(getattr(etac, args.constant))


class AuthCommand(Command):
    '''Tools for configuring authentication credentials.'''

    @staticmethod
    def setup(parser):
        subparsers = parser.add_subparsers(title="available commands")
        _register_command(subparsers, "show", ShowAuthCommand)
        _register_command(subparsers, "activate", ActivateAuthCommand)
        _register_command(subparsers, "deactivate", DeactivateAuthCommand)


class ShowAuthCommand(Command):
    '''Show info about active credentials.

    Examples:
        # Print info about all active credentials
        eta auth show

        # Print info about active Google credentials
        eta auth show --google

        # Print info about active AWS credentials
        eta auth show --aws

        # Print info about active SSH credentials
        eta auth show --ssh
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "--google", action="store_true",
            help="show info about Google credentials")
        parser.add_argument(
            "--aws", action="store_true",
            help="show info about AWS credentials")
        parser.add_argument(
            "--ssh", action="store_true",
            help="show info about SSH credentials")

    @staticmethod
    def run(args):
        if args.google:
            _print_google_credentials_info()

        if args.aws:
            _print_aws_credentials_info()

        if args.ssh:
            _print_ssh_credentials_info()

        show_all_credentials = not any((args.google, args.aws, args.ssh))
        if show_all_credentials:
            try:
                _print_google_credentials_info()
            except etas.GoogleCredentialsError:
                pass

            try:
                _print_aws_credentials_info()
            except etas.AWSCredentialsError:
                pass

            try:
                _print_ssh_credentials_info()
            except etas.SSHCredentialsError:
                pass


def _print_google_credentials_info():
    credentials, path = etas.NeedsGoogleCredentials.load_credentials_json()
    contents = [
        ("project id", credentials["project_id"]),
        ("client email", credentials["client_email"]),
        ("private key id", credentials["private_key_id"]),
        ("path", path),
    ]
    table_str = tabulate(
        contents, headers=["Google credentials", ""], tablefmt="simple")
    logger.info(table_str + "\n")


def _print_aws_credentials_info():
    credentials, path = etas.NeedsAWSCredentials.load_credentials()
    contents = []
    for key, value in iteritems(credentials):
        contents.append((key.lower().replace("_", " "), value))

    if path:
        contents.append(("path", path))

    table_str = tabulate(
        contents, headers=["AWS credentials", ""], tablefmt="simple")
    logger.info(table_str + "\n")


def _print_ssh_credentials_info():
    path = etas.NeedsSSHCredentials.get_private_key_path()
    contents = [
        ("path", path),
    ]
    table_str = tabulate(
        contents, headers=["SSH credentials", ""], tablefmt="simple")
    logger.info(table_str + "\n")


class ActivateAuthCommand(Command):
    '''Activate authentication credentials.

    Examples:
        # Activate Google credentials
        eta auth activate --google '/path/to/service-account.json'

        # Activate AWS credentials
        eta auth activate --aws '/path/to/credentials.ini'
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "--google", metavar="PATH",
            help="path to Google service account JSON file")
        parser.add_argument(
            "--aws", metavar="PATH", help="path to AWS credentials file")
        parser.add_argument(
            "--ssh", metavar="PATH", help="path to SSH private key")

    @staticmethod
    def run(args):
        if args.google:
            etas.NeedsGoogleCredentials.activate_credentials(args.google)

        if args.aws:
            etas.NeedsAWSCredentials.activate_credentials(args.aws)

        if args.ssh:
            etas.NeedsSSHCredentials.activate_credentials(args.ssh)


class DeactivateAuthCommand(Command):
    '''Deactivate authentication credentials.'''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "--google", action="store_true",
            help="delete the active Google credentials")
        parser.add_argument(
            "--aws", action="store_true",
            help="delete the active AWS credentials")
        parser.add_argument(
            "--ssh", action="store_true",
            help="delete the active SSH credentials")

    @staticmethod
    def run(args):
        if args.google:
            etas.NeedsGoogleCredentials.deactivate_credentials()

        if args.aws:
            etas.NeedsAWSCredentials.deactivate_credentials()

        if args.ssh:
            etas.NeedsSSHCredentials.deactivate_credentials()


class S3Command(Command):
    '''Tools for working with S3.'''

    @staticmethod
    def setup(parser):
        subparsers = parser.add_subparsers(title="available commands")
        _register_command(subparsers, "info", S3InfoCommand)
        _register_command(subparsers, "list", S3ListCommand)
        _register_command(subparsers, "upload", S3UploadCommand)
        _register_command(subparsers, "upload-dir", S3UploadDirectoryCommand)
        _register_command(subparsers, "download", S3DownloadCommand)
        _register_command(
            subparsers, "download-dir", S3DownloadDirectoryCommand)
        _register_command(subparsers, "delete", S3DeleteCommand)
        _register_command(subparsers, "delete-dir", S3DeleteDirCommand)


class S3InfoCommand(Command):
    '''Get information about files in S3.

    Examples:
        # Get file info
        eta s3 info <path> [...]
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "paths", nargs="+", metavar="CLOUD_PATH",
            help="the path(s) of the files of interest in S3")

    @staticmethod
    def run(args):
        client = etas.S3StorageClient()

        metadata = [client.get_file_metadata(path) for path in args.paths]
        _print_s3_info_table(metadata)


class S3ListCommand(Command):
    '''List contents of an S3 folder.

    Examples:
        # List folder contents
        eta s3 list <folder>

        # List folder contents recursively
        eta s3 list <folder> --recursive

        # List folder contents according to the given query
        eta s3 list <folder>
            [--limit <limit>]
            [--search [<field>:]<str>]
            [--sort-by <field>]
            [--ascending]
            [--count]
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "folder", metavar="CLOUD_DIR", help="the S3 folder to list")
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="whether to "
            "recursively list the contents of subfolders")
        parser.add_argument(
            "-l", "--limit", metavar="LIMIT", type=int, default=-1,
            help="limit the number of files listed")
        parser.add_argument(
            "-s", "--search", metavar="[FIELD:]STR",
            help="search to limit results when listing files")
        parser.add_argument(
            "--sort-by", metavar="FIELD",
            help="field to sort by when listing files")
        parser.add_argument(
            "--ascending", action="store_true",
            help="whether to sort in ascending order")
        parser.add_argument(
            "-c", "--count", action="store_true",
            help="whether to show the number of files in the list")

    @staticmethod
    def run(args):
        client = etas.S3StorageClient()

        metadata = client.list_files_in_folder(
            args.folder, recursive=args.recursive, return_metadata=True)

        metadata = _apply_search(
            metadata, args.limit, args.search, args.sort_by, args.ascending,
            _S3_SEARCH_FIELDS_MAP)

        _print_s3_info_table(metadata, show_count=args.count)


_S3_SEARCH_FIELDS_MAP = {
    "name": "object_name",
    "size": "size",
    "type": "mime_type",
    "last_modified": "last_modified",
}


class S3UploadCommand(Command):
    '''Upload file to S3.

    Examples:
        # Upload file
        eta s3 upload <local-path> <cloud-path>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "local_path", metavar="LOCAL_PATH", help="the path to the file to "
            "upload")
        parser.add_argument(
            "cloud_path", metavar="CLOUD_PATH", help="the path to the S3 "
            "object to create")
        parser.add_argument(
            "-t", "--content-type", metavar="TYPE", help="an optional content "
            "type of the file. By default, the type is guessed from the "
            "filename")

    @staticmethod
    def run(args):
        client = etas.S3StorageClient()

        logger.info("Uploading '%s' to '%s'", args.local_path, args.cloud_path)
        client.upload(
            args.local_path, args.cloud_path, content_type=args.content_type)


class S3UploadDirectoryCommand(Command):
    '''Upload directory to S3.

    Examples:
        # Upload directory
        eta s3 upload-dir <local-dir> <cloud-dir>

        # Upload-sync directory
        eta s3 upload-dir --sync <local-dir> <cloud-dir>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "local_dir", metavar="LOCAL_DIR", help="the directory of files to "
            "upload")
        parser.add_argument(
            "cloud_dir", metavar="CLOUD_DIR", help="the S3 directory to "
            "upload into")
        parser.add_argument(
            "--sync", action="store_true", help="whether to sync the S3 "
            "directory to match the contents of the local directory")
        parser.add_argument(
            "-o", "--overwrite", action="store_true", help="whether to "
            "overwrite existing files; only valid in `--sync` mode")
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="whether to "
            "recursively upload the contents of subdirecotires")

    @staticmethod
    def run(args):
        client = etas.S3StorageClient()

        if args.sync:
            client.upload_dir_sync(
                args.local_dir, args.cloud_dir, overwrite=args.overwrite,
                recursive=args.recursive)
        else:
            client.upload_dir(
                args.local_dir, args.cloud_dir, recursive=args.recursive)


class S3DownloadCommand(Command):
    '''Download file from S3.

    Examples:
        # Download file
        eta s3 download <cloud-path> <local-path>

        # Print download to stdout
        eta s3 download <cloud-path> --print
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "cloud_path", metavar="CLOUD_PATH", help="the S3 object to "
            "download")
        parser.add_argument(
            "local_path", nargs="?", metavar="LOCAL_PATH", help="the path to "
            "which to write the downloaded file. If not provided, the "
            "filename of the file in S3 is used")
        parser.add_argument(
            "--print", action="store_true", help="whether to print the "
            "download to stdout. If true, a file is NOT written to disk")

    @staticmethod
    def run(args):
        client = etas.S3StorageClient()

        if args.print:
            logger.info(client.download_bytes(args.cloud_path))
        else:
            local_path = args.local_path
            if local_path is None:
                local_path = client.get_file_metadata(args.cloud_path)["name"]

            logger.info(
                "Downloading '%s' to '%s'", args.cloud_path, local_path)
            client.download(args.cloud_path, local_path)


class S3DownloadDirectoryCommand(Command):
    '''Download directory from S3.

    Examples:
        # Download directory
        eta s3 download-dir <cloud-folder> <local-dir>

        # Download directory sync
        eta s3 download-dir --sync <cloud-folder> <local-dir>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "cloud_dir", metavar="CLOUD_DIR", help="the S3 directory to "
            "download")
        parser.add_argument(
            "local_dir", metavar="LOCAL_DIR", help="the directory to which to "
            "download files into")
        parser.add_argument(
            "--sync", action="store_true", help="whether to sync the local"
            "directory to match the contents of the S3 directory")
        parser.add_argument(
            "-o", "--overwrite", action="store_true", help="whether to "
            "overwrite existing files; only valid in `--sync` mode")
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="whether to "
            "recursively download the contents of subdirecotires")

    @staticmethod
    def run(args):
        client = etas.S3StorageClient()

        if args.sync:
            client.download_dir_sync(
                args.cloud_dir, args.local_dir, overwrite=args.overwrite,
                recursive=args.recursive)
        else:
            client.download_dir(
                args.cloud_dir, args.local_dir, recursive=args.recursive)


class S3DeleteCommand(Command):
    '''Delete file from S3.

    Examples:
        # Delete file
        eta s3 delete <cloud-path>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "cloud_path", metavar="CLOUD_PATH", help="the S3 file to delete")

    @staticmethod
    def run(args):
        client = etas.S3StorageClient()

        logger.info("Deleting '%s'", args.cloud_path)
        client.delete(args.cloud_path)


class S3DeleteDirCommand(Command):
    '''Delete directory from S3.

    Examples:
        # Delete directory
        eta s3 delete-dir <cloud-dir>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "cloud_dir", metavar="CLOUD_DIR", help="the S3 folder to delete")

    @staticmethod
    def run(args):
        client = etas.S3StorageClient()

        logger.info("Deleting '%s'", args.cloud_dir)
        client.delete_folder(args.cloud_dir)


class GCSCommand(Command):
    '''Tools for working with Google Cloud Storage.'''

    @staticmethod
    def setup(parser):
        subparsers = parser.add_subparsers(title="available commands")
        _register_command(subparsers, "info", GCSInfoCommand)
        _register_command(subparsers, "list", GCSListCommand)
        _register_command(subparsers, "upload", GCSUploadCommand)
        _register_command(subparsers, "upload-dir", GCSUploadDirectoryCommand)
        _register_command(subparsers, "download", GCSDownloadCommand)
        _register_command(
            subparsers, "download-dir", GCSDownloadDirectoryCommand)
        _register_command(subparsers, "delete", GCSDeleteCommand)
        _register_command(subparsers, "delete-dir", GCSDeleteDirCommand)


class GCSInfoCommand(Command):
    '''Get information about files in GCS.

    Examples:
        # Get file info
        eta gcs info <path> [...]
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "paths", nargs="+", metavar="CLOUD_PATH",
            help="path(s) to GCS files")

    @staticmethod
    def run(args):
        client = etas.GoogleCloudStorageClient()

        metadata = [client.get_file_metadata(path) for path in args.paths]
        _print_gcs_info_table(metadata)


class GCSListCommand(Command):
    '''List contents of a GCS folder.

    Examples:
        # List folder contents
        eta gcs list <folder>

        # List folder contents recursively
        eta gcs list <folder> --recursive

        # List folder contents according to the given query
        eta gcs list <folder>
            [--limit <limit>]
            [--search [<field>:]<str>]
            [--sort-by <field>]
            [--ascending]
            [--count]
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "folder", metavar="CLOUD_DIR", help="the GCS folder to list")
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="whether to "
            "recursively list the contents of subfolders")
        parser.add_argument(
            "-l", "--limit", metavar="LIMIT", type=int, default=-1,
            help="limit the number of files listed")
        parser.add_argument(
            "-s", "--search", metavar="[FIELD:]STR",
            help="search to limit results when listing files")
        parser.add_argument(
            "--sort-by", metavar="FIELD",
            help="field to sort by when listing files")
        parser.add_argument(
            "--ascending", action="store_true",
            help="whether to sort in ascending order")
        parser.add_argument(
            "-c", "--count", action="store_true",
            help="whether to show the number of files in the list")

    @staticmethod
    def run(args):
        client = etas.GoogleCloudStorageClient()

        metadata = client.list_files_in_folder(
            args.folder, recursive=args.recursive, return_metadata=True)

        metadata = _apply_search(
            metadata, args.limit, args.search, args.sort_by, args.ascending,
            _GOOGLE_CLOUD_STORAGE_SEARCH_FIELDS_MAP)

        _print_gcs_info_table(metadata, show_count=args.count)


_GOOGLE_CLOUD_STORAGE_SEARCH_FIELDS_MAP = {
    "name": "object_name",
    "size": "size",
    "type": "mime_type",
    "last_modified": "last_modified",
}


class GCSUploadCommand(Command):
    '''Upload file to GCS.

    Examples:
        # Upload file
        eta gcs upload <local-path> <cloud-path>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "local_path", metavar="LOCAL_PATH", help="the path to the file to "
            "upload")
        parser.add_argument(
            "cloud_path", metavar="CLOUD_PATH", help="the path to the GCS "
            "object to create")
        parser.add_argument(
            "-t", "--content-type", metavar="TYPE", help="an optional content "
            "type of the file. By default, the type is guessed from the "
            "filename")
        parser.add_argument(
            "-s", "--chunk-size", metavar="SIZE", type=int, help="an optional "
            "chunk size (in bytes) to use")

    @staticmethod
    def run(args):
        client = etas.GoogleCloudStorageClient(chunk_size=args.chunk_size)

        logger.info("Uploading '%s' to '%s'", args.local_path, args.cloud_path)
        client.upload(
            args.local_path, args.cloud_path, content_type=args.content_type)


class GCSUploadDirectoryCommand(Command):
    '''Upload directory to GCS.

    Examples:
        # Upload directory
        eta gcs upload-dir <local-dir> <cloud-dir>

        # Upload-sync directory
        eta gcs upload-dir --sync <local-dir> <cloud-dir>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "local_dir", metavar="LOCAL_DIR", help="the directory of files to "
            "upload")
        parser.add_argument(
            "cloud_dir", metavar="CLOUD_DIR", help="the GCS directory to "
            "upload into")
        parser.add_argument(
            "--sync", action="store_true", help="whether to sync the GCS"
            "directory to match the contents of the local directory")
        parser.add_argument(
            "-o", "--overwrite", action="store_true", help="whether to "
            "overwrite existing files; only valid in `--sync` mode")
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="whether to "
            "recursively upload the contents of subdirecotires")
        parser.add_argument(
            "-s", "--chunk-size", metavar="SIZE", type=int, help="an optional "
            "chunk size (in bytes) to use")

    @staticmethod
    def run(args):
        client = etas.GoogleCloudStorageClient(chunk_size=args.chunk_size)

        if args.sync:
            client.upload_dir_sync(
                args.local_dir, args.cloud_dir, overwrite=args.overwrite,
                recursive=args.recursive)
        else:
            client.upload_dir(
                args.local_dir, args.cloud_dir, recursive=args.recursive)


class GCSDownloadCommand(Command):
    '''Download file from GCS.

    Examples:
        # Download file
        eta gcs download <cloud-path> <local-path>

        # Print download to stdout
        eta gcs download <cloud-path> --print
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "cloud_path", metavar="CLOUD_PATH", help="the GCS object to "
            "download")
        parser.add_argument(
            "local_path", nargs="?", metavar="LOCAL_PATH", help="the path to "
            "which to write the downloaded file. If not provided, the "
            "filename of the file in GCS is used")
        parser.add_argument(
            "--print", action="store_true", help="whether to print the "
            "download to stdout. If true, a file is NOT written to disk")
        parser.add_argument(
            "-s", "--chunk-size", metavar="SIZE", type=int, help="an optional "
            "chunk size (in bytes) to use")

    @staticmethod
    def run(args):
        client = etas.GoogleCloudStorageClient(chunk_size=args.chunk_size)

        if args.print:
            logger.info(client.download_bytes(args.cloud_path))
        else:
            local_path = args.local_path
            if local_path is None:
                local_path = client.get_file_metadata(args.cloud_path)["name"]

            logger.info(
                "Downloading '%s' to '%s'", args.cloud_path, local_path)
            client.download(args.cloud_path, local_path)


class GCSDownloadDirectoryCommand(Command):
    '''Download directory from GCS.

    Examples:
        # Download directory
        eta gcs download-dir <cloud-folder> <local-dir>

        # Download directory sync
        eta gcs download-dir --sync <cloud-folder> <local-dir>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "cloud_dir", metavar="CLOUD_DIR", help="the GCS directory to "
            "download")
        parser.add_argument(
            "local_dir", metavar="LOCAL_DIR", help="the directory to which to "
            "download files into")
        parser.add_argument(
            "--sync", action="store_true", help="whether to sync the local"
            "directory to match the contents of the GCS directory")
        parser.add_argument(
            "-o", "--overwrite", action="store_true", help="whether to "
            "overwrite existing files; only valid in `--sync` mode")
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="whether to "
            "recursively download the contents of subdirecotires")
        parser.add_argument(
            "-s", "--chunk-size", metavar="SIZE", type=int, help="an optional "
            "chunk size (in bytes) to use")

    @staticmethod
    def run(args):
        client = etas.GoogleCloudStorageClient(chunk_size=args.chunk_size)

        if args.sync:
            client.download_dir_sync(
                args.cloud_dir, args.local_dir, overwrite=args.overwrite,
                recursive=args.recursive)
        else:
            client.download_dir(
                args.cloud_dir, args.local_dir, recursive=args.recursive)


class GCSDeleteCommand(Command):
    '''Delete file from GCS.

    Examples:
        # Delete file
        eta gcs delete <cloud-path>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "cloud_path", metavar="CLOUD_PATH", help="the GCS file to delete")

    @staticmethod
    def run(args):
        client = etas.GoogleCloudStorageClient()

        logger.info("Deleting '%s'", args.cloud_path)
        client.delete(args.cloud_path)


class GoogleDriveStorageCommand(Command):
    '''Tools for working with Google Drive.'''

    @staticmethod
    def setup(parser):
        subparsers = parser.add_subparsers(title="available commands")
        _register_command(subparsers, "info", GoogleDriveInfoCommand)
        _register_command(subparsers, "list", GoogleDriveListCommand)
        _register_command(subparsers, "upload", GoogleDriveUploadCommand)
        _register_command(
            subparsers, "upload-dir", GoogleDriveUploadDirectoryCommand)
        _register_command(subparsers, "download", GoogleDriveDownloadCommand)
        _register_command(
            subparsers, "download-dir", GoogleDriveDownloadDirectoryCommand)
        _register_command(subparsers, "delete", GoogleDriveDeleteCommand)


class GoogleDriveInfoCommand(Command):
    '''Get information about files in Google Drive.

    Examples:
        # Get file info
        eta gdrive info <id> [...]
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "ids", nargs="+", metavar="ID",
            help="the ID(s) of the files of interest in Google Drive")

    @staticmethod
    def run(args):
        client = etas.GoogleDriveStorageClient()

        metadata = [client.get_file_metadata(fid) for fid in args.ids]
        _print_google_drive_info_table(metadata)


class GoogleDriveListCommand(Command):
    '''List contents of a Google Drive folder.

    Examples:
        # List folder contents
        eta gdrive list <id>

        # List folder contents recursively
        eta gdrive list <id> --recursive

        # List folder contents according to the given query
        eta gdrive list <id>
            [--limit <limit>]
            [--search [<field>:]<str>]
            [--sort-by <field>]
            [--ascending]
            [--count]
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "folder_id", metavar="ID", help="the ID of the folder to list")
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="whether to "
            "recursively list the contents of subfolders")
        parser.add_argument(
            "-l", "--limit", metavar="LIMIT", type=int, default=-1,
            help="limit the number of files listed")
        parser.add_argument(
            "-s", "--search", metavar="[FIELD:]STR",
            help="search to limit results when listing files")
        parser.add_argument(
            "--sort-by", metavar="FIELD",
            help="field to sort by when listing files")
        parser.add_argument(
            "--ascending", action="store_true",
            help="whether to sort in ascending order")
        parser.add_argument(
            "-c", "--count", action="store_true",
            help="whether to show the number of files in the list")

    @staticmethod
    def run(args):
        client = etas.GoogleDriveStorageClient()

        metadata = client.list_files_in_folder(
            args.folder_id, recursive=args.recursive)

        metadata = _apply_search(
            metadata, args.limit, args.search, args.sort_by, args.ascending,
            _GOOGLE_DRIVE_SEARCH_FIELDS_MAP)

        _print_google_drive_info_table(metadata, show_count=args.count)


_GOOGLE_DRIVE_SEARCH_FIELDS_MAP = {
    "id": "id",
    "name": "name",
    "size": "size",
    "type": "mime_type",
    "last_modified": "last_modified",
}


class GoogleDriveUploadCommand(Command):
    '''Upload file to Google Drive.

    Examples:
        # Upload file
        eta gdrive upload <local-path> <folder-id>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "path", metavar="LOCAL_PATH", help="the path to the file to upload")
        parser.add_argument(
            "folder_id", metavar="ID", help="the ID of the folder to upload "
            "the file into")
        parser.add_argument(
            "-f", "--filename", metavar="FILENAME", help="an optional "
            "filename to include in the request. By default, the name of the "
            "local file is used")
        parser.add_argument(
            "-t", "--content-type", metavar="TYPE", help="an optional content "
            "type of the file. By default, the type is guessed from the "
            "filename")
        parser.add_argument(
            "-s", "--chunk-size", metavar="SIZE", type=int, help="an optional "
            "chunk size (in bytes) to use")

    @staticmethod
    def run(args):
        client = etas.GoogleDriveStorageClient(chunk_size=args.chunk_size)

        logger.info("Uploading '%s' to '%s'", args.path, args.folder_id)
        client.upload(
            args.path, args.folder_id, filename=args.filename,
            content_type=args.content_type)


class GoogleDriveUploadDirectoryCommand(Command):
    '''Upload directory to Google Drive.

    Examples:
        # Upload directory
        eta gdrive upload-dir <local-dir> <folder-id>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "local_dir", metavar="LOCAL_DIR", help="the directory of files to "
            "upload")
        parser.add_argument(
            "folder_id", metavar="ID", help="the ID of the folder to upload "
            "the files into")
        parser.add_argument(
            "-f", "--skip-failures", action="store_true", help="whether to "
            "skip failures")
        parser.add_argument(
            "-e", "--skip-existing", action="store_true", help="whether to "
            "skip existing files")
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="whether to "
            "recursively upload the contents of subdirecotires")
        parser.add_argument(
            "-s", "--chunk-size", metavar="SIZE", type=int, help="an optional "
            "chunk size (in bytes) to use")

    @staticmethod
    def run(args):
        client = etas.GoogleDriveStorageClient(chunk_size=args.chunk_size)

        client.upload_files_in_folder(
            args.local_dir, args.folder_id, skip_failures=args.skip_failures,
            skip_existing_files=args.skip_existing, recursive=args.recursive)


class GoogleDriveDownloadCommand(Command):
    '''Download file from Google Drive.

    Examples:
        # Download file
        eta gdrive download <file-id> <local-path>

        # Print download to stdout
        eta gdrive download <file-id> --print

        # Download file with link sharing turned on (no credentials required)
        eta gdrive download --public <file-id>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "file_id", metavar="ID", help="the ID of the file to download")
        parser.add_argument(
            "path", nargs="?", metavar="LOCAL_PATH", help="the path to which "
            "to write the downloaded file. If not provided, the filename of "
            "the file in Google Drive is used")
        parser.add_argument(
            "--public", action="store_true", help="whether the file has "
            "public link sharing turned on and can therefore be downloaded "
            "with no credentials")
        parser.add_argument(
            "--print", action="store_true", help="whether to print the "
            "download to stdout. If true, a file is NOT written to disk")
        parser.add_argument(
            "-s", "--chunk-size", metavar="SIZE", type=int, help="an optional "
            "chunk size (in bytes) to use")

    @staticmethod
    def run(args):
        #
        # Download publicly available file
        #

        if args.public:
            if args.print:
                logger.info(
                    etaw.download_google_drive_file(
                        args.file_id, chunk_size=args.chunk_size))
            elif args.path is None:
                raise ValueError(
                    "Must provide `path` when `--public` flag is set")
            else:
                logger.info(
                    "Downloading '%s' to '%s'", args.file_id, args.path)
                etaw.download_google_drive_file(
                    args.file_id, path=args.path, chunk_size=args.chunk_size)

            return

        #
        # Download via GoogleDriveStorageClient
        #

        client = etas.GoogleDriveStorageClient(chunk_size=args.chunk_size)

        if args.print:
            logger.info(client.download_bytes(args.file_id))
        else:
            local_path = args.path
            if local_path is None:
                local_path = client.get_file_metadata(args.file_id)["name"]

            logger.info("Downloading '%s' to '%s'", args.file_id, local_path)
            client.download(args.file_id, local_path)


class GoogleDriveDownloadDirectoryCommand(Command):
    '''Download directory from Google Drive.

    Examples:
        # Download directory
        eta gdrive download-dir <folder-id> <local-dir>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "folder_id", metavar="ID", help="the ID of the folder to download")
        parser.add_argument(
            "local_dir", metavar="LOCAL_DIR", help="the directory to download "
            "the files into")
        parser.add_argument(
            "-f", "--skip-failures", action="store_true", help="whether to "
            "skip failures")
        parser.add_argument(
            "-e", "--skip-existing", action="store_true", help="whether to "
            "skip existing files")
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="whether to "
            "recursively download the contents of subdirecotires")
        parser.add_argument(
            "-s", "--chunk-size", metavar="SIZE", type=int, help="an optional "
            "chunk size (in bytes) to use")

    @staticmethod
    def run(args):
        client = etas.GoogleDriveStorageClient(chunk_size=args.chunk_size)

        client.download_files_in_folder(
            args.folder_id, args.local_dir, skip_failures=args.skip_failures,
            skip_existing_files=args.skip_existing, recursive=args.recursive)


class GoogleDriveDeleteCommand(Command):
    '''Delete file from Google Drive.

    Examples:
        # Delete file
        eta gdrive delete <id>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "id", metavar="ID", help="the ID of the file to delete")

    @staticmethod
    def run(args):
        client = etas.GoogleDriveStorageClient()

        logger.info("Deleting '%s'", args.id)
        client.delete(args.id)


class GoogleDriveDeleteDirCommand(Command):
    '''Delete directory from Google Drive.

    Examples:
        # Delete directory
        eta gdrive delete-dir <id>

        # Delete the contents (only) of a directory
        eta gdrive delete-dir <id> --contents-only
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "id", metavar="ID", help="the ID of the folder to delete")
        parser.add_argument(
            "-c", "--contents-only", action="store_true", help="whether to "
            "delete only the contents of the folder (not the folder itself)")
        parser.add_argument(
            "-s", "--skip-failures", action="store_true", help="whether to "
            "skip failures")

    @staticmethod
    def run(args):
        client = etas.GoogleDriveStorageClient()

        if args.contents_only:
            client.delete_folder_contents(
                args.id, skip_failures=args.skip_failures)
        else:
            logger.info("Deleting '%s'", args.id)
            client.delete_folder(args.id)


class HTTPStorageCommand(Command):
    '''Tools for working with HTTP storage.'''

    @staticmethod
    def setup(parser):
        subparsers = parser.add_subparsers(title="available commands")
        _register_command(subparsers, "upload", HTTPUploadCommand)
        _register_command(subparsers, "download", HTTPDownloadCommand)
        _register_command(subparsers, "delete", HTTPDeleteCommand)


class HTTPUploadCommand(Command):
    '''Upload file via HTTP.

    Examples:
        # Upload file
        eta http upload <local-path> <url>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "path", metavar="LOCAL_PATH", help="the path to the file to "
            "upload")
        parser.add_argument(
            "url", metavar="URL", help="the URL to which to PUT the file")
        parser.add_argument(
            "-f", "--filename", metavar="FILENAME", help="an optional "
            "filename to include in the request. By default, the name of the "
            "local file is used")
        parser.add_argument(
            "-t", "--content-type", metavar="TYPE", help="an optional content "
            "type of the file. By default, the type is guessed from the "
            "filename")

    @staticmethod
    def run(args):
        set_content_type = bool(args.content_type)
        client = etas.HTTPStorageClient(set_content_type=set_content_type)

        logger.info("Uploading '%s' to '%s'", args.path, args.url)
        client.upload(
            args.path, args.url, filename=args.filename,
            content_type=args.content_type)


class HTTPDownloadCommand(Command):
    '''Download file via HTTP.

    Examples:
        # Download file
        eta http download <url> <local-path>

        # Print download to stdout
        eta http download <url> --print
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "url", metavar="URL", help="the URL from which to GET the file")
        parser.add_argument(
            "path", nargs="?", metavar="LOCAL_PATH", help="the path to which "
            "to write the downloaded file. If not provided, the filename is "
            "guessed from the URL")
        parser.add_argument(
            "--print", action="store_true", help="whether to print the "
            "download to stdout. If true, a file is NOT written to disk")
        parser.add_argument(
            "-s", "--chunk-size", metavar="SIZE", type=int, help="an optional "
            "chunk size (in bytes) to use")

    @staticmethod
    def run(args):
        client = etas.HTTPStorageClient(chunk_size=args.chunk_size)

        if args.print:
            logger.info(client.download_bytes(args.url))
        else:
            local_path = args.path or client.get_filename(args.url)
            logger.info("Downloading '%s' to '%s'", args.url, local_path)
            client.download(args.url, local_path)


class HTTPDeleteCommand(Command):
    '''Delete file via HTTP.

    Examples:
        # Delete file
        eta http delete <url>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument("url", metavar="URL", help="the URL to DELETE")

    @staticmethod
    def run(args):
        client = etas.HTTPStorageClient()
        logger.info("Deleting '%s'", args.url)
        client.delete(args.url)


class SFTPStorageCommand(Command):
    '''Tools for working with SFTP storage.'''

    @staticmethod
    def setup(parser):
        subparsers = parser.add_subparsers(title="available commands")
        _register_command(subparsers, "upload", SFTPUploadCommand)
        _register_command(subparsers, "download", SFTPDownloadCommand)
        _register_command(subparsers, "delete", SFTPDeleteCommand)


class SFTPUploadCommand(Command):
    '''Upload file via SFTP.

    Examples:
        # Upload file
        eta sftp upload <local-path> <user>@<host>:<remote-path>
        eta sftp upload -u <user> -h <host> <local-path> <remote-path>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "local_path", metavar="PATH", help="the path to the file to "
            "upload")
        parser.add_argument(
            "remote_path", metavar="PATH", help="the remote path to write the "
            "file")
        parser.add_argument(
            "-u", "--username", metavar="USER", help="the username")
        parser.add_argument(
            "-h", "--host", metavar="HOST", help="the hostname to connect to")
        parser.add_argument(
            "-p", "--port", metavar="PORT", help="the port to use "
            "(default = 22)")

    @staticmethod
    def run(args):
        hostname, username, remote_path = _parse_remote_path(
            args.remote_path, args.host, args.user)

        client = etas.SFTPStorageClient(hostname, username, port=args.port)

        logger.info("Uploading '%s' to '%s'", args.local_path, remote_path)
        client.upload(args.local_path, remote_path)


class SFTPDownloadCommand(Command):
    '''Download file via SFTP.

    Examples:
        # Download file
        eta sftp download <user>@<host>:<remote-path> <local-path>
        eta sftp download -u <user> -h <host> <remote-path> <local-path>

        # Print download to stdout
        eta sftp download <remote-path> --print
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "remote_path", metavar="PATH", help="the remote file to download")
        parser.add_argument(
            "local_path", nargs="?", metavar="PATH", help="the path to which "
            "to write the downloaded file. If not provided, the filename is "
            "guessed from the remote path")
        parser.add_argument(
            "-u", "--user", metavar="USER", help="the username")
        parser.add_argument(
            "-h", "--host", metavar="HOST", help="the hostname to connect to")
        parser.add_argument(
            "-p", "--port", metavar="PORT", help="the port to use "
            "(default = 22)")
        parser.add_argument(
            "--print", action="store_true", help="whether to print the "
            "download to stdout. If true, a file is NOT written to disk")

    @staticmethod
    def run(args):
        hostname, username, remote_path = _parse_remote_path(
            args.remote_path, args.host, args.user)

        client = etas.SFTPStorageClient(hostname, username, port=args.port)

        if args.print:
            logger.info(client.download_bytes(remote_path))
        else:
            local_path = args.local_path or os.path.basename(remote_path)
            logger.info("Downloading '%s' to '%s'", remote_path, local_path)
            client.download(remote_path, local_path)


class SFTPDeleteCommand(Command):
    '''Delete file via SFTP.

    Examples:
        # Delete file
        eta sftp delete <user>@<host>:<remote-path>
        eta sftp delete -u <user> -h <host> <remote-path>
    '''

    @staticmethod
    def setup(parser):
        parser.add_argument(
            "remote_path", metavar="PATH", help="the remote file to delete")
        parser.add_argument(
            "-u", "--user", metavar="USER", help="the username")
        parser.add_argument(
            "-h", "--host", metavar="HOST", help="the hostname to connect to")
        parser.add_argument(
            "-p", "--port", metavar="PORT", help="the port to use "
            "(default = 22)")

    @staticmethod
    def run(args):
        hostname, username, remote_path = _parse_remote_path(
            args.remote_path, args.host, args.user)

        client = etas.SFTPStorageClient(hostname, username, port=args.port)

        logger.info("Deleting '%s'", remote_path)
        client.delete(remote_path)


def _parse_remote_path(remote_path, hostname, username):
    if "@" in remote_path:
        username, remote_path = remote_path.split("@")
    if ":" in remote_path:
        hostname, remote_path = remote_path.split(":")
    return hostname, username, remote_path



def _apply_search(records, limit, search, sort_by, ascending, field_map):
    if search:
        for match in _parse_search(search, field_map):
            records = [r for r in records if match(r)]

    reverse = not ascending
    if sort_by:
        records = sorted(records, key=lambda r: r[sort_by], reverse=reverse)
    elif reverse:
        records = reversed(records)

    if limit > 0:
        records = records[:limit]

    return records


def _parse_search(search, field_map):
    for s in _split_on_char(search, ","):
        chunks = [_remove_escapes(c, ":,") for c in _split_on_char(s, ":")]
        if len(chunks) == 1:
            # Match any field
            value = chunks[0]
            yield lambda record: any(value in str(record[f]) for f in record)
        else:
            # Match specific field
            key = chunks[0]
            value = ":".join(chunks[1:])
            yield lambda record: value in str(record[field_map[key]])


def _split_on_char(s, char):
    return re.split(r"(?<!\\)" + char, s)


def _remove_escapes(s, chars):
    return re.sub("\\\(" + "|".join(chars) + ")", "\\1", s)


def _print_s3_info_table(metadata, show_count=False):
    records = [(
        m["bucket"], _parse_name(m["object_name"]), _parse_size(m["size"]),
        m["mime_type"], _parse_datetime(m["last_modified"])
    ) for m in metadata]

    table_str = tabulate(
        records,
        headers=["bucket", "name", "size", "type", "last modified"],
        tablefmt=TABLE_FORMAT)

    logger.info(table_str)
    if show_count:
        logger.info("\nFound %d files\n", len(records))


def _print_gcs_info_table(metadata, show_count=False):
    records = [(
        m["bucket"], _parse_name(m["object_name"]), _parse_size(m["size"]),
        m["mime_type"], _parse_datetime(m["last_modified"])
    ) for m in metadata]

    table_str = tabulate(
        records,
        headers=["bucket", "name", "size", "type", "last modified"],
        tablefmt=TABLE_FORMAT)

    logger.info(table_str)
    if show_count:
        logger.info("\nFound %d files\n", len(records))


def _print_google_drive_info_table(metadata, show_count=False):
    records = [(
        m["id"], _parse_name(m["name"]), _parse_size(m["size"]),
        _parse_google_drive_mime_type(m["mime_type"]),
        _parse_datetime(m["last_modified"])
    ) for m in metadata]

    table_str = tabulate(
        records,
        headers=["id", "name", "size", "type", "last modified"],
        tablefmt=TABLE_FORMAT)

    logger.info(table_str)
    if show_count:
        logger.info("\nFound %d files\n", len(records))


def _parse_google_drive_mime_type(mime_type):
    if mime_type == "application/vnd.google-apps.folder":
        mime_type = "(folder)"

    return mime_type


def _parse_name(name):
    if MAX_NAME_COLUMN_WIDTH is not None and len(name) > MAX_NAME_COLUMN_WIDTH:
        name = name[:(MAX_NAME_COLUMN_WIDTH - 4)] + " ..."
    return name


def _parse_size(size):
    if size is None or size < 0:
        return "-"

    return etau.to_human_bytes_str(size)


def _parse_datetime(dt):
    return dt.astimezone(get_localzone()).strftime("%Y-%m-%d %H:%M:%S %Z")


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
