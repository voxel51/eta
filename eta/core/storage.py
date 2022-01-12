"""
Core infrastructure for working with remote storage resources.

See `docs/storage_dev_guide.md` for more information and best practices for
using this module to work with remote storage resources.

This module currently provides clients for the following storage resources:
- S3 buckets via the `boto3` package
- Google Cloud buckets via the `google.cloud.storage` package
- Google Drive via the `googleapiclient` package
- Remote servers via the `pysftp` package
- Web storage via HTTP requests
- Local disk storage

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
from future.utils import iteritems
import six

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import configparser
import datetime
import io
import logging
import os
import re

try:
    import urllib.parse as urlparse  # Python 3
except ImportError:
    import urlparse  # Python 2

import dateutil.parser
import requests

try:
    import boto3
    import botocore
    import botocore.config as bcc
    import google.api_core.exceptions as gae
    import google.api_core.retry as gar
    import google.cloud.storage as gcs
    from google.cloud.storage._signing import generate_signed_url_v4
    import google.oauth2.service_account as gos
    import googleapiclient.discovery as gad
    import googleapiclient.http as gah
    import pysftp
except ImportError as e:
    six.raise_from(
        ImportError(
            "The requested operation requires extra dependencies; install "
            '"voxel51-eta[storage]" to use it'
        ),
        e,
    )

import eta.constants as etac
import eta.core.serial as etas
import eta.core.utils as etau


logger = logging.getLogger(__name__)


# Suppress non-critical logging from third-party libraries
logging.getLogger("googleapiclient").setLevel(logging.ERROR)
logging.getLogger("paramiko").setLevel(logging.ERROR)


def ensure_file(filepath, remote_path):
    """Ensures that the file exists at the given `filepath` by downloading it
    from the remote location, if necessary.

    Args:
        filepath: the local path
        remote_path: the remote location of the file, which can be a URL, cloud
            storage object, or Google Drive ID
    """
    if os.path.exists(filepath):
        return

    client, _ = _parse_remote_path(remote_path)
    client.download(remote_path, filepath)


def ensure_archive(dirpath, remote_path):
    """Ensures that the directory exists at the given `dirpath` by downloading
    and extracting an archived version of it from the remote location, if
    necessary.

    Args:
        dirpath: the local directory
        remote_path: the remote location of the archived directory, which can
            be a URL, cloud storage object, or Google Drive ID
    """
    if os.path.exists(dirpath):
        return

    client, filename = _parse_remote_path(remote_path)

    archive_path = dirpath + os.path.splitext(filename)[1]
    if not os.path.isfile(archive_path):
        client.download(remote_path, archive_path)

    etau.extract_archive(archive_path, delete_archive=True)


def _parse_remote_path(remote_path):
    if remote_path.startswith("http"):
        client = HTTPStorageClient()
        filename = client.get_filename(remote_path)
    elif remote_path.startswith("gs://"):
        client = GoogleCloudStorageClient()
        filename = os.path.basename(remote_path)
    elif remote_path.startswith("s3://"):
        client = S3StorageClient()
        filename = os.path.basename(remote_path)
    elif len(remote_path) == 33:
        client = GoogleDriveStorageClient()
        filename = client.get_file_metadata(remote_path)["name"]
    else:
        raise ValueError("Unknown remote path '%s'" % remote_path)

    return client, filename


class StorageClient(object):
    """Interface for storage clients that read/write files from remote storage
    locations.

    Depending on the nature of the concrete StorageClient, `remote_path` may be
    a cloud object, a file ID, or some other construct that represents the path
    to which the file will be written.
    """

    def upload(self, local_path, remote_path):
        """Uploads the file to the given remote path.

        Args:
            local_path: the path to the file to upload
            remote_path: the remote path to write the file
        """
        raise NotImplementedError("subclass must implement upload()")

    def upload_bytes(self, bytes_str, remote_path):
        """Uploads the given bytes to the given remote path.

        Args:
            bytes_str: the bytes string to upload
            remote_path: the remote path to write the file
        """
        raise NotImplementedError("subclass must implement upload_bytes()")

    def upload_stream(self, file_obj, remote_path):
        """Uploads the contents of the given file-like object to the given
        remote path.

        Args:
            file_obj: the file-like object to upload, which must be open for
                reading
            remote_path: the remote path to write the file
        """
        raise NotImplementedError("subclass must implement upload_stream()")

    def download(self, remote_path, local_path):
        """Downloads the remote file to the given local path.

        Args:
            remote_path: the remote file to download
            local_path: the path to which to write the downloaded file
        """
        raise NotImplementedError("subclass must implement download()")

    def download_bytes(self, remote_path):
        """Downloads bytes from the given remote path.

        Args:
            remote_path: the remote file to download

        Returns:
            the downloaded bytes string
        """
        raise NotImplementedError("subclass must implement download_bytes()")

    def download_stream(self, remote_path, file_obj):
        """Downloads the file from the given remote path to the given
        file-like object.

        Args:
            remote_path: the remote file to download
            file_obj: the file-like object to which to write the download,
                which must be open for writing
        """
        raise NotImplementedError("subclass must implement download_stream()")

    def is_file(self, remote_path):
        """Determines whether the given remote file exists.

        Args:
            remote_path: the remote path

        Returns:
            True/False
        """
        raise NotImplementedError("subclass must implement is_file()")

    def delete(self, remote_path):
        """Deletes the file at the remote path.

        Args:
            remote_path: the remote path to delete
        """
        raise NotImplementedError("subclass must implement delete()")


class CanSyncDirectories(object):
    """Mixin class for `StorageClient`s that can sync directories to/from
    remote storage.

    Depending on the nature of the concrete StorageClient, `remote_dir` may be
    a cloud bucket or prefix, the ID of a directory, or some other construct
    that contains a collection of files of interest.
    """

    def is_folder(self, remote_dir):
        """Determines whether the given remote folder exists.

        Args:
            remote_dir: the remote directory

        Returns:
            True/False
        """
        raise NotImplementedError("subclass must implement is_folder()")

    def list_files_in_folder(self, remote_dir, recursive=True):
        """Returns a list of the files in the given remote directory.

        Args:
            remote_dir: the remote directory
            recursive: whether to recursively traverse sub-directories. By
                default, this is True

        Returns:
            a list of full paths to the files in the folder
        """
        raise NotImplementedError(
            "subclass must implement list_files_in_folder()"
        )

    def upload_dir(
        self, local_dir, remote_dir, recursive=True, skip_failures=False
    ):
        """Uploads the contents of the given directory to the given remote
        storage directory.

        The remote paths are created by appending the relative paths of all
        files inside the local directory to the provided remote directory.

        Args:
            local_dir: the local directory to upload
            remote_dir: the remote directory to upload into
            recursive: whether to recursively traverse subdirectories. By
                default, this is True
            skip_failures: whether to skip failures. By default, this is False
        """
        files = etau.list_files(local_dir, recursive=recursive)
        if not files:
            return

        logger.info("Uploading %d files to '%s'", len(files), remote_dir)
        for f in files:
            local_path = os.path.join(local_dir, f)
            remote_path = os.path.join(remote_dir, f)
            self._do_upload_sync(local_path, remote_path, skip_failures)

    def upload_dir_sync(
        self,
        local_dir,
        remote_dir,
        overwrite=False,
        recursive=True,
        skip_failures=False,
    ):
        """Syncs the contents of the given local directory to the given remote
        storage directory.

        This method is similar to `upload_dir()`, except that files in the
        remote diretory that are not present in the local directory will be
        deleted, and files that are already present in the remote directory
        are not uploaded if `overwrite` is False.

        Args:
            local_dir: the local directory to sync
            remote_dir: the remote directory to sync to
            overwrite: whether or not to upload files that are already present
                in the remote directory, thus overwriting them. By default,
                this is False
            recursive: whether to recursively traverse subdirectories. By
                default, this is True
            skip_failures: whether to skip failures. By default, this is False
        """
        local_files = set(etau.list_files(local_dir, recursive=recursive))
        remote_files = set(
            os.path.relpath(f, remote_dir)
            for f in self.list_files_in_folder(remote_dir, recursive=recursive)
        )

        # Files to delete remotely
        delete_files = remote_files - local_files

        # Files to upload to remote directory
        if overwrite:
            upload_files = local_files
        else:
            upload_files = local_files - remote_files

        if delete_files:
            logger.info(
                "Deleting %d files from '%s'", len(delete_files), remote_dir
            )
            for f in delete_files:
                remote_path = os.path.join(remote_dir, f)
                self._do_remote_delete_sync(remote_path, skip_failures)

        if upload_files:
            logger.info(
                "Uploading %d files to '%s'", len(upload_files), remote_dir
            )
            for f in upload_files:
                local_path = os.path.join(local_dir, f)
                remote_path = os.path.join(remote_dir, f)
                self._do_upload_sync(local_path, remote_path, skip_failures)

    def download_dir(
        self, remote_dir, local_dir, recursive=True, skip_failures=False
    ):
        """Downloads the contents of the remote directory to the given local
        directory.

        The files are written inside the specified local directory according
        to their relative paths w.r.t. the provided remote directory.

        Args:
            remote_dir: the remote directory to download
            local_dir: the local directory in which to write the files
            recursive: whether to recursively traverse subdirectories. By
                default, this is True
            skip_failures: whether to skip failures. By default, this is False
        """
        remote_paths = self.list_files_in_folder(
            remote_dir, recursive=recursive
        )
        if not remote_paths:
            return

        logger.info(
            "Downloading %d files from '%s'", len(remote_paths), remote_dir
        )
        for remote_path in remote_paths:
            local_path = os.path.join(
                local_dir, os.path.relpath(remote_path, remote_dir)
            )
            self._do_download_sync(remote_path, local_path, skip_failures)

    def download_dir_sync(
        self,
        remote_dir,
        local_dir,
        overwrite=False,
        recursive=True,
        skip_failures=False,
    ):
        """Syncs the contents of the given remote directory to the given local
        directory.

        This method is similar to `download_dir()`, except that files in the
        local diretory that are not present in the remote directory will be
        deleted, and files that are already present in the local directory
        are not downloaded if `overwrite` is False.

        Args:
            remote_dir: the remote directory to sync
            local_dir: the local directory to sync to
            overwrite: whether or not to download files that are already
                present in the local directory, thus overwriting them. By
                default, this is False
            recursive: whether to recursively traverse subdirectories. By
                default, this is True
            skip_failures: whether to skip failures. By default, this is False
        """
        remote_files = set(
            os.path.relpath(f, remote_dir)
            for f in self.list_files_in_folder(remote_dir, recursive=recursive)
        )
        local_files = set(etau.list_files(local_dir, recursive=recursive))

        # Files to delete locally
        delete_files = local_files - remote_files

        # Files to download locally
        if overwrite:
            download_files = remote_files
        else:
            download_files = remote_files - local_files

        if delete_files:
            logger.info(
                "Deleting %d files from '%s'", len(delete_files), local_dir
            )
            for f in delete_files:
                local_path = os.path.join(local_dir, f)
                self._do_local_delete_sync(local_path, skip_failures)

        if download_files:
            logger.info(
                "Downloading %d files to '%s'", len(download_files), local_dir
            )
            for f in download_files:
                remote_path = os.path.join(remote_dir, f)
                local_path = os.path.join(local_dir, f)
                self._do_download_sync(remote_path, local_path, skip_failures)

    def _do_upload_sync(self, local_path, remote_path, skip_failures):
        try:
            self.upload(local_path, remote_path)  # pylint: disable=no-member
            logger.info("Uploaded '%s'", remote_path)
        except Exception as e:
            if not skip_failures:
                raise SyncDirectoriesError(e)
            logger.warning(
                "Failed to upload '%s' to '%s'; skipping",
                local_path,
                remote_path,
            )

    def _do_download_sync(self, remote_path, local_path, skip_failures):
        try:
            self.download(remote_path, local_path)  # pylint: disable=no-member
            logger.info("Downloaded '%s'", local_path)
        except Exception as e:
            if not skip_failures:
                raise SyncDirectoriesError(e)
            logger.warning(
                "Failed to download '%s' to '%s'; skipping",
                remote_path,
                local_path,
            )

    def _do_local_delete_sync(self, local_path, skip_failures):
        try:
            etau.delete_file(local_path)
            logger.info("Deleted '%s'", local_path)
        except Exception as e:
            if not skip_failures:
                raise SyncDirectoriesError(e)
            logger.warning("Failed to delete '%s'; skipping", local_path)

    def _do_remote_delete_sync(self, remote_path, skip_failures):
        try:
            self.delete(remote_path)  # pylint: disable=no-member
            logger.info("Deleted '%s'", remote_path)
        except Exception as e:
            if not skip_failures:
                raise SyncDirectoriesError(e)
            logger.warning("Failed to delete '%s'; skipping", remote_path)


class SyncDirectoriesError(Exception):
    """Error raised when a CanSyncDirectories method fails."""

    pass


class LocalStorageClient(StorageClient, CanSyncDirectories):
    """Client for reading/writing data from local disk storage.

    Since this class encapsulates local disk storage, the `storage_path`
    arguments refer to local storage paths on disk.

    Attributes:
        chunk_size: the chunk size (in bytes) that will be used for streaming
            uploads and downloads. A negative value implies that the entire
            file is uploaded/downloaded at once
    """

    #
    # The chunk size (in bytes) to use when streaming. If a negative value
    # is supplied, then the entire file is uploaded/downloaded at once
    #
    DEFAULT_CHUNK_SIZE = -1

    def __init__(self, chunk_size=None):
        """Creates a LocalStorageClient instance.

        chunk_size: an optional chunk size (in bytes) to use for streaming
            uploads and downloads. By default, `DEFAULT_CHUNK_SIZE` is used
        """
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE

    def upload(self, local_path, storage_path):
        """Uploads the file to storage.

        Args:
            local_path: the path to the file to upload
            storage_path: the path to the storage location
        """
        etau.copy_file(local_path, storage_path)

    def upload_bytes(self, bytes_str, storage_path):
        """Uploads the given bytes to storage.

        Args:
            bytes_str: the bytes string to upload
            storage_path: the path to the storage location
        """
        etau.ensure_basedir(storage_path)
        with open(storage_path, "wb") as f:
            f.write(bytes_str)

    def upload_stream(self, file_obj, storage_path):
        """Uploads the contents of the given file-like object to storage.

        Args:
            file_obj: the file-like object to upload, which must be open for
                reading
            storage_path: the path to the storage location
        """
        etau.ensure_basedir(storage_path)
        with open(storage_path, "wb") as f:
            for chunk in _read_file_in_chunks(file_obj, self.chunk_size):
                f.write(chunk)

    def download(self, storage_path, local_path):
        """Downloads the file from storage.

        Args:
            storage_path: the path to the storage location
            local_path: the path to store the downloaded file locally
        """
        etau.copy_file(storage_path, local_path)

    def download_bytes(self, storage_path):
        """Downloads bytes from storage.

        Args:
            storage_path: the path to the storage location

        Returns:
            the downloaded bytes string
        """
        with open(storage_path, "rb") as f:
            return f.read()

    def download_stream(self, storage_path, file_obj):
        """Downloads the file from storage to the given file-like object.

        Args:
            storage_path: the path to the storage location
            file_obj: the file-like object to which to write the download,
                which must be open for writing
        """
        with open(storage_path, "rb") as f:
            for chunk in _read_file_in_chunks(f, self.chunk_size):
                file_obj.write(chunk)

    def delete(self, storage_path):
        """Deletes the file from storage.

        Args:
            storage_path: the path to the storage location
        """
        etau.delete_file(storage_path)

    def is_file(self, storage_path):
        """Determines whether the given file exists in storage.

        Args:
            storage_path: the path to the storage location

        Returns:
            True/False
        """
        return os.path.isfile(storage_path)

    def is_folder(self, storage_dir):
        """Determines whether the given storage directory exists.

        Args:
            storage_dir: the storage directory

        Returns:
            True/False
        """
        return os.path.isdir(storage_dir)

    def list_files_in_folder(self, storage_dir, recursive=True):
        """Returns a list of the files in the given storage directory.

        Args:
            storage_dir: the storage directory
            recursive: whether to recursively traverse sub-directories. By
                default, this is True

        Returns:
            a list of full paths to the files in the storage directory
        """
        return etau.list_files(
            storage_dir, abs_paths=True, recursive=recursive
        )


class _BotoStorageClient(StorageClient, CanSyncDirectories):
    """Base client for reading/writing data using Amazon's ``boto3`` API."""

    def __init__(
        self,
        credentials,
        alias=None,
        endpoint_url=None,
        max_pool_connections=None,
        **kwargs,
    ):
        """Creates a _BotoStorageClient instance.

        Args:
            credentials: a credentials dictionary to be passed to `boto3` via
                `boto3.client("s3", **credentials)`
            alias: a prefix for all cloud path strings, e.g., "s3"
            endpoint_url: the storage endpoint, if different from the default
                AWS service endpoint
            max_pool_connections: an optional maximum number of connections to
                keep in the connection pool
            **kwargs: optional configuration options for
                `botocore.config.Config(**kwargs)`
        """
        self.alias = alias
        self.endpoint_url = endpoint_url

        prefixes = []

        if alias is not None:
            prefixes.append(alias + "://")

        if endpoint_url is not None:
            prefixes.append(endpoint_url.rstrip("/") + "/")

        if not prefixes:
            raise ValueError(
                "At least one of `alias` and `endpoint_url` must be provided"
            )

        self._prefixes = tuple(prefixes)

        if "retries" not in kwargs:
            kwargs["retries"] = {"max_attempts": 10, "mode": "standard"}

        if "signature_version" not in kwargs:
            kwargs["signature_version"] = "s3v4"

        if (
            max_pool_connections is not None
            and "max_pool_connections" not in kwargs
        ):
            kwargs["max_pool_connections"] = max_pool_connections

        #
        # We allow users to provide credentials via numerous options, which
        # have already been parsed from above. However, the AWS libraries seem
        # to complain when a profile is specified whose credentials aren't
        # configured via environment variables or `~/.aws/credentials`.
        #
        # Temporarily hiding the `AWS_PROFILE` env var when creating the client
        # with manually provided credentials as kwargs bypasses the issue
        #
        aws_profile = os.environ.pop("AWS_PROFILE", None)

        try:
            config = bcc.Config(**kwargs)
            self._client = boto3.client(
                "s3", endpoint_url=endpoint_url, config=config, **credentials,
            )
        finally:
            if aws_profile is not None:
                os.environ["AWS_PROFILE"] = aws_profile

    def upload(self, local_path, cloud_path, content_type=None, metadata=None):
        """Uploads the file to the cloud.

        Args:
            local_path: the path to the file to upload
            cloud_path: the cloud path to create
            content_type: the optional type of the content being uploaded. If
                no value is provided, it is guessed from the filename
            metadata: an optional dictionary of custom object metadata to store
        """
        self._do_upload(
            cloud_path,
            local_path=local_path,
            content_type=content_type,
            metadata=metadata,
        )

    def upload_stream(
        self, file_obj, cloud_path, content_type=None, metadata=None
    ):
        """Uploads the contents of the given file-like object to the cloud.

        Args:
            file_obj: the file-like object to upload, which must be open for
                reading in binary (not text) mode
            cloud_path: the cloud path to create
            content_type: the optional type of the content being uploaded. If
                no value is provided but the object already exists, then the
                same value is used. Otherwise, the default value
                ("application/octet-stream") is used
            metadata: an optional dictionary of custom object metadata to store
        """
        self._do_upload(
            cloud_path,
            file_obj=file_obj,
            content_type=content_type,
            metadata=metadata,
        )

    def upload_bytes(
        self, bytes_str, cloud_path, content_type=None, metadata=None
    ):
        """Uploads the given bytes to the cloud.

        Args:
            bytes_str: the bytes string to upload
            cloud_path: the cloud path to create
            content_type: the optional type of the content being uploaded. If
                no value is provided but the object already exists, then the
                same value is used. Otherwise, the default value
                ("application/octet-stream") is used
            metadata: an optional dictionary of custom object metadata to store
        """
        with io.BytesIO(_to_bytes(bytes_str)) as f:
            self._do_upload(
                cloud_path,
                file_obj=f,
                content_type=content_type,
                metadata=metadata,
            )

    def download(self, cloud_path, local_path):
        """Downloads the cloud file to the given location.

        Args:
            cloud_path: the cloud path
            local_path: the local disk path to store the downloaded file
        """
        self._do_download(cloud_path, local_path=local_path)

    def download_stream(self, cloud_path, file_obj):
        """Downloads the cloud file to the given file-like object.

        Args:
            cloud_path: the cloud path
            file_obj: the file-like object to which to write the download,
                which must be open for writing in binary mode
        """
        self._do_download(cloud_path, file_obj=file_obj)

    def download_bytes(self, cloud_path, start=None, end=None):
        """Downloads the cloud file as a bytes string.

        Args:
            cloud_path: the cloud path
            start: an optional first byte in a range to download
            end: an optional last byte in a range to download

        Returns:
            the downloaded bytes string
        """
        if start is not None or end is not None:
            return self._do_download(cloud_path, start=start, end=end)

        with io.BytesIO() as f:
            self.download_stream(cloud_path, f)
            return f.getvalue()

    def delete(self, cloud_path):
        """Deletes the given cloud file.

        Args:
            cloud_path: the cloud path
        """
        bucket, object_name = self._parse_path(cloud_path)
        self._client.delete_object(Bucket=bucket, Key=object_name)

    def delete_folder(self, cloud_folder):
        """Deletes all files in the given cloud "folder".

        Args:
            cloud_folder: a cloud "folder" path
        """
        bucket, folder_name = self._parse_path(cloud_folder)
        if folder_name and not folder_name.endswith("/"):
            folder_name += "/"

        kwargs = {"Bucket": bucket, "Prefix": folder_name}
        while True:
            resp = self._client.list_objects_v2(**kwargs)
            contents = resp.get("Contents", [])
            if contents:
                delete = {"Objects": [{"Key": obj["Key"]} for obj in contents]}
                self._client.delete_objects(Bucket=bucket, Delete=delete)

            try:
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            except KeyError:
                break

    def get_file_metadata(self, cloud_path):
        """Returns metadata about the given cloud file.

        Args:
            cloud_path: the cloud path

        Returns:
            a dictionary containing metadata about the file, including its
                `bucket`, `object_name`, `name`, `size`, `mime_type`,
                `last_modified`, `etag`, and `metadata`
        """
        bucket, object_name = self._parse_path(cloud_path)
        return self._get_file_metadata(bucket, object_name)

    def get_folder_metadata(self, cloud_folder):
        """Returns metadata about the given cloud "folder".

        Note that this method is *expensive*; the only way to compute this
        information is to call `list_files_in_folder(..., recursive=True)` and
        compute stats from individual files!

        Args:
            cloud_folder: a cloud "folder" path

        Returns:
            a dictionary containing metadata about the "folder", including its
                `bucket`, `path`, `num_files`, `size`, and `last_modified`
        """
        files = self.list_files_in_folder(
            cloud_folder, recursive=True, return_metadata=True
        )

        bucket, path = self._parse_path(cloud_folder)
        path = path.rstrip("/")

        if files:
            num_files = len(files)
            size = sum(f["size"] for f in files)
            last_modified = max(f["last_modified"] for f in files)
        else:
            num_files = 0
            size = 0
            last_modified = "-"

        return {
            "bucket": bucket,
            "path": path,
            "num_files": num_files,
            "size": size,
            "last_modified": last_modified,
        }

    def is_file(self, cloud_path):
        """Determines whether the given cloud file exists.

        Args:
            cloud_path: the cloud path

        Returns:
            True/False
        """
        bucket, object_name = self._parse_path(cloud_path)

        try:
            self._client.head_object(Bucket=bucket, Key=object_name)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False

            raise

    def is_folder(self, cloud_folder):
        """Determines whether the given cloud "folder" contains at least one
        object.

        Args:
            cloud_folder: a cloud "folder" path

        Returns:
            True/False
        """
        bucket, folder_name = self._parse_path(cloud_folder)
        if folder_name and not folder_name.endswith("/"):
            folder_name += "/"

        resp = self._client.list_objects_v2(
            Bucket=bucket, Prefix=folder_name, MaxKeys=1
        )
        return bool(resp.get("Contents", []))

    def list_files_in_folder(
        self, cloud_folder, recursive=True, return_metadata=False
    ):
        """Returns a list of the files in the given cloud "folder".

        Args:
            cloud_folder: a cloud "folder" path
            recursive: whether to recursively traverse sub-"folders". By
                default, this is True
            return_metadata: whether to return a metadata dictionary for each
                file, including its `bucket`, `object_name`, `name`, `size`,
                `mime_type`, `last_modified`, `etag`, and `metadata`. By
                default, only the paths to the files are returned

        Returns:
            a list of full cloud paths (when `return_metadata == False`) or a
                list of metadata dictionaries (when `return_metadata == True`)
                for the files in the folder
        """
        bucket, folder_name = self._parse_path(cloud_folder)
        if folder_name and not folder_name.endswith("/"):
            folder_name += "/"

        kwargs = {"Bucket": bucket, "Prefix": folder_name}
        if not recursive:
            kwargs["Delimiter"] = "/"

        paths_or_metadata = []
        prefix = self._prefixes[0] + bucket
        while True:
            resp = self._client.list_objects_v2(**kwargs)

            for obj in resp.get("Contents", []):
                path = obj["Key"]
                if not path.endswith("/"):
                    if return_metadata:
                        paths_or_metadata.append(
                            self._get_object_metadata(bucket, obj)
                        )
                    else:
                        paths_or_metadata.append(os.path.join(prefix, path))

            try:
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            except KeyError:
                break

        return paths_or_metadata

    def generate_signed_url(
        self, cloud_path, method="GET", hours=24, content_type=None
    ):
        """Generates a signed URL for accessing the given object.

        Anyone with the URL can access the object with the permission until it
        expires.

        Note that you should use `PUT`, not `POST`, to upload objects!

        Args:
            cloud_path: the cloud path
            method: the HTTP verb (GET, PUT, DELETE) to authorize
            hours: the number of hours that the URL is valid
            content_type: (PUT actions only) the optional type of the content
                being uploaded

        Returns:
            a URL for accessing the object via HTTP request
        """
        client_method = method.lower() + "_object"
        bucket, object_name = self._parse_path(cloud_path)
        params = {"Bucket": bucket, "Key": object_name}
        if client_method == "put_object" and content_type:
            params["ContentType"] = content_type

        expiration = int(3600 * hours)
        return self._client.generate_presigned_url(
            ClientMethod=client_method, Params=params, ExpiresIn=expiration
        )

    def add_metadata(self, cloud_path, metadata):
        """Adds custom metadata key-value pairs to the given object.

        Note that the S3 API fundamentally treats metadata as immutable, so
        this method must use `botocore.client.S3.copy_object()` to copy the
        object, which only supports file sizes up to 5GB. For larger files, you
        must manually re-upload the file with the desired metadata.

        Args:
            cloud_path: the cloud path
            metadata: a dictionary of custom metadata
        """
        bucket, object_name = self._parse_path(cloud_path)
        response = self._client.head_object(Bucket=bucket, Key=object_name)

        response["Metadata"].update(metadata)

        self._client.copy_object(
            Bucket=bucket,
            Key=object_name,
            CopySource={"Bucket": bucket, "Key": object_name},
            Metadata=response["Metadata"],
            MetadataDirective="REPLACE",
            TaggingDirective="COPY",
            ContentType=response["ContentType"],
        )

    def _do_upload(
        self,
        cloud_path,
        local_path=None,
        file_obj=None,
        content_type=None,
        metadata=None,
    ):
        bucket, object_name = self._parse_path(cloud_path)

        if local_path and not content_type:
            content_type = etau.guess_mime_type(local_path)

        extra_args = {}

        if content_type:
            extra_args["ContentType"] = content_type

        if metadata:
            extra_args["Metadata"] = metadata

        if local_path:
            self._client.upload_file(
                local_path, bucket, object_name, ExtraArgs=extra_args
            )

        if file_obj is not None:
            self._client.upload_fileobj(
                file_obj, bucket, object_name, ExtraArgs=extra_args
            )

    def _do_download(
        self, cloud_path, local_path=None, file_obj=None, start=None, end=None
    ):
        bucket, object_name = self._parse_path(cloud_path)

        if start is not None or end is not None:
            response = self._client.get_object(
                Bucket=bucket,
                Key=object_name,
                Range="bytes=%s-%s" % (start or "", end or ""),
            )
            return response["Body"].read()

        if local_path:
            etau.ensure_basedir(local_path)
            self._client.download_file(bucket, object_name, local_path)

        if file_obj is not None:
            self._client.download_fileobj(bucket, object_name, file_obj)

    def _get_file_metadata(self, bucket, object_name):
        response = self._client.head_object(Bucket=bucket, Key=object_name)

        mime_type = response["ContentType"]
        if not mime_type:
            mime_type = etau.guess_mime_type(object_name)

        return {
            "bucket": bucket,
            "object_name": object_name,
            "name": os.path.basename(object_name),
            "size": response["ContentLength"],
            "mime_type": mime_type,
            "last_modified": response["LastModified"],
            "etag": response["ETag"][1:-1],
            "metadata": response.get("Metadata", {}),
        }

    @staticmethod
    def _get_object_metadata(bucket, obj):
        # @todo is there a way to get the MIME type without guessing or making
        # an expensive call to `head_object`?
        path = obj["Key"]
        return {
            "bucket": bucket,
            "object_name": path,
            "name": os.path.basename(path),
            "size": obj["Size"],
            "mime_type": etau.guess_mime_type(path),
            "last_modified": obj["LastModified"],
            "etag": obj["ETag"][1:-1],
            "metadata": obj.get("Metadata", {}),
        }

    def _strip_prefix(self, cloud_path):
        _cloud_path = _strip_prefix(cloud_path, self._prefixes)

        if _cloud_path is None:
            if len(self._prefixes) == 1:
                valid_prefixes = "'%s'" % self._prefixes[0]
            else:
                valid_prefixes = self._prefixes

            raise ValueError(
                "Invalid path '%s'; must start with %s"
                % (cloud_path, valid_prefixes)
            )

        return _cloud_path

    def _parse_path(self, cloud_path):
        _cloud_path = self._strip_prefix(cloud_path)

        chunks = _cloud_path.split("/", 1)

        if len(chunks) != 2:
            return chunks[0], ""

        return chunks[0], chunks[1]


class NeedsAWSCredentials(object):
    """Mixin for classes that need AWS credentials to take authenticated
    actions.

    Storage clients that derive from this class should allow users to provide
    credentials in the following ways (in order of precedence):

        (1) manually constructing an instance of the class via the
            `cls.from_ini()` method by providing a path to a valid credentials
            `.ini` file

        (2) setting the following environment variables directly:

            -   `AWS_ACCESS_KEY_ID`
            -   `AWS_SECRET_ACCESS_KEY`
            -   `AWS_SESSION_TOKEN` (if applicable)
            -   `AWS_DEFAULT_REGION`

        (3) setting the `AWS_SHARED_CREDENTIALS_FILE` environment variable to
            point to a valid credentials `.ini` file

        (4) setting the `AWS_CONFIG_FILE` environment variable to point to a
            valid credentials `.ini` file

        (5) loading credentials from `~/.eta/aws-credentials.ini` that have
            been activated via `cls.activate_credentials()`

    In the above, the `.ini` file should have syntax similar to the following::

        [default]
        aws_access_key_id = ...
        aws_secret_access_key = ...
        aws_session_token = ...  # if applicable
        region = ...

    See the following link for more information:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#configuration
    """

    CREDENTIALS_PATH = os.path.join(etac.ETA_CONFIG_DIR, "aws-credentials.ini")

    @classmethod
    def activate_credentials(cls, credentials_path):
        """Activate the credentials by copying them to
        `~/.eta/aws-credentials.ini`.

        Args:
            credentials_path: the path to a credentials `.ini` file
        """
        etau.copy_file(credentials_path, cls.CREDENTIALS_PATH)
        logger.info(
            "AWS credentials successfully activated at '%s'",
            cls.CREDENTIALS_PATH,
        )

    @classmethod
    def deactivate_credentials(cls):
        """Deactivates (deletes) the currently active credentials, if any.

        Active credentials (if any) are at `~/.eta/aws-credentials.ini`.
        """
        try:
            os.remove(cls.CREDENTIALS_PATH)
            logger.info(
                "AWS credentials '%s' successfully deactivated",
                cls.CREDENTIALS_PATH,
            )
        except OSError:
            logger.info("No AWS credentials to deactivate")

    @classmethod
    def has_active_credentials(cls):
        """Determines whether there are any active credentials stored at
        `~/.eta/aws-credentials.ini`.

        Returns:
            True/False
        """
        return os.path.isfile(cls.CREDENTIALS_PATH)

    @classmethod
    def load_credentials(cls, credentials_path=None, profile=None):
        """Loads the AWS credentials as a dictionary.

        Args:
            credentials_path: an optional path to a credentials `.ini` file.
                If omitted, active credentials are located using the strategy
                described in the class docstring of `NeedsAWSCredentials`
            profile: an optional profile to load when a credentials `.ini` file
                is loaded (if applicable). If not provided, the "default"
                section is loaded

        Returns:
            a (credentials, path) tuple containing the credentials and the path
                from which they were loaded. If the credentials were loaded
                from environment variables, `path` will be None
        """
        if profile is None and "AWS_PROFILE" in os.environ:
            logger.debug(
                "Loading profile from 'AWS_PROFILE' environment variable"
            )
            profile = os.environ["AWS_PROFILE"]

        if credentials_path:
            logger.debug(
                "Loading AWS credentials from manually provided path '%s'",
                credentials_path,
            )
            credentials = cls._load_credentials_ini(
                credentials_path, profile=profile
            )
            return credentials, credentials_path

        if (
            "AWS_ACCESS_KEY_ID" in os.environ
            and "AWS_SECRET_ACCESS_KEY" in os.environ
        ):
            logger.debug(
                "Loading access key and secret key from 'AWS_ACCESS_KEY_ID' "
                "and 'AWS_SECRET_ACCESS_KEY' environment variables"
            )
            credentials = {
                "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
                "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
            }

            if "AWS_DEFAULT_REGION" in os.environ:
                logger.debug(
                    "Loading region from 'AWS_DEFAULT_REGION' environment "
                    "variable"
                )
                credentials["region"] = os.environ["AWS_DEFAULT_REGION"]

            if "AWS_SESSION_TOKEN" in os.environ:
                logger.debug(
                    "Loading session token from 'AWS_SESSION_TOKEN' "
                    "environment variable"
                )
                credentials["aws_session_token"] = os.environ[
                    "AWS_SESSION_TOKEN"
                ]

            return credentials, None

        if "AWS_SHARED_CREDENTIALS_FILE" in os.environ:
            credentials_path = os.environ["AWS_SHARED_CREDENTIALS_FILE"]
            logger.debug(
                "Loading AWS credentials from environment variable "
                "AWS_SHARED_CREDENTIALS_FILE='%s'",
                credentials_path,
            )
        elif "AWS_CONFIG_FILE" in os.environ:
            credentials_path = os.environ["AWS_CONFIG_FILE"]
            logger.debug(
                "Loading AWS credentials from environment variable "
                "AWS_CONFIG_FILE='%s'",
                credentials_path,
            )
        elif cls.has_active_credentials():
            credentials_path = cls.CREDENTIALS_PATH
            logger.debug(
                "Loading activated AWS credentials from '%s'", credentials_path
            )
        else:
            raise AWSCredentialsError("No AWS credentials found")

        credentials = cls._load_credentials_ini(
            credentials_path, profile=profile
        )
        return credentials, credentials_path

    @classmethod
    def from_ini(cls, credentials_path, profile=None):
        """Creates a `cls` instance from the given credentials `.ini` file.

        Args:
            credentials_path: the path to a credentials `.ini` file
            profile: an optional profile to load. If not provided the "default"
                section is loaded

        Returns:
            an instance of cls with the given credentials
        """
        credentials, _ = cls.load_credentials(
            credentials_path=credentials_path, profile=profile
        )
        return cls(credentials=credentials)

    @classmethod
    def _load_credentials_ini(cls, credentials_path, profile=None):
        if profile is None:
            profile = "default"

        if not os.path.isfile(credentials_path):
            raise FileNotFoundError(
                "File '%s' does not exist" % credentials_path
            )

        config = configparser.ConfigParser()
        config.read(credentials_path)

        # Config file syntax
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#aws-config-file
        if "profile " + profile in config:
            return dict(config["profile " + profile])

        # Credentials file syntax
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#shared-credentials-file
        return dict(config[profile])


class AWSCredentialsError(Exception):
    """Error raised when a problem with AWS credentials is encountered."""

    def __init__(self, message):
        """Creates an AWSCredentialsError instance.

        Args:
            message: the error message
        """
        super(AWSCredentialsError, self).__init__(
            "%s. Read the documentation for "
            "`eta.core.storage.NeedsAWSCredentials` for more information "
            "about authenticating with AWS services." % message
        )


class S3StorageClient(_BotoStorageClient, NeedsAWSCredentials):
    """Client for reading/writing data from Amazon S3 buckets.

    All cloud path strings used by this class should have the form
    "s3://<bucket>/<path/to/object>".

    See `NeedsAWSCredentials` for more information about the authentication
    strategy used by this class.
    """

    def __init__(self, credentials=None, max_pool_connections=None, **kwargs):
        """Creates an S3StorageClient instance.

        Args:
            credentials: an optional AWS credentials dictionary. If not
                provided, credentials must be loadable as described in
                `NeedsAWSCredentials`
            max_pool_connections: an optional maximum number of connections to
                keep in the connection pool
            **kwargs: optional configuration options for
                `botocore.config.Config(**kwargs)`
        """
        if credentials is None:
            credentials, _ = self.load_credentials()

        # The .ini files use `region` but `boto3.client` uses `region_name`
        credentials["region_name"] = credentials.pop("region", None)

        super(S3StorageClient, self).__init__(
            credentials,
            alias="s3",
            max_pool_connections=max_pool_connections,
            **kwargs,
        )


class NeedsMinIOCredentials(object):
    """Mixin for classes that need MinIO credentials to take authenticated
    actions.

    Storage clients that derive from this class should allow users to provide
    credentials in the following ways (in order of precedence):

        (1) manually constructing an instance of the class via the
            `cls.from_ini()` method by providing a path to a valid credentials
            `.ini` file

        (2) setting the following environment variables directly:

            -   `MINIO_ACCESS_KEY`
            -   `MINIO_SECRET_ACCESS_KEY`
            -   `MINIO_ENDPOINT_URL`
            -   `MINIO_ALIAS`  # optional
            -   `MINIO_REGION`  # if applicable

        (3) setting the `MINIO_CONFIG_FILE` environment variable to point to a
            valid credentials `.ini` file

        (4) setting the `MINIO_SHARED_CREDENTIALS_FILE` environment variable to
            point to a valid credentials `.ini` file

        (5) loading credentials from `~/.eta/minio-credentials.ini` that have
            been activated via `cls.activate_credentials()`

    In the above, the `.ini` file should have syntax similar to the following::

        [default]
        access_key = ...
        secret_access_key = ...
        endpoint_url = ...
        alias = ...  # optional
        region = ...  # if applicable

    See the following link for more information:
    https://docs.min.io/docs/minio-quickstart-guide.html
    """

    CREDENTIALS_PATH = os.path.join(
        etac.ETA_CONFIG_DIR, "minio-credentials.ini"
    )

    @classmethod
    def activate_credentials(cls, credentials_path):
        """Activate the credentials by copying them to
        `~/.eta/minio-credentials.ini`.

        Args:
            credentials_path: the path to a credentials `.ini` file
        """
        etau.copy_file(credentials_path, cls.CREDENTIALS_PATH)
        logger.info(
            "MinIO credentials successfully activated at '%s'",
            cls.CREDENTIALS_PATH,
        )

    @classmethod
    def deactivate_credentials(cls):
        """Deactivates (deletes) the currently active credentials, if any.

        Active credentials (if any) are at `~/.eta/minio-credentials.ini`.
        """
        try:
            os.remove(cls.CREDENTIALS_PATH)
            logger.info(
                "MinIO credentials '%s' successfully deactivated",
                cls.CREDENTIALS_PATH,
            )
        except OSError:
            logger.info("No MinIO credentials to deactivate")

    @classmethod
    def has_active_credentials(cls):
        """Determines whether there are any active credentials stored at
        `~/.eta/minio-credentials.ini`.

        Returns:
            True/False
        """
        return os.path.isfile(cls.CREDENTIALS_PATH)

    @classmethod
    def load_credentials(cls, credentials_path=None, profile=None):
        """Loads the MinIO credentials as a dictionary.

        Args:
            credentials_path: an optional path to a credentials `.ini` file.
                If omitted, active credentials are located using the strategy
                described in the class docstring of `NeedsMinIOCredentials`
            profile: an optional profile to load when a credentials `.ini` file
                is loaded (if applicable). If not provided, the "default"
                section is loaded

        Returns:
            a (credentials, path) tuple containing the credentials and the path
                from which they were loaded. If the credentials were loaded
                from environment variables, `path` will be None
        """
        if profile is None and "MINIO_PROFILE" in os.environ:
            logger.debug(
                "Loading profile from 'MINIO_PROFILE' environment variable"
            )
            profile = os.environ["MINIO_PROFILE"]

        if credentials_path:
            logger.debug(
                "Loading MinIO credentials from manually provided path '%s'",
                credentials_path,
            )
            credentials = cls._load_credentials_ini(
                credentials_path, profile=profile
            )
            return credentials, credentials_path

        if (
            "MINIO_ACCESS_KEY" in os.environ
            and "MINIO_SECRET_ACCESS_KEY" in os.environ
        ):
            logger.debug(
                "Loading access key and secret key from 'MINIO_ACCESS_KEY' "
                "and 'MINIO_SECRET_ACCESS_KEY' environment variables"
            )
            credentials = {
                "access_key": os.environ["MINIO_ACCESS_KEY"],
                "secret_access_key": os.environ["MINIO_SECRET_ACCESS_KEY"],
            }

            if "MINIO_ENDPOINT_URL" in os.environ:
                logger.debug(
                    "Loading endpoint URL from 'MINIO_ENDPOINT_URL' "
                    "environment variable"
                )
                credentials["endpoint_url"] = os.environ["MINIO_ENDPOINT_URL"]

            if "MINIO_ALIAS" in os.environ:
                logger.debug(
                    "Loading region from 'MINIO_ALIAS' environment variable"
                )
                credentials["alias"] = os.environ["MINIO_ALIAS"]

            if "MINIO_REGION" in os.environ:
                logger.debug(
                    "Loading region from 'MINIO_REGION' environment variable"
                )
                credentials["region"] = os.environ["MINIO_REGION"]

            return credentials, None

        if "MINIO_SHARED_CREDENTIALS_FILE" in os.environ:
            credentials_path = os.environ["MINIO_SHARED_CREDENTIALS_FILE"]
            logger.debug(
                "Loading MinIO credentials from environment variable "
                "MINIO_SHARED_CREDENTIALS_FILE='%s'",
                credentials_path,
            )
        elif "MINIO_CONFIG_FILE" in os.environ:
            credentials_path = os.environ["MINIO_CONFIG_FILE"]
            logger.debug(
                "Loading MinIO credentials from environment variable "
                "MINIO_CONFIG_FILE='%s'",
                credentials_path,
            )
        elif cls.has_active_credentials():
            credentials_path = cls.CREDENTIALS_PATH
            logger.debug(
                "Loading activated MinIO credentials from '%s'",
                credentials_path,
            )
        else:
            raise MinIOCredentialsError("No MinIO credentials found")

        credentials = cls._load_credentials_ini(
            credentials_path, profile=profile
        )
        return credentials, credentials_path

    @classmethod
    def from_ini(cls, credentials_path, profile=None):
        """Creates a `cls` instance from the given credentials `.ini` file.

        Args:
            credentials_path: the path to a credentials `.ini` file
            profile: an optional profile to load. If not provided the "default"
                section is loaded

        Returns:
            an instance of cls with the given credentials
        """
        credentials, _ = cls.load_credentials(
            credentials_path=credentials_path, profile=profile
        )
        return cls(credentials=credentials)

    @classmethod
    def _load_credentials_ini(cls, credentials_path, profile=None):
        if profile is None:
            profile = "default"

        if not os.path.isfile(credentials_path):
            raise FileNotFoundError(
                "File '%s' does not exist" % credentials_path
            )

        config = configparser.ConfigParser()
        config.read(credentials_path)

        # Config file syntax
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#aws-config-file
        if "profile " + profile in config:
            return dict(config["profile " + profile])

        # Credentials file syntax
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#shared-credentials-file
        return dict(config[profile])


class MinIOCredentialsError(Exception):
    """Error raised when a problem with MinIO credentials is encountered."""

    def __init__(self, message):
        """Creates an MinIOCredentialsError instance.

        Args:
            message: the error message
        """
        super(MinIOCredentialsError, self).__init__(
            "%s. Read the documentation for "
            "`eta.core.storage.NeedsMinIOCredentials` for more information "
            "about authenticating with MinIO services." % message
        )


class MinIOStorageClient(_BotoStorageClient, NeedsMinIOCredentials):
    """Client for reading/writing data from MinIO.

    All cloud path strings used by this class should have the form
    "<alias>://<bucket>/<path/to/object>".

    See `NeedsMinIOCredentials` for more information about the authentication
    strategy used by this class.
    """

    def __init__(self, credentials=None, max_pool_connections=None, **kwargs):
        """Creates a MinIOStorageClient instance.

        Args:
            credentials: an optional MinIO credentials dictionary. If not
                provided, credentials must be loadable as described in
                `NeedsMinIOCredentials`
            max_pool_connections: an optional maximum number of connections to
                keep in the connection pool
            **kwargs: optional configuration options for
                `botocore.config.Config(**kwargs)`
        """
        if credentials is None:
            credentials, _ = self.load_credentials()

        _credentials = {
            "aws_access_key_id": credentials["access_key"],
            "aws_secret_access_key": credentials["secret_access_key"],
            "region_name": credentials.get("region", "us-east-1"),
        }

        alias = credentials.pop("alias", None)
        endpoint_url = credentials.pop("endpoint_url")

        super(MinIOStorageClient, self).__init__(
            _credentials,
            alias=alias,
            endpoint_url=endpoint_url,
            max_pool_connections=max_pool_connections,
            **kwargs,
        )


class NeedsGoogleCredentials(object):
    """Mixin for classes that need a `google.auth.credentials.Credentials`
    instance to take authenticated actions.

    Storage clients that derive from this class should allow users to provide
    credentials in the following ways (in order of precedence):

        (1) manually constructing an instance of the class via the
            `cls.from_json()` method by providing a path to a valid service
            account JSON file

        (2) setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable
            to point to a service account JSON file

        (3) loading credentials from `~/.eta/google-credentials.json` that have
            been activated via `cls.activate_credentials()`

    In the above, the service account JSON file should have syntax similar to
    the following::

        {
          "type": "service_account",
          "project_id": "<project-id>",
          "private_key_id": "<private-key-id>",
          "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
          "client_email": "<account-name>@<project-id>.iam.gserviceaccount.com",
          "client_id": "<client-id>",
          "auth_uri": "https://accounts.google.com/o/oauth2/auth",
          "token_uri": "https://oauth2.googleapis.com/token",
          "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
          "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/..."
        }

    See the following page for more information:
    https://cloud.google.com/docs/authentication/getting-started
    """

    CREDENTIALS_PATH = os.path.join(
        etac.ETA_CONFIG_DIR, "google-credentials.json"
    )

    @classmethod
    def activate_credentials(cls, credentials_path):
        """Activate the credentials by copying them to
        `~/.eta/google-credentials.json`.

        Args:
            credentials_path: the path to a service account JSON file
        """
        etau.copy_file(credentials_path, cls.CREDENTIALS_PATH)
        logger.info(
            "Google credentials successfully activated at '%s'",
            cls.CREDENTIALS_PATH,
        )

    @classmethod
    def deactivate_credentials(cls):
        """Deactivates (deletes) the currently active credentials, if any.

        Active credentials (if any) are at `~/.eta/google-credentials.json`.
        """
        try:
            os.remove(cls.CREDENTIALS_PATH)
            logger.info(
                "Google credentials '%s' successfully deactivated",
                cls.CREDENTIALS_PATH,
            )
        except OSError:
            logger.info("No Google credentials to deactivate")

    @classmethod
    def has_active_credentials(cls):
        """Determines whether there are any active credentials stored at
        `~/.eta/google-credentials.json`.

        Returns:
            True/False
        """
        return os.path.isfile(cls.CREDENTIALS_PATH)

    @classmethod
    def load_credentials(cls, credentials_path=None):
        """Loads Google credentials as an `google.auth.credentials.Credentials`
        instance.

        Args:
            credentials_path: an optional path to a service account JSON file.
                If omitted, the strategy described in the class docstring of
                `NeedsGoogleCredentials` is used to locate credentials

        Returns:
            a (credentials, path) tuple containing the
                `google.auth.credentials.Credentials` instance and the path
                from which it was loaded
        """
        info, credentials_path = cls.load_credentials_json(
            credentials_path=credentials_path
        )
        credentials = gos.Credentials.from_service_account_info(info)
        return credentials, credentials_path

    @classmethod
    def load_credentials_json(cls, credentials_path=None):
        """Loads the Google credentials as a JSON dictionary.

        Args:
            credentials_path: an optional path to a service account JSON file.
                If omitted, the strategy described in the class docstring of
                `NeedsGoogleCredentials` is used to locate credentials

        Returns:
            a (credentials_dict, path) tuple containing the service account
                dictionary and the path from which it was loaded
        """
        if credentials_path is not None:
            logger.debug(
                "Loading Google credentials from manually provided path '%s'",
                credentials_path,
            )
        else:
            credentials_path = cls._find_active_credentials()

        credentials_dict = etas.read_json(credentials_path)
        return credentials_dict, credentials_path

    @classmethod
    def from_json(cls, credentials_path):
        """Creates a `cls` instance from the given service account JSON file.

        Args:
            credentials_path: the path to a service account JSON file

        Returns:
            an instance of cls
        """
        credentials, _ = cls.load_credentials(
            credentials_path=credentials_path
        )
        return cls(credentials=credentials)

    @classmethod
    def _find_active_credentials(cls):
        credentials_path = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS", None
        )
        if credentials_path is not None:
            logger.debug(
                "Loading Google credentials from environment variable "
                "'GOOGLE_APPLICATION_CREDENTIALS=%s'",
                credentials_path,
            )
        elif cls.has_active_credentials():
            credentials_path = cls.CREDENTIALS_PATH
            logger.debug(
                "Loading activated Google credentials from '%s'",
                credentials_path,
            )
        else:
            raise GoogleCredentialsError("No Google credentials found")

        return credentials_path


class GoogleCredentialsError(Exception):
    """Error raised when a problem with Google credentials is encountered."""

    def __init__(self, message):
        """Creates a GoogleCredentialsError instance.

        Args:
            message: the error message
        """
        super(GoogleCredentialsError, self).__init__(
            "%s. Read the documentation for "
            "`eta.core.storage.NeedsGoogleCredentials` for more information "
            "about authenticating with Google services." % message
        )


class GoogleCloudStorageClient(
    StorageClient, CanSyncDirectories, NeedsGoogleCredentials
):
    """Client for reading/writing data from Google Cloud Storage buckets.

    All cloud path strings used by this class should have the form
    "gs://<bucket>/<path/to/object>".

    See `NeedsGoogleCredentials` for more information about the authentication
    strategy used by this class.
    """

    def __init__(
        self,
        credentials=None,
        chunk_size=None,
        max_pool_connections=None,
        **kwargs,
    ):
        """Creates a GoogleCloudStorageClient instance.

        Args:
            credentials: an optional `google.auth.credentials.Credentials`
                instance. If not provided, active credentials are automatically
                loaded as described in `NeedsGoogleCredentials`
            chunk_size: an optional chunk size (in bytes) to use for uploads
                and downloads
            max_pool_connections: an optional maximum number of connections to
                keep in the connection pool
            **kwargs: optional keyword arguments for
                `google.cloud.storage.Client(**kwargs)`
        """
        if credentials is None:
            credentials, _ = self.load_credentials()

        client = gcs.Client(
            credentials=credentials, project=credentials.project_id, **kwargs
        )

        # https://github.com/googleapis/python-bigquery/issues/59#issuecomment-650432896
        if max_pool_connections is not None:
            adapter = requests.adapters.HTTPAdapter(
                pool_maxsize=max_pool_connections
            )

            try:
                client._http.mount("http://", adapter)
                client._http.mount("https://", adapter)
                client._http_internal.mount("http://", adapter)
                client._http_internal.mount("https://", adapter)
                client._http._auth_request.session.mount("http://", adapter)
                client._http._auth_request.session.mount("https://", adapter)
            except Exception as e:
                logger.debug(e)

        retriable_types = (
            gae.TooManyRequests,  # 429
            gae.InternalServerError,  # 500
            gae.BadGateway,  # 502
            gae.ServiceUnavailable,  # 503
        )

        retry = gar.Retry(predicate=lambda e: isinstance(e, retriable_types))

        self.chunk_size = chunk_size
        self._client = client
        self._retry = retry

    def upload(self, local_path, cloud_path, content_type=None, metadata=None):
        """Uploads the file to GCS.

        Args:
            local_path: the path to the file to upload
            cloud_path: the path to the GCS object to create
            content_type: the optional type of the content being uploaded. If
                no value is provided, it is guessed from the filename
            metadata: an optional dictionary of custom object metadata to store
        """
        content_type = content_type or etau.guess_mime_type(local_path)
        blob = self._get_blob(cloud_path)
        if metadata:
            blob.metadata = metadata

        blob.upload_from_filename(local_path, content_type=content_type)

    def upload_bytes(
        self, bytes_str, cloud_path, content_type=None, metadata=None
    ):
        """Uploads the given bytes to GCS.

        Args:
            bytes_str: the bytes string to upload
            cloud_path: the path to the GCS object to create
            content_type: the optional type of the content being uploaded. If
                no value is provided but the object already exists in GCS, then
                the same value is used. Otherwise, the default value
                ("application/octet-stream") is used
            metadata: an optional dictionary of custom object metadata to store
        """
        blob = self._get_blob(cloud_path)
        if metadata:
            blob.metadata = metadata

        blob.upload_from_string(bytes_str, content_type=content_type)

    def upload_stream(
        self, file_obj, cloud_path, content_type=None, metadata=None
    ):
        """Uploads the contents of the given file-like object to GCS.

        Args:
            file_obj: the file-like object to upload, which must be open for
                reading
            cloud_path: the path to the GCS object to create
            content_type: the optional type of the content being uploaded. If
                no value is provided but the object already exists in GCS, then
                the same value is used. Otherwise, the default value
                ("application/octet-stream") is used
            metadata: an optional dictionary of custom object metadata to store
        """
        blob = self._get_blob(cloud_path)
        if metadata:
            blob.metadata = metadata

        blob.upload_from_file(file_obj, content_type=content_type)

    def download(self, cloud_path, local_path):
        """Downloads the file from GCS to the given location.

        Args:
            cloud_path: the path to the GCS object to download
            local_path: the local disk path to store the downloaded file
        """
        blob = self._get_blob(cloud_path)
        etau.ensure_basedir(local_path)
        blob.download_to_filename(local_path, checksum=None)

    def download_bytes(self, cloud_path, start=None, end=None):
        """Downloads the file from GCS and returns the bytes string.

        Args:
            cloud_path: the path to the GCS object to download
            start: an optional first byte in a range to download
            end: an optional last byte in a range to download

        Returns:
            the downloaded bytes string
        """
        blob = self._get_blob(cloud_path)
        return blob.download_as_bytes(start=start, end=end, checksum=None)

    def download_stream(self, cloud_path, file_obj):
        """Downloads the file from GCS to the given file-like object.

        Args:
            cloud_path: the path to the GCS object to download
            file_obj: the file-like object to which to write the download,
                which must be open for writing
        """
        blob = self._get_blob(cloud_path)
        blob.download_to_file(file_obj, checksum=None)

    def delete(self, cloud_path):
        """Deletes the given file from GCS.

        Args:
            cloud_path: the path to the GCS object to delete
        """
        blob = self._get_blob(cloud_path)
        blob.delete()

    def delete_folder(self, cloud_folder):
        """Deletes all files in the given GCS "folder".

        Args:
            cloud_folder: a string like `gs://<bucket-name>/<folder-path>`
        """
        bucket_name, folder_name = self._parse_path(cloud_folder)
        if folder_name and not folder_name.endswith("/"):
            folder_name += "/"

        blobs = self._client.list_blobs(
            bucket_name, prefix=folder_name, retry=self._retry
        )

        for blob in blobs:
            blob.delete()

    def get_file_metadata(self, cloud_path):
        """Returns metadata about the given file in GCS.

        Args:
            cloud_path: the path to the GCS object

        Returns:
            a dictionary containing metadata about the file, including its
                `bucket`, `object_name`, `name`, `size`, `mime_type`,
                `last_modified`, `etag`, and `metadata`
        """
        blob = self._get_blob(cloud_path, include_metadata=True)
        return self._get_file_metadata(blob)

    def get_folder_metadata(self, cloud_folder):
        """Returns metadata about the given "folder" in GCS.

        Note that this method is *expensive*; the only way to compute this
        information is to call `list_files_in_folder(..., recursive=True)` and
        compute stats from individual files!

        Args:
            cloud_folder: a string like `gs://<bucket-name>/<folder-path>`

        Returns:
            a dictionary containing metadata about the "folder", including its
                `bucket`, `path`, `num_files`, `size`, and `last_modified`
        """
        files = self.list_files_in_folder(
            cloud_folder, recursive=True, return_metadata=True
        )

        bucket, path = self._parse_path(cloud_folder)
        path = path.rstrip("/")

        if files:
            num_files = len(files)
            size = sum(f["size"] for f in files)
            last_modified = max(f["last_modified"] for f in files)
        else:
            num_files = 0
            size = 0
            last_modified = "-"

        return {
            "bucket": bucket,
            "path": path,
            "num_files": num_files,
            "size": size,
            "last_modified": last_modified,
        }

    def is_file(self, cloud_path):
        """Determines whether the given GCS file exists.

        Args:
            cloud_path: the path to the GCS object

        Returns:
            True/False
        """
        blob = self._get_blob(cloud_path)
        return blob.exists()

    def is_folder(self, cloud_folder):
        """Determines whether the given GCS "folder" contains at least one
        object.

        Args:
            cloud_folder: a string like `gs://<bucket-name>/<folder-path>`

        Returns:
            True/False
        """
        bucket_name, folder_name = self._parse_path(cloud_folder)
        if folder_name and not folder_name.endswith("/"):
            folder_name += "/"

        blobs = self._client.list_blobs(
            bucket_name, max_results=1, prefix=folder_name, retry=self._retry
        )
        return bool(list(blobs))

    def list_files_in_folder(
        self, cloud_folder, recursive=True, return_metadata=False
    ):
        """Returns a list of the files in the given "folder" in GCS.

        Args:
            cloud_folder: a string like `gs://<bucket-name>/<folder-path>`
            recursive: whether to recursively traverse sub-"folders". By
                default, this is True
            return_metadata: whether to return a metadata dictionary for each
                file, including its  `bucket`, `object_name`, `name`, `size`,
                `mime_type`, `last_modified`, `etag`, and `metadata`. By
                default, only the paths to the files are returned

        Returns:
            a list of full cloud paths (when `return_metadata == False`) or a
                list of metadata dictionaries (when `return_metadata == True`)
                for the files in the folder
        """
        bucket_name, folder_name = self._parse_path(cloud_folder)
        if folder_name and not folder_name.endswith("/"):
            folder_name += "/"

        delimiter = "/" if not recursive else None
        blobs = self._client.list_blobs(
            bucket_name,
            prefix=folder_name,
            delimiter=delimiter,
            retry=self._retry,
        )

        # Return metadata dictionaries for each file
        if return_metadata:
            metadata = []
            for blob in blobs:
                if not blob.name.endswith("/"):
                    metadata.append(self._get_file_metadata(blob))

            return metadata

        # Return paths for each file
        paths = []
        prefix = "gs://" + bucket_name
        for blob in blobs:
            if not blob.name.endswith("/"):
                paths.append(os.path.join(prefix, blob.name))

        return paths

    def generate_signed_url(
        self, cloud_path, method="GET", hours=24, content_type=None
    ):
        """Generates a signed URL for accessing the given GCS object.

        Anyone with the URL can access the object with the permission until it
        expires.

        Note that you should use `PUT`, not `POST`, to upload objects!

        Args:
            cloud_path: the path to the GCS object
            method: the HTTP verb (GET, PUT, DELETE) to authorize
            hours: the number of hours that the URL is valid
            content_type: (PUT actions only) the optional type of the content
                being uploaded

        Returns:
            a URL for accessing the object via HTTP request
        """
        bucket, name = self._parse_path(cloud_path)
        resource = "/%s/%s" % (bucket, urlparse.quote(name, safe=b"/~"))
        return generate_signed_url_v4(
            self._client._credentials,
            resource=resource,
            expiration=datetime.timedelta(hours=hours),
            method=method.upper(),
            content_type=content_type,
        )

    def add_metadata(self, cloud_path, metadata):
        """Adds custom metadata key-value pairs to the given GCS object.

        Args:
            cloud_path: the path to the GCS object
            metadata: a dictionary of custom metadata
        """
        blob = self._get_blob(cloud_path, include_metadata=True)

        if blob.metadata:
            blob.metadata.update(metadata)
        else:
            blob.metadata = metadata

        blob.patch()

    def _get_blob(self, cloud_path, include_metadata=False):
        bucket_name, object_name = self._parse_path(cloud_path)
        bucket = self._client.get_bucket(bucket_name, retry=self._retry)
        blob = bucket.blob(object_name, chunk_size=self.chunk_size)

        if include_metadata:
            blob.reload()

        return blob

    @staticmethod
    def _get_file_metadata(blob):
        mime_type = blob.content_type
        if not mime_type:
            mime_type = etau.guess_mime_type(blob.name)

        return {
            "bucket": blob.bucket.name,
            "object_name": blob.name,
            "name": os.path.basename(blob.name),
            "size": blob.size,
            "mime_type": mime_type,
            "last_modified": blob.updated,
            "etag": blob.etag,
            "metadata": blob.metadata or {},
        }

    def _strip_prefix(self, cloud_path):
        prefix = "gs://"

        if not cloud_path.startswith(prefix):
            raise ValueError(
                "Invalid path '%s'; must start with %s" % (cloud_path, prefix)
            )

        return cloud_path[len(prefix) :]

    def _parse_path(self, cloud_path):
        _cloud_path = self._strip_prefix(cloud_path)

        chunks = _cloud_path.split("/", 1)

        if len(chunks) != 2:
            return chunks[0], ""

        return chunks[0], chunks[1]


class GoogleDriveStorageClient(StorageClient, NeedsGoogleCredentials):
    """Client for reading/writing data from Google Drive.

    The service account credentials you use must have access permissions for
    any Drive folders you intend to access.

    See `NeedsGoogleCredentials` for more information about the authentication
    strategy used by this class.

    Attributes:
        chunk_size: the chunk size (in bytes) that will be used for uploads and
            downloads
    """

    #
    # The default chunk size to use when uploading and downloading files.
    # Note that this gives the Drive API the right to use up to this
    # much memory as a buffer during read/write
    #
    DEFAULT_CHUNK_SIZE = 256 * 1024 * 1024  # in bytes

    def __init__(self, credentials=None, chunk_size=None):
        """Creates a GoogleDriveStorageClient instance.

        Args:
            credentials: an optional `google.auth.credentials.Credentials`
                instance. If not provided, active credentials are automatically
                loaded as described in `NeedsGoogleCredentials`
            chunk_size: an optional chunk size (in bytes) to use for uploads
                and downloads. By default, `DEFAULT_CHUNK_SIZE` is used
        """
        if credentials is None:
            credentials, _ = self.load_credentials()

        self._service = gad.build(
            "drive", "v3", credentials=credentials, cache_discovery=False
        )
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE

    def upload(self, local_path, folder_id, filename=None, content_type=None):
        """Uploads the file to Google Drive.

        Note that `filename` can contain directories like `path/to/filename`.
        Such directories will be created (if necessary) in Google Drive.

        Args:
            local_path: the path to the file to upload
            folder_id: the ID of the Drive folder to upload the file into
            filename: the optional name to give the file in Drive. By default,
                the name of the local file is used
            content_type: the optional content type of the file. By default,
                the content type is guessed from the filename

        Returns:
            file_id: the ID of the uploaded file
        """
        name = filename or os.path.basename(local_path)
        with open(local_path, "rb") as f:
            return self._do_upload(f, folder_id, name, content_type)

    def upload_bytes(self, bytes_str, folder_id, filename, content_type=None):
        """Uploads the given bytes to Google Drive.

        Note that `filename` can contain directories like `path/to/filename`.
        Such directories will be created (if necessary) in Google Drive.

        Args:
            bytes_str: the bytes string to upload
            folder_id: the ID of the Drive folder to upload the file into
            filename: the name to give the file in Drive
            content_type: the optional content type of the file. By default,
                the content type is guessed from the filename

        Returns:
            file_id: the ID of the uploaded file
        """
        with io.BytesIO(_to_bytes(bytes_str)) as f:
            return self._do_upload(f, folder_id, filename, content_type)

    def upload_stream(self, file_obj, folder_id, filename, content_type=None):
        """Uploads the contents of the given file-like object to Google Drive.

        Note that `filename` can contain directories like `path/to/filename`.
        Such directories will be created (if necessary) in Google Drive.

        Args:
            file_obj: the file-like object to upload, which must be open for
                reading
            folder_id: the ID of the Drive folder to upload the file into
            filename: the name to give the file in Drive
            content_type: the optional content type of the file. By default,
                the content type is guessed from the filename

        Returns:
            file_id: the ID of the uploaded file
        """
        return self._do_upload(file_obj, folder_id, filename, content_type)

    def download(self, file_id, local_path):
        """Downloads the file from Google Drive.

        Args:
            file_id: the ID of the file to download
            local_path: the path to which to download the file
        """
        etau.ensure_basedir(local_path)
        with open(local_path, "wb") as f:
            self._do_download(file_id, f)

    def download_bytes(self, file_id):
        """Downloads a file from Google Drive and returns the bytes.

        Args:
            file_id: the ID of the file to download

        Returns:
            the downloaded bytes string
        """
        with io.BytesIO() as f:
            self._do_download(file_id, f)
            return f.getvalue()

    def download_stream(self, file_id, file_obj):
        """Downloads the file from Google Drive to the given file-like object.

        Args:
            file_id: the ID of the file to download
            file_obj: the file-like object to which to write the download,
                which must be open for writing
        """
        self._do_download(file_id, file_obj)

    def delete(self, file_id):
        """Deletes the file from Google Drive.

        Args:
            file_id: the ID of the file to delete
        """
        self._service.files().delete(  # pylint: disable=no-member
            fileId=file_id, supportsTeamDrives=True
        ).execute()

    def delete_folder(self, folder_id):
        """Deletes the folder from Google Drive.

        Args:
            folder_id: the ID of the folder to delete
        """
        self._service.files().delete(  # pylint: disable=no-member
            fileId=folder_id, supportsTeamDrives=True
        ).execute()

    def delete_folder_contents(self, folder_id, skip_failures=False):
        """Deletes the contents (files and subfolders) of the given Google
        Drive folder, retaining the (now empty) parent folder.

        Args:
            folder_id: the ID of the folder from which the files are to be
                deleted
            skip_failures: whether to gracefully skip delete errors. By
                default, this is False
        """
        files, folders = self._list_folder_contents(folder_id)
        files += folders

        num_files = len(files)
        if num_files > 0:
            logger.info(
                "Deleting %d files and folders from '%s'", num_files, folder_id
            )
        for f in files:
            try:
                self.delete(f["id"])  # works for files and folders
                logger.info("Deleted '%s'", f["name"])
            except Exception as e:
                if not skip_failures:
                    raise GoogleDriveStorageClientError(e)
                logger.warning(
                    "Failed to delete '%s' from '%s'; skipping",
                    f["name"],
                    folder_id,
                )

    def get_file_metadata(self, file_id, include_path=False):
        """Gets metadata about the file with the given ID.

        Args:
            file_id: the ID of the file
            include_path: whether to include information about the absolute
                path to the file. By default, this is False

        Returns:
            a dictionary containing the available metadata about the file,
                including its `id`, `name`, `size`, `mime_type`, and
                `last_modified`. When `include_path == True`, additional fields
                `drive_id`, `drive_name`, and `path` are included
        """
        fields = ["id", "name", "size", "mimeType", "modifiedTime"]
        _meta = self._get_file_metadata(file_id, fields=fields)
        metadata = self._parse_file_metadata(_meta)

        if include_path:
            metadata.update(self._get_filepath_metadata(file_id))

        return metadata

    def get_folder_metadata(self, folder_id):
        """Returns metadata about the given folder.

        Note that this method is *expensive*; the only way to compute this
        information is to call `list_files_in_folder(..., recursive=True)` and
        compute stats from individual files!

        Args:
            folder_id: the ID of the folder

        Returns:
            a dictionary containing metadata about the folder, including its
                `folder_id`, `drive_id`, `drive_name`, `path`, `num_files`,
                `size`, and `last_modified`
        """
        # Get folder path metadata
        metadata = self._get_filepath_metadata(folder_id)

        # Compute (at great cost) metadata about folder contents
        files = self.list_files_in_folder(folder_id, recursive=True)
        if files:
            num_files = len(files)
            size = sum(f["size"] for f in files)
            last_modified = max(f["last_modified"] for f in files)
        else:
            num_files = 0
            size = 0
            last_modified = "-"

        return {
            "folder_id": folder_id,
            "drive_id": metadata["drive_id"],
            "drive_name": metadata["drive_name"],
            "path": metadata["path"],
            "num_files": num_files,
            "size": size,
            "last_modified": last_modified,
        }

    def get_team_drive_id(self, name):
        """Get the ID of the Team Drive with the given name.

        Args:
            name: the name of the Team Drive

        Returns:
            the ID of the root folder in the Team Drive

        Raises:
            GoogleDriveStorageClientError: if the Team Drive was not found
        """
        response = (
            self._service.teamdrives()  # pylint: disable=no-member
            .list(fields="teamDrives(id, name)")
            .execute()
        )
        for team_drive in response["teamDrives"]:
            if name == team_drive["name"]:
                return team_drive["id"]

        raise GoogleDriveStorageClientError("Team Drive '%s' not found" % name)

    def get_team_drive_name(self, drive_id):
        """Gets the name of the Team Drive with the given ID.

        Args:
            drive_id: the ID of the Team Drive

        Returns:
            the name of the Team Drive

        Raises:
            GoogleDriveStorageClientError: if the Team Drive was not found
        """
        try:
            response = (
                self._service.teamdrives().get(teamDriveId=drive_id).execute()
            )
            return response["name"]
        except:
            raise GoogleDriveStorageClientError(
                "Team Drive with ID '%s' not found" % drive_id
            )

    def get_root_team_drive_id(self, file_id):
        """Returns the ID of the root Team Drive in which this file lives.

        Args:
            file_id: the ID of the file (or folder)

        Returns:
            the ID of the Team Drive, or None if the file does not live in a
                Team Drive
        """
        metadata = self._get_file_metadata(file_id, ["teamDriveId"])
        return metadata.get("teamDriveId", None)

    def is_folder(self, file_id):
        """Determines whether the file with the given ID is a folder.

        Args:
            file_id: the ID of a file (or folder)

        Returns:
            True/False
        """
        metadata = self._get_file_metadata(file_id, ["mimeType"])
        return self._is_folder(metadata)

    def create_folder_if_necessary(self, folder_name, parent_folder_id):
        """Creates the given folder within the given parent folder, if
        necessary.

        Args:
            folder_name: a `folder` or `a/nested/folder` specifying the folder
                to create, if necessary
            parent_folder_id: the ID of the parent folder in which to operate

        Returns:
            the ID of the last created folder
        """
        folder_id = parent_folder_id
        for folder in folder_name.split("/"):
            folder_id = self._create_folder_if_necessary(folder, folder_id)
        return folder_id

    def create_folder(self, folder_name, parent_folder_id):
        """Creates the given folder within the given parent folder. The folder
        is assumed not to exist.

        Args:
            folder_name: a `folder` or `a/nested/folder` specifying the folder
                to create
            parent_folder_id: the ID of the parent folder in which to operate

        Returns:
            the ID of the last created folder
        """
        folder_id = parent_folder_id
        for folder in folder_name.split("/"):
            folder_id = self._create_folder(folder, folder_id)
        return folder_id

    def list_subfolders(self, folder_id, recursive=False):
        """Returns a list of the subfolders of the folder with the given ID.

        Args:
            folder_id: the ID of a folder
            recursive: whether to recursively traverse subfolders. By
                default, this is False

        Returns:
            a list of dicts containing the `id`, `name`, and `last_modified` of
                the subfolders
        """
        # List folder contents
        _, folders = self._list_folder_contents(folder_id)

        if recursive:
            # Recursively traverse subfolders
            for folder in folders:
                for f in self.list_subfolders(folder["id"], recursive=True):
                    # Embed <root>/<subdir> namespace in folder name
                    f["name"] = os.path.join(folder["name"], f["name"])
                    folders.append(f)

        return [self._parse_folder_metadata(f) for f in folders]

    def list_files_in_folder(self, folder_id, recursive=False):
        """Returns a list of the files in the folder with the given ID.

        Args:
            folder_id: the ID of a folder
            recursive: whether to recursively traverse sub-"folders". By
                default, this is False

        Returns:
            a list of dicts containing the `id`, `name`, `size`, `mime_type`,
                and `last_modified` of the files in the folder
        """
        # List folder contents
        files, folders = self._list_folder_contents(folder_id)
        files = [self._parse_file_metadata(f) for f in files]

        if recursive:
            # Recursively traverse subfolders
            for folder in folders:
                contents = self.list_files_in_folder(
                    folder["id"], recursive=True
                )
                for f in contents:
                    # Embed <folder-name>/<file-name> namespace in filename
                    f["name"] = os.path.join(folder["name"], f["name"])
                    files.append(f)

        return files

    def upload_files_in_folder(
        self,
        local_dir,
        folder_id,
        skip_failures=False,
        skip_existing_files=False,
        recursive=False,
    ):
        """Uploads the files in the given folder to Google Drive.

        Note that this function uses `eta.core.utils.list_files()` to determine
        which files to upload. This means that hidden files are skipped.

        Args:
            local_dir: the directory of files to upload
            folder_id: the ID of the Drive folder to upload the files into
            skip_failures: whether to gracefully skip upload errors. By
                default, this is False
            skip_existing_files: whether to skip files whose names match
                existing files in the Google Drive folder. By default, this is
                False
            recursive: whether to recursively upload the contents of
                subdirectories. By default, this is False

        Returns:
            a dict mapping filenames to IDs of the uploaded files

        Raises:
            GoogleDriveStorageClientError if an upload error occured and
                failure skipping is turned off
        """
        # Get local files to upload
        files = etau.list_files(local_dir)

        # Get existing Drive files
        if skip_existing_files or recursive:
            existing_files, existing_folders = self._list_folder_contents(
                folder_id
            )

        # Skip existing files, if requested
        if skip_existing_files:
            _existing_files = set(f["name"] for f in existing_files)
            _files = [f for f in files if f not in _existing_files]
            num_skipped = len(files) - len(_files)
            if num_skipped > 0:
                logger.info("Skipping %d existing files", num_skipped)
                files = _files

        # Upload files
        num_files = len(files)
        if num_files > 0:
            logger.info("Uploading %d files to '%s'", num_files, folder_id)
        file_ids = {}
        for filename in files:
            try:
                local_path = os.path.join(local_dir, filename)
                file_id = self.upload(local_path, folder_id)
                file_ids[filename] = file_id
                logger.info("Uploaded '%s'", local_path)
            except Exception as e:
                if not skip_failures:
                    raise GoogleDriveStorageClientError(e)

                logger.warning(
                    "Failed to upload '%s' to '%s'; skipping",
                    local_path,
                    folder_id,
                )

        # Recursively traverse subfolders, if requested
        if recursive:
            for subfolder in etau.list_subdirs(local_dir):
                subfolder_id = self._create_folder_if_necessary(
                    subfolder, folder_id, existing_folders=existing_folders
                )
                logger.info(
                    "Recursively uploading contents of '%s' to '%s'",
                    subfolder,
                    subfolder_id,
                )

                sublocal_dir = os.path.join(local_dir, subfolder)
                subfile_ids = self.upload_files_in_folder(
                    sublocal_dir,
                    subfolder_id,
                    skip_failures=skip_failures,
                    skip_existing_files=skip_existing_files,
                    recursive=True,
                )

                for subname, subid in iteritems(subfile_ids):
                    file_ids[os.path.join(subfolder, subname)] = subid

        return file_ids

    def download_files_in_folder(
        self,
        folder_id,
        local_dir,
        skip_failures=False,
        skip_existing_files=False,
        recursive=True,
    ):
        """Downloads the files in the Google Drive folder to the given local
        directory.

        Args:
            folder_id: the ID of the Drive folder to download files from
            local_dir: the directory to download the files into
            skip_failures: whether to gracefully skip download errors. By
                default, this is False
            skip_existing_files: whether to skip files whose names match
                existing files in the local directory. By default, this is
                False
            recursive: whether to recursively traverse sub-"folders". By
                default, this is True

        Returns:
            a list of filenames of the downloaded files

        Raises:
            GoogleDriveStorageClientError if a download error occured and
                failure skipping is turned off
        """
        # Get Drive files in folder
        files, folders = self._list_folder_contents(folder_id)

        # Skip existing files, if requested
        if skip_existing_files:
            etau.ensure_dir(local_dir)
            existing_files = set(etau.list_files(local_dir))
            _files = [f for f in files if f["name"] not in existing_files]
            num_skipped = len(files) - len(_files)
            if num_skipped > 0:
                logger.info("Skipping %d existing files", num_skipped)
                files = _files

        # Download files
        num_files = len(files)
        if num_files > 0:
            logger.info("Downloading %d files to '%s'", num_files, local_dir)
        filenames = []
        for f in files:
            filename = f["name"]
            file_id = f["id"]
            try:
                local_path = os.path.join(local_dir, filename)
                self.download(file_id, local_path)
                filenames.append(filename)
                logger.info("Downloaded '%s'", local_path)
            except Exception as e:
                if not skip_failures:
                    raise GoogleDriveStorageClientError(e)

                logger.warning(
                    "Failed to download '%s' to '%s'; skipping",
                    file_id,
                    local_path,
                )

        # Recursively download folders, if requested
        if recursive:
            for folder in folders:
                subdir_name = folder["name"]
                subdir_id = folder["id"]
                local_subdir = os.path.join(local_dir, subdir_name)
                logger.info(
                    "Recursively downloading contents of '%s' to '%s'",
                    subdir_id,
                    local_subdir,
                )

                subfiles = self.download_files_in_folder(
                    subdir_id,
                    local_subdir,
                    skip_failures=skip_failures,
                    skip_existing_files=skip_existing_files,
                    recursive=True,
                )

                for subfile in subfiles:
                    filenames.append(os.path.join(subdir_name, subfile))

        return filenames

    def delete_duplicate_files_in_folder(
        self, folder_id, skip_failures=False, recursive=False
    ):
        """Deletes any duplicate files (i.e., files with the same filename) in
        the given Google Drive folder.

        Args:
            folder_id: the ID of the Drive folder to process
            skip_failures: whether to gracefully skip deletion errors. By
                default, this is False
            recursive: whether to recursively traverse subfolders. By default,
                this is False

        Returns:
            num_deleted: the number of deleted files

        Raises:
            GoogleDriveStorageClientError if a deletion error occured and
                failure skipping is turned off
        """
        existing_files = set()
        num_deleted = 0
        for f in self.list_files_in_folder(folder_id, recursive=recursive):
            filename = f["name"]
            if filename not in existing_files:
                existing_files.add(filename)
                continue

            # Delete duplicate file
            try:
                self.delete(f["id"])
                num_deleted += 1
                logger.info("Deleted '%s' from '%s'", filename, folder_id)
            except Exception as e:
                if not skip_failures:
                    raise GoogleDriveStorageClientError(e)

                logger.warning(
                    "Failed to delete '%s' from '%s'; skipping",
                    filename,
                    folder_id,
                )

        return num_deleted

    def _do_upload(self, file_obj, folder_id, filename, content_type):
        # Handle any leading directories
        chunks = filename.rsplit("/", 1)
        if len(chunks) == 2:
            folder_id = self.create_folder_if_necessary(chunks[0], folder_id)
            filename = chunks[1]

        mime_type = content_type or etau.guess_mime_type(filename)
        media = gah.MediaIoBaseUpload(
            file_obj, mime_type, chunksize=self.chunk_size, resumable=True
        )
        body = {
            "name": filename,
            "mimeType": mime_type,
            "parents": [folder_id],
        }
        stored_file = (
            self._service.files()  # pylint: disable=no-member
            .create(
                body=body,
                media_body=media,
                supportsTeamDrives=True,
                fields="id",
            )
            .execute()
        )
        return stored_file["id"]

    def _do_download(self, file_id, file_obj):
        request = self._service.files().get_media(  # pylint: disable=no-member
            fileId=file_id, supportsTeamDrives=True
        )
        downloader = gah.MediaIoBaseDownload(
            file_obj, request, chunksize=self.chunk_size
        )

        done = False
        while not done:
            _, done = downloader.next_chunk()

    @staticmethod
    def _is_folder(f):
        return f.get("mimeType", "") == "application/vnd.google-apps.folder"

    def _create_folder_if_necessary(
        self, new_folder, parent_folder_id, existing_folders=None
    ):
        if existing_folders is None:
            _, existing_folders = self._list_folder_contents(parent_folder_id)

        folder_id = None
        for f in existing_folders:
            if new_folder == f["name"]:
                folder_id = f["id"]
                break

        if folder_id is None:
            folder_id = self._create_folder(new_folder, parent_folder_id)

        return folder_id

    def _create_folder(self, new_folder, parent_folder_id):
        body = {
            "name": new_folder,
            "parents": [parent_folder_id],
            "mimeType": "application/vnd.google-apps.folder",
        }
        folder = (
            self._service.files()  # pylint: disable=no-member
            .create(body=body, supportsTeamDrives=True, fields="id")
            .execute()
        )
        return folder["id"]

    def _list_folder_contents(self, folder_id):
        # Handle Team Drives
        team_drive_id = self.get_root_team_drive_id(folder_id)
        if team_drive_id:
            params = {
                "corpora": "teamDrive",
                "supportsTeamDrives": True,
                "includeTeamDriveItems": True,
                "teamDriveId": team_drive_id,
            }
        else:
            params = {}

        # Perform query
        folders = []
        files = []
        page_token = None
        query = "'%s' in parents and trashed=false" % folder_id
        fields = "id,name,size,mimeType,modifiedTime"
        while True:
            # Get the next page of files
            response = (
                self._service.files()  # pylint: disable=no-member
                .list(
                    q=query,
                    fields="files(%s),nextPageToken" % fields,
                    pageSize=256,
                    pageToken=page_token,
                    **params,
                )
                .execute()
            )
            page_token = response.get("nextPageToken", None)

            for f in response["files"]:
                if self._is_folder(f):
                    folders.append(f)
                else:
                    files.append(f)

            # Check for end of list
            if not page_token:
                break

        return files, folders

    def _get_file_metadata(self, file_id, fields=None, all_fields=False):
        if all_fields:
            fields = "*"
        else:
            fields = ",".join(fields)

        return (
            self._service.files()  # pylint: disable=no-member
            .get(fileId=file_id, fields=fields, supportsTeamDrives=True)
            .execute()
        )

    def _get_filepath_metadata(self, file_id):
        parts = []
        drive_id = None
        part_id = file_id
        while True:
            metadata = self._get_file_metadata(part_id, ["name", "parents"])
            if "parents" in metadata:
                parts.append(metadata["name"])
                part_id = metadata["parents"][0]
            else:
                drive_id = part_id
                break

        try:
            drive_name = self.get_team_drive_name(drive_id)
        except:
            drive_name = ""

        return {
            "drive_id": drive_id,
            "drive_name": drive_name,
            "path": "/".join(reversed(parts)),
        }

    @staticmethod
    def _parse_file_metadata(metadata):
        return {
            "id": metadata["id"],
            "name": metadata["name"],
            "size": int(metadata.get("size", -1)),
            "mime_type": metadata["mimeType"],
            "last_modified": dateutil.parser.parse(metadata["modifiedTime"]),
        }

    @staticmethod
    def _parse_folder_metadata(metadata):
        return {
            "id": metadata["id"],
            "name": metadata["name"],
            "last_modified": dateutil.parser.parse(metadata["modifiedTime"]),
        }


class GoogleDriveStorageClientError(Exception):
    """Error raised when a problem occurred in a GoogleDriveStorageClient."""

    pass


class HTTPStorageClient(StorageClient):
    """Client for reading/writing files via HTTP requests.

    Attributes:
        set_content_type: whether to set the `Content-Type` in the request
            header of uploads. The Google Cloud documentation requires that
            the `Content-Type` be *OMITTED* from PUT requests to Google Cloud
            Storage, so set this attribute to False for use with GCS
        chunk_size: the chunk size (in bytes) that will be used for streaming
            downloads

    Examples::

        # Use client to perform a series of tasks without closing and
        # reopening the request session each time
        client = HTTPStorageClient(...)
        client.upload(...)
        client.download(...)
        ...
        client.close()  # make sure the session is closed

        # Automatic connection management via context manager
        with HTTPStorageClient(...) as client:
            client.upload(...)
            client.download(...)
            ...
    """

    #
    # The default chunk size to use when downloading files.
    # Note that this gives the requests toolbelt the right to use up to this
    # much memory as a buffer during read/write
    #
    DEFAULT_CHUNK_SIZE = 32 * 1024 * 1024  # in bytes

    def __init__(
        self,
        set_content_type=False,
        chunk_size=None,
        max_pool_connections=None,
    ):
        """Creates an HTTPStorageClient instance.

        Args:
            set_content_type: whether to specify the `Content-Type` during
                upload requests. By default, this is False
            chunk_size: an optional chunk size (in bytes) to use for downloads.
                By default, `DEFAULT_CHUNK_SIZE` is used
            max_pool_connections: an optional maximum number of connections to
                keep in the connection pool
        """
        session = requests.Session()

        if max_pool_connections is not None:
            adapter = requests.adapters.HTTPAdapter(
                pool_maxsize=max_pool_connections
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)

        self.set_content_type = set_content_type
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self._session = session

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Closes the HTTP session."""
        self._session.close()

    def upload(self, local_path, url, filename=None, content_type=None):
        """Uploads the file to the given URL via a PUT request.

        Args:
            local_path: the path to the file to upload
            url: the URL to which to PUT the file
            filename: an optional filename to include in the request. By
                default, the name of the local file is used
            content_type: an optional content type of the file. By default,
                the type is guessed from the filename. Note that this is only
                added to the request when `set_content_type` is True

        Raises:
            `requests.exceptions.HTTPError`: if the request resulted in an HTTP
                error
        """
        filename = filename or os.path.basename(local_path)
        content_type = content_type or etau.guess_mime_type(filename)
        with open(local_path, "rb") as f:
            self._do_upload(f, url, filename, content_type)

    def upload_bytes(self, bytes_str, url, filename=None, content_type=None):
        """Uploads the given bytes to the given URL via a PUT request.

        Args:
            bytes_str: the bytes string to upload
            url: the URL to which to PUT the file
            filename: an optional filename to include in the request
            content_type: an optional content type to include in the request.
                Note that this is only added when `set_content_type` is True

        Raises:
            `requests.exceptions.HTTPError`: if the request resulted in an HTTP
                error
        """
        with io.BytesIO(_to_bytes(bytes_str)) as f:
            self._do_upload(f, url, filename, content_type)

    def upload_stream(self, file_obj, url, filename=None, content_type=None):
        """Uploads the contents of the given file-like object to the given
        URL via a PUT request.

        Args:
            file_obj: the file-like object to upload, which must be open for
                reading
            url: the URL to which to PUT the file
            filename: an optional filename to include in the request
            content_type: an optional content type to include in the request.
                Note that this is only added when `set_content_type` is True

        Raises:
            `requests.exceptions.HTTPError`: if the request resulted in an HTTP
                error
        """
        self._do_upload(file_obj, url, filename, content_type)

    def download(self, url, local_path):
        """Downloads the file from the given URL via a GET request.

        Args:
            url: the URL from which to GET the file
            local_path: the path to which to write the downloaded file

        Raises:
            `requests.exceptions.HTTPError`: if the request resulted in an HTTP
                error
        """
        etau.ensure_basedir(local_path)
        with open(local_path, "wb") as f:
            self._do_download(url, f)

    def download_bytes(self, url, start=None, end=None):
        """Downloads bytes from the given URL via a GET request.

        Args:
            url: the URL from which to GET the file
            start: an optional first byte in a range to download
            end: an optional last byte in a range to download

        Returns:
            the downloaded bytes string

        Raises:
            `requests.exceptions.HTTPError`: if the request resulted in an HTTP
                error
        """
        with io.BytesIO() as f:
            self._do_download(url, f, start=start, end=end)
            return f.getvalue()

    def download_stream(self, url, file_obj):
        """Downloads the file from the given URL via a GET request to the given
        file-like object.

        Args:
            url: the URL from which to GET the file
            file_obj: the file-like object to which to write the download,
                which must be open for writing

        Raises:
            `requests.exceptions.HTTPError`: if the request resulted in an HTTP
                error
        """
        self._do_download(url, file_obj)

    def is_file(self, url):
        """Determines whether a file exists at the given URL.

        Args:
            url: the URL of the file

        Returns:
            True/False
        """
        res = self._session.head(url)

        if res.status_code < 300:
            return True

        if res.status_code == 404:
            return False

        res.raise_for_status()

    def delete(self, url):
        """Deletes the file at the given URL via a DELETE request.

        Args:
            url: the URL of the file to DELETE
        """
        self._session.delete(url)

    @staticmethod
    def get_filename(url):
        """Gets the filename for the given URL by first trying to extract it
        from the "Content-Disposition" header field, and, if that fails, by
        returning the base name of the path portion of the URL.
        """
        filename = None
        try:
            res = requests.head(url)
            cd = res.headers["Content-Disposition"]
            filename = re.findall("filename=([^;]+)", cd)[0].strip("\"'")
        except (KeyError, IndexError):
            pass

        if not filename:
            filename = os.path.basename(urlparse.urlparse(url).path)

        return filename

    def _do_upload(self, file_obj, url, filename, content_type):
        if not self.set_content_type:
            # Upload without setting content type
            res = self._session.put(url, data=file_obj)
            res.raise_for_status()
            return

        if filename:
            files = {"file": (filename, file_obj, content_type)}
        else:
            files = {"file": file_obj}

        headers = {"Content-Type": content_type}
        res = self._session.put(url, files=files, headers=headers)
        res.raise_for_status()

    def _do_download(self, url, file_obj, start=None, end=None):
        if start is not None or end is not None:
            headers = {"Range": "bytes=%s-%s" % (start or "", end or "")}
        else:
            headers = None

        with self._session.get(url, headers=headers, stream=True) as res:
            for chunk in res.iter_content(chunk_size=self.chunk_size):
                file_obj.write(chunk)
            res.raise_for_status()


class NeedsSSHCredentials(object):
    """Mixin for classes that need an SSH private key to take authenticated
    actions.

    The SSH key used must have _no password_.

    Storage clients that derive from this class should allow users to provide
    their private key in the following ways (in order of precedence):

        (1) manually providing a private key path when contructing an instance
            of the class

        (2) setting the `SSH_PRIVATE_KEY_PATH` environment variable to point to
            a private key file

        (3) loading credentials from `~/.eta/id_rsa` that have been activated
            via `cls.activate_credentials()`
    """

    CREDENTIALS_PATH = os.path.join(etac.ETA_CONFIG_DIR, "id_rsa")

    @classmethod
    def activate_credentials(cls, credentials_path):
        """Activate the credentials by copying them to `~/.eta/id_rsa`.

        Args:
            credentials_path: the path to an SSH private key file
        """
        etau.copy_file(credentials_path, cls.CREDENTIALS_PATH)
        logger.info(
            "SSH credentials successfully activated at '%s'",
            cls.CREDENTIALS_PATH,
        )

    @classmethod
    def deactivate_credentials(cls):
        """Deactivates (deletes) the currently active credentials, if any.

        Active credentials (if any) are at `~/.eta/id_rsa`.
        """
        try:
            os.remove(cls.CREDENTIALS_PATH)
            logger.info(
                "SSH credentials '%s' successfully deactivated",
                cls.CREDENTIALS_PATH,
            )
        except OSError:
            logger.info("No SSH credentials to deactivate")

    @classmethod
    def has_active_credentials(cls):
        """Determines whether there are any active credentials stored at
        `~/.eta/id_rsa`.

        Returns:
            True/False
        """
        return os.path.isfile(cls.CREDENTIALS_PATH)

    @classmethod
    def get_private_key_path(cls, private_key_path=None):
        """Gets the path to the SSH private key.

        Args:
            private_key_path: an optional path to an SSH private key. If no
                value is provided, the active credentials are located via the
                strategy described in the `NeedsSSHCredentials` class docstring

        Returns:
            path to the private key file

        Raises:
            SSHCredentialsError: if no private key file was found
        """
        if private_key_path is not None:
            logger.debug(
                "Using manually provided SSH private key path '%s'",
                private_key_path,
            )
            return private_key_path

        private_key_path = os.environ.get("SSH_PRIVATE_KEY_PATH", None)
        if private_key_path is not None:
            logger.debug(
                "Using SSH private key from environment variable "
                "'SSH_PRIVATE_KEY_PATH=%s'",
                private_key_path,
            )
        elif cls.has_active_credentials():
            private_key_path = cls.CREDENTIALS_PATH
            logger.debug(
                "Using activated SSH private key from '%s'", private_key_path
            )
        else:
            raise SSHCredentialsError("No SSH credentials found")

        return private_key_path


class SSHCredentialsError(Exception):
    """Error raised when a problem with SSH credentials is encountered."""

    def __init__(self, message):
        """Creates an SSHCredentialsError instance.

        Args:
            message: the error message
        """
        super(SSHCredentialsError, self).__init__(
            "%s. Read the documentation for "
            "`eta.core.storage.NeedsSSHCredentials` for more information "
            "about authenticating with SSH keys." % message
        )


class SFTPStorageClient(StorageClient, NeedsSSHCredentials):
    """Client for reading/writing files from remote servers via SFTP.

    Attributes:
        hostname: the host name of the remote server
        username: the username to login to
        private_key_path: the path to the SSH private key used
        port: the remote port to use for the SFTP connection
        keep_open: whether the connection is being held open between API calls

    Examples::

        # Use client to perform a one-off task
        client = SFTPStorageClient(...)
        client.upload(...)

        # Use client to perform a series of tasks without closing and
        # reopening the SFTP connection each time
        client = SFTPStorageClient(..., keep_open=True)
        client.upload(...)
        client.download(...)
        ...
        client.close()  # must manually close the connection

        # Automatic connection management via context manager
        with SFTPStorageClient(...) as client:
            client.upload(...)
            client.download(...)
            ...
    """

    def __init__(
        self,
        hostname,
        username,
        private_key_path=None,
        port=None,
        keep_open=False,
    ):
        """Creates an SFTPStorageClient instance.

        Args:
            hostname: the host name of the remote server
            username: the username to login to
            private_key_path: an optional path to an SSH private key to use. If
                not provided, the active private key is located using the
                strategy described in `NeedsSSHCredentials`
            port: an optional remote port to use for the SFTP connection. The
                default value is 22
            keep_open: whether to keep the connection open between API calls.
                The default value is False
        """
        self.hostname = hostname
        self.username = username
        self.private_key_path = self.get_private_key_path(
            private_key_path=private_key_path
        )
        self.port = port or 22
        self.keep_open = keep_open

        self._connection = _SFTPConnection(
            self.hostname,
            self.username,
            self.private_key_path,
            self.port,
            keep_open=self.keep_open,
        )

    def __enter__(self):
        if not self.keep_open:
            self._connection.set_keep_open(True)
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Closes the SFTP connection. Only needs to be called when
        `keep_open=True` is passed to the constructor.
        """
        self._connection.close()

    def upload(self, local_path, remote_path):
        """Uploads the file to the given remote path.

        Args:
            local_path: the path to the file to upload
            remote_path: the remote path to write the file
        """
        with self._connection as sftp:
            sftp.put(local_path, remotepath=remote_path)

    def upload_bytes(self, bytes_str, remote_path):
        """Uploads the given bytes to the given remote path.

        Args:
            bytes_str: the bytes string to upload
            remote_path: the remote path to write the file
        """
        with io.BytesIO(_to_bytes(bytes_str)) as f:
            self.upload_stream(f, remote_path)

    def upload_stream(self, file_obj, remote_path):
        """Uploads the contents of the given file-like object to the given
        remote path.

        Args:
            file_obj: the file-like object to upload, which must be open for
                reading
            remote_path: the remote path to write the file
        """
        with self._connection as sftp:
            sftp.putfo(file_obj, remotepath=remote_path)

    def download(self, remote_path, local_path):
        """Uploads the file to the given remote path.

        Args:
            remote_path: the remote file to download
            local_path: the path to which to write the downloaded file
        """
        with self._connection as sftp:
            sftp.get(remote_path, localpath=local_path)

    def download_bytes(self, remote_path):
        """Downloads bytes from the given remote path.

        Args:
            remote_path: the remote file to download

        Returns:
            the downloaded bytes string
        """
        with io.BytesIO() as f:
            self.download_stream(remote_path, f)
            return f.getvalue()

    def download_stream(self, remote_path, file_obj):
        """Downloads the file from the given remote path to the given
        file-like object.

        Args:
            remote_path: the remote file to download
            file_obj: the file-like object to which to write the download,
                which must be open for writing
        """
        with self._connection as sftp:
            sftp.getfo(remote_path, file_obj)

    def delete(self, remote_path):
        """Deletes the given file from the remote server.

        Args:
            remote_path: the remote path to delete. Must be a file
        """
        with self._connection as sftp:
            sftp.remove(remote_path)

    def upload_dir(self, local_dir, remote_dir):
        """Uploads the local directory to the given remote directory.

        Args:
            local_dir: the path to the local directory to upload
            remote_dir: the remote directory to write the uploaded directory
        """
        with self._connection as sftp:
            sftp.put_r(local_dir, remote_dir)

    def download_dir(self, remote_dir, local_dir):
        """Downloads the remote directory to the given local directory.

        Args:
            remote_dir: the remote directory to download
            local_dir: the local directory to write the downloaded directory
        """
        with self._connection as sftp:
            sftp.get_r(remote_dir, local_dir)

    def make_dir(self, remote_dir, mode=777):
        """Makes the specified remote directory, recursively if necessary.

        Args:
            remote_dir: the remote directory to create
            mode: int representation of the octal permissions for directory
        """
        with self._connection as sftp:
            sftp.makedirs(remote_dir, mode=mode)

    def delete_dir(self, remote_dir):
        """Deletes the remote directory, which must be empty.

        Args:
            remote_dir: the (empty) remote directory to delete
        """
        with self._connection as sftp:
            sftp.rmdir(remote_dir)


class _SFTPConnection(object):
    """An internal class for managing a pysftp.Connection that can either be
    kept open manually controlled or automatically opened and closed on a
    per-context basis.

    Attributes:
        hostname: the host name of the remote server
        username: the username to login to
        private_key_path: the path to the SSH private key used
        port: the remote port to use for the SFTP connection
        keep_open: whether the connection is being held open between API calls

    Examples::

        # Automatic usage
        conn = _SFTPConnection(..., keep_open=False)
        # no pysftp.Connection is opened yet
        with conn as pyconn:
            # pyconn is a open pysftp.Connection object that is automatically
            #closed when this context is exited
        with conn as pyconn:
            # pyconn is a new pysftp.Connection
        # no need to call conn.close()

        # Manual usage
        conn = _SFTPConnection(..., keep_open=True)
        # an underlying pysftp.Connection is immediately opened
        with conn as pyconn:
            # pyconn is the opened pysftp.Connection
        with conn as pyconn:
            # pyconn is the same pysftp.Connection
        conn.close()
    """

    def __init__(
        self, hostname, username, private_key_path, port, keep_open=False,
    ):
        """Creates an _SFTPConnection instance.

        Args:
            hostname: the host name of the remote server
            username: the username to login to
            private_key_path: the path to an SSH private key to use
            port: the remote port to use for the SFTP connection
            keep_open: whether to keep the connection open between API calls
        """
        self.hostname = hostname
        self.username = username
        self.private_key_path = private_key_path
        self.port = port
        self.keep_open = False

        self._pyconn = None
        self.set_keep_open(keep_open)

    def __enter__(self):
        if not self.keep_open:
            self._open()
        return self._pyconn

    def __exit__(self, *args):
        if not self.keep_open:
            self._close()

    def set_keep_open(self, keep_open):
        """Sets the keep open status of the connection.

        Args:
            keep_open: True/False
        """
        if keep_open != self.keep_open:
            self.keep_open = keep_open
            if self.keep_open:
                self._open()
            else:
                self._close()

    def close(self):
        """Closes the SFTP connection, if necessary. Only needs to be called
        when `keep_open=True` is passed to the constructor.
        """
        if self._pyconn is not None:
            self._close()

    def _open(self):
        self._pyconn = pysftp.Connection(
            self.hostname,
            username=self.username,
            private_key=self.private_key_path,
            port=self.port,
        )

    def _close(self):
        if self._pyconn is not None:
            self._pyconn.close()
        self._pyconn = None


def _read_file_in_chunks(file_obj, chunk_size):
    while True:
        chunk = file_obj.read(chunk_size)
        if not chunk:
            break

        yield chunk


def _to_bytes(val, encoding="utf-8"):
    bytes_str = val.encode(encoding) if isinstance(val, six.text_type) else val
    if not isinstance(bytes_str, six.binary_type):
        raise TypeError("Failed to convert %r to bytes" % bytes_str)

    return bytes_str


def _strip_prefix(path, prefixes):
    if not path:
        return None

    for prefix in prefixes:
        if path.startswith(prefix):
            return path[len(prefix) :]

    return None
