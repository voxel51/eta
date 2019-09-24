'''
Core infrastructure for working with remote storage resources.

See `docs/storage_dev_guide.md` for more information and best practices for
using this module to work with remote storage resources.

This module currently provides clients for the following storage resources:
- Google Cloud buckets via the `google.cloud.storage` API
- Google Drive via the `googleapiclient` Drive API
- Web storage via HTTP requests
- Remote servers via SFTP
- Local disk storage

Copyright 2018-2019, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
Ravali Pinnaka, ravali@voxel51.com
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
import six
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import datetime
import io
import logging
import os
import re
try:
    import urllib.parse as urlparse  # Python 3
except ImportError:
    import urlparse  # Python 2

import google.api_core.exceptions as gae
import google.cloud.storage as gcs
import google.oauth2.service_account as gos
import googleapiclient.discovery as gad
import googleapiclient.http as gah
import pysftp
from retrying import retry
import requests

import eta.core.serial as etas
import eta.core.utils as etau


logger = logging.getLogger(__name__)


# Suppress non-critical logging from third-party libraries
logging.getLogger("googleapiclient").setLevel(logging.ERROR)
logging.getLogger("paramiko").setLevel(logging.ERROR)


def google_cloud_api_retry(func):
    '''Decorator for handling retry of Google API errors.

    Follows recommendations from:
        https://cloud.google.com/apis/design/errors#error_retries
    '''

    def is_500_or_503(exception):
        return (isinstance(exception, gae.InternalServerError)
                or isinstance(exception, gae.ServiceUnavailable))

    def is_429(exception):
        return isinstance(exception, gae.TooManyRequests)

    stop_max_attempt_number = 10
    # wait times below are in milliseconds

    @retry(retry_on_exception=is_500_or_503,
           stop_max_attempt_number=stop_max_attempt_number,
           wait_exponential_multiplier=1000, wait_exponential_max=1 * 1000)
    @retry(retry_on_exception=is_429,
           stop_max_attempt_number=stop_max_attempt_number,
           wait_fixed=30 * 1000)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class StorageClient(object):
    '''Interface for storage clients.'''

    def upload(self, *args, **kwargs):
        raise NotImplementedError("subclass must implement upload()")

    def upload_bytes(self, *args, **kwargs):
        raise NotImplementedError("subclass must implement upload_bytes()")

    def upload_stream(self, *args, **kwargs):
        raise NotImplementedError("subclass must implement upload_stream()")

    def download(self, *args, **kwargs):
        raise NotImplementedError("subclass must implement download()")

    def download_bytes(self, *args, **kwargs):
        raise NotImplementedError("subclass must implement download_bytes()")

    def download_stream(self, *args, **kwargs):
        raise NotImplementedError("subclass must implement download_stream()")

    def delete(self, *args, **kwargs):
        raise NotImplementedError("subclass must implement delete()")


class LocalStorageClient(StorageClient):
    '''Client for reading/writing data from local disk storage.

    Since this class encapsulates local disk storage, the `storage_path`
    arguments refer to local storage paths on disk.

    Attributes:
        chunk_size: the chunk size (in bytes) that will be used for streaming
            uploads and downloads. A negative value implies that the entire
            file is uploaded/downloaded at once
    '''

    #
    # The chunk size (in bytes) to use when streaming. If a negative value
    # is supplied, then the entire file is uploaded/downloaded at once
    #
    DEFAULT_CHUNK_SIZE = -1

    def __init__(self, chunk_size=None):
        '''Creates a LocalStorageClient instance.

        chunk_size: an optional chunk size (in bytes) to use for uploads
            and downloads. By default, `DEFAULT_CHUNK_SIZE` is used
        '''
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE

    def upload(self, local_path, storage_path):
        '''Uploads the file to storage.

        Args:
            local_path: the path to the file to upload
            storage_path: the path to the storage location
        '''
        etau.copy_file(local_path, storage_path)

    def upload_bytes(self, bytes_str, storage_path):
        '''Uploads the given bytes to storage.

        Args:
            bytes_str: the bytes string to upload
            storage_path: the path to the storage location
        '''
        etau.ensure_basedir(storage_path)
        with open(storage_path, "wb") as f:
            f.write(bytes_str)

    def upload_stream(self, file_obj, storage_path):
        '''Uploads the contents of the given file-like object to storage.

        Args:
            file_obj: the file-like object to upload, which must be open for
                reading
            storage_path: the path to the storage location
        '''
        etau.ensure_basedir(storage_path)
        with open(storage_path, "wb") as f:
            for chunk in _read_file_in_chunks(file_obj, self.chunk_size):
                f.write(chunk)

    def download(self, storage_path, local_path):
        '''Downloads the file from storage.

        Args:
            storage_path: the path to the storage location
            local_path: the path to store the downloaded file locally
        '''
        etau.copy_file(storage_path, local_path)

    def download_bytes(self, storage_path):
        '''Downloads bytes from storage.

        Args:
            storage_path: the path to the storage location

        Returns:
            the downloaded bytes string
        '''
        with open(storage_path, "rb") as f:
            return f.read()

    def download_stream(self, storage_path, file_obj):
        '''Downloads the file from storage to the given file-like object.

        Args:
            storage_path: the path to the storage location
            file_obj: the file-like object to which to write the download,
                which must be open for writing
        '''
        with open(storage_path, "rb") as f:
            for chunk in _read_file_in_chunks(f, self.chunk_size):
                file_obj.write(chunk)

    def delete(self, storage_path):
        '''Deletes the file from storage.

        Args:
            storage_path: the path to the storage location
        '''
        etau.delete_file(storage_path)


class NeedsGoogleCredentials(object):
    '''Mixin for classes that need a google.auth.credentials.Credentials
    instance in order to take authenticated actions.

    By convention, storage client classes that derive from this class should
    allow users to set the `GOOGLE_APPLICATION_CREDENTIALS` environment
    variable to point to a valid service account JSON file rather than
    constructing an instance using the `from_json()` method.
    '''

    @classmethod
    def from_json(cls, credentials_json_path):
        '''Creates a cls instance from the given service account JSON file.

        Args:
            credentials_json_path: the path to a service account JSON file

        Returns:
            an instance of cls with the given credentials
        '''
        info = etas.read_json(credentials_json_path)
        credentials = gos.Credentials.from_service_account_info(info)
        return cls(credentials)


class GoogleCloudStorageClient(StorageClient, NeedsGoogleCredentials):
    '''Client for reading/writing data from Google Cloud Storage buckets.

    `cloud_path` should have form "gs://<bucket>/<path/to/object>".
    '''

    #
    # The default chunk size to use when uploading and downloading files.
    # Note that this gives the GCS API the right to use up to this much memory
    # as a buffer during read/write
    #
    DEFAULT_CHUNK_SIZE = 256 * 1024 * 1024  # in bytes

    def __init__(self, credentials=None, chunk_size=None):
        '''Creates a GoogleCloudStorageClient instance.

        Args:
            credentials: a google.auth.credentials.Credentials instance. If no
                credentials are provided, the `GOOGLE_APPLICATION_CREDENTIALS`
                environment variable must be set to point to a valid service
                account JSON file
            chunk_size: an optional chunk size (in bytes) to use for uploads
                and downloads. By default, `DEFAULT_CHUNK_SIZE` is used
        '''
        if credentials:
            self._client = gcs.Client(
                credentials=credentials, project=credentials.project_id)
        else:
            # Uses credentials from GOOGLE_APPLICATION_CREDENTIALS
            self._client = gcs.Client()
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE

    @google_cloud_api_retry
    def upload(self, local_path, cloud_path, content_type=None):
        '''Uploads the file to Google Cloud Storage.

        Args:
            local_path: the path to the file to upload
            cloud_path: the path to the Google Cloud object to create
            content_type: the optional type of the content being uploaded. If
                no value is provided, it is guessed from the filename
        '''
        content_type = content_type or etau.guess_mime_type(local_path)
        blob = self._get_blob(cloud_path)
        blob.upload_from_filename(local_path, content_type=content_type)

    @google_cloud_api_retry
    def upload_bytes(self, bytes_str, cloud_path, content_type=None):
        '''Uploads the given bytes to Google Cloud Storage.

        Args:
            bytes_str: the bytes string to upload
            cloud_path: the path to the Google Cloud object to create
            content_type: the optional type of the content being uploaded. If
                no value is provided but the object already exists in GCS, then
                the same value is used. Otherwise, the default value
                ("application/octet-stream") is used
        '''
        blob = self._get_blob(cloud_path)
        blob.upload_from_string(bytes_str, content_type=content_type)

    @google_cloud_api_retry
    def upload_stream(self, file_obj, cloud_path, content_type=None):
        '''Uploads the contents of the given file-like object to Google Cloud
        Storage.

        Args:
            file_obj: the file-like object to upload, which must be open for
                reading
            cloud_path: the path to the Google Cloud object to create
            content_type: the optional type of the content being uploaded. If
                no value is provided but the object already exists in GCS, then
                the same value is used. Otherwise, the default value
                ("application/octet-stream") is used
        '''
        blob = self._get_blob(cloud_path)
        blob.upload_from_file(file_obj, content_type=content_type)

    def upload_dir(self, local_dir, cloud_dir):
        '''Uploads the contents of the given directory (recursively) to Google
        Cloud Storage.

        The cloud paths are created by appending the relative paths of all
        files inside the local directory to the provided base "directory" in
        Google Cloud Storage.

        Args:
            local_dir: the local directory to upload
            cloud_dir: the base "directory" to use when creating Google Cloud
                object paths
        '''
        files = etau.list_files(local_dir, recursive=True)
        for f in files:
            local_path = os.path.join(local_dir, f)
            cloud_path = os.path.join(cloud_dir, f)
            self.upload(local_path, cloud_path)

    @google_cloud_api_retry
    def download(self, cloud_path, local_path):
        '''Downloads the file from Google Cloud Storage.

        Args:
            cloud_path: the path to the Google Cloud object to download
            local_path: the local disk path to store the downloaded file
        '''
        blob = self._get_blob(cloud_path)
        etau.ensure_basedir(local_path)
        blob.download_to_filename(local_path)

    @google_cloud_api_retry
    def download_bytes(self, cloud_path):
        '''Downloads the file from Google Cloud Storage and returns the bytes
        string.

        Args:
            cloud_path: the path to the Google Cloud object to download
            local_path: the local disk path to store the downloaded file

        Returns:
            the downloaded bytes string
        '''
        blob = self._get_blob(cloud_path)
        return blob.download_as_string()

    @google_cloud_api_retry
    def download_stream(self, cloud_path, file_obj):
        '''Downloads the file from Google Cloud Storage to the given file-like
        object.

        Args:
            cloud_path: the path to the Google Cloud object to download
            file_obj: the file-like object to which to write the download,
                which must be open for writing
        '''
        blob = self._get_blob(cloud_path)
        blob.download_to_file(file_obj)

    def download_dir(self, cloud_dir, local_dir):
        '''Downloads the contents of the "directory" in Google Cloud Storage to
        the given local directory.

        The files are written inside the specified local directory according
        to their relative paths w.r.t. the provided cloud "directory".

        Args:
            cloud_dir: the Google Cloud Storage "directory" to download
            local_dir: the local directory in which to write the files
        '''
        cloud_paths = self.list_files_in_folder(cloud_dir)
        for cloud_path in cloud_paths:
            local_path = os.path.join(
                local_dir, os.path.relpath(cloud_path, cloud_dir))
            self.download(cloud_path, local_path)

    @google_cloud_api_retry
    def delete(self, cloud_path):
        '''Deletes the given file from Google Cloud Storage.

        Args:
            cloud_path: the path to the Google Cloud object to delete
        '''
        blob = self._get_blob(cloud_path)
        blob.delete()

    @google_cloud_api_retry
    def get_file_metadata(self, cloud_path):
        '''Returns metadata about the given file in Google Cloud Storage.

        Args:
            cloud_path: the path to the Google Cloud object

        Returns:
            a dictionary containing metadata about the file, including its
                `name`, `bucket`, `creation_date`, `size`, `mime_type`, and
                `encoding`
        '''
        blob = self._get_blob(cloud_path)
        #
        # WARNING:
        #   If `patch()` isn't called, the blob's properties will not be
        #   populated
        #
        blob.patch()
        return {
            "name": os.path.basename(blob.name),
            "bucket": blob.bucket,
            "creation_date": blob.updated,
            "size": blob.size,
            "mime_type": blob.content_type,
            "encoding": blob.content_encoding
        }

    @google_cloud_api_retry
    def list_files_in_folder(self, cloud_folder, recursive=True):
        '''Returns a list of the files in the given "folder" in Google Cloud
        Storage.

        Args:
            cloud_folder: a string like `gs://<bucket-name>/<folder-path>`
            recursive: whether to recursively traverse sub-"folders". By
                default, this is True

        Returns:
            a list of full cloud paths to the files in the folder
        '''
        bucket_name, folder_name = self._parse_cloud_storage_path(cloud_folder)
        bucket = self._client.get_bucket(bucket_name)

        if folder_name and not folder_name.endswith("/"):
            folder_name += "/"
        delimiter = "/" if not recursive else None
        blobs = bucket.list_blobs(prefix=folder_name, delimiter=delimiter)

        paths = []
        prefix = "gs://" + bucket_name
        for blob in blobs:
            if not blob.name.endswith("/"):
                paths.append(os.path.join(prefix, blob.name))
        return paths

    @google_cloud_api_retry
    def generate_signed_url(self, cloud_path, method="GET", hours=24):
        '''Generates a signed URL for accessing the given storage object.

        Anyone with the URL can access the object with the permission until it
        expires.

        Note that the Google Cloud documentation strongly recommends using PUT
        rather than POST to upload objects.

        Args:
            cloud_path: the path to the Google Cloud object
            hours: the number of hours that the URL is valid
            method: the HTTP verb (GET, PUT, DELETE) to authorize

        Returns:
            a URL for accessing the object via HTTP request
        '''
        blob = self._get_blob(cloud_path)
        expiration = datetime.timedelta(hours=hours)
        return blob.generate_signed_url(expiration=expiration, method=method)

    def _get_blob(self, cloud_path):
        bucket_name, object_name = self._parse_cloud_storage_path(cloud_path)
        bucket = self._client.get_bucket(bucket_name)
        return bucket.blob(object_name, chunk_size=self.chunk_size)

    @staticmethod
    def _parse_cloud_storage_path(cloud_path):
        '''Parses a cloud storage path string.

        Args:
            cloud_path: a string of the form gs://<bucket_name>/<object_name>

        Returns:
            bucket_name: the name of the Google Cloud Storage bucket
            object_name: the name of the object

        Raises:
            GoogleCloudStorageClientError: if the cloud storage path string was
                invalid
        '''
        if not cloud_path.startswith("gs://"):
            raise GoogleCloudStorageClientError(
                "Cloud storage path '%s' must start with gs://" % cloud_path)
        chunks = cloud_path[5:].split("/", 1)
        if len(chunks) != 2:
            return chunks[0], ""
        return chunks[0], chunks[1]


class GoogleCloudStorageClientError(Exception):
    pass


class GoogleDriveStorageClient(StorageClient, NeedsGoogleCredentials):
    '''Client for reading/writing data from Google Drive.

    The service account credentials you use must have access permissions for
    any Drive folders you intend to access.

    Attributes:
        chunk_size: the chunk size (in bytes) that will be used for streaming
            uploads and downloads
    '''

    #
    # The default chunk size to use when uploading and downloading files.
    # Note that this gives the Drive API the right to use up to this
    # much memory as a buffer during read/write
    #
    DEFAULT_CHUNK_SIZE = 256 * 1024 * 1024  # in bytes

    def __init__(self, credentials=None, chunk_size=None):
        '''Creates a GoogleDriveStorageClient instance.

        Args:
            credentials: a google.auth.credentials.Credentials instance. If no
                credentials are provided, the `GOOGLE_APPLICATION_CREDENTIALS`
                environment variable must be set to point to a valid service
                account JSON file
            chunk_size: an optional chunk size (in bytes) to use for uploads
                and downloads. By default, `DEFAULT_CHUNK_SIZE` is used
        '''
        if credentials:
            self._service = gad.build(
                "drive", "v3", credentials=credentials, cache_discovery=False)
        else:
            # Uses credentials from GOOGLE_APPLICATION_CREDENTIALS
            self._service = gad.build("drive", "v3", cache_discovery=False)

        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE

    def upload(self, local_path, folder_id, filename=None, content_type=None):
        '''Uploads the file to Google Drive.

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
        '''
        name = filename or os.path.basename(local_path)
        with open(local_path, "rb") as f:
            return self._do_upload(f, folder_id, name, content_type)

    def upload_bytes(self, bytes_str, folder_id, filename, content_type=None):
        '''Uploads the given bytes to Google Drive.

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
        '''
        with io.BytesIO(_to_bytes(bytes_str)) as f:
            return self._do_upload(f, folder_id, filename, content_type)

    def upload_stream(self, file_obj, folder_id, filename, content_type=None):
        '''Uploads the contents of the given file-like object to Google Drive.

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
        '''
        return self._do_upload(file_obj, folder_id, filename, content_type)

    def download(self, file_id, local_path):
        '''Downloads the file from Google Drive.

        Args:
            file_id: the ID of the file to download
            local_path: the path to the storage location
        '''
        etau.ensure_basedir(local_path)
        with open(local_path, "wb") as f:
            self._do_download(file_id, f)

    def download_bytes(self, file_id):
        '''Downloads a file from Google Drive and returns the bytes.

        Args:
            file_id: the ID of the file to download

        Returns:
            the downloaded bytes string
        '''
        with io.BytesIO() as f:
            self._do_download(file_id, f)
            return f.getvalue()

    def download_stream(self, file_id, file_obj):
        '''Downloads the file from Google Drive to the given file-like object.

        Args:
            file_id: the ID of the file to download
            file_obj: the file-like object to which to write the download,
                which must be open for writing
        '''
        self._do_download(file_id, file_obj)

    def delete(self, file_or_folder_id):
        '''Deletes the file or folder from Google Drive.

        Args:
            file_or_folder_id: the ID of the file or folder to delete
        '''
        self._service.files().delete(
            fileId=file_or_folder_id, supportsTeamDrives=True).execute()

    def delete_all_files_in_folder(self, folder_id, skip_failures=False):
        '''Deletes all the files in the Google Drive folder and retains the
        original folder.

        Args:
            folder_id: the ID of the Google Drive folder from which the files
                are to be deleted
            skip_failures: whether to gracefully skip delete errors. By
                default, this is False
        '''
        files = self.list_files_in_folder(folder_id)
        for f in files:
            filename = f["name"]
            try:
                with etau.Timer() as t:
                    self.delete(f["id"])
                    logger.info(
                        "File '%s' deleted from '%s' (%s)", filename,
                        folder_id, t.elapsed_time_str)
            except Exception as e:
                if not skip_failures:
                    raise GoogleDriveStorageClientError(e)
                logger.info(
                    "Failed to delete file '%s' from '%s' (%s)", filename,
                    folder_id, t.elapsed_time_str)

        if not self.count_files_in_folder(folder_id):
            logger.info("All files deleted in %s", folder_id)

    def count_files_in_folder(self, folder_id):
        '''Returns count of number of files in the Google Drive folder

        Args:
            folder_id: the ID of the Google Drive folder to be processed

        Returns:
            the count of files in folder
        '''
        return len(self.list_files_in_folder(folder_id))

    def get_team_drive_id(self, name):
        '''Get the ID of the Team Drive with the given name.

        Args:
            name: the name of the Team Drive

        Returns:
            the ID of the root folder in the Team Drive

        Raises:
            GoogleDriveStorageClientError: if the Team Drive was not found
        '''
        response = self._service.teamdrives().list(
            fields="teamDrives(id, name)").execute()
        for team_drive in response["teamDrives"]:
            if name == team_drive["name"]:
                return team_drive["id"]

        raise GoogleDriveStorageClientError("Team Drive '%s' not found" % name)

    def get_file_metadata(self, file_id):
        '''Gets metadata about the file with the given ID.

        Args:
            file_id: the ID of a file (or folder)

        Returns:
            a dictionary containing the available metadata about the file,
                including at least `name`, `kind`, and `mimeType`
        '''
        return self._service.files().get(
            fileId=file_id, supportsTeamDrives=True).execute()

    def get_root_team_drive_id(self, file_id):
        '''Returns the ID of the root Team Drive in which this file lives.

        Args:
            file_id: the ID of the file (or folder)

        Returns:
            the ID of the Team Drive, or None if the file does not live in a
                Team Drive
        '''
        return self.get_file_metadata(file_id).get("teamDriveId", None)

    def is_folder(self, file_id):
        '''Determines whether the file with the given ID is a folder.

        Args:
            file_id: the ID of a file (or folder)

        Returns:
            True/False whether the file is a folder
        '''
        mime_type = self.get_file_metadata(file_id).get("mimeType", None)
        return mime_type == "application/vnd.google-apps.folder"

    def create_folder_if_necessary(self, folder_name, parent_folder_id):
        '''Creates the given folder within the given parent folder, if
        necessary.

        Args:
            folder_name: a `folder` or `a/nested/folder` specifying the folder
                to create, if necessary
            parent_folder_id: the ID of the parent folder in which to operate

        Returns:
            the ID of the last created folder
        '''
        folder_id = parent_folder_id
        for folder in folder_name.split("/"):
            folder_id = self._create_folder_if_necessary(folder, folder_id)
        return folder_id

    def create_folder(self, folder_name, parent_folder_id):
        '''Creates the given folder within the given parent folder. The folder
        is assumed not to exist.

        Args:
            folder_name: a `folder` or `a/nested/folder` specifying the folder
                to create
            parent_folder_id: the ID of the parent folder in which to operate

        Returns:
            the ID of the last created folder
        '''
        folder_id = parent_folder_id
        for folder in folder_name.split("/"):
            folder_id = self._create_folder(folder, folder_id)
        return folder_id

    def list_files_in_folder(
            self, folder_id, include_folders=False, recursive=False):
        '''Returns a list of the files in the folder with the given ID.

        Args:
            folder_id: the ID of a folder
            include_folders: whether to include "folders" in the list of
                returned files. By default, this is False
            recursive: whether to recursively traverse sub-"folders". By
                default, this is False

        Returns:
            A list of dicts containing the `id`, `name`, and `mimeType` of the
                files/sub-folders in the folder
        '''
        team_drive_id = self.get_root_team_drive_id(folder_id)
        if team_drive_id:
            # Parameters required to list Team Drives
            params = {
                "corpora": "teamDrive",
                "supportsTeamDrives": True,
                "includeTeamDriveItems": True,
                "teamDriveId": team_drive_id,
            }
        else:
            params = {}

        # Build file list
        folders = []
        files = []
        page_token = None
        query = "'%s' in parents and trashed=false" % folder_id
        while True:
            # Get the next page of files
            response = self._service.files().list(
                q=query,
                fields="files(id, name, mimeType),nextPageToken",
                pageSize=256,
                pageToken=page_token,
                **params
            ).execute()
            page_token = response.get("nextPageToken", None)

            for f in response["files"]:
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    folders.append(f)
                else:
                    files.append(f)

            # Check for end of list
            if not page_token:
                break

        if recursive:
            # Recursively traverse subfolders
            for folder in folders:
                if include_folders:
                    # Include folder in list just before its contents
                    files.append(folder)

                contents = self.list_files_in_folder(
                    folder["id"], include_folders=include_folders,
                    recursive=True)
                for f in contents:
                    # Embed <folder-name>/<file-name> namespace in filename
                    f["name"] = os.path.join(folder["name"], f["name"])
                    files.append(f)

        elif include_folders:
            files.extend(folders)

        return files

    def upload_files_in_folder(
            self, local_dir, folder_id, skip_failures=False,
            skip_existing_files=False, recursive=False):
        '''Uploads the files in the given folder to Google Drive.

        Note that this function uses `eta.core.utils.list_files` to determine
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
        '''
        # @todo retry failures? exponential backoff? rate limit requests?
        files = etau.list_files(local_dir, recursive=recursive)

        # Skip existing files, if requested
        if skip_existing_files:
            existing_files = set(
                f["name"] for f in self.list_files_in_folder(
                    folder_id, recursive=recursive))
            _files = [f for f in files if f not in existing_files]
            num_skipped = len(files) - len(_files)
            if num_skipped > 0:
                logger.info("Skipping %d existing files", num_skipped)
                files = _files

        num_files = len(files)
        logger.info("Uploading %d files to '%s'", num_files, folder_id)
        file_ids = {}
        for idx, filename in enumerate(files, 1):
            try:
                local_path = os.path.join(local_dir, filename)
                with etau.Timer() as t:
                    file_id = self.upload(local_path, folder_id)
                    file_ids[filename] = file_id
                logger.info(
                    "File '%s' uploaded to '%s' (%s) (%d/%d)", local_path,
                    folder_id, t.elapsed_time_str, idx, num_files)
            except Exception as e:
                if not skip_failures:
                    raise GoogleDriveStorageClientError(e)
                logger.info(
                    "Failed to upload file '%s' to '%s'; skipping", local_path,
                    folder_id)
        return file_ids

    def download_files_in_folder(
            self, folder_id, local_dir, skip_failures=False,
            skip_existing_files=False, recursive=True):
        '''Downloads the files in the Google Drive folder to the given local
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
        '''
        # @todo retry failures? exponential backoff? rate limit requests?
        files = self.list_files_in_folder(folder_id, include_folders=True)

        # Skip existing files, if requested
        if skip_existing_files:
            etau.ensure_dir(local_dir)
            existing_files = set(etau.list_files(local_dir))
            _files = [f for f in files if f["name"] not in existing_files]
            num_skipped = len(files) - len(_files)
            if num_skipped > 0:
                logger.info("Skipping %d existing files", num_skipped)
                files = _files

        num_files = len(files)
        logger.info("Downloading %d files to '%s'", num_files, local_dir)
        filenames = []
        for idx, f in enumerate(files, 1):
            filename = f["name"]
            file_id = f["id"]
            try:
                if (recursive and
                        f["mimeType"] == "application/vnd.google-apps.folder"):
                    self.download_files_in_folder(
                        file_id,
                        os.path.join(local_dir, filename),
                        skip_failures=skip_failures,
                        skip_existing_files=skip_existing_files,
                        recursive=True)
                else:
                    local_path = os.path.join(local_dir, filename)
                    with etau.Timer() as t:
                        self.download(file_id, local_path)
                        filenames.append(filename)
                    logger.info(
                        "File '%s' downloaded to '%s' (%s) (%d/%d)", filename,
                        local_path, t.elapsed_time_str, idx, num_files)
            except Exception as e:
                if not skip_failures:
                    raise GoogleDriveStorageClientError(e)
                logger.info(
                    "Failed to download file '%s' to '%s'; skipping", file_id,
                    local_path)
        return filenames

    def delete_duplicate_files_in_folder(
            self, folder_id, skip_failures=False, recursive=False):
        '''Deletes any duplicate files (files with the same filename) in the
        given Google Drive folder.

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
        '''
        files = self.list_files_in_folder(folder_id, recursive=recursive)
        existing_files = set()
        num_deleted = 0
        for f in files:
            filename = f["name"]
            if filename not in existing_files:
                existing_files.add(filename)
                continue

            # Delete duplicate file
            try:
                with etau.Timer() as t:
                    self.delete(f["id"])
                    num_deleted += 1
                logger.info(
                    "File '%s' deleted from '%s' (%s)", filename, folder_id,
                    t.elapsed_time_str)
            except Exception as e:
                if not skip_failures:
                    raise GoogleDriveStorageClientError(e)
                logger.info(
                    "Failed to delete file '%s' in '%s'; skipping",
                    filename, folder_id)

        return num_deleted

    def _do_upload(self, file_obj, folder_id, filename, content_type):
        # Handle any leading directories
        chunks = filename.rsplit("/", 1)
        if len(chunks) == 2:
            folder_id = self.create_folder_if_necessary(chunks[0], folder_id)
            filename = chunks[1]

        mime_type = content_type or etau.guess_mime_type(filename)
        media = gah.MediaIoBaseUpload(
            file_obj, mime_type, chunksize=self.chunk_size, resumable=True)
        body = {
            "name": filename,
            "mimeType": mime_type,
            "parents": [folder_id],
        }
        stored_file = self._service.files().create(
            body=body, media_body=media,
            supportsTeamDrives=True, fields="id").execute()
        return stored_file["id"]

    def _do_download(self, file_id, file_obj):
        request = self._service.files().get_media(
            fileId=file_id, supportsTeamDrives=True)
        downloader = gah.MediaIoBaseDownload(
            file_obj, request, chunksize=self.chunk_size)

        done = False
        while not done:
            status, done = downloader.next_chunk()

    def _create_folder_if_necessary(self, new_folder, parent_folder_id):
        folder_id = None
        for f in self.list_files_in_folder(parent_folder_id):
            if f["name"] == new_folder:
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
        folder = self._service.files().create(
            body=body, supportsTeamDrives=True, fields="id").execute()
        return folder["id"]


class GoogleDriveStorageClientError(Exception):
    pass


class HTTPStorageClient(StorageClient):
    '''Client for reading/writing files via HTTP requests.

    Attributes:
        set_content_type: whether to set the `Content-Type` in the request
            header of uploads. The Google Cloud documentation requires that
            the `Content-Type` be *OMITTED* from PUT requests to Google Cloud
            Storage, so set this attribute to False for use with GCS
        chunk_size: the chunk size (in bytes) that will be used for streaming
            downloads
        keep_alive: whether the request session should be kept alive between
            requests

    Examples:
        ```
        # Use client to perform a one-off task
        client = HTTPStorageClient(...)
        client.upload(...)

        # Use client to perform a series of tasks without closing and
        # reopening the request session each time
        client = HTTPStorageClient(..., keep_alive=True)
        client.upload(...)
        client.download(...)
        ...
        client.close()  # make sure the session is closed

        # Automatic connection management via context manager
        with HTTPStorageClient(..., keep_alive=True) as client:
            client.upload(...)
            client.download(...)
            ...
        ```
    '''

    #
    # The default chunk size to use when downloading files.
    # Note that this gives the requests toolbelt the right to use up to this
    # much memory as a buffer during read/write
    #
    DEFAULT_CHUNK_SIZE = 32 * 1024 * 1024  # in bytes

    def __init__(
            self, set_content_type=False, chunk_size=None, keep_alive=False):
        '''Creates an HTTPStorageClient instance.

        Args:
            set_content_type: whether to specify the `Content-Type` during
                upload requests. By default, this is False
            chunk_size: an optional chunk size (in bytes) to use for downloads.
                By default, `DEFAULT_CHUNK_SIZE` is used
            keep_alive: whether to keep the request session alive between
                requests. By default, this is False
        '''
        self.set_content_type = set_content_type
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self.keep_alive = keep_alive
        self._requests = requests.Session() if keep_alive else requests

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        '''Closes the HTTP session. Only needs to be called when
        `keep_alive=True` is passed to the constructor.
        '''
        if self.keep_alive:
            self._requests.close()

    def upload(self, local_path, url, filename=None, content_type=None):
        '''Uploads the file to the given URL via a PUT request.

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
        '''
        filename = filename or os.path.basename(local_path)
        content_type = content_type or etau.guess_mime_type(filename)
        with open(local_path, "rb") as f:
            self._do_upload(f, url, filename, content_type)

    def upload_bytes(self, bytes_str, url, filename=None, content_type=None):
        '''Uploads the given bytes to the given URL via a PUT request.

        Args:
            bytes_str: the bytes string to upload
            url: the URL to which to PUT the file
            filename: an optional filename to include in the request
            content_type: an optional content type to include in the request.
                Note that this is only added when `set_content_type` is True

        Raises:
            `requests.exceptions.HTTPError`: if the request resulted in an HTTP
                error
        '''
        with io.BytesIO(_to_bytes(bytes_str)) as f:
            self._do_upload(f, url, filename, content_type)

    def upload_stream(self, file_obj, url, filename=None, content_type=None):
        '''Uploads the contents of the given file-like object to the given
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
        '''
        self._do_upload(file_obj, url, filename, content_type)

    def download(self, url, local_path):
        '''Downloads the file from the given URL via a GET request.

        Args:
            url: the URL from which to GET the file
            local_path: the path to which to write the downloaded file

        Raises:
            `requests.exceptions.HTTPError`: if the request resulted in an HTTP
                error
        '''
        etau.ensure_basedir(local_path)
        with open(local_path, "wb") as f:
            self._do_download(url, f)

    def download_bytes(self, url):
        '''Downloads bytes from the given URL via a GET request.

        Args:
            url: the URL from which to GET the file

        Returns:
            the downloaded bytes string

        Raises:
            `requests.exceptions.HTTPError`: if the request resulted in an HTTP
                error
        '''
        with io.BytesIO() as f:
            self._do_download(url, f)
            return f.getvalue()

    def download_stream(self, url, file_obj):
        '''Downloads the file from the given URL via a GET request to the given
        file-like object.

        Args:
            url: the URL from which to GET the file
            file_obj: the file-like object to which to write the download,
                which must be open for writing

        Raises:
            `requests.exceptions.HTTPError`: if the request resulted in an HTTP
                error
        '''
        self._do_download(url, file_obj)

    def delete(self, url):
        '''Deletes the file at the given URL via a DELETE request.

        Args:
            url: the URL of the file to DELETE
        '''
        self._requests.delete(url)

    @staticmethod
    def get_filename(url):
        '''Gets the filename for the given URL by first trying to extract it
        from the "Content-Disposition" header field, and, if that fails, by
        returning the base name of the path portion of the URL.
        '''
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
            res = self._requests.put(url, data=file_obj)
            res.raise_for_status()
            return

        if filename:
            files = {"file": (filename, file_obj, content_type)}
        else:
            files = {"file": file_obj}
        headers = {"Content-Type": content_type}
        res = self._requests.put(url, files=files, headers=headers)
        res.raise_for_status()

    def _do_download(self, url, file_obj):
        with self._requests.get(url, stream=True) as res:
            for chunk in res.iter_content(chunk_size=self.chunk_size):
                file_obj.write(chunk)
            res.raise_for_status()


class NeedsSSHCredentials(object):
    '''Mixin for classes that need an SSH private key to take authenticated
    actions.

    The SSH key used must have _no password_.

    This class provides the ability to locate an SSH key either from a
    `private_key_path` argument or by reading the `SSH_PRIVATE_KEY_PATH`
    environment variable that points to a valid SSH key file.
    '''

    SSH_PRIVATE_KEY_PATH_ENVIRON_VAR = "SSH_PRIVATE_KEY_PATH"

    @staticmethod
    def parse_private_key_path(private_key_path=None):
        '''Parses the private key path.

        Args:
            private_key_path: an optional path to an SSH private key. If no
                value is provided, the `SSH_PRIVATE_KEY_PATH` environment
                variable must be set to point to the key

        Returns:
            the path to the private key file

        Raises:
            SSHKeyError: if no SSH key file was found
        '''
        pkp = (
            private_key_path or os.environ.get(
                NeedsSSHCredentials.SSH_PRIVATE_KEY_PATH_ENVIRON_VAR)
        )
        if not pkp or not os.path.isfile(pkp):
            raise SSHKeyError("No SSH key found")

        return pkp


class SSHKeyError(Exception):
    pass


class SFTPStorageClient(StorageClient, NeedsSSHCredentials):
    '''Client for reading/writing files from remote servers via SFTP.

    Attributes:
        hostname: the host name of the remote server
        username: the username to login to
        private_key_path: the path to the SSH private key used
        port: the remote port to use for the SFTP connection
        keep_open: whether the connection is being held open between API calls

    Examples:
        ```
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
        ```
    '''

    def __init__(
            self, hostname, username, private_key_path=None, port=22,
            keep_open=False,
        ):
        '''Creates an SFTPStorageClient instance.

        Args:
            hostname: the host name of the remote server
            username: the username to login to
            private_key_path: the path to an SSH private key to use. If not
                provided, the `SSH_PRIVATE_KEY_PATH` environment variable must
                be set to point to a private key file
            port: the remote port to use for the SFTP connection. The default
                value is 22
            keep_open: whether to keep the connection open between API calls.
                The default value is False
        '''
        self.hostname = hostname
        self.username = username
        self.private_key_path = self.parse_private_key_path(private_key_path)
        self.port = port
        self.keep_open = keep_open

        self._connection = _SFTPConnection(
            self.hostname, self.username, self.private_key_path, self.port,
            keep_open=self.keep_open,
        )

    def __enter__(self):
        if not self.keep_open:
            self._connection.set_keep_open(True)
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        '''Closes the SFTP connection. Only needs to be called when
        `keep_open=True` is passed to the constructor.
        '''
        self._connection.close()

    def upload(self, local_path, remote_path):
        '''Uploads the file to the given remote path.

        Args:
            local_path: the path to the file to upload
            remote_path: the remote path to write the file
        '''
        with self._connection as sftp:
            sftp.put(local_path, remotepath=remote_path)

    def upload_bytes(self, bytes_str, remote_path):
        '''Uploads the given bytes to the given remote path.

        Args:
            bytes_str: the bytes string to upload
            remote_path: the remote path to write the file
        '''
        with io.BytesIO(_to_bytes(bytes_str)) as f:
            self.upload_stream(f, remote_path)

    def upload_stream(self, file_obj, remote_path):
        '''Uploads the contents of the given file-like object to the given
        remote path.

        Args:
            file_obj: the file-like object to upload, which must be open for
                reading
            remote_path: the remote path to write the file
        '''
        with self._connection as sftp:
            sftp.putfo(file_obj, remotepath=remote_path)

    def download(self, remote_path, local_path):
        '''Uploads the file to the given remote path.

        Args:
            remote_path: the remote file to download
            local_path: the path to which to write the downloaded file
        '''
        with self._connection as sftp:
            sftp.get(remote_path, localpath=local_path)

    def download_bytes(self, remote_path):
        '''Downloads bytes from the given remote path.

        Args:
            remote_path: the remote file to download

        Returns:
            the downloaded bytes string
        '''
        with io.BytesIO() as f:
            self.download_stream(remote_path, f)
            return f.getvalue()

    def download_stream(self, remote_path, file_obj):
        '''Downloads the file from the given remote path to the given
        file-like object.

        Args:
            remote_path: the remote file to download
            file_obj: the file-like object to which to write the download,
                which must be open for writing
        '''
        with self._connection as sftp:
            sftp.getfo(remote_path, file_obj)

    def delete(self, remote_path):
        '''Deletes the given file from the remote server.

        Args:
            remote_path: the remote path to delete. Must be a file
        '''
        with self._connection as sftp:
            sftp.remove(remote_path)

    def upload_dir(self, local_dir, remote_dir):
        '''Uploads the local directory to the given remote directory.

        Args:
            local_dir: the path to the local directory to upload
            remote_dir: the remote directory to write the uploaded directory
        '''
        with self._connection as sftp:
            sftp.put_r(local_dir, remote_dir)

    def download_dir(self, remote_dir, local_dir):
        '''Downloads the remote directory to the given local directory.

        Args:
            remote_dir: the remote directory to download
            local_dir: the local directory to write the downloaded directory
        '''
        with self._connection as sftp:
            sftp.get_r(remote_dir, local_dir)

    def make_dir(self, remote_dir, mode=777):
        '''Makes the specified remote directory, recursively if necessary.

        Args:
            remote_dir: the remote directory to create
            mode: int representation of the octal permissions for directory
        '''
        with self._connection as sftp:
            sftp.makedirs(remote_dir, mode=mode)

    def delete_dir(self, remote_dir):
        '''Deletes the remote directory, which must be empty.

        Args:
            remote_dir: the (empty) remote directory to delete
        '''
        with self._connection as sftp:
            sftp.rmdir(remote_dir)


class _SFTPConnection(object):
    '''An internal class for managing a pysftp.Connection that can either be
    kept open manually controlled or automatically opened and closed on a
    per-context basis.

    Attributes:
        hostname: the host name of the remote server
        username: the username to login to
        private_key_path: the path to the SSH private key used
        port: the remote port to use for the SFTP connection
        keep_open: whether the connection is being held open between API calls

    Examples:
        ```
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
        ```
    '''

    def __init__(
            self, hostname, username, private_key_path, port,
            keep_open=False,
        ):
        '''Creates an _SFTPConnection instance.

        Args:
            hostname: the host name of the remote server
            username: the username to login to
            private_key_path: the path to an SSH private key to use
            port: the remote port to use for the SFTP connection
            keep_open: whether to keep the connection open between API calls
        '''
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
        '''Sets the keep open status of the connection.

        Args:
            keep_open: True/False
        '''
        if keep_open != self.keep_open:
            self.keep_open = keep_open
            if self.keep_open:
                self._open()
            else:
                self._close()

    def close(self):
        '''Closes the SFTP connection, if necessary. Only needs to be called
        when `keep_open=True` is passed to the constructor.
        '''
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
    bytes_str = (
        val.encode(encoding) if isinstance(val, six.text_type) else val
    )
    if isinstance(bytes_str, six.binary_type):
        return bytes_str
    else:
        raise TypeError("Failed to convert %r to bytes" % bytes_str)
