# Storage Developer's Guide

This document describes ETA's infrastructure for accessing resources in remote
storage, which is contained in the [eta.core.storage](
https://github.com/voxel51/eta/blob/develop/eta/core/storage.py) module.


## Supported Clients

ETA supports the following storage clients:

| Storage Type | ETA class | Underlying Python package |
| ------------ | --------- | ------------------------- |
| Amazon S3 | `eta.core.storage.S3StorageClient` | `boto3` |
| Google Cloud Storage | `eta.core.storage.GoogleCloudStorageClient` | `google.cloud.storage` |
| Google Drive | `eta.core.storage.GoogleDriveStorageClient` | `googleapiclient` |
| HTTP | `eta.core.storage.HTTPStorageClient` | `requests` |
| SFTP | `eta.core.storage.SFTPStorageClient` | `pysftp` |


## Amazon S3 Client

The `eta.core.storage.S3StorageClient` class provides programmatic access to
Amazon S3 buckets.

### Authentication

All instances of this client must be provided with AWS credentials with the
appropriate permissions to perform the file manipulations that you request.
This can be done in the following ways (in order of precedence):

- using the `eta.core.storage.S3StorageClient.from_ini()` method to manually
specify the credentials `.ini` file to use

- setting the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_SESSION_TOKEN` (if applicable), and `AWS_DEFAULT_REGION` environment
variables directly

- setting the `AWS_SHARED_CREDENTIALS_FILE` environment variable to point to
a valid credentials `.ini` file

- setting the `AWS_CONFIG_FILE` environment variable to point to a valid
credentials `.ini` file

- automatically loading credentials from `~/.eta/aws-credentials.ini` that have
been activated via `eta.core.storage.S3StorageClient.activate_credentials()`

In the above, the `.ini` file should have syntax similar to the following:

```
[default]
aws_access_key_id = WWW
aws_secret_access_key = XXX
aws_session_token = YYY
region = ZZZ
```

See the following link for more information:
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#configuration

### Example usage

```py
import eta.core.storage as etas

credentials_path = "/path/to/credentials.ini"
local_path1 = "/path/to/file.txt"
local_path2 = "/path/to/file2.txt"
cloud_path = "s3://bucket-name/file.txt"

client = etas.S3StorageClient.from_ini(credentials_path)
client.upload(local_path1, cloud_path)
client.download(cloud_path, local_path2)
client.delete(cloud_path)
```

### References

- https://docs.aws.amazon.com/AmazonS3/latest/dev/Introduction.html
- https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html


## Google Cloud Storage Client

The `eta.core.storage.GoogleCloudStorageClient` class provides programmatic
access to Google Cloud Storage buckets.

### Authentication

All instances of this client must be provided with Google service account
credentials with the appropriate permissions to perform the file manipulations
that you request. This can be done in the following ways (in order of
precedence):

- using the `eta.core.storage.GoogleCloudStorageClient.from_json()` method to
manually specify the service account JSON file to use

- setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to
a valid service account JSON file

- automatically loading credentials from `~/.eta/google-credentials.json` that
have been activated via
`eta.core.storage.GoogleCloudStorageClient.activate_credentials()`

In the above, the service account JSON file should have syntax similar to the
following:

```json
{
  "type": "service_account",
  "project_id": "<project-id>",
  "private_key_id": "WWWWW",
  "private_key": "-----BEGIN PRIVATE KEY-----\nXXXXX\n-----END PRIVATE KEY-----\n",
  "client_email": "<account-name>@<project-id>.iam.gserviceaccount.com",
  "client_id": "YYYYY",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/ZZZZZ"
}
```

### Example usage

```py
import eta.core.storage as etas

credentials_path = "/path/to/service-account.json"
local_path1 = "/path/to/file.txt"
local_path2 = "/path/to/file2.txt"
cloud_path = "gs://bucket-name/file.txt"

client = etas.GoogleCloudStorageClient.from_json(credentials_path)
client.upload(local_path1, cloud_path)
client.download(cloud_path, local_path2)
client.delete(cloud_path)
```

### References

- https://cloud.google.com/storage/docs/reference/libraries
- https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python
- https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/storage/cloud-client/snippets.py


## Google Drive Client

The `eta.core.storage.GoogleDriveStorageClient` class provides programmatic
access to Google Drive.

### Authentication

All instances of this client must be provided with Google service account
credentials with the appropriate permissions to perform the file manipulations
that you request. This can be done in the following ways (in order of
precedence):

- using the `eta.core.storage.GoogleDriveStorageClient.from_json()` method to
manually specify the service account JSON file to use

- setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to
a valid service account JSON file

- automatically loading credentials from `~/.eta/google-credentials.json` that
have been activated via
`eta.core.storage.GoogleDriveStorageClient.activate_credentials()`

In the above, the service account JSON file should have syntax similar to the
following:

```json
{
  "type": "service_account",
  "project_id": "<project-id>",
  "private_key_id": "WWWWW",
  "private_key": "-----BEGIN PRIVATE KEY-----\nXXXXX\n-----END PRIVATE KEY-----\n",
  "client_email": "<account-name>@<project-id>.iam.gserviceaccount.com",
  "client_id": "YYYYY",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/ZZZZZ"
}
```

### Enable the Google Drive API

In order to access Google Drive files via the API, you must enable API use
at the following link:
https://console.developers.google.com/apis/api/drive.googleapis.com

> Note: for organizations, this activation only needs to be performed once

### Giving a service account access to a Team Drive

- Navigate to the folder at drive.google.com
- Click Share...
- Add service account's `client_email` address to the membership list

### Example usage

```py
import eta.core.storage as etas

team_drive_name = "public"
credentials_path = "/path/to/service-account.json"
local_path1 = "/path/to/file.txt"
local_path2 = "/path/to/file2.txt"

client = etas.GoogleDriveStorageClient.from_json(credentials_path)
team_drive_id = client.get_team_drive_id(team_drive_name)
file_id = client.upload(local_path1, team_drive_id)
client.download(file_id, local_path2)
client.delete(file_id)
```

### Access via authenticated HTTP requests

One can also read/write/delete files in Google Drive using standard HTTP
requests. The basic format of the requests is:

```
# Route
GET https://www.googleapis.com/drive/v3/files/${FILE_ID}?alt=media

# Header
Authorization: Bearer ${ACCESS_TOKEN}
```

where `ACCESS_TOKEN` is an access token that can be generated for an authorized
service account using the `gcloud` utility as follows:

```shell
# Activate the service account
gcloud auth activate-service-account --key-file "/path/to/service-account.json"

# Get the access token
ACCESS_TOKEN="$(gcloud auth print-access-token)"
```

### References

- https://cloud.google.com/vision/docs/auth
- https://developers.google.com/api-client-library/python/apis/drive/v3


## HTTP Storage Client

The `eta.core.storage.HTTPStorageClient` class provides a convenient API for
issuing HTTP requests to upload and download files of arbitrary sizes from
URLs.

### Example usage

The following snippet demonstrates an example of
`eta.core.storage.GoogleCloudStorageClient` to generate signed URLs for
accessing objects in GCS and then using `eta.core.storage.HTTPStorageClient` to
access the resources via HTTP requests.

```py
import eta.core.storage as etas

credentials_path = "/path/to/service-account.json"
local_path1 = "/path/to/file.txt"
local_path2 = "/path/to/file2.txt"
cloud_path = "gs://bucket-name/file.txt"

gsclient = etas.GoogleCloudStorageClient.from_json(credentials_path)
client = etas.HTTPStorageClient()

put_url = gsclient.generate_signed_url(cloud_path, method="PUT")
client.upload(local_path1, put_url)

get_url = gsclient.generate_signed_url(cloud_path, method="GET")
client.download(get_url, local_path2)

delete_url = gsclient.generate_signed_url(cloud_path, method="DELETE")
client.delete(delete_url)
```


## SFTP Storage Client

The `eta.core.storage.SFTPStorageClient` class provides an SFTP client to
perform file transfers to/from a remote file server using ssh key-based
authentication.

### Authentication

All instances of this client must be provided with an SSH private key with
access to the remote host of interest. This can be done in the following ways
(in order of precedence):

- providing the `private_key_path` argument to
`eta.core.storage.SFTPStorageClient()` to manually specify the private key file
to use

- setting the `SSH_PRIVATE_KEY_PATH` environment variable to point to a private
key file to use

- automatically loading credentials from `~/.eta/id_rsa` that have been
activated via `eta.core.storage.SFTPStorageClient.activate_credentials()`

In the above, the private key file should have syntax similar to the following:

```
-----BEGIN RSA PRIVATE KEY-----
...
-----END RSA PRIVATE KEY-----
```

### Example usage

```py
import eta.core.storage as etas

username = "user"
hostname = "remote.server.com"
private_key_path = "~/.ssh/id_rsa"

local_path1 = "/path/to/file.txt"
local_path2 = "/path/to/file2.txt"
remote_dir = "/home/user/dir"
remote_path = "/home/user/dir/file.txt"

with etas.SFTPStorageClient(hostname, username, private_key_path) as client:
    client.make_dir(remote_dir)
    client.upload(local_path1, remote_path)
    client.download(remote_path, local_path2)
    client.delete(remote_path)
    client.delete_dir(remote_dir)
```

### References

- https://pysftp.readthedocs.io/en/release_0.2.9


## Copyright

Copyright 2017-2022, Voxel51, Inc.<br>
voxel51.com
