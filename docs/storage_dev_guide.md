# Remote Storage Developer's Guide

This document describes best practices for accessing remote storage resources
in ETA. The functionality is implemented in the `eta.core.storage` module.


## Installation

The `eta.core.storage` module depends on the following third-party packages,
which are installed by ETA's `requirements.txt` file:

```shell
# Google Drive
pip install --upgrade google-api-python-client
pip install --upgrade google-auth-httplib2

# Google Cloud
pip install --upgrade google-cloud-storage

# HTTP
pip install --upgrade requests
pip install --upgrade requests-toolbelt

# SFTP
pip install --upgrade pysftp
```


## Google Drive Storage API

The `GoogleDriveStorageClient` class provides programmatic access to Google
Drive.

All instances of this client must be provided with Google service account
credentials with the appropriate permissions to perform the file manipulations
that you request. This can be done either by passing the path to the service
account JSON file to the constructor or by setting the
`GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to it.

### Enable the Google Drive API

https://console.developers.google.com/apis/api/drive.googleapis.com
(once per organization)

### Giving a service account access to a Team Drive

- Navigate to the folder at drive.google.com
- Click Share...
- Add service account's `client_email` address to the membership list

### Example usage

```py
import eta.core.storage as etas

team_drive_name = "public"
credentials_json_path = "/path/to/service-account.json"
local_path1 = "/path/to/file.txt"
local_path2 = "/path/to/file2.txt"

client = etas.GoogleDriveStorageClient.from_json(credentials_json_path)
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


## Google Cloud Storage API

The `GoogleCloudStorageClient` class provides programmatic access to Google
Cloud Storage buckets.

All instances of this client must be provided with Google service account
credentials with the appropriate permissions to perform the file manipulations
that you request. This can be done either by passing the path to the service
account JSON file to the constructor or by setting the
`GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to it.

### Example usage

```py
import eta.core.storage as etas

credentials_json_path = "/path/to/service-account.json"
local_path1 = "/path/to/file.txt"
local_path2 = "/path/to/file2.txt"
cloud_path = "gs://bucket-name/file.txt"

client = etas.GoogleCloudStorageClient.from_json(credentials_json_path)
client.upload(local_path1, cloud_path)
client.download(cloud_path, local_path2)
client.delete(cloud_path)
```

### References

- https://cloud.google.com/storage/docs/reference/libraries
- https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python
- https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/storage/cloud-client/snippets.py


## HTTP Storage API

The `HTTPStorageClient` class provides a convenient API for issuing HTTP
requests to upload and download files of arbitrary sizes from URLs.

### Example usage

The following snippet demonstrates an example of `GoogleCloudStorageClient` to
generate signed URLs for accessing objects in Google Cloud storage and then
using `HTTPStorageClient` to access the resources via HTTP requests.

```py
import eta.core.storage as etas

credentials_json_path = "/path/to/service-account.json"
local_path1 = "/path/to/file.txt"
local_path2 = "/path/to/file2.txt"
cloud_path = "gs://bucket-name/file.txt"

gsclient = etas.GoogleCloudStorageClient.from_json(credentials_json_path)
client = etas.HTTPStorageClient()

put_url = gsclient.generate_signed_url(cloud_path, method="PUT")
client.upload(local_path1, put_url)

get_url = gsclient.generate_signed_url(cloud_path, method="GET")
client.download(get_url, local_path2)

delete_url = gsclient.generate_signed_url(cloud_path, method="DELETE")
client.delete(delete_url)
```


## SFTP Storage API

The `SFTPStorageClient` class provides an SFTP client to perform file transfers
to/from a remote file server using ssh key-based authentication.

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
