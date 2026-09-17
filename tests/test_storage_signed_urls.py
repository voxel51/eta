"""Tests for signing cloud storage URLs.

Every client signs a URL for a HEAD request. A caller that needs only an
object's size asks for one, so a client that cannot sign it fails before
reading a byte.

The credentials here are fabricated and the clients are never used against a
provider: constructing a client and signing a URL are both offline, so these
tests exercise real signing code with no network and no account.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import base64
import importlib.util
import urllib.parse as urlparse
from unittest import mock

import pytest


def _is_installed(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except ImportError:
        # A missing parent package
        return False


# The cloud SDKs live behind the `storage` extra, which the ffmpeg CI jobs do
# not install. Probing them keeps an ImportError raised by the module under
# test a failure: catching one here would skip this file, and report a missing
# extra, for a broken import in the code it covers.
if not all(
    _is_installed(module_name)
    for module_name in ["azure.storage.blob", "boto3", "google.cloud.storage"]
):
    pytest.skip("requires the `storage` extra", allow_module_level=True)

import eta.core.storage as etast  # noqa: E402


AZURE_ACCOUNT = "myaccount"
AZURE_PATH = (
    "https://%s.blob.core.windows.net/mycontainer/myblob.mcap" % AZURE_ACCOUNT
)
S3_PATH = "s3://mybucket/myobject.mcap"
MINIO_PATH = "minio://mybucket/myobject.mcap"
GCS_PATH = "gs://mybucket/myobject.mcap"

# `generate_blob_sas()` requires a base64-encoded key
AZURE_ACCOUNT_KEY = base64.b64encode(b"a" * 32).decode("ascii")


def _azure_client(**kwargs):
    credentials = {
        "account_name": AZURE_ACCOUNT,
        "account_key": AZURE_ACCOUNT_KEY,
    }
    credentials.update(kwargs)
    return etast.AzureStorageClient(credentials=credentials)


def _s3_client():
    return etast.S3StorageClient(
        credentials={
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "aws_secret_access_key": "wJalrXUtnFEMI/bPxRfiCYEXAMPLEKEY",
            "region": "us-east-1",
        }
    )


def _minio_client():
    return etast.MinIOStorageClient(
        credentials={
            "access_key": "minioadmin",
            "secret_access_key": "minioadmin",
            "endpoint_url": "https://minio.example.com",
            "alias": "minio",
        }
    )


def _gcs_client():
    # A signed GCS URL is signed with a service account's private key, so the
    # test needs a real key pair. Nothing else about the account has to exist.
    crypto_serialization = pytest.importorskip(
        "cryptography.hazmat.primitives.serialization"
    )
    rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    service_account = pytest.importorskip("google.oauth2.service_account")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=crypto_serialization.Encoding.PEM,
        format=crypto_serialization.PrivateFormat.PKCS8,
        encryption_algorithm=crypto_serialization.NoEncryption(),
    ).decode("ascii")

    credentials = service_account.Credentials.from_service_account_info(
        {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "testkeyid",
            "private_key": pem,
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "1",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )

    return etast.GoogleCloudStorageClient(credentials=credentials)


def _query(url):
    return urlparse.parse_qs(urlparse.urlparse(url).query)


# Azure's SAS permission letters: `r` read, `c` create, `d` delete. A HEAD
# returns a blob's properties, which `r` authorizes.
@pytest.mark.parametrize(
    "method,permission",
    [
        ("GET", "r"),
        ("HEAD", "r"),
        ("head", "r"),
        ("PUT", "c"),
        ("DELETE", "d"),
    ],
)
def test_azure_signs_each_method_it_supports(method, permission):
    url = _azure_client().generate_signed_url(AZURE_PATH, method=method)

    query = _query(url)
    assert query["sig"], "the URL carries no signature"
    assert query["sp"] == [permission]


def test_azure_head_and_get_authorize_the_same_permission():
    client = _azure_client()

    get_url = client.generate_signed_url(AZURE_PATH, method="GET")
    head_url = client.generate_signed_url(AZURE_PATH, method="HEAD")

    assert _query(head_url)["sp"] == _query(get_url)["sp"]


def test_azure_rejects_a_method_it_cannot_sign():
    with pytest.raises(ValueError, match="PATCH"):
        _azure_client().generate_signed_url(AZURE_PATH, method="PATCH")


def test_azure_sas_token_needs_no_permission_lookup():
    # A SAS token carries its own permissions, so the client appends it rather
    # than signing anything. Every method resolves, including ones it cannot
    # sign for itself.
    client = _azure_client(sas_token="sv=2020-08-04&sig=fake")

    for method in ["GET", "HEAD", "PATCH"]:
        url = client.generate_signed_url(AZURE_PATH, method=method)
        assert url.endswith("?sv=2020-08-04&sig=fake")


@pytest.mark.parametrize(
    "client_fn,path",
    [(_s3_client, S3_PATH), (_minio_client, MINIO_PATH)],
    ids=["s3", "minio"],
)
@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_boto_signs_read_methods(client_fn, path, method):
    url = client_fn().generate_signed_url(path, method=method)

    assert _query(url)["X-Amz-Signature"]


@pytest.mark.parametrize(
    "client_fn,path",
    [(_s3_client, S3_PATH), (_minio_client, MINIO_PATH)],
    ids=["s3", "minio"],
)
@pytest.mark.parametrize(
    "method,client_method", [("GET", "get_object"), ("HEAD", "head_object")]
)
def test_boto_signs_the_method_it_is_given(
    client_fn, path, method, client_method
):
    # A signature is present whatever the verb, so assert the verb reaches
    # botocore: signing every method as a GET would satisfy the test above.
    client = client_fn()

    with mock.patch.object(
        client._client, "generate_presigned_url"
    ) as presign:
        client.generate_signed_url(path, method=method)

    assert presign.call_args.kwargs["ClientMethod"] == client_method


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_gcs_signs_read_methods(method):
    url = _gcs_client().generate_signed_url(GCS_PATH, method=method)

    assert _query(url)["X-Goog-Signature"]


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_gcs_signs_the_method_it_is_given(method):
    client = _gcs_client()

    with mock.patch.object(etast, "generate_signed_url_v4") as sign:
        client.generate_signed_url(GCS_PATH, method=method)

    assert sign.call_args.kwargs["method"] == method
