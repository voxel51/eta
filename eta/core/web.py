"""
Tools for downloading files from the web.

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

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import io
import logging
import re
import requests

import eta.constants as etac
import eta.core.utils as etau


logger = logging.getLogger(__name__)


URL_REGEX = re.compile(r"http://|https://|ftp://|file://|file:\\")


def is_url(filename):
    """Return True if string is an http or ftp path.

    Args:
        filename: a string

    Returns:
        True/False if the filename is a url
    """
    return etau.is_str(filename) and URL_REGEX.match(filename) is not None


def download_file(url, path=None, chunk_size=None, verify=True, quiet=False):
    """Downloads a file from a URL. If a path is specified, the file is written
    there. Otherwise, the content is returned as a binary string.

    Args:
        url: the URL to get
        path: an optional path to write the file to
        chunk_size: an optional chunk size (in bytes) to use
        verify: whether to verify SSL certificates before downloading. Set this
            parameter to `False` to bypasses certificate validation
        quiet: whether to NOT show a progress bar tracking the download

    Returns:
        the binary string if path is not specified; otherwise None

    Raises:
        WebSessionError: if the download failed
    """
    sess = WebSession(chunk_size=chunk_size, verify=verify, quiet=quiet)
    return sess.write(path, url) if path else sess.get(url)


def download_google_drive_file(fid, path=None, chunk_size=None, quiet=False):
    """Downloads the Google Drive file with the given ID. If a path is
    specified, the file is written there. Otherwise, the file contents are
    returned as a binary string.

    Args:
        fid: the ID of the Google Drive file
        path: an optional path to write the file to
        chunk_size: an optional chunk size (in bytes) to use
        quiet: whether to NOT show a progress bar tracking the download

    Returns:
        the binary string if path is not specified; otherwise None

    Raises:
        WebSessionError: if the download failed
    """
    sess = GoogleDriveSession(chunk_size=chunk_size, quiet=quiet)
    return sess.write(path, fid) if path else sess.get(fid)


class WebSession(object):
    """Class for downloading files from the web."""

    # Chunk size, in bytes
    DEFAULT_CHUNK_SIZE = 64 * 1024

    def __init__(self, chunk_size=None, verify=True, quiet=False):
        """Creates a WebSession instance.

        chunk_size: an optional chunk size (in bytes) to use for downloads.
            By default, `DEFAULT_CHUNK_SIZE` is used
        verify: whether to verify SSL certificates before downloading. Set this
            parameter to `False` to bypasses certificate validation
        quiet: whether to NOT show progress bars tracking downloads
        """
        self.sess = requests.Session()
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self.verify = verify
        self.quiet = quiet

        # Tell the website who is downloading
        header = "%s v%s, %s" % (etac.NAME, etac.VERSION, etac.AUTHOR)
        self.sess.headers.update({"User-Agent": header})

    def get(self, url, params=None):
        """Gets the content from the URL.

        Args:
            url: the URL to get
            params: optional dictionary of parameters

        Returns:
            a string containing the raw bytes from the URL

        Raises:
            WebSessionError: if the download failed
        """
        r = self._get_streaming_response(url, params=params)
        with io.BytesIO() as f:
            self._do_download(r, f)
            return f.getvalue()

    def write(self, path, url, params=None):
        """Writes the URL content to the given local path.

        The download is performed in chunks, so arbitrarily large files can
        be downloaded. The output directory is created, if necessary.

        Args:
            path: the output path
            url: the URL to get
            params: optional dictionary of parameters

        Raises:
            WebSessionError: if the download failed
        """
        r = self._get_streaming_response(url, params=params)
        etau.ensure_basedir(path)
        with open(path, "wb") as f:
            self._do_download(r, f)

    def _get_streaming_response(self, url, headers=None, params=None):
        r = self.sess.get(
            url,
            headers=headers,
            params=params,
            stream=True,
            verify=self.verify,
        )

        if r.status_code not in (200, 206):
            raise WebSessionError("Unable to get '%s'" % url)

        return r

    def _do_download(self, r, f):
        size_bytes = _get_content_length(r)
        size_bits = 8 * size_bytes if size_bytes is not None else None
        with etau.ProgressBar(
            size_bits, use_bits=True, quiet=self.quiet
        ) as pb:
            for chunk in r.iter_content(chunk_size=self.chunk_size):
                f.write(chunk)
                pb.update(8 * len(chunk))


class WebSessionError(Exception):
    """Exception raised when there is a problem with a web session."""

    pass


class GoogleDriveSession(WebSession):
    """Class for downloading Google Drive files."""

    BASE_URL = "https://drive.google.com/uc?export=download"

    def get(self, fid):
        return super(GoogleDriveSession, self).get(
            self.BASE_URL, params={"id": fid}
        )

    def write(self, path, fid):
        return super(GoogleDriveSession, self).write(
            path, self.BASE_URL, params={"id": fid}
        )

    def _get_streaming_response(self, url, params=None):
        # Include `Range` in request so that returned header will contain
        # `Content-Range`
        # https://stackoverflow.com/a/52044629
        r = super(GoogleDriveSession, self)._get_streaming_response(
            url, headers={"Range": "bytes=0-"}, params=params
        )

        # Handle download warning for large files
        # https://stackoverflow.com/a/39225272
        for key, token in iteritems(r.cookies):
            if key.startswith("download_warning"):
                logger.debug("Retrying request with a confirm parameter")
                r.close()
                new = params.copy()
                new["confirm"] = token
                r = self._get_streaming_response(url, params=new)
                break

        return r


def _get_content_length(r):
    try:
        return int(r.headers["Content-Length"])
    except KeyError:
        pass

    try:
        # <unit> <range-start>-<range-end>/<size>
        range_str = r.headers["Content-Range"]
        return int(range_str.partition("/")[-1])
    except:
        pass

    return None
