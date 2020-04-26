"""
Core tools for downloading files from the web.

Copyright 2017-2020, Voxel51, Inc.
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

import logging
import re
import requests
from time import time

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


def download_file(url, path=None, chunk_size=None):
    """Downloads a file from a URL. If a path is specified, the file is written
    there. Otherwise, the content is returned as a binary string.

    Args:
        url: the URL to get
        path: an optional path to write the file to
        chunk_size: an optional chunk size (in bytes) to use

    Returns:
        the binary string if path is not specified; otherwise None

    Raises:
        WebSessionError: if the download failed
    """
    sess = WebSession(chunk_size=chunk_size)
    return sess.write(path, url) if path else sess.get(url)


def download_google_drive_file(fid, path=None, chunk_size=None):
    """Downloads the Google Drive file with the given ID. If a path is
    specified, the file is written there. Otherwise, the file contents are
    returned as a binary string.

    Args:
        fid: the ID of the Google Drive file (usually a 28 character string)
        path: an optional path to write the file to
        chunk_size: an optional chunk size (in bytes) to use

    Returns:
        the binary string if path is not specified; otherwise None

    Raises:
        WebSessionError: if the download failed
    """
    sess = GoogleDriveSession(chunk_size=chunk_size)
    return sess.write(path, fid) if path else sess.get(fid)


class WebSession(object):
    """Class for downloading files from the web."""

    DEFAULT_CHUNK_SIZE = None

    def __init__(self, chunk_size=None):
        """Creates a WebSession instance.

        chunk_size: an optional chunk size (in bytes) to use for downloads.
            By default, `DEFAULT_CHUNK_SIZE` is used
        """
        self.sess = requests.Session()
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE

        # Tell the website who is downloading
        header = "%s v%s, %s" % (etac.NAME, etac.VERSION, etac.AUTHOR)
        self.sess.headers.update({"User-Agent": header})

    def get(self, url, params=None):
        """Gets the content from the URL.

        Args:
            url: the URL to get
            params: optional dictionary of parameters for the URL

        Returns:
            a string containing the raw bytes from the URL

        Raises:
            WebSessionError: if the download failed
        """
        r = self._get_streaming_response(url, params=params)
        return r.content

    def write(self, path, url, params=None):
        """Writes the URL content to the given local path.

        The download is performed in chunks, so arbitrarily large files can
        be downloaded. The output directory is created, if necessary.

        Args:
            path: the output path
            url: the URL to get
            params: optional dictionary of parameters for the URL

        Raises:
            WebSessionError: if the download failed
        """
        r = self._get_streaming_response(url, params=params)
        etau.ensure_basedir(path)

        num_bytes = 0
        start_time = time()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=self.chunk_size):
                num_bytes += len(chunk)
                f.write(chunk)

        time_elapsed = time() - start_time
        _log_download_stats(num_bytes, time_elapsed)

    def _get_streaming_response(self, url, params=None):
        r = self.sess.get(url, params=params, stream=True)
        if r.status_code != 200:
            raise WebSessionError("Unable to get '%s'" % url)

        return r


class WebSessionError(Exception):
    """Exception raised when there is a problem with a web session."""

    pass


class GoogleDriveSession(WebSession):
    """Class for downloading Google Drive files."""

    BASE_URL = "https://drive.google.com/uc?export=download"

    def get(self, fid):
        return WebSession.get(self, self.BASE_URL, params={"id": fid})

    def write(self, path, fid):
        return WebSession.write(self, path, self.BASE_URL, params={"id": fid})

    def _get_streaming_response(self, url, params=None):
        r = WebSession._get_streaming_response(self, url, params=params)

        # Handle download warning for large files
        for key, token in iteritems(r.cookies):
            if key.startswith("download_warning"):
                logger.debug("Retrying request with a confirm parameter")
                r.close()
                new = params.copy()
                new["confirm"] = token
                r = WebSession._get_streaming_response(self, url, params=new)
                break

        return r


def _log_download_stats(num_bytes, time_elapsed):
    bytes_str = etau.to_human_bytes_str(num_bytes)
    time_str = etau.to_human_time_str(time_elapsed)
    avg_speed = etau.to_human_bits_str(8 * num_bytes / time_elapsed) + "/s"
    logger.info("%s downloaded in %s (%s)", bytes_str, time_str, avg_speed)
