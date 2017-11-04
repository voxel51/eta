'''
Core web tools.

Copyright 2017, Voxel51, LLC
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
from future.utils import iteritems
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import requests

from eta import constants


def download_file(url, path=None):
    '''Downloads a file from a URL. If a path is specified, the file is written
    there. Otherwise, the content is returned as a binary string.

    Args:
        url: the URL to get
        path: an optional path to write the file to

    Raises:
        WebSessionError: if the download failed
    '''
    sess = WebSession()
    return sess.write(path, url) if path else sess.get(url)


def download_google_drive_file(fid, path=None):
    '''Downloads the Google Drive file with the given ID. If a path is
    specified, the file is written there. Otherwise, the file contents are
    returned as a binary string.

    Args:
        fid: the ID of the Google Drive file (usually a 28 character string)
        path: an optional path to write the file to

    Raises:
        WebSessionError: if the download failed
    '''
    sess = GoogleDriveSession()
    return sess.write(path, fid) if path else sess.get(fid)


class WebSession(object):
    '''Class for downloading files from the web.'''

    CHUNK_SIZE = 32768  # 32MB

    def __init__(self):
        '''Constructs a WebSession instance.'''
        self.sess = requests.Session()

        # Tell the website who is downloading
        header = "%s v%s, %s" % (
            constants.NAME, constants.VERSION, constants.AUTHOR)
        self.sess.headers.update({"User-Agent": header})

    def get(self, url, params=None):
        '''Gets the content from the URL.

        Args:
            url: the URL to get
            params: optional dictionary of parameters for the URL

        Returns:
            a string containing the raw bytes from the URL

        Raises:
            WebSessionError: if the download failed
        '''
        r = self._get_streaming_response(url, params=params)
        return r.content

    def write(self, path, url, params=None):
        '''Writes the URL content to the given local path.

        The download is performed in chunks, so arbitrarily large files can
        be downloaded.

        Args:
            path: the output path
            url: the URL to get
            params: optional dictionary of parameters for the URL

        Raises:
            WebSessionError: if the download failed
        '''
        r = self._get_streaming_response(url, params=params)
        with open(path, "wb") as f:
            for chunk in r.iter_content(self.CHUNK_SIZE):
                # @todo log progress periodically?
                f.write(chunk)

    def _get_streaming_response(self, url, params=None):
        r = self.sess.get(url, params=params, stream=True)
        if r.status_code != 200:
            raise WebSessionError("Unable to get '%s'" % url)

        return r


class WebSessionError(Exception):
    pass


class GoogleDriveSession(WebSession):
    '''Class for downloading Google Drive files.'''

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
                # Retry the get request with a confirm parameter
                # @todo log this
                r.close()
                new = params.copy()
                new["confirm"] = token
                r = WebSession._get_streaming_response(self, url, params=new)
                break

        return r
