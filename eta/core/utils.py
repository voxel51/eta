'''
Core system and file I/O utilities.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
Jason Corso, jjc@voxel51.com
'''
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import requests
from BeautifulSoup import BeautifulSoup

import numpy as np

import serial


def run_cmd(args):
    '''Runs command and returns True if successful and False otherwise.

    Args:
        args: command and any flags given as a ["list", "of", "strings"]
    '''
    return subprocess.call(args, shell=False) == 0


def download_large_google_drive_file(url):
    '''Scrapes the Google Drive web page to confirm that we want to download
    the file even though it is too big to scan for viruses.
    This function assumes the url is valid and will link to the confirmation
    page via the first request.
    '''
    session = requests.Session()

    session.headers.update({'User-Agent':'ETA v0.1.0, Voxel51, LLC'})

    page = session.get(url)
    if (page.status_code != 200):
        # @todo log an error here
        return None

    soup = BeautifulSoup(page.content)
    d = soup.find('a',{'id':'uc-download-link'})['href']
    assert(d != '')
    # is it bad practice to return large chunks of data?
    page2 = return session.get("https://drive.google.com"+d)
    if (page2.status_code != 200):
        # @todo log an error here
        return None
    return page2.content


def download_file(url):
    '''Downloads a file from a URL checking the status to be correct.
    Returns the download file (a binary chunk) or None if failed.
    '''
    session = requests.Session()
    session.headers.update({'User-Agent':'ETA v0.1.0, Voxel51, LLC'})
    page = session.get(url)
    if (page.status_code != 200):
        # @todo log an error here
        return None
    return page.content


def ensure_path(path):
    '''Ensures that the given path is ready for writing by deleting any existing
    file and ensuring that the base directory exists.
    '''
    if os.path.isfile(path):
        os.remove(path)
    ensure_dir(path)


def ensure_dir(path):
    '''Makes base directory, if necessary.'''
    dirname = os.path.dirname(path)
    if dirname and not os.path.isdir(dirname):
        os.makedirs(dirname)


def copy_file(inpath, outpath):
    '''Copies input file to the output file (the actual file, not the
    directory), creating the output directory if necessary.
    '''
    ensure_dir(outpath)
    shutil.copy(inpath, outpath)


def read_json(path):
    '''Reads JSON from file.'''
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    '''Writes JSON object to file, creating the output directory if necessary.

    Args:
        obj: is either an object that can be directly dumped to a JSON file or
            an instance of a subclass of serial.Serializable
        path: the output path
    '''
    if serial.is_serializable(obj):
        obj = obj.serialize()
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, cls=JSONNumpyEncoder)


class FileHasher(object):
    '''Base class for file hashers.'''

    EXT = ""

    def __init__(self, path):
        '''Constructs a FileHasher instance based on the current version of
        the input file.'''
        self.path = path
        self._new_hash = self.hash(path)
        self._cur_hash = self.read()

    @property
    def record_path(self):
        '''The path to the hash record file.'''
        return os.path.splitext(self.path)[0] + self.EXT

    @property
    def has_record(self):
        '''True if the file has an existing hash record.'''
        return self._cur_hash is not None

    @property
    def has_changed(self):
        '''True if the file's current hash differs from it's last hash record.
        Always returns False if the file has no existing hash record.
        '''
        return self.has_record and self._new_hash != self._cur_hash

    def read(self):
        '''Returns the current hash record, or None if there is no record.'''
        try:
            with open(self.record_path, "r") as f:
                return f.read()
        except:
            return None

    def write(self):
        '''Writes the current hash record.'''
        with open(self.record_path, "w") as f:
            f.write(self._new_hash)

    @staticmethod
    def hash(path):
        raise NotImplementedError("subclass must implement hash()")


class MD5FileHasher(FileHasher):
    '''MD5 file hasher.'''

    EXT = ".md5"

    @staticmethod
    def hash(path):
        '''Computes the MD5 hash of the file contents.'''
        with open(path, "r") as f:
            return hashlib.md5(f.read()).hexdigest()


class TempDir(object):
    '''Context manager that creates and destroys a temporary directory.'''

    def __enter__(self):
        self.name = tempfile.mkdtemp()
        return self.name

    def __exit__(self, *args):
        shutil.rmtree(self.name)


class WorkingDir(object):
    '''Context manager that temporarily changes working directories.'''

    def __init__(self, working_dir):
        self.working_dir = working_dir

    def __enter__(self):
        self.orig_dir = os.getcwd()
        os.chdir(self.working_dir)
        return self

    def __exit__(self, *args):
        os.chdir(self.orig_dir)


class JSONNumpyEncoder(json.JSONEncoder):
    '''Extends basic JSONEncoder to handle numpy scalars/arrays.'''

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSONNumpyEncoder, self).default(obj)


class ExecutableNotFoundError(Exception):
    def __init__(self, executable):
        message = "Executable '%s' not found" % executable
        super(ExecutableNotFoundError, self).__init__(message)


class ExecutableRuntimeError(Exception):
    def __init__(self, cmd, err):
        message = "Command '%s' failed with error:\n%s" % (cmd, err)
        super(ExecutableRuntimeError, self).__init__(message)

