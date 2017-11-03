'''
Core system and file I/O utilities.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
Jason Corso, jjc@voxel51.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import hashlib
import os
import random
import shutil
import string
import subprocess
import tempfile


def run_cmd(args):
    '''Runs command and returns True if successful and False otherwise.

    Args:
        args: command and any flags given as a ["list", "of", "strings"]
    '''
    return subprocess.call(args, shell=False) == 0


def ensure_path(path):
    '''Ensures that the given path is ready for writing by deleting any
    existing file and ensuring that the base directory exists.
    '''
    if os.path.isfile(path):
        os.remove(path)
    ensure_basedir(path)


def ensure_basedir(path):
    '''Makes the base directory of the given path, if necessary.'''
    ensure_dir(os.path.dirname(path))


def ensure_dir(dirname):
    '''Makes the given directory, if necessary.'''
    if dirname and not os.path.isdir(dirname):
        os.makedirs(dirname)


def copy_file(inpath, outpath):
    '''Copies input file to the output file (the actual file, not the
    directory), creating the output directory if necessary.
    '''
    ensure_basedir(outpath)
    shutil.copy(inpath, outpath)


def random_key(n):
    '''Generates an n-len random key of lowercase characters and digits.'''
    return ''.join(random.SystemRandom().choice(
        string.ascii_lowercase + string.digits) for _ in range(n))


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
        '''Returns the current hash record (bytes), or None if there is no
        record.
        '''
        try:
            with open(self.record_path, "rb") as f:
                return f.read()
        except:
            return None

    def write(self):
        '''Writes the current hash record (bytes).'''
        with open(self.record_path, "wb") as f:
            f.write(self._new_hash)

    @staticmethod
    def hash(path):
        raise NotImplementedError("subclass must implement hash()")


class MD5FileHasher(FileHasher):
    '''MD5 file hasher.'''

    EXT = ".md5"

    @staticmethod
    def hash(path):
        '''Computes the MD5 hash (bytes) of the file contents.'''
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()


class TempDir(object):
    '''Context manager that creates and destroys a temporary directory.'''

    def __init__(self):
        self._name = None

    def __enter__(self):
        self._name = tempfile.mkdtemp()
        return self._name

    def __exit__(self, *args):
        shutil.rmtree(self._name)


class WorkingDir(object):
    '''Context manager that temporarily changes working directories.'''

    def __init__(self, working_dir):
        self._orig_dir = None
        self._working_dir = working_dir

    def __enter__(self):
        self._orig_dir = os.getcwd()
        os.chdir(self._working_dir)
        return self

    def __exit__(self, *args):
        os.chdir(self._orig_dir)


class ExecutableNotFoundError(Exception):
    def __init__(self, executable):
        message = "Executable '%s' not found" % executable
        super(ExecutableNotFoundError, self).__init__(message)


class ExecutableRuntimeError(Exception):
    def __init__(self, cmd, err):
        message = "Command '%s' failed with error:\n%s" % (cmd, err)
        super(ExecutableRuntimeError, self).__init__(message)
