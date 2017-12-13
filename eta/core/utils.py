'''
Core system and file I/O utilities.

Copyright 2017, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
Jason Corso, jjc@voxel51.com
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import errno
import hashlib
import logging
import os
import random
import shutil
import string
import subprocess
import sys
import tempfile

import eta.constants as c


logger = logging.getLogger(__name__)


def get_eta_rev():
    '''Returns the hash of the last commit to the current ETA branch or "" if
    something went wrong with git.'''
    with WorkingDir(c.ETA_DIR):
        success, rev, _ = communicate(
            ["git", "rev-parse", "HEAD"], decode=True)
    return rev.strip() if success else ""


def has_gpu():
    '''Determine if the current device has a GPU'''
    if sys.platform == "darwin":
        # No GPU on mac
        return False
    try:
        return "NVIDIA" in communicate(["lspci"])[1]
    except OSError:
        # couldn't find lspci command...
        return False


def get_full_class_name(obj):
    '''Returns the fully-qualified class name of the given object.'''
    return obj.__module__ + "." + obj.__class__.__name__


def get_class(class_name, module_name=None):
    '''Returns the class specified by the given string.

    Loads the parent module if necessary.

    Args:
        class_name: the "ClassName" or a fully-qualified class name like
            "eta.core.utils.ClassName"
        module_name: the fully-qualified module name like "eta.core.utils", or
            None if class_name includes the module name. Set module_name to
            __name__ to load a class from the calling module
    '''
    if module_name is None:
        module_name, class_name = class_name.rsplit(".", 1)

    __import__(module_name)  # does nothing if module is already imported
    return getattr(sys.modules[module_name], class_name)


def get_function(function_name, module_name=None):
    '''Returns the function specified by the given string.

    Loads the parent module if necessary.

    Args:
        function_name: local function name by string fully-qualified name
            like "eta.core.utils.get_function"
        module_name: the fully-qualified module name like "eta.core.utils", or
            None if function_name includes the module name. Set module_name to
            __name__ to load a function from the calling module
    '''
    return get_class(function_name, module_name)


def communicate(args, decode=False):
    '''Runs the command via subprocess.communicate()

    Args:
        args: the command specified as a ["list", "of", "strings"]
        decode: whether to decode the output bytes into utf-8 strings. By
            default, the raw bytes are returned

    Returns:
        True/False: if the command executed successfully
        out: the command's stdout
        err: the command's stderr
    '''
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if decode:
        out = out.decode()
        err = err.decode()
    return p.returncode == 0, out, err


def call(args):
    '''Runs the command via subprocess.call()

    Args:
        args: the command specified as a ["list", "of", "strings"]

    Returns:
        True/False: if the command executed successfully
    '''
    return subprocess.call(args) == 0


def ensure_path(path):
    '''Ensures that the given path is ready for writing by deleting any
    existing file and ensuring that the base directory exists.
    '''
    if os.path.isfile(path):
        logger.debug("Deleting '%s'", path)
        os.remove(path)
    ensure_basedir(path)


def ensure_basedir(path):
    '''Makes the base directory of the given path, if necessary.'''
    ensure_dir(os.path.dirname(path))


def ensure_dir(dirname):
    '''Makes the given directory, if necessary.'''
    if dirname and not os.path.isdir(dirname):
        logger.debug("Making directory '%s'", dirname)
        os.makedirs(dirname)


def copy_file(inpath, outpath):
    '''Copies the input file to the output location, creating the base output
    directory if necessary.
    '''
    ensure_basedir(outpath)
    shutil.copy(inpath, outpath)


def move_file(inpath, outpath):
    '''Copies the input file to the output location, creating the base output
    directory if necessary.
    '''
    ensure_basedir(outpath)
    shutil.move(inpath, outpath)


def random_key(n):
    '''Generates an n-len random key of lowercase characters and digits.'''
    return "".join(random.SystemRandom().choice(
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
        '''Returns the current hash record, or None if there is no record.'''
        try:
            with open(self.record_path, "rt") as f:
                logger.debug("Found hash record '%s'", self.record_path)
                return f.read()
        except EnvironmentError as e:
            if e.errno == errno.ENOENT:
                logger.debug("No hash record '%s'", self.record_path)
                return None
            else:
                raise

    def write(self):
        '''Writes the current hash record.'''
        with open(self.record_path, "wt") as f:
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
        with open(path, "rb") as f:
            return str(hashlib.md5(f.read()).hexdigest())


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
