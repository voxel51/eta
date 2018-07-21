'''
Core system and file I/O utilities.

Copyright 2017-2018, Voxel51, LLC
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
from future.utils import iteritems
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import datetime
import errno
import glob
import hashlib
import inspect
import itertools as it
import logging
import os
import random
import re
import shutil
import six
import string
import subprocess
import sys
import tarfile
import tempfile

import eta.constants as etac


logger = logging.getLogger(__name__)


def is_str(val):
    '''Returns True/False whether the given value is a string.'''
    return isinstance(val, six.string_types)


def get_isotime():
    '''Gets the local time in ISO 8601 format: "YYYY-MM-DD HH:MM:SS".'''
    return str(datetime.datetime.now().replace(microsecond=0))


def get_eta_rev():
    '''Returns the hash of the last commit to the current ETA branch or "" if
    something went wrong with git.'''
    with WorkingDir(etac.ETA_DIR):
        success, rev, _ = communicate(
            ["git", "rev-parse", "HEAD"], decode=True)
    return rev.strip() if success else ""


def has_gpu():
    '''Determine if the current device has a GPU'''
    if sys.platform == "darwin":
        # No GPU on mac
        return False
    try:
        return "NVIDIA" in communicate(["lspci"], decode=True)[1]
    except OSError:
        # couldn't find lspci command...
        return False


def get_class_name(cls_or_obj):
    '''Returns the fully-qualified class name for the given input, which can
    be a class or class instance.

    Args:
        cls_or_obj: a class or class instance

    Returns:
        class_name: a fully-qualified class name string like
            "eta.core.utils.ClassName"
    '''
    cls = cls_or_obj if inspect.isclass(cls_or_obj) else cls_or_obj.__class__
    return cls_or_obj.__module__ + "." + cls.__name__


def get_function_name(fcn):
    '''Returns the fully-qualified function name for the given function.

    Args:
        fcn: a function

    Returns:
        function_name: a fully-qualified function name string like
            "eta.core.utils.function_name"
    '''
    return fcn.__module__ + "." + fcn.__name__


def get_class(class_name, module_name=None):
    '''Returns the class specified by the given class string, loading the
    parent module if necessary.

    Args:
        class_name: the "ClassName" or a fully-qualified class name like
            "eta.core.utils.ClassName"
        module_name: the fully-qualified module name like "eta.core.utils", or
            None if class_name includes the module name. Set module_name to
            __name__ to load a class from the calling module

    Raises:
        ImportError: if the class could not be imported
    '''
    if module_name is None:
        try:
            module_name, class_name = class_name.rsplit(".", 1)
        except ValueError:
            raise ImportError(
                "Class name '%s' must be fully-qualified when no module "
                "name is provided" % class_name)

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
    # reuse implementation for getting a class
    return get_class(function_name, module_name=module_name)


def query_yes_no(question, default=None):
    '''Asks a yes/no question via raw_input() and returns the answer.

    This function is case insensitive and partially matches are allowed.

    Args:
        question: the question to ask
        default: the default answer, which can be "yes", "no", or None (a
            response is required). The default is None

    Returns:
        True/False whether the user replied "yes" or "no"

    Raises:
        ValueError: if the default value was invalid
    '''
    valid = {"y": True, "ye": True, "yes": True, "n": False, "no": False}

    if default:
        default = default.lower()
        try:
            prompt = " [Y/n] " if valid[default] else " [y/N] "
        except KeyError:
            raise ValueError("Invalid default value '%s'" % default)
    else:
        prompt = " [y/n] "

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default and not choice:
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no'")


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


def communicate_or_die(args, decode=False):
    '''Wrapper around communicate() that raises an exception if any error
    occurs.

    Args:
        same as communicate()

    Returns:
        out: the command's stdout

    Raises:
        ExecutableNotFoundError: if the executable in the command was not found
        ExecutableRuntimeError: if an error occurred while executing the
            command
    '''
    try:
        success, out, err = communicate(args, decode=decode)
        if not success:
            raise ExecutableRuntimeError(" ".join(args), err)

        return out
    except EnvironmentError as e:
        if e.errno == errno.ENOENT:
            raise ExecutableNotFoundError(args[0])
        else:
            raise


def call(args):
    '''Runs the command via subprocess.call()

    Args:
        args: the command specified as a ["list", "of", "strings"]

    Returns:
        True/False: if the command executed successfully
    '''
    return subprocess.call(args) == 0


def copy_file(inpath, outpath):
    '''Copies the input file to the output location, which can be a filepath or
    a directory in which to write the file. The base output directory is
    created if necessary, and any existing file will be overwritten.
    '''
    ensure_basedir(outpath)
    shutil.copy(inpath, outpath)


def symlink_file(filepath, linkpath):
    '''Creates a symlink at the given location that points to the given file.
    The base output directory is created if necessary, and any existing file
    will be overwritten.
    '''
    ensure_basedir(linkpath)
    if os.path.exists(linkpath):
        os.remove(linkpath)
    os.symlink(os.path.realpath(filepath), linkpath)


def move_file(inpath, outpath):
    '''Copies the input file to the output location, creating the base output
    directory if necessary.
    '''
    ensure_basedir(outpath)
    shutil.move(inpath, outpath)


def copy_dir(indir, outdir):
    '''Copies the input directory to the output directory. The base output
    directory is created if necessary, and any existing output directory will
    be deleted.
    '''
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)

    shutil.copytree(indir, outdir)


def delete_file(path):
    '''Deletes the file at the given path and recursively deletes any empty
    directories from the resulting directory tree.
    '''
    os.remove(path)
    try:
        os.removedirs(os.path.dirname(path))
    except OSError:
        # found a non-empty directory or directory with no write access
        pass


def delete_dir(dir_):
    '''Deletes the given directory and recursively deletes any empty
    directories from the resulting directory tree.
    '''
    dir_ = os.path.normpath(dir_)
    shutil.rmtree(dir_)
    try:
        os.removedirs(os.path.dirname(dir_))
    except OSError:
        # found a non-empty directory or directory with no write access
        pass


def make_search_path(dirs):
    '''Makes a search path for the given directories by doing the following:
        - converting all paths to absolute paths
        - removing directories that don't exist
        - removing duplicate directories

    The order of the original directories is preserved.

    Args:
        dirs: a list of relative or absolute directory paths

    Returns:
        a list of absolute paths with duplicates and non-existent directories
            removed
    '''
    search_dirs = []
    for d in dirs:
        adir = os.path.abspath(d)
        if os.path.isdir(adir) and adir not in search_dirs:
            search_dirs.append(adir)

    if not search_dirs:
        logger.warning("Search path is empty")

    return search_dirs


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


def has_extension(filename, *args):
    '''Determines whether the filename has any of the given extensions.

    Args:
        filename: a file name
        *args: extensions like ".txt" or ".json"
    '''
    ext = os.path.splitext(filename)[1]
    return any(ext == a for a in args)


def to_human_bytes_str(num_bytes):
    '''Returns a human-readable string represntation of the given number of
    bytes.
    '''
    return _to_human_binary_str(num_bytes, "B")


def to_human_bits_str(num_bits):
    '''Returns a human-readable string represntation of the given number of
    bits.
    '''
    return _to_human_binary_str(num_bits, "b")


def _to_human_binary_str(num, suffix):
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            break
        num /= 1024.0
    return "%3.1f %s%s" % (num, unit, suffix)


def extract_tar(inpath, outdir=None, delete_tar=False):
    '''Extracts the .tar, tar.gz, .tgz, .tar.bz, and .tbz files into the same
    directory and then optionally deletes the tarball.

    Args:
        inpath: the path to the tar or compressed tar file
        outdir: the directory into which to extract the archive contents. By
            default, the directory containing the tar file is used
        delete_tar: whether to delete the tar archive after extraction. By
            default, this is False
    '''
    if inpath.endswith("tar"):
        fmt = "r:"
    elif inpath.endswith("tar.gz") or inpath.endswith("tgz"):
        fmt = "r:gz"
    elif inpath.endswith("tar.bz") or inpath.endswith("tbz"):
        fmt = "r:bz2"
    else:
        raise ValueError(
            "Expected file '%s' to have extension .tar, .tar.gz, .tgz,"
            ".tar.bz, or .tbz in order to extract it" % inpath)

    outdir = outdir or os.path.dirname(inpath) or "."
    with tarfile.open(inpath, fmt) as tar:
        tar.extractall(path=outdir)

    if delete_tar:
        delete_file(inpath)


def multiglob(*patterns, **kwargs):
    ''' Returns an iterable for globbing over multiple patterns.

    Args:
        patterns is the set of patterns to search for
        kwargs["root"] allows for a `root` path to be specified once and
            applied to all patterns

    Note that this does not us os.path.join if a root=FOO is provided. So, if
    you want to just search by extensions, you can use root="path/*" and
    provide only extensions in the patterns.
    '''
    root = kwargs.get("root", "")
    return it.chain.from_iterable(
        glob.iglob(root + pattern) for pattern in patterns)


def list_files(dir_path):
    '''Lists the files in the given directory, sorted alphabetically and
    excluding directories and hidden files.

    Args:
        dir_path: the path to the directory to list

    Returns:
        a sorted list of the non-hidden files in the directory
    '''
    return sorted(
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and not f.startswith(".")
    )


def parse_dir_pattern(dir_path):
    '''Inspects the contents of the given directory, returning the numeric
    pattern in use and the associated indexes.

    The numeric pattern is guessed by analyzing the first file (alphabetically)
    in the directory.

    For example, if the directory contains:
        "frame-00040-object-5.json"
        "frame-00041-object-6.json"
    then the pattern "frame-%05d-object-%d.json" will be inferred along with
    the associated indices [(40, 5), (41, 6)]

    Args:
        dir_path: the path to the directory to inspect

    Returns:
        a tuple containing:
            - the numeric pattern used in the directory
            - a list (or list of tuples if the pattern contains multiple
                numbers) describing the numeric indices in the directory

    Raises:
        OSError: if the directory is empty
    '''
    files = list_files(dir_path)
    if not files:
        raise OSError("Directory %s contains no files" % dir_path)

    def _guess_patt(m):
        s = m.group()
        leading_zeros = len(s) - len(str(int(s)))
        return "%%0%dd" % len(s) if leading_zeros > 0 else "%d"

    # Guess pattern from first file
    name, ext = os.path.splitext(os.path.basename(files[0]))
    regex = re.compile(r"\d+")
    patt = os.path.join(dir_path, re.sub(regex, _guess_patt, name) + ext)

    def _unpack(l):
        return int(l[0]) if len(l) == 1 else tuple(map(int, l))

    # Infer indices
    indices = [_unpack(re.findall(regex, f)) for f in files]

    return patt, indices


def random_key(n):
    '''Generates an n-lenth random key of lowercase characters and digits.'''
    return "".join(
        random.SystemRandom().choice(string.ascii_lowercase + string.digits)
        for _ in range(n)
    )


def replace_strings(s, replacers):
    '''Performs a sequence of find-replace operations on the given string.

    Args:
        s: the input string
        replacers: a list of (find, replace) strings

    Returns:
        a copy of the input strings with all of the find-and-replacements made
    '''
    sout = s
    for sfind, srepl in replacers:
        sout = sout.replace(sfind, srepl)

    return sout


def join_dicts(*args):
    '''Joins any number of dictionaries into a new single dictionary.

    Args:
        *args: one or more dictionaries

    Returns:
        a single dictionary containing all items.
    '''
    d = {}
    for di in args:
        d.update(di)
    return d


def remove_none_values(d):
    '''Returns a copy of the input dictionary with any keys with value None
    removed.

    Args:
        d: a dictionary

    Returns:
        a copy of the input dictionary with keys whose value was None ommitted
    '''
    return {k: v for k, v in iteritems(d) if v is not None}


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
        delete_dir(self._name)


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
    '''Exception raised when an executable file is not found.'''

    def __init__(self, executable):
        message = "Executable '%s' not found" % executable
        super(ExecutableNotFoundError, self).__init__(message)


class ExecutableRuntimeError(Exception):
    '''Exception raised when an executable call throws a runtime error.'''

    def __init__(self, cmd, err):
        message = "Command '%s' failed with error:\n%s" % (cmd, err)
        super(ExecutableRuntimeError, self).__init__(message)
