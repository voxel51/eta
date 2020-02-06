'''
Core system and file I/O utilities.

Copyright 2017-2020, Voxel51, Inc.
voxel51.com

Brian Moore, brian@voxel51.com
Jason Corso, jason@voxel51.com
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from future.utils import iteritems, itervalues
import six
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict
from datetime import datetime
import dateutil.parser
import errno
import glob
import glob2
import hashlib
import inspect
import itertools as it
import logging
import math
import mimetypes
import os
import pytz
import random
import re
import shutil
import string
import subprocess
import sys
import tarfile
import tempfile
import timeit
import zipfile as zf

import eta
import eta.constants as etac


logger = logging.getLogger(__name__)


def is_str(val):
    '''Returns True/False whether the given value is a string.'''
    return isinstance(val, six.string_types)


def standarize_strs(arg):
    '''Standardizes any strings in the given object by casting them via
    `str()`. Dictionaries and lists are processed recursively.

    Args:
        arg: an object

    Returns:
        a copy (only if necessary) of the input object with any strings casted
            via str()
    '''
    if isinstance(arg, dict):
        return {
            standarize_strs(k): standarize_strs(v) for k, v in iteritems(arg)
        }

    if isinstance(arg, list):
        return [standarize_strs(e) for e in arg]

    if isinstance(arg, six.string_types):
        return str(arg)

    return arg


def get_localtime():
    '''Gets the local time in "YYYY-MM-DD HH:MM:SS" format.

    Returns:
        "YYYY-MM-DD HH:MM:SS"
    '''
    return str(datetime.now().replace(microsecond=0))


def parse_isotime(isostr_or_none):
    '''Parses the ISO time string into a datetime.

    If the input string has a timezone ("Z" or "+HH:MM"), a timezone-aware
    datetime will be returned. Otherwise, a naive datetime will be returned.
    If the input is falsey, None is returned.

    Args:
        isostr_or_none: an ISO time string like "YYYY-MM-DD HH:MM:SS", or None

    Returns:
        a datetime, or None if the input was empty
    '''
    if not isostr_or_none:
        return None

    return dateutil.parser.parse(isostr_or_none)


def datetime_delta_seconds(time1, time2):
    '''Computes the difference between the two datetimes, in seconds.

    If one (but not both) of the datetimes are timezone-aware, the other
    datetime is assumed to be expressed in UTC time.

    Args:
        time1: a datetime
        time2: a datetime

    Returns:
        the time difference, in seconds
    '''
    try:
        return (time2 - time1).total_seconds()
    except (TypeError, ValueError):
        time1 = add_utc_timezone_if_necessary(time1)
        time2 = add_utc_timezone_if_necessary(time2)
        return (time2 - time1).total_seconds()


def to_naive_local_datetime(dt):
    '''Converts the datetime to a naive (no timezone) datetime with its time
    expressed in the local timezone.

    The conversion is performed as follows:
        (1a) if the input datetime has no timezone, assume it is UTC
        (1b) if the input datetime has a timezone, convert to UTC
         (2) convert to local time
         (3) remove the timezone info

    Args:
        dt: a datetime

    Returns:
        a naive datetime in local time
    '''
    dt = add_utc_timezone_if_necessary(dt)
    return dt.astimezone().replace(tzinfo=None)


def to_naive_utc_datetime(dt):
    '''Converts the datetime to a naive (no timezone) datetime with its time
    expressed in UTC.

    The conversion is performed as follows:
        (1a) if the input datetime has no timezone, assume it is UTC
        (1b) if the input datetime has a timezone, convert to UTC
         (2) remove the timezone info

    Args:
        dt: a datetime

    Returns:
        a naive datetime in UTC
    '''
    dt = add_utc_timezone_if_necessary(dt)
    return dt.astimezone(pytz.utc).replace(tzinfo=None)


def add_local_timezone_if_necessary(dt):
    '''Makes the datetime timezone-aware, if necessary, by setting its timezone
    to the local timezone.

    Args:
        dt: a datetime

    Returns:
        a timezone-aware datetime
    '''
    if dt.tzinfo is None:
        dt = dt.astimezone()  # empty ==> local timezone

    return dt


def add_utc_timezone_if_necessary(dt):
    '''Makes the datetime timezone-aware, if necessary, by setting its timezone
    to UTC.

    Args:
        dt: a datetime

    Returns:
        a timezone-aware datetime
    '''
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.utc)

    return dt


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


def get_int_pattern_with_capacity(max_number):
    '''Gets a zero-padded integer pattern like "%%02d" or "%%03d" with
    sufficient capacity for the given number.

    Args:
        max_number: the maximum number you intend to pass to the pattern

    Returns:
        a zero-padded integer formatting pattern
    '''
    num_digits = max(1, math.ceil(math.log10(1 + max_number)))
    return "%%0%dd" % num_digits


def fill_patterns(string, patterns):
    '''Fills the patterns, if any, in the given string.

    Args:
        string: a string
        patterns: a dictionary of key -> replace pairs

    Returns:
        a copy of string with any patterns replaced
    '''
    for patt, val in iteritems(patterns):
        string = string.replace(patt, val)
    return string


def fill_config_patterns(string):
    '''Fills the patterns from ``eta.config.patterns``, if any, in the given
    string.

    Args:
        string: a string

    Returns:
        a copy of string with any patterns replaced
    '''
    return fill_patterns(string, eta.config.patterns)


def parse_kvps(kvps_str):
    '''Parses the comma-separated list of `key=value` pairs from the given
    string.

    Args:
        kvps_str: a string of the form `"key1=val1,key2=val2,..."

    Returns:
        a dict of key-value pair strings

    Raises:
        ValueError: if the string was invalid
    '''
    kvps = {}
    if kvps_str:
        try:
            for pair in kvps_str.split(","):
                k, v = pair.strip().split("=")
                kvps[k.strip()] = v.strip()
        except ValueError:
            raise ValueError("Invalid key-value pair string '%s'" % kvps_str)
    return kvps


def parse_categorical_string(value, choices, ignore_case=True):
    '''Parses a categorical string value, which must take a value from among
    the given choices.

    Args:
        value: the string to parse
        choices: either an iterable of possible values or an enum-like class
            whose attributes define the possible values
        ignore_case: whether to perform case insensitive matches. By default,
            this is True

    Returns:
        the raw (untouched) value of the given field

    Raises:
        ValueError: if the value was not an allowed choice
    '''
    if inspect.isclass(choices):
        choices = set(
            v for k, v in iteritems(vars(choices)) if not k.startswith("_"))

    orig_value = value
    orig_choices = choices
    if ignore_case:
        value = value.lower()
        choices = set(c.lower() for c in choices)

    if value not in choices:
        raise ValueError(
            "Unsupported value '%s'; choices are %s" % (orig_value, orig_choices))

    return orig_value


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
    '''Asks a yes/no question via the command-line and returns the answer.

    This function is case insensitive and partial matches are allowed.

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
        choice = six.moves.input().lower()
        if default and not choice:
            return valid[default]
        if choice in valid:
            return valid[choice]
        print("Please respond with 'y[es]' or 'n[o]'")


def call(args):
    '''Runs the command via `subprocess.call()`.

    stdout and stderr are streamed live during execution. If you want to
    capture these streams, use `communicate()`.

    Args:
        args: the command specified as a ["list", "of", "strings"]

    Returns:
        True/False: if the command executed successfully
    '''
    return subprocess.call(args) == 0


def communicate(args, decode=False):
    '''Runs the command via `subprocess.communicate()`

    Args:
        args: the command specified as a ["list", "of", "strings"]
        decode: whether to decode the output bytes into utf-8 strings. By
            default, the raw bytes are returned

    Returns:
        True/False: if the command executed successfully
        out: the command's stdout
        err: the command's stderr
    '''
    logger.debug("Executing '%s'", " ".join(args))
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if decode:
        out = out.decode()
        err = err.decode()
    return p.returncode == 0, out, err


def communicate_or_die(args, decode=False):
    '''Wrapper around `communicate()` that raises an exception if any error
    occurs.

    Args:
        args: the command specified as a ["list", "of", "strings"]
        decode: whether to decode the output bytes into utf-8 strings. By
            default, the raw bytes are returned

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

        raise


class Timer(object):
    '''Class for timing things that supports the context manager interface.

    Example usage:
        ```
        with Timer() as t:
            # your commands here

        print("Request took %s" % t.elapsed_time_str)
        ```
    '''

    def __init__(self):
        '''Creates a Timer instance.'''
        self.start_time = None
        self.stop_time = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    @property
    def elapsed_time(self):
        '''The number of elapsed seconds.'''
        return self.stop_time - self.start_time

    @property
    def elapsed_time_str(self):
        '''The human-readable elapsed time string.'''
        return self.get_elapsed_time_str()

    def get_elapsed_time_str(self, decimals=1):
        '''Gets the elapsed time as a human-readable string.

        Args:
            decimals: the desired number of decimal points to show in the
                string. The default is 1
        '''
        return to_human_time_str(self.elapsed_time, decimals=decimals)

    def start(self):
        '''Starts the timer.'''
        self.start_time = timeit.default_timer()

    def stop(self):
        '''Stops the timer.'''
        self.stop_time = timeit.default_timer()


def guess_mime_type(filepath):
    '''Guess the MIME type for the given file path. If no reasonable guess can
    be determined, `application/octet-stream` is returned.

    Args:
        filepath: path to the file

    Returns:
        the MIME type string
    '''
    return mimetypes.guess_type(filepath)[0] or "application/octet-stream"


def read_file(inpath, binary=False):
    '''Reads the file from disk.

    Args:
        inpath: the path to the file to read
        binary: whether to read the file in binary mode. By default, this is
            False (text mode)

    Returns:
        the contents of the file as a string
    '''
    mode = "rb" if binary else "rt"
    with open(inpath, mode) as f:
        return f.read()


def write_file(str_or_bytes, outpath):
    '''Writes the given string/bytes to disk.

    If a string is provided, it is encoded via `.encode()`.

    Args:
        str_or_bytes: the string or bytes to write to disk
        outpath: the desired output filepath
    '''
    ensure_basedir(outpath)
    if is_str(str_or_bytes):
        str_or_bytes = str_or_bytes.encode()

    with open(outpath, "wb") as f:
        f.write(str_or_bytes)


def copy_file(inpath, outpath, check_ext=False):
    '''Copies the input file to the output location.

    The output location can be a filepath or a directory in which to write the
    file. The base output directory is created if necessary, and any existing
    file will be overwritten.

    Args:
        inpath: the input path
        outpath: the output location (file or directory)
        check_ext: whether to check if the extensions of the input and output
            paths match. Only applicable if the output path is not a directory

    Raises:
        OSError: if check_ext is True and the input and output paths have
            different extensions
    '''
    if not os.path.isdir(outpath) and check_ext:
        assert_same_extensions(inpath, outpath)
    ensure_basedir(outpath)
    communicate_or_die(["cp", inpath, outpath])


def link_file(filepath, linkpath, check_ext=False):
    '''Creates a hard link at the given location using the given file.

    The base output directory is created if necessary, and any existing file
    will be overwritten.

    Args:
        filepath: a file or directory
        linkpath: the desired symlink path
        check_ext: whether to check if the extensions (or lack thereof, for
            directories) of the input and output paths match

    Raises:
        OSError: if check_ext is True and the input and output paths have
            different extensions
    '''
    if check_ext:
        assert_same_extensions(filepath, linkpath)
    ensure_basedir(linkpath)
    if os.path.exists(linkpath):
        delete_file(linkpath)
    os.link(os.path.realpath(filepath), linkpath)


def symlink_file(filepath, linkpath, check_ext=False):
    '''Creates a symlink at the given location that points to the given file.

    The base output directory is created if necessary, and any existing file
    will be overwritten.

    Args:
        filepath: a file or directory
        linkpath: the desired symlink path
        check_ext: whether to check if the extensions (or lack thereof, for
            directories) of the input and output paths match

    Raises:
        OSError: if check_ext is True and the input and output paths have
            different extensions
    '''
    if check_ext:
        assert_same_extensions(filepath, linkpath)
    ensure_basedir(linkpath)
    if os.path.exists(linkpath):
        delete_file(linkpath)
    os.symlink(os.path.realpath(filepath), linkpath)


def move_file(inpath, outpath, check_ext=False):
    '''Moves the input file to the output location.

    The output location can be a filepath or a directory in which to move the
    file. The base output directory is created if necessary, and any existing
    file will be overwritten.

    Args:
        inpath: the input path
        outpath: the output location (file or directory)
        check_ext: whether to check if the extensions of the input and output
            paths match. Only applicable if the output path is not a directory

    Raises:
        OSError: if check_ext is True and the input and output paths have
            different extensions
    '''
    if not os.path.splitext(outpath)[1]:
        # Output location is a directory
        ensure_dir(outpath)
    else:
        # Output location is a file
        if check_ext:
            assert_same_extensions(inpath, outpath)
        ensure_basedir(outpath)
    communicate_or_die(["mv", inpath, outpath])


def move_dir(indir, outdir):
    '''Moves the input directory to the given output location.

    The base output directory is created, if necessary. Any existing directory
    will be deleted.

    Args:
        indir: the input directory
        outdir: the output directory to create
    '''
    if os.path.isdir(outdir):
        delete_dir(outdir)
    ensure_basedir(outdir)
    communicate_or_die(["mv", indir, outdir])


def partition_files(indir, outdir=None, num_parts=None, dir_size=None):
    '''Partitions the files in the input directory into the specified number
    of (equal-sized) subdirectories.

    Exactly one of `num_parts` and `dir_size` must be specified.

    Args:
        indir: the input directory of files to partition
        outdir: the output directory in which to create the subdirectories and
            move the files. By default, the input directory is manipulated
            in-place
        num_parts: the number of subdirectories to create. If omitted,
            `dir_size` must be provided
        dir_size: the number of files per subdirectory. If omitted,
            `num_parts` must be provided
    '''
    if not outdir:
        outdir = indir

    files = list_files(indir)

    num_files = len(files)
    if num_parts:
        dir_size = int(math.ceil(num_files / num_parts))
    elif dir_size:
        num_parts = int(math.ceil(num_files / dir_size))

    part_patt = "part-%%0%dd" % int(math.ceil(math.log10(num_parts)))
    for idx, f in enumerate(files):
        inpath = os.path.join(indir, f)
        chunk = 1 + idx // dir_size
        outpath = os.path.join(outdir, part_patt % chunk, f)
        move_file(inpath, outpath)


def copy_sequence(inpatt, outpatt, check_ext=False):
    '''Copies the input sequence to the output sequence.

    The base output directory is created if necessary, and any existing files
    will be overwritten.

    Args:
        inpatt: the input sequence
        outpatt: the output sequence
        check_ext: whether to check if the extensions of the input and output
            sequences match

    Raises:
        OSError: if check_ext is True and the input and output sequences have
            different extensions
    '''
    if check_ext:
        assert_same_extensions(inpatt, outpatt)
    for idx in parse_pattern(inpatt):
        copy_file(inpatt % idx, outpatt % idx)


def link_sequence(inpatt, outpatt, check_ext=False):
    '''Creates hard links at the given locations using the given sequence.

    The base output directory is created if necessary, and any existing files
    will be overwritten.

    Args:
        inpatt: the input sequence
        outpatt: the output sequence
        check_ext: whether to check if the extensions of the input and output
            sequences match

    Raises:
        OSError: if check_ext is True and the input and output sequences have
            different extensions
    '''
    if check_ext:
        assert_same_extensions(inpatt, outpatt)
    for idx in parse_pattern(inpatt):
        link_file(inpatt % idx, outpatt % idx)


def symlink_sequence(inpatt, outpatt, check_ext=False):
    '''Creates symlinks at the given locations that point to the given
    sequence.

    The base output directory is created if necessary, and any existing files
    will be overwritten.

    Args:
        inpatt: the input sequence
        outpatt: the output sequence
        check_ext: whether to check if the extensions of the input and output
            sequences match

    Raises:
        OSError: if check_ext is True and the input and output sequences have
            different extensions
    '''
    if check_ext:
        assert_same_extensions(inpatt, outpatt)
    for idx in parse_pattern(inpatt):
        symlink_file(inpatt % idx, outpatt % idx)


def move_sequence(inpatt, outpatt, check_ext=False):
    '''Moves the input sequence to the output sequence.

    The base output directory is created if necessary, and any existing files
    will be overwritten.

    Args:
        inpatt: the input sequence
        outpatt: the output sequence
        check_ext: whether to check if the extensions of the input and output
            sequences match

    Raises:
        OSError: if check_ext is True and the input and output sequences have
            different extensions
    '''
    if check_ext:
        assert_same_extensions(inpatt, outpatt)
    for idx in parse_pattern(inpatt):
        move_file(inpatt % idx, outpatt % idx)


def is_in_root_dir(path, rootdir):
    '''Determines if the given path is a file or subdirectory (any levels deep)
    within the given root directory.

    Args:
        path: the input path (relative or absolute)
        rootdir: the root directory
    '''
    path = os.path.abspath(path)
    rootdir = os.path.abspath(rootdir)
    return path.startswith(rootdir)


def copy_dir(indir, outdir):
    '''Copies the input directory to the output directory.

    The base output directory is created if necessary, and any existing output
    directory will be deleted.

    Args:
        indir: the input directory
        outdir: the output directory
    '''
    if os.path.isdir(outdir):
        communicate_or_die(["rm", "-rf", outdir])
    ensure_dir(outdir)

    for filepath in list_files(indir, include_hidden_files=True, sort=False):
        copy_file(
            os.path.join(indir, filepath),
            os.path.join(outdir, filepath)
        )

    for subdir in list_subdirs(indir):
        outsubdir = os.path.join(outdir, subdir)
        insubdir = os.path.join(indir, subdir)
        copy_dir(insubdir, outsubdir)


def delete_file(path):
    '''Deletes the file at the given path and recursively deletes any empty
    directories from the resulting directory tree.

    Args:
        path: the filepath
    '''
    communicate_or_die(["rm", "-f", path])
    try:
        os.removedirs(os.path.dirname(path))
    except OSError:
        # found a non-empty directory or directory with no write access
        pass


def delete_dir(dir_):
    '''Deletes the given directory and recursively deletes any empty
    directories from the resulting directory tree.

    Args:
        dir_: the directory path
    '''
    dir_ = os.path.normpath(dir_)
    communicate_or_die(["rm", "-rf", dir_])
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


def ensure_empty_dir(dirname, cleanup=False):
    '''Ensures that the given directory exists and is empty.

    Args:
        dirname: the directory path
        cleanup: whether to delete any existing directory contents. By default,
            this is False

    Raises:
        ValueError: if the directory is not empty and `cleanup` is False
    '''
    if os.path.isdir(dirname):
        if cleanup:
            delete_dir(dirname)
        elif os.listdir(dirname):
            raise ValueError("%s not empty" % dirname)

    ensure_dir(dirname)


def ensure_path(path):
    '''Ensures that the given path is ready for writing by deleting any
    existing file and ensuring that the base directory exists.

    Args:
        path: the filepath
    '''
    if os.path.isfile(path):
        logger.debug("Deleting '%s'", path)
        delete_file(path)

    ensure_basedir(path)


def ensure_basedir(path):
    '''Makes the base directory of the given path, if necessary.

    Args:
        path: the filepath
    '''
    ensure_dir(os.path.dirname(path))


def ensure_dir(dirname):
    '''Makes the given directory, if necessary.

    Args:
        dirname: the directory path
    '''
    if dirname and not os.path.isdir(dirname):
        logger.debug("Making directory '%s'", dirname)
        os.makedirs(dirname)


def has_extension(filename, *args):
    '''Determines whether the filename has any of the given extensions.

    Args:
        filename: a file name
        *args: extensions like ".txt" or ".json"

    Returns:
        True/False
    '''
    ext = os.path.splitext(filename)[1]
    return any(ext == a for a in args)


def have_same_extesions(*args):
    '''Determines whether all of the input paths have the same extension.

    Args:
        *args: filepaths

    Returns:
        True/False
    '''
    exts = [os.path.splitext(path)[1] for path in args]
    return exts[1:] == exts[:-1]


def assert_same_extensions(*args):
    '''Asserts that all of the input paths have the same extension.

    Args:
        *args: filepaths

    Raises:
        OSError: if all input paths did not have the same extension
    '''
    if not have_same_extesions(*args):
        raise OSError("Expected %s to have the same extensions" % str(args))


def split_path(path):
    '''Splits a path into a list of its individual parts.

    E.g. split_path("/path/to/file") = ["/", "path", "to", "file"]

    Taken from
    https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html

    Args:
        path: a path to a file or directory

    Returns:
        all_parts: the path split into its individual components (directory
            and file names)
    '''
    all_parts = []
    while True:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            all_parts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            all_parts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            all_parts.insert(0, parts[1])

    return all_parts


_TIME_UNITS = [
    "ns", "us", "ms", " second", " minute", " hour", " day", " week", " month",
    " year"]
_TIME_CONVERSIONS = [
    1000, 1000, 1000, 60, 60, 24, 7, 52 / 12, 12, float("inf")]
_TIME_IN_SECS = [
    1e-9, 1e-6, 1e-3, 1, 60, 3600, 86400, 606461.5384615385, 2628000, 31536000]
_TIME_PLURALS = [False, False, False, True, True, True, True, True, True, True]
_DECIMAL_UNITS = ["", "K", "M", "B", "T"]
_BYTES_UNITS = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
_BITS_UNITS = ["b", "Kb", "Mb", "Gb", "Tb", "Pb", "Eb", "Zb", "Yb"]


def to_human_time_str(num_seconds, decimals=1, max_unit=None):
    '''Converts the given number of seconds to a human-readable time string.

    The supported units are ["ns", "us", "ms", "second", "minute", "hour",
    "day", "week", "month", "year"].

    Examples:
        0.001 => "1ms"
        60 => "1 minute"
        65 => "1.1 minutes"
        60123123 => "1.9 years"

    Args:
        num_seconds: the number of seconds
        decimals: the desired number of decimal points to show in the string.
            The default is 1
        max_unit: an optional max unit, e.g., "hour", beyond which to stop
            converting to larger units, e.g., "day". By default, no maximum
            unit is used

    Returns:
        a human-readable time string like "1.5 minutes" or "20.1 days"
    '''
    if num_seconds == 0:
        return "0 seconds"

    if max_unit and not any(u.strip() == max_unit for u in _TIME_UNITS):
        logger.warning("Unsupported max_unit = %s; ignoring", max_unit)
        max_unit = None

    num = 1e9 * num_seconds  # start with smallest unit
    for unit, conv, plural in zip(
            _TIME_UNITS, _TIME_CONVERSIONS, _TIME_PLURALS):
        if abs(num) < conv:
            break
        if max_unit and unit.strip() == max_unit:
            break
        num /= conv

    # Convert to string with the desired number of decimals, UNLESS those
    # decimals are zeros, in which case they are removed
    str_fmt = "%." + str(decimals) + "f"
    num_only_str = (str_fmt % num).rstrip("0").rstrip(".")

    # Add units
    num_str = num_only_str + unit
    if plural and num_only_str != "1":
        num_str += "s"  # handle pluralization

    return num_str


def from_human_time_str(time_str):
    '''Parses the number of seconds from the given human-readable time string.

    The supported units are ["ns", "us", "ms", "second", "minute", "hour",
    "day", "week", "month", "year"].

    Examples:
        "1ms" => 0.001
        "1 minute" => 60.0
        "1.1 minutes" => 66.0
        "1.9 years" => 59918400.0

    Args:
        time_str: a human-readable time string

    Returns:
        the number of seconds
    '''
    # Handle unit == "" outside loop
    for idx in reversed(range(len(_TIME_UNITS))):
        unit = _TIME_UNITS[idx].strip()
        can_be_plural = _TIME_PLURALS[idx]
        if time_str.endswith(unit):
            return float(time_str[:-len(unit)]) * _TIME_IN_SECS[idx]

        if can_be_plural and time_str.endswith(unit + "s"):
            return float(time_str[:-(len(unit) + 1)]) * _TIME_IN_SECS[idx]

    return float(time_str)


def to_human_decimal_str(num, decimals=1, max_unit=None):
    '''Returns a human-readable string representation of the given decimal
    (base-10) number.

    Supported units are ["", "K", "M", "B", "T"].

    Examples:
        65 => "65"
        123456 => "123.5K"
        1e7 => "10M"

    Args:
        num: a number
        decimals: the desired number of digits after the decimal point to show.
            The default is 1
        max_unit: an optional max unit, e.g., "M", beyond which to stop
            converting to larger units, e.g., "B". By default, no maximum unit
            is used

    Returns:
        a human-readable decimal string
    '''
    if max_unit is not None and max_unit not in _DECIMAL_UNITS:
        logger.warning("Unsupported max_unit = %s; ignoring", max_unit)
        max_unit = None

    for unit in _DECIMAL_UNITS:
        if abs(num) < 1000:
            break
        if max_unit is not None and unit == max_unit:
            break
        num /= 1000

    str_fmt = "%." + str(decimals) + "f"
    return (str_fmt % num).rstrip("0").rstrip(".") + unit


def from_human_decimal_str(num_str):
    '''Parses the decimal number from the given human-readable decimal string.

    Supported units are ["", "K", "M", "B", "T"].

    Examples:
        "65" => 65.0
        "123.5K" => 123450.0
        "10M" => 1e7

    Args:
        num_str: a human-readable decimal string

    Returns:
        the decimal number
    '''
    # Handle unit == "" outside loop
    for idx in reversed(range(len(_DECIMAL_UNITS))[1:]):
        unit = _DECIMAL_UNITS[idx]
        if num_str.endswith(unit):
            return float(num_str[:-len(unit)]) * (1000 ** idx)

    return float(num_str)


def to_human_bytes_str(num_bytes, decimals=1, max_unit=None):
    '''Returns a human-readable string representation of the given number of
    bytes.

    Supported units are ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"].

    Examples:
        123 => "123B"
        123000 => "120.1KB"
        1024 ** 4 => "1TB"

    Args:
        num_bytes: a number of bytes
        decimals: the desired number of digits after the decimal point to show.
            The default is 1
        max_unit: an optional max unit, e.g., "TB", beyond which to stop
            converting to larger units, e.g., "PB". By default, no maximum
            unit is used

    Returns:
        a human-readable bytes string
    '''
    if max_unit is not None and max_unit not in _BYTES_UNITS:
        logger.warning("Unsupported max_unit = %s; ignoring", max_unit)
        max_unit = None

    for unit in _BYTES_UNITS:
        if abs(num_bytes) < 1024:
            break
        if max_unit is not None and unit == max_unit:
            break
        num_bytes /= 1024

    str_fmt = "%." + str(decimals) + "f"
    return (str_fmt % num_bytes).rstrip("0").rstrip(".") + unit


def from_human_bytes_str(bytes_str):
    '''Parses the number of bytes from the given human-readable bytes string.

    Supported units are ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"].

    Examples:
        "123B" => 123
        "120.1KB" => 122982
        "1TB" => 1024 ** 4

    Args:
        bytes_str: a human-readable bytes string

    Returns:
        the number of bytes
    '''
    for idx in reversed(range(len(_BYTES_UNITS))):
        unit = _BYTES_UNITS[idx]
        if bytes_str.endswith(unit):
            return int(float(bytes_str[:-len(unit)]) * 1024 ** idx)

    return int(bytes_str)


def to_human_bits_str(num_bits, decimals=1, max_unit=None):
    '''Returns a human-readable string representation of the given number of
    bits.

    Supported units are ["b", "Kb", "Mb", "Gb", "Tb", "Pb", "Eb", "Zb", "Yb"].

    Examples:
        123 => "123b"
        123000 => "120.1Kb"
        1024 ** 4 => "1Tb"

    Args:
        num_bits: a number of bits
        decimals: the desired number of digits after the decimal point to show.
            The default is 1
        max_unit: an optional max unit, e.g., "Tb", beyond which to stop
            converting to larger units, e.g., "Pb". By default, no maximum
            unit is used

    Returns:
        a human-readable bits string
    '''
    if max_unit is not None and max_unit not in _BITS_UNITS:
        logger.warning("Unsupported max_unit = %s; ignoring", max_unit)
        max_unit = None

    for unit in _BITS_UNITS:
        if abs(num_bits) < 1024:
            break
        if max_unit is not None and unit == max_unit:
            break
        num_bits /= 1024

    str_fmt = "%." + str(decimals) + "f"
    return (str_fmt % num_bits).rstrip("0").rstrip(".") + unit


def from_human_bits_str(bits_str):
    '''Parses the number of bits from the given human-readable bits string.

    Supported units are ["b", "Kb", "Mb", "Gb", "Tb", "Pb", "Eb", "Zb", "Yb"].

    Examples:
        "123b" => 123
        "120.1Kb" => 122982
        "1Tb" => 1024 ** 4

    Args:
        bits_str: a human-readable bits string

    Returns:
        the number of bits
    '''
    for idx in reversed(range(len(_BITS_UNITS))):
        unit = _BITS_UNITS[idx]
        if bits_str.endswith(unit):
            return int(float(bits_str[:-len(unit)]) * 1024 ** idx)

    return int(bits_str)


def _get_archive_format(archive_path):
    basepath, ext = os.path.splitext(archive_path)
    if basepath.endswith(".tar"):
        # Handle .tar.gz and .tar.bz
        basepath, ext2 = os.path.splitext(basepath)
        ext = ext2 + ext

    if ext == ".zip":
        return basepath, "zip"
    if ext == ".tar":
        return basepath, "tar"
    if ext in (".tar.gz", ".tgz"):
        return basepath, "gztar"
    if ext in (".tar.bz", ".tbz"):
        return basepath, "bztar"

    raise ValueError("Unsupported archive format '%s'" % archive_path)


def make_archive(dir_path, archive_path):
    '''Makes an archive containing the given directory.

    Supported formats include `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz`,
    and `.tbz`.

    Args:
        dir_path: the directory to archive
        archive_path: the path + filename of the archive to create
    '''
    outpath, format = _get_archive_format(archive_path)
    if format == "zip" and eta.is_python2():
        make_zip64(dir_path, archive_path)
        return

    rootdir, basedir = os.path.split(os.path.realpath(dir_path))
    shutil.make_archive(outpath, format, rootdir, basedir)


def make_tar(dir_path, tar_path):
    '''Makes a tarfile containing the given directory.

    Supported formats include `.tar`, `.tar.gz`, `.tgz`, `.tar.bz`, and `.tbz`.

    Args:
        dir_path: the directory to tar
        tar_path: the path + filename of the .tar.gz file to create
    '''
    make_archive(dir_path, tar_path)


def make_zip(dir_path, zip_path):
    '''Makes a zipfile containing the given directory.

    Python 2 users must use `make_zip64` when making large zip files.
    `shutil.make_archive` does not offer Zip64 in Python 2, and is therefore
    limited to 4GiB archives with less than 65,536 entries.

    Args:
        dir_path: the directory to zip
        zip_path: the path + filename of the zip file to create
    '''
    make_archive(dir_path, zip_path)


def make_zip64(dir_path, zip_path):
    '''Makes a zip file containing the given directory in Zip64 format.

    Args:
        dir_path: the directory to zip
        zip_path: the path with extension of the zip file to create
    '''
    dir_path = os.path.realpath(dir_path)
    rootdir = os.path.dirname(dir_path)
    with zf.ZipFile(zip_path, "w", zf.ZIP_DEFLATED, allowZip64=True) as f:
        for root, _, filenames in os.walk(dir_path):
            base = os.path.relpath(root, rootdir)
            for name in filenames:
                src_path = os.path.join(root, name)
                dest_path = os.path.join(base, name)
                f.write(src_path, dest_path)


def extract_archive(archive_path, outdir=None, delete_archive=False):
    '''Extracts the contents of an archive.

    Supported formats include `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz`,
    and `.tbz`.

    Args:
        archive_path: the path to the archive file
        outdir: the directory into which to extract the archive contents. By
            default, the directory containing the archive is used
        delete_archive: whether to delete the archive after extraction. By
            default, this is False
    '''
    #
    # One could use `shutil.unpack_archive` in Python 3...
    # https://docs.python.org/3/library/shutil.html#shutil.unpack_archive
    #
    if archive_path.endswith(".zip"):
        extract_zip(archive_path, outdir=outdir, delete_zip=delete_archive)
    else:
        extract_tar(archive_path, outdir=outdir, delete_tar=delete_archive)


def extract_zip(zip_path, outdir=None, delete_zip=False):
    '''Extracts the contents of a .zip file.

    Args:
        zip_path: the path to the zip file
        outdir: the directory into which to extract the zip contents. By
            default, the directory containing the zip file is used
        delete_zip: whether to delete the zip after extraction. By default,
            this is False
    '''
    outdir = outdir or os.path.dirname(zip_path) or "."

    with zf.ZipFile(zip_path, "r", allowZip64=True) as f:
        f.extractall(outdir)

    if delete_zip:
        delete_file(zip_path)


def extract_tar(tar_path, outdir=None, delete_tar=False):
    '''Extracts the contents of a tarfile.

    Supported formats include `.tar`, `.tar.gz`, `.tgz`, `.tar.bz`, and `.tbz`.

    Args:
        tar_path: the path to the tarfile
        outdir: the directory into which to extract the archive contents. By
            default, the directory containing the tar file is used
        delete_tar: whether to delete the tar archive after extraction. By
            default, this is False
    '''
    if tar_path.endswith(".tar"):
        fmt = "r:"
    elif tar_path.endswith(".tar.gz") or tar_path.endswith(".tgz"):
        fmt = "r:gz"
    elif tar_path.endswith(".tar.bz") or tar_path.endswith(".tbz"):
        fmt = "r:bz2"
    else:
        raise ValueError(
            "Expected file '%s' to have extension .tar, .tar.gz, .tgz,"
            ".tar.bz, or .tbz in order to extract it" % tar_path)

    outdir = outdir or os.path.dirname(tar_path) or "."
    with tarfile.open(tar_path, fmt) as f:
        f.extractall(path=outdir)

    if delete_tar:
        delete_file(tar_path)


def multiglob(*patterns, **kwargs):
    '''Returns an iterable over the glob mathces for multiple patterns.

    Note that if a given file matches multiple patterns that you provided, it
    will appear multiple times in the output iterable.

    Examples:
        Find all .py or .pyc files in a directory
        ```py
        multiglob(".py", ".pyc", root="/path/to/dir/*")
        ```

        Find all JSON files recursively in a given directory:
        ```py
        multiglob(".json", root="/path/to/dir/**/*")
        ```

    Args:
        *patterns: the patterns to search for
        root: an optional root path to be applied to all patterns. This root is
            directly prepended to each pattern; `os.path.join` is NOT used

    Returns:
        an iteratable over the glob matches
    '''
    root = kwargs.get("root", "")
    return it.chain.from_iterable(glob2.iglob(root + p) for p in patterns)


def list_files(dir_path, abs_paths=False, recursive=False,
               include_hidden_files=False, sort=True):
    '''Lists the files in the given directory, sorted alphabetically and
    excluding directories and hidden files.

    Args:
        dir_path: the path to the directory to list
        abs_paths: whether to return the absolute paths to the files. By
            default, this is False
        recursive: whether to recursively traverse subdirectories. By default,
            this is False
        include_hidden_files: whether to include dot files
        sort: whether to sort the list of files

    Returns:
        a sorted list of the non-hidden files in the directory
    '''
    if recursive:
        files = []
        for root, _, filenames in os.walk(dir_path):
            files.extend([
                os.path.relpath(os.path.join(root, f), dir_path)
                for f in filenames if not f.startswith(".")])
    else:
        files = [
            f for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
            and (not f.startswith(".") or include_hidden_files)]

    if sort:
        files = sorted(files)

    if abs_paths:
        basedir = os.path.abspath(os.path.realpath(dir_path))
        files = [os.path.join(basedir, f) for f in files]

    return files


def list_subdirs(dir_path, abs_paths=False, recursive=False):
    '''Lists the subdirectories in the given directory, sorted alphabetically
    and excluding hidden directories.

    Args:
        dir_path: the path to the directory to list
        abs_paths: whether to return the absolute paths to the dirs. By
            default, this is False
        recursive: whether to recursively traverse subdirectories. By default,
            this is False

    Returns:
        a sorted list of the non-hidden subdirectories in the directory
    '''
    if recursive:
        dirs = []
        for root, dirnames, _ in os.walk(dir_path):
            dirs.extend([
                os.path.relpath(os.path.join(root, d), dir_path)
                for d in dirnames if not d.startswith(".")])
    else:
        dirs = [
            d for d in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, d))
            and not d.startswith(".")]

    dirs = sorted(dirs)

    if abs_paths:
        basedir = os.path.abspath(os.path.realpath(dir_path))
        dirs = [os.path.join(basedir, d) for d in dirs]

    return dirs


def parse_pattern(patt):
    '''Inspects the files matching the given pattern and returns the numeric
    indicies of the sequence.

    Args:
        patt: a pattern with a one or more numeric sequences like
            "/path/to/frame-%05d.jpg" or `/path/to/clips/%02d-%d.mp4`

    Returns:
        a list (or list of tuples if the pattern contains multiple sequences)
            describing the numeric indices of the files matching the pattern.
            The indices are returned in alphabetical order of their
            corresponding files
    '''
    # Extract indices from exactly matching patterns
    inds = []
    for _, match, num_inds in _iter_pattern_matches(patt):
        idx = tuple(map(int, match.groups()))
        inds.append(idx[0] if num_inds == 1 else idx)

    return inds


def get_pattern_matches(patt):
    '''Returns a list of file paths matching the given pattern.

    Args:
        patt: a pattern with one or more numeric sequences like
            "/path/to/frame-%05d.jpg" or "/path/to/clips/%02d-%d.mp4"

    Returns:
        a list of file paths that match the pattern `patt`
    '''
    return [path for path, _, _ in _iter_pattern_matches(patt)]


def fill_partial_pattern(patt, vals):
    '''Partially fills a pattern with the given values.

    Only supports integer ("%05d", "%4d", or "%d") and string ("%s") patterns.

    Args:
        patt: a pattern with one or more numeric or string sequences like
            "/path/to/features/%05d.npy" or "/path/to/features/%s-%d.npy"
        vals: a tuple of values whose length matches the number of patterns in
            `patt`, with Nones as placeholders for patterns that should not be
            filled

    Returns:
        the partially filled pattern
    '''
    if not vals:
        return patt

    exp = re.compile(r"(%[0-9]*[ds])")
    chunks = re.split(exp, patt)
    for i, j in enumerate(range(1, len(chunks), 2)):
        if vals[i] is not None:
            chunks[j] = chunks[j] % vals[i]

    return "".join(chunks)


def _iter_pattern_matches(patt):
    def _glob_escape(s):
        return re.sub(r"([\*\?\[])", r"\\\1", s)

    # Use glob to extract approximate matches
    seq_exp = re.compile(r"(%[0-9]*d)")
    glob_str = re.sub(seq_exp, "*", _glob_escape(patt))
    files = sorted(glob.glob(glob_str))

    # Create validation functions
    seq_patts = re.findall(seq_exp, patt)
    fcns = [parse_int_sprintf_pattern(sp) for sp in seq_patts]
    full_exp, num_inds = re.subn(seq_exp, "(\\s*\\d+)", patt)

    # Iterate over exactly matching patterns and files
    for f in files:
        m = re.match(full_exp, f)
        if m and all(f(p) for f, p in zip(fcns, m.groups())):
            yield f, m, num_inds


def parse_bounds_from_pattern(patt):
    '''Inspects the files satisfying the given pattern and returns the minimum
    and maximum indices satisfying it.

    Args:
        patt: a pattern with a single numeric sequence like
            "/path/to/frames/frame-%05d.jpg"

    Returns:
        a (first, last) tuple describing the first and last indices satisfying
            the pattern, or (None, None) if no matches were found
    '''
    inds = parse_pattern(patt)
    if not inds or isinstance(inds[0], tuple):
        return None, None
    return min(inds), max(inds)


def parse_dir_pattern(dir_path):
    '''Inspects the contents of the given directory, returning the numeric
    pattern in use and the associated indexes.

    The numeric pattern is guessed by analyzing the first file (alphabetically)
    in the directory.

    For example, if the directory contains:
        "frame-00040-object-5.json"
        "frame-00041-object-6.json"
    then the pattern "frame-%05d-object-%d.json" will be inferred along with
    the associated indices [(40, 5), (41, 6)].

    Args:
        dir_path: the path to the directory to inspect

    Returns:
        a tuple containing:
            - the numeric pattern used in the directory (the full path), or
                None if the directory was empty or non-existent
            - a list (or list of tuples if the pattern contains multiple
                numbers) describing the numeric indices in the directory. The
                indices are returned in alphabetical order of their
                corresponding filenames. If no files were found or the
                directory was non-existent, an empty list is returned
    '''
    try:
        files = list_files(dir_path)
    except OSError:
        return None, []

    if not files:
        return None, []

    def _guess_patt(m):
        s = m.group()
        leading_zeros = len(s) - len(str(int(s)))
        return "%%0%dd" % len(s) if leading_zeros > 0 else "%d"

    # Guess pattern from first file
    name, ext = os.path.splitext(os.path.basename(files[0]))
    regex = re.compile(r"\d+")
    patt = os.path.join(dir_path, re.sub(regex, _guess_patt, name) + ext)

    # Parse pattern
    inds = parse_pattern(patt)

    return patt, inds


def parse_sequence_idx_from_pattern(patt):
    '''Extracts the (first) numeric sequence index from the given pattern.

    Args:
        patt: a pattern like "/path/to/frames/frame-%05d.jpg"

    Returns:
        the numeric sequence string like "%05d", or None if no sequence was
            found
    '''
    m = re.search("%[0-9]*d", patt)
    return m.group() if m else None


def parse_int_sprintf_pattern(patt):
    '''Parses the integer sprintf pattern and returns a function that can
    detect whether a string matches the pattern.

    Args:
        patt: a sprintf pattern like "%05d", "%4d", or "%d"

    Returns:
        a function that returns True/False whether a given string matches the
            input numeric pattern
    '''
    # zero-padded: e.g., "%05d"
    zm = re.match(r"%0(\d+)d$", patt)
    if zm:
        n = int(zm.group(1))

        def _is_zero_padded_int_str(s):
            try:
                num_digits = len(str(int(s)))
            except ValueError:
                return False
            if num_digits > n:
                return True
            return len(s) == n and s[:-num_digits] == "0" * (n - num_digits)

        return _is_zero_padded_int_str

    # whitespace-padded: e.g., "%5d"
    wm = re.match(r"%(\d+)d$", patt)
    if wm:
        n = int(wm.group(1))

        def _is_whitespace_padded_int_str(s):
            try:
                num_digits = len(str(int(s)))
            except ValueError:
                return False
            if num_digits > n:
                return True
            return len(s) == n and s[:-num_digits] == " " * (n - num_digits)

        return _is_whitespace_padded_int_str

    # tight: "%d"
    if patt == "%d":
        def _is_tight_int_str(s):
            try:
                return s == str(int(s))
            except ValueError:
                return False

        return _is_tight_int_str

    raise ValueError("Unsupported integer sprintf pattern '%s'" % patt)


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


def find_duplicate_files(path_list):
    '''Returns a list of lists of file paths from the input, that have
    identical contents to each other.

    Args:
        path_list: list of file paths in which to look for duplicate files

    Returns:
        duplicates: a list of lists, where each list contains a group of
            file paths that all have identical content. File paths in
            `path_list` that don't have any duplicates will not appear in
            the output.
    '''
    hash_buckets = _get_file_hash_buckets(path_list)

    duplicates = []
    for file_group in itervalues(hash_buckets):
        if len(file_group) >= 2:
            duplicates.extend(_find_duplicates_brute_force(file_group))

    return duplicates


def find_matching_file_pairs(path_list1, path_list2):
    '''Returns a list of pairs of paths that have identical contents, where
    the paths in each pair aren't from the same path list.

    Args:
        path_list1: list of file paths
        path_list2: another list of file paths

    Returns:
        pairs: a list of pairs of file paths that have identical content,
            where one member of the pair is from `path_list1` and the other
            member is from `path_list2`
    '''
    hash_buckets1 = _get_file_hash_buckets(path_list1)
    pairs = []
    for path in path_list2:
        with open(path, "rb") as f:
            content = f.read()
        candidate_matches = hash_buckets1.get(hash(content), [])
        for candidate_path in candidate_matches:
            if not _diff_paths(candidate_path, path, content2=content):
                pairs.append((candidate_path, path))

    return pairs


def _get_file_hash_buckets(path_list):
    hash_buckets = defaultdict(list)
    for path in path_list:
        if not os.path.isfile(path):
            logger.warning(
                "File '%s' is a directory or does not exist. "
                "Skipping.", path)
            continue

        with open(path, "rb") as f:
            hash_buckets[hash(f.read())].append(path)

    return hash_buckets


def _find_duplicates_brute_force(path_list):
    if len(path_list) < 2:
        return []

    candidate_file_path = path_list[0]
    candidate_duplicates = [candidate_file_path]
    with open(candidate_file_path, "rb") as f:
        candidate_content = f.read()

    remaining_paths = []
    for path in path_list[1:]:
        if _diff_paths(
                candidate_file_path, path, content1=candidate_content):
            remaining_paths.append(path)
        else:
            candidate_duplicates.append(path)

    duplicates = []
    if len(candidate_duplicates) >= 2:
        duplicates.append(candidate_duplicates)

    duplicates.extend(_find_duplicates_brute_force(remaining_paths))

    return duplicates


def _diff_paths(path1, path2, content1=None, content2=None):
    '''Returns whether or not the files at `path1` and `path2` are different
    without using hashing.

    Since hashing is not used, this is a good function to use when two paths
    are in the same hash bucket.

    Args:
        path1: path to the first file
        path2: path to the second file
        content1: optional binary contents of the file at `path1`. If not
            provided, the contents will be read from disk.
        content2: optional binary contents of the file at `path2`. If not
            provided, the contents will be read from disk.

    Returns:
        `True` if the files at `path1` and `path2` are different, otherwise
            returns `False`
    '''
    if os.path.normpath(path1) == os.path.normpath(path2):
        return False

    if content1 is None:
        with open(path1, "rb") as f:
            content1 = f.read()

    if content2 is None:
        with open(path2, "rb") as f:
            content2 = f.read()

    return content1 != content2


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


def make_temp_dir(basedir=None):
    '''Makes a temporary directory.

    Args:
        basedir: an optional directory in which to create the new directory

    Returns:
        the path to the temporary directory
    '''
    if not basedir:
        basedir = tempfile.gettempdir()

    ensure_dir(basedir)
    return tempfile.mkdtemp(dir=basedir)


class TempDir(object):
    '''Context manager that creates and destroys a temporary directory.'''

    def __init__(self, basedir=None):
        '''Creates a TempDir instance.

        Args:
            basedir: an optional base directory in which to create the temp dir
        '''
        self._basedir = basedir
        self._name = None

    def __enter__(self):
        self._name = make_temp_dir(basedir=self._basedir)
        return self._name

    def __exit__(self, *args):
        delete_dir(self._name)


class WorkingDir(object):
    '''Context manager that temporarily changes working directories.'''

    def __init__(self, working_dir):
        self._orig_dir = None
        self._working_dir = working_dir

    def __enter__(self):
        if self._working_dir:
            self._orig_dir = os.getcwd()
            os.chdir(self._working_dir)
        return self

    def __exit__(self, *args):
        if self._working_dir:
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
