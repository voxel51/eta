"""
Utility functions for working with zipped directories of files.

Note that this module is not intended to provide general purpose zip-related
utilities such as zipping and extracting arbitrary files and directories.
For such functionality, see `eta.core.utils`. Rather, this module is
specifically designed to automate the processing of parallel files and
directories that arise when ETA modules support both both single inputs or zip
files containing multiple inputs of the same type.

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

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import os
import shutil
from zipfile import ZipFile


def make_zip(zip_path):
    """Makes the given zip file by zipping the directory of the same base name.

    For example, if zip_path is `/path/to/dir.zip`, the created zip file will
    contain the `/path/to/dir` directory.

    Args:
        zip_path: the output zip file path
    """
    outpath = os.path.splitext(zip_path)[0]
    rootdir, basedir = os.path.split(outpath)
    shutil.make_archive(outpath, "zip", rootdir, basedir)


def extract_zip(zip_path):
    """Extracts the given zip file.

    Only the folder with the same name as the base zip name is extracted, and
    any hidden directories or files (starting with ".") within it are skipped.

    For example, if zip_path is `/path/to/dir.zip` it must contain a directory
    called `dir`, and its contents will be extracted to `/path/to/dir`.

    Args:
        zip_path: the input zip file path

    Returns:
        contents: a list of the top-level files and directories that were
            extracted from the zip file (i.e. files within subdirectories are
            omitted
    """
    zdir, zfile = os.path.split(zip_path)
    zname = os.path.splitext(zfile)[0] + os.sep
    contents = []
    with ZipFile(zip_path, "r") as zf:
        for f in zf.namelist():
            if _is_legal_name(f, zname):
                zf.extract(f, zdir)
                if _is_top_level_name(f, zname):
                    contents.append(os.path.join(zdir, f))

    return contents


def make_parallel_dirs(zip_path, ref_paths):
    """Makes a list of directories based on the given zip path that parallel
    the provided reference paths.

    For example, if zip_path is "/path/to/dir.zip" and
    ref_paths is ["/a/b/video1.mp4", "/a/b/video2.mp4"], then the parallel
    directories will be ["/path/to/dir/video1", "/path/to/dir/video2"].

    Args:
        zip_path: the zip file path for which to generate the parallel
            directories
        ref_paths: a list of reference paths, which may be files or directories
    """
    base = os.path.splitext(zip_path)[0]
    return [os.path.join(base, _get_basename_no_ext(p)) for p in ref_paths]


def make_parallel_files(zip_path, ref_files):
    """Makes a list of filepaths based on the given zip path that parallel
    the provided reference files.

    For example, if zip_path is "/path/to/dir.zip" and
    ref_files is ["/a/b/video1.mp4", "/a/b/video2.mp4"], then the parallel
    filepaths will be ["/path/to/dir/video1.mp4", "/path/to/dir/video2.mp4"].

    Args:
        zip_path: the zip file path for which to generate the parallel files
        ref_files: a list of reference filepaths
    """
    base = os.path.splitext(zip_path)[0]
    return [os.path.join(base, _get_basename(p)) for p in ref_files]


def make_parallel_paths(zip_path, ref_paths, filename):
    """Makes a list of paths based on the given zip path to parallel the
    provided reference paths.

    For example, if zip_path is "/path/to/dir.zip" and
    ref_paths is ["/a/b/video1.mp4", "/a/b/video2.mp4"] and
    filename is "%05d.png", then the parallel paths will be:
    ["/path/to/dir/video1/%05d.png", "/path/to/dir/video2/%05d.png"].

    Args:
        zip_path: the zip file path for which to generate the parallel paths
        ref_paths: a list of reference paths, which may be files or directories
        filename: a filename to give to each parallel path
    """
    parallel_dirs = make_parallel_dirs(zip_path, ref_paths)
    return [os.path.join(d, filename) for d in parallel_dirs]


def _is_legal_name(fname, zname):
    # Must start with zname
    if not fname.startswith(zname):
        return False

    # Must not contain hidden components
    parts = fname[len(zname) :].split(os.sep)
    return all(not part.startswith(".") for part in parts)


def _is_top_level_name(fname, zname):
    if fname == zname:
        return False

    # Must be top-level file or directory within zname
    seps = fname.count(os.sep)
    return (seps == 1) or (seps == 2 and fname.endswith(os.sep))


def _get_basename(path):
    return os.path.basename(path)


def _get_basename_no_ext(path):
    return os.path.splitext(_get_basename(path))[0]
