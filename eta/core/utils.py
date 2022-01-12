"""
Core system and file I/O utilities.

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
from future.utils import iteritems, itervalues
import six

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict, deque
from datetime import datetime
import dateutil.parser
import errno
import glob
import glob2
import hashlib
import importlib
import inspect

try:
    # Although StringIO.StringIO's handling of unicode vs bytes is imperfect,
    # we import it here for use when a text-buffer replacement for `print` in
    # Python 2.X is required
    from StringIO import StringIO as _StringIO  # Python 2
except ImportError:
    from io import StringIO as _StringIO  # Python 3

try:
    import urllib.parse as urlparse  # Python 3
except ImportError:
    import urlparse  # Python 2

import itertools as it
import logging
import math
import mimetypes
import numbers
import os
from packaging.requirements import Requirement
import patoolib
import pkg_resources
import pytz
import random
import re
import shutil
import signal
import string
import subprocess
import sys
import tarfile
import tempfile
import timeit
import types
import zipfile as zf

import eta
import eta.constants as etac


logger = logging.getLogger(__name__)


def is_str(val):
    """Returns True/False whether the given value is a string."""
    return isinstance(val, six.string_types)


def is_numeric(val):
    """Returns True/False whether the given value is numeric."""
    return isinstance(val, numbers.Number)


def is_container(val):
    """Returns True/False whether the given value is a container.

    Here "container" means any non-string iterable object.
    """
    if is_str(val):
        return False

    try:
        iter(val)
        return True
    except:
        return False


def standarize_strs(arg):
    """Standardizes any strings in the given object by casting them via
    `str()`. Dictionaries and lists are processed recursively.

    Args:
        arg: an object

    Returns:
        a copy (only if necessary) of the input object with any strings casted
            via str()
    """
    if isinstance(arg, dict):
        return {
            standarize_strs(k): standarize_strs(v) for k, v in iteritems(arg)
        }

    if isinstance(arg, list):
        return [standarize_strs(e) for e in arg]

    if is_str(arg):
        return str(arg)

    return arg


def summarize_long_str(s, max_len, mode="middle"):
    """Renders a shorter version of a long string (if necessary) to meet a
    given length requirement by replacing part of the string with "..."

    Args:
        s: a string
        max_len: the desired maximum length
        mode: the summary mode, which controls which portion of long strings
            are deleted. Supported values are ("first", "middle", "last"). The
            default is "middle"

    Returns:
        the summarized string
    """
    if len(s) <= max_len:
        return s

    _mode = mode.lower()

    if _mode == "first":
        return "... " + s[-(max_len - 4) :]

    if _mode == "middle":
        len1 = math.ceil(0.5 * (max_len - 5))
        len2 = math.floor(0.5 * (max_len - 5))
        return s[:len1] + " ... " + s[-len2:]

    if _mode == "last":
        return s[: (max_len - 4)] + " ..."

    raise ValueError("Unsupported mode '%s'" % mode)


def get_localtime():
    """Gets the local time in "YYYY-MM-DD HH:MM:SS" format.

    Returns:
        "YYYY-MM-DD HH:MM:SS"
    """
    return str(datetime.now().replace(microsecond=0))


def parse_isotime(isostr_or_none):
    """Parses the ISO time string into a datetime.

    If the input string has a timezone ("Z" or "+HH:MM"), a timezone-aware
    datetime will be returned. Otherwise, a naive datetime will be returned.
    If the input is falsey, None is returned.

    Args:
        isostr_or_none: an ISO time string like "YYYY-MM-DD HH:MM:SS", or None

    Returns:
        a datetime, or None if the input was empty
    """
    if not isostr_or_none:
        return None

    return dateutil.parser.parse(isostr_or_none)


def datetime_delta_seconds(time1, time2):
    """Computes the difference between the two datetimes, in seconds.

    If one (but not both) of the datetimes are timezone-aware, the other
    datetime is assumed to be expressed in UTC time.

    Args:
        time1: a datetime
        time2: a datetime

    Returns:
        the time difference, in seconds
    """
    try:
        return (time2 - time1).total_seconds()
    except (TypeError, ValueError):
        time1 = add_utc_timezone_if_necessary(time1)
        time2 = add_utc_timezone_if_necessary(time2)
        return (time2 - time1).total_seconds()


def to_naive_local_datetime(dt):
    """Converts the datetime to a naive (no timezone) datetime with its time
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
    """
    dt = add_utc_timezone_if_necessary(dt)
    return dt.astimezone().replace(tzinfo=None)


def to_naive_utc_datetime(dt):
    """Converts the datetime to a naive (no timezone) datetime with its time
    expressed in UTC.

    The conversion is performed as follows:
        (1a) if the input datetime has no timezone, assume it is UTC
        (1b) if the input datetime has a timezone, convert to UTC
         (2) remove the timezone info

    Args:
        dt: a datetime

    Returns:
        a naive datetime in UTC
    """
    dt = add_utc_timezone_if_necessary(dt)
    return dt.astimezone(pytz.utc).replace(tzinfo=None)


def add_local_timezone_if_necessary(dt):
    """Makes the datetime timezone-aware, if necessary, by setting its timezone
    to the local timezone.

    Args:
        dt: a datetime

    Returns:
        a timezone-aware datetime
    """
    if dt.tzinfo is None:
        dt = dt.astimezone()  # empty ==> local timezone

    return dt


def add_utc_timezone_if_necessary(dt):
    """Makes the datetime timezone-aware, if necessary, by setting its timezone
    to UTC.

    Args:
        dt: a datetime

    Returns:
        a timezone-aware datetime
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.utc)

    return dt


def get_eta_rev():
    """Returns the hash of the last commit to the current ETA branch or "" if
    something went wrong with git.

    Returns:
        the current ETA revision hash
    """
    with WorkingDir(etac.ETA_DIR):
        success, rev, _ = communicate(
            ["git", "rev-parse", "HEAD"], decode=True
        )
    return rev.strip() if success else ""


def has_gpu():
    """Determine if the current device has a GPU.

    Returns:
        True/False
    """
    if sys.platform == "darwin":
        # No GPU on mac
        return False
    try:
        return "NVIDIA" in communicate(["lspci"], decode=True)[1]
    except OSError:
        # couldn't find lspci command...
        return False


def get_int_pattern_with_capacity(max_number, zero_padded=True):
    """Gets an integer pattern like "%%03d" or "%%4d" with sufficient capacity
    for the given number.

    Args:
        max_number: the maximum number you intend to pass to the pattern
        zero_padded: whether to left-pad with zeros. By default, this is True

    Returns:
        an integer formatting pattern
    """
    num_digits = max(1, math.ceil(math.log10(1 + max_number)))
    if zero_padded:
        return "%%0%dd" % num_digits

    return "%%%dd" % num_digits


def fill_patterns(string, patterns):
    """Fills the patterns, if any, in the given string.

    Args:
        string: a string
        patterns: a dictionary of key -> replace pairs

    Returns:
        a copy of string with any patterns replaced
    """
    if string is None:
        return None

    for patt, val in iteritems(patterns):
        string = string.replace(patt, val)

    return string


def fill_config_patterns(string):
    """Fills the patterns from ``eta.config.patterns``, if any, in the given
    string.

    Args:
        string: a string

    Returns:
        a copy of string with any patterns replaced
    """
    return fill_patterns(string, eta.config.patterns)


def parse_kvps(kvps_str):
    """Parses the comma-separated list of `key=value` pairs from the given
    string.

    Args:
        kvps_str: a string of the form `"key1=val1,key2=val2,..."

    Returns:
        a dict of key-value pair strings

    Raises:
        ValueError: if the string was invalid
    """
    kvps = {}
    if kvps_str:
        try:
            for pair in kvps_str.split(","):
                k, v = remove_escape_chars(pair, ",").strip().split("=")
                kvps[k.strip()] = v.strip()

        except ValueError:
            raise ValueError("Invalid key-value pair string '%s'" % kvps_str)

    return kvps


def parse_categorical_string(value, choices, ignore_case=True):
    """Parses a categorical string value, which must take a value from among
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
    """
    if inspect.isclass(choices):
        choices = set(
            v for k, v in iteritems(vars(choices)) if not k.startswith("_")
        )

    orig_value = value
    orig_choices = choices
    if ignore_case:
        value = value.lower()
        choices = set(c.lower() for c in choices)

    if value not in choices:
        raise ValueError(
            "Unsupported value '%s'; choices are %s"
            % (orig_value, orig_choices)
        )

    return orig_value


def parse_bool(val):
    """Parses the boolean value as per the table below.

    | Input                                         | Output  |
    | --------------------------------------------- | ------- |
    | None, "None", ""                              | None    |
    | True, 1, "t", "T", "true", "True", "TrUe"     | True    |
    | False, 0, "f", "F", "false", "False", "FaLsE" | False   |

    Args:
        val: the value to parse

    Returns:
        True, False, or None

    Raises:
        ValueError: if the provided value is not a valid boolean representation
    """
    if val is None:
        return None

    if isinstance(val, bool):
        return val

    if is_str(val):
        val = val.lower()

        if val in ("", "none"):
            return None

        if val in ("t", "true", "1"):
            return True

        if val in ("f", "false", "0"):
            return False

    if is_numeric(val):
        if val == 0:
            return False

        if val == 1:
            return True

    raise ValueError("Failed to parse boolean from '%s'" % val)


class FunctionEnum(object):
    """Base class for enums that support string-based lookup into a set of
    functions.

    Subclasses must implement the `_FUNCTIONS_MAP` constant.
    """

    #
    # A dictionary mapping string values to functions
    #
    # Subclasses MUST implement this constant
    #
    _FUNCTIONS_MAP = {}

    @classmethod
    def get_function(cls, value):
        """Gets the function for the given value.

        Args:
            value: the FunctionEnum value

        Returns:
            the function
        """
        cls.validate_value(value)
        return cls._FUNCTIONS_MAP[value]

    @classmethod
    def is_valid_value(cls, value):
        """Determines whether the given value is valid.

        Args:
            value: the FunctionEnum value

        Returns:
            True/False
        """
        try:
            cls.validate_value(value)
            return True
        except ValueError:
            return False

    @classmethod
    def validate_value(cls, value):
        """Validates that the given value is valid.

        Args:
            value: the FunctionEnum value

        Raises:
            ValueError: if the value is invalid
        """
        if value not in cls._FUNCTIONS_MAP:
            raise ValueError(
                "'%s' is not a valid value for %s; supported values are %s"
                % (value, get_class_name(cls), list(cls._FUNCTIONS_MAP))
            )


def get_class_name(cls_or_obj):
    """Returns the fully-qualified class name for the given input, which can
    be a class or class instance.

    Args:
        cls_or_obj: a class or class instance

    Returns:
        the fully-qualified class name string, such as
            "eta.core.utils.ClassName"
    """
    cls = cls_or_obj if inspect.isclass(cls_or_obj) else cls_or_obj.__class__
    return cls_or_obj.__module__ + "." + cls.__name__


def get_function_name(fcn):
    """Returns the fully-qualified function name for the given function.

    Args:
        fcn: a function

    Returns:
        the fully-qualified function name string, such as
            "eta.core.utils.function_name"
    """
    return fcn.__module__ + "." + fcn.__name__


def get_class(class_name, module_name=None):
    """Returns the class specified by the given class string, loading the
    parent module if necessary.

    Args:
        class_name: the "ClassName" or a fully-qualified class name like
            "eta.core.utils.ClassName"
        module_name: the fully-qualified module name like "eta.core.utils", or
            None if class_name includes the module name. Set module_name to
            __name__ to load a class from the calling module

    Returns:
        the class

    Raises:
        ImportError: if the class could not be imported
    """
    if module_name is None:
        try:
            module_name, class_name = class_name.rsplit(".", 1)
        except ValueError:
            raise ImportError(
                "Class name '%s' must be fully-qualified when no module "
                "name is provided" % class_name
            )

    __import__(module_name)  # does nothing if module is already imported
    return getattr(sys.modules[module_name], class_name)


def get_function(function_name, module_name=None):
    """Returns the function specified by the given string.

    Loads the parent module if necessary.

    Args:
        function_name: local function name by string fully-qualified name
            like "eta.core.utils.get_function"
        module_name: the fully-qualified module name like "eta.core.utils", or
            None if function_name includes the module name. Set module_name to
            __name__ to load a function from the calling module

    Returns:
        the function

    Raises:
        ImportError: if the function could not be imported
    """
    return get_class(function_name, module_name=module_name)


def install_package(
    requirement_str, error_level=0, error_msg=None, error_suffix=None,
):
    """Installs the newest compliant version of the package.

    Args:
        requirement_str: a PEP 440 compliant package requirement, like
            "tensorflow", "tensorflow<2", "tensorflow==2.3.0", or
            "tensorflow>=1.13,<1.15"
        error_level: the error level to use, defined as:

            0: raise error if the install fails
            1: log warning if the install fails
            2: ignore install fails

        error_msg: an optional error message to use if the installation fails
        error_suffix: an optional message to append to the error if the
            installation fails and ``error_level == 0``

    Returns:
        True/False whether the requirement was successfully installed
    """
    if "|" in requirement_str:
        full_requirement_str = requirement_str
        requirement_str = full_requirement_str.split("|", 1)[0]
        logger.warning(
            "***** Requirement '%s' has multiple options. We're going to "
            "install the first one: '%s' *****",
            full_requirement_str,
            requirement_str,
        )

    args = [sys.executable, "-m", "pip", "install", requirement_str]
    p = subprocess.Popen(args, stderr=subprocess.PIPE)
    _, err = p.communicate()

    success = p.returncode == 0

    if not success:
        if error_msg is None:
            error_msg = "Failed to install package '%s'\n\n%s" % (
                requirement_str,
                err.decode(),
            )

        if error_suffix is not None and error_level <= 0:
            error_msg += "\n" + error_suffix

        handle_error(PackageError(error_msg), error_level)

    return success


def ensure_package(
    requirement_str,
    error_level=0,
    error_msg=None,
    error_suffix=None,
    log_success=False,
):
    """Ensures that the given package is installed.

    This function uses `pkg_resources.get_distribution` to locate the package
    by its pip name and does not actually import the module.

    Therefore, unlike `ensure_import()`, `requirement_str` should refer to the
    package name (e.g., "tensorflow-gpu"), not the module name
    (e.g., "tensorflow").

    Args:
        requirement_str: a PEP 440 compliant package requirement, like
            "tensorflow", "tensorflow<2", "tensorflow==2.3.0", or
            "tensorflow>=1.13,<1.15". This can also be an iterable of multiple
            requirements, all of which must be installed, or this can be a
            single "|"-delimited string specifying multiple requirements, at
            least one of which must be installed
        error_level: the error level to use, defined as:

            0: raise error if requirement is not satisfied
            1: log warning if requirement is not satisifed
            2: ignore unsatisifed requirements

        error_msg: an optional error message to use if the requirement is not
            satisifed
        error_suffix: an optional message to append to the error if the
            requirement is not satisifed and ``error_level == 0``
        log_success: whether to generate a log message if the requirement is
            satisifed

    Returns:
        True/False whether the requirement is installed
    """
    parse_requirement = lambda req_str: _get_package_version(req_str)
    error_cls = PackageError

    return _ensure_requirements(
        requirement_str,
        parse_requirement,
        error_level,
        error_cls,
        error_msg,
        error_suffix,
        log_success,
    )


class PackageError(ImportError):
    """Exception raised when a requested package is not installed."""


def _get_package_version(requirement_str):
    req = Requirement(requirement_str)

    try:
        version = pkg_resources.get_distribution(req.name).version
        error = None
    except pkg_resources.DistributionNotFound as e:
        version = None
        error = e

    return req, version, error


def ensure_import(
    requirement_str,
    error_level=0,
    error_msg=None,
    error_suffix=None,
    log_success=False,
):
    """Ensures that the given package is installed and importable.

    This function imports the module the specified name and optionally enforces
    any version requirements included in `requirement_str`.

    Therefore, unlike `ensure_package()`, `requirement_str` should refer to the
    module name (e.g., "tensorflow"), not the package name (e.g.,
    "tensorflow-gpu").

    Args:
        requirement_str: a PEP 440-like module requirement, like "tensorflow",
            "tensorflow<2", "tensorflow==2.3.0", or "tensorflow>=1.13,<1.15".
            This can also be an iterable of multiple requirements, all of which
            must be installed, or this can be a single "|"-delimited string
            specifying multiple requirements, at least one of which must be
            installed
        error_level: the error level to use, defined as:

            0: raise error if requirement is not satisfied
            1: log warning if requirement is not satisifed
            2: ignore unsatisifed requirements

        error_msg: an optional error message to use if the requirement is not
            satisifed
        error_suffix: an optional message to append to the error if the
            requirement is not satisifed and ``error_level == 0``
        log_success: whether to generate a log message if the requirement is
            satisifed

    Returns:
        True/False whether the package is installed and importable
    """
    parse_requirement = lambda req_str: _get_module_version(
        req_str, error_level
    )
    error_cls = ImportError

    return _ensure_requirements(
        requirement_str,
        parse_requirement,
        error_level,
        error_cls,
        error_msg,
        error_suffix,
        log_success,
    )


def _get_module_version(requirement_str, error_level):
    req = Requirement(requirement_str)

    try:
        mod = importlib.import_module(req.name)
        version = getattr(mod, "__version__", None)
        error = None
    except ImportError as e:
        version = None
        error = e

    if error is None and version is None and req.specifier:
        handle_error(
            ImportError(
                "Unable to determine the installed version of '%s'; required "
                "'%s'" % (req.name, req)
            ),
            error_level + 1,
        )

    return req, version, error


def _ensure_requirements(
    requirement_str,
    parse_requirement,
    error_level,
    error_cls,
    error_msg,
    error_suffix,
    log_success,
):
    if is_str(requirement_str):
        # Require any
        if "|" in requirement_str:
            results = [
                parse_requirement(req_str)
                for req_str in requirement_str.split("|")
            ]
            return _ensure_any_requirement(
                results,
                error_level,
                log_success,
                error_cls=error_cls,
                error_msg=error_msg,
                error_suffix=error_suffix,
            )

        requirement_str = [requirement_str]

    # Require all
    results = [parse_requirement(req_str) for req_str in requirement_str]
    return _ensure_all_requirements(
        results,
        error_level,
        log_success,
        error_cls=error_cls,
        error_msg=error_msg,
        error_suffix=error_suffix,
    )


def _ensure_all_requirements(
    results,
    error_level,
    log_success,
    error_cls=ImportError,
    error_msg=None,
    error_suffix=None,
):
    successes = []
    fails = []
    for req, version, error in results:
        if error is not None or (
            req.specifier
            and (version is None or not req.specifier.contains(version))
        ):
            fails.append((req, version, error))
        else:
            successes.append((req, version))

    if not fails:
        if log_success:
            for req, version in successes:
                ver = version or "???"
                logger.info(
                    "Requirement satisfied: %s (found %s)", str(req), ver
                )

        return True

    req_strs = []
    found_strs = []
    last_error = None
    for req, version, error in fails:
        req_strs.append(str(req))

        if error is None and version is None:
            found_strs.append("%s==???" % req.name)

        if version is not None:
            found_strs.append("%s==%s" % (req.name, version))

        if error is not None:
            last_error = error

    if error_msg is None:
        num_req = len(req_strs)
        num_found = len(found_strs)

        if num_req == 1:
            reqs = "'%s' is" % req_strs[0]
        else:
            reqs = "%s are" % (req_strs,)

        if num_found == 1:
            founds = ", but found '%s'." % found_strs[0]
        elif num_found > 1:
            founds = ", but found %s." % (found_strs,)
        else:
            founds = "."

        error_msg = (
            "The requested operation requires that %s installed on "
            "your machine%s" % (reqs, founds)
        )

    if error_suffix is not None and error_level <= 0:
        error_msg += "\n\n" + error_suffix

    handle_error(error_cls(error_msg), error_level, base_error=last_error)

    return False


def _ensure_any_requirement(
    results,
    error_level,
    log_success,
    error_cls=ImportError,
    error_msg=None,
    error_suffix=None,
):
    successes = []
    fails = []
    for req, version, error in results:
        if error is not None or (
            req.specifier
            and (version is None or not req.specifier.contains(version))
        ):
            fails.append((req, version, error))
        else:
            successes.append((req, version))

    if successes:
        if log_success:
            for req, version in successes:
                ver = version or "???"
                logger.info(
                    "Requirement satisfied: %s (found %s)", str(req), ver
                )

        return True

    req_strs = []
    found_strs = []
    last_error = None
    for req, version, error in fails:
        req_strs.append(str(req))

        if error is None and version is None:
            found_strs.append("%s==???" % req.name)

        if version is not None:
            found_strs.append("%s==%s" % (req.name, version))

        if error is not None:
            last_error = error

    if error_msg is None:
        req_str = "\n".join("-   %s" % r for r in req_strs)
        found_str = "\n".join("-   %s" % f for f in found_strs)
        error_msg = (
            "The requested operation requires that one of the following is "
            "installed on your machine:\n%s\n\nbut found:\n%s"
        ) % (req_str, found_str)

    if error_suffix is not None and error_level <= 0:
        error_msg += "\n\n" + error_suffix

    handle_error(error_cls(error_msg), error_level, base_error=last_error)

    return False


def handle_error(error, error_level, base_error=None):
    """Handles the error at the specified error level.

    Args:
        error: an Exception instance
        error_level: the error level to use, defined as:

            0: raise the error
            1: log the error as a warning
            2: ignore the error

        base_error: (optional) a base Exception from which to raise `error`
    """
    if error_level <= 0:
        if base_error is not None:
            six.raise_from(error, base_error)
        else:
            raise error

    if error_level == 1:
        logger.warning(error)


def ensure_cuda_version(
    requirement_str,
    error_level=0,
    error_msg=None,
    error_suffix=None,
    log_success=False,
):
    """Ensures that a compliant version of CUDA is installed.

    Args:
        requirement_str: the version component of a PEP 440 compliant package
            requirement, like "", "<10", "==9.1.0", or ">=9,<10"
        error_level: the error level to use, defined as:

            0: raise error if the requirement is not satisfied
            1: log warning if the requirement is not satisifed
            2: ignore unsatisifed requirements

        error_msg: an optional error message to use if the requirement is not
            satisifed
        error_suffix: an optional message to append to the error if the
            requirement is not satisifed and ``error_level == 0``
        log_success: whether to generate a log message if the requirement is
            satisifed

    Returns:
        True/False whether a compliant CUDA version was found
    """
    req = Requirement("CUDA" + requirement_str)
    version = get_cuda_version()
    if version is None:
        error = CUDAError("CUDA not found")
    else:
        error = None

    return _ensure_all_requirements(
        [(req, version, error)],
        error_level,
        log_success,
        error_cls=CUDAError,
        error_msg=error_msg,
        error_suffix=error_suffix,
    )


def ensure_cudnn_version(
    requirement_str,
    error_level=0,
    error_msg=None,
    error_suffix=None,
    log_success=False,
):
    """Ensures that a compliant version of cuDNN is installed.

    Args:
        requirement_str: the version component of a PEP 440 compliant package
            requirement, like "", "<8", "==7.5.1", or ">=7,<8"
        error_level: the error level to use, defined as:

            0: raise error if the requirement is not satisfied
            1: log warning if the requirement is not satisifed
            2: ignore unsatisifed requirements

        error_msg: an optional error message to use if the requirement is not
            satisifed
        error_suffix: an optional message to append to the error if the
            requirement is not satisifed and ``error_level == 0``
        log_success: whether to generate a log message if the requirement is
            satisifed

    Returns:
        True/False whether a compliant CUDA version was found
    """
    req = Requirement("cuDNN" + requirement_str)
    version = get_cudnn_version()
    if version is None:
        error = CUDAError("cuDNN not found")
    else:
        error = None

    return _ensure_all_requirements(
        [(req, version, error)],
        error_level,
        log_success,
        error_cls=CUDAError,
        error_msg=error_msg,
        error_suffix=error_suffix,
    )


class CUDAError(Exception):
    """An error raised when a problem with CUDA installation occurs."""

    pass


def get_cuda_version():
    """Gets the CUDA version installed on the machine, if possible.

    The `CUDA_HOME` environment variable will be used, if set, to locate the
    CUDA installation.

    Returns:
        the CUDA version string, or None if CUDA is not installed or the
        installation could not be located
    """
    cuda_home_dir = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cuda_version_path = os.path.join(cuda_home_dir, "version.txt")
    if not os.path.isfile(cuda_version_path):
        return None

    contents = read_file(cuda_version_path)
    return contents[len("CUDA Version ") :].strip()


def get_cudnn_version():
    """Gets the cuDNN version installed on the machine, if possible.

    The `CUDA_HOME` environment variable will be used, if set, to locate the
    CUDA installation.

    Returns:
        the cuDNN version string, or None if cuDNN is not installed or the
        installation could not be located
    """
    cuda_home_dir = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cudnn_header_path = os.path.join(cuda_home_dir, "include", "cudnn.h")
    if not os.path.isfile(cudnn_header_path):
        return None

    major = None
    minor = None
    patch = None
    with open(cudnn_header_path, "rt") as f:
        for line in f.readlines():
            if line.startswith("#define CUDNN_MAJOR"):
                major = line[len("#define CUDNN_MAJOR") :].strip()

            if line.startswith("#define CUDNN_MINOR"):
                minor = line[len("#define CUDNN_MINOR") :].strip()

            if line.startswith("#define CUDNN_PATCHLEVEL"):
                patch = line[len("#define CUDNN_PATCHLEVEL") :].strip()

    ver = [v for v in (major, minor, patch) if v is not None]
    return ".".join(ver)


def lazy_import(module_name, callback=None, error_msg=None):
    """Returns a proxy module object that will lazily import the given module
    the first time it is used.

    Example usage::

        import eta.core.utils as etau

        # Lazy version of `import tensorflow as tf`
        tf = etau.lazy_import("tensorflow")

        # Other commands

        # Now the module is loaded
        tf.__version__

    Args:
        module_name: the fully-qualified module name to import
        callback (None): a callback function to call before importing the
            module
        error_msg (None): an error message to print if the import fails

    Returns:
        a LazyModule
    """
    return LazyModule(module_name, callback=callback, error_msg=error_msg)


def lazy_object(_callable):
    """Returns a proxy object that will lazily be created by calling the
    provided callable the first time it is used.

    Example usage::

        #
        # Calls `import_tf1()` to import the TF 1.X namespace the first time
        # that `tf` is used
        #

        import eta.core.utils as etau

        def import_tf1():
            try:
                import tensorflow.compat.v1 as tf
            except:
                import tensorflow as tf

            return tf

        tf = etau.lazy_object(import_tf1)

    Args:
        _callable: a callable that returns the object when called

    Returns:
        a LazyObject
    """
    return LazyObject(_callable)


class LazyModule(types.ModuleType):
    """Proxy module that lazily imports the underlying module the first time it
    is actually used.

    Args:
        module_name: the fully-qualified module name to import
        callback (None): a callback function to call before importing the
            module
        error_msg (None): an error message to print if the import fails
    """

    def __init__(self, module_name, callback=None, error_msg=None):
        super(LazyModule, self).__init__(module_name)
        self._module = None
        self._callback = callback
        self._error_msg = error_msg

    def __getattr__(self, item):
        if self._module is None:
            self._import_module()

        return getattr(self._module, item)

    def __dir__(self):
        if self._module is None:
            self._import_module()

        return dir(self._module)

    def _import_module(self):
        # Execute callback, if any
        if self._callback is not None:
            self._callback()

        # Actually import the module
        try:
            module = importlib.import_module(self.__name__)
            self._module = module
        except ImportError as e:
            if self._error_msg is not None:
                six.raise_from(ImportError(self._error_msg), e)

            raise

        # Update this object's dict so that attribute references are efficient
        # (__getattr__ is only called on lookups that fail)
        self.__dict__.update(module.__dict__)


class LazyObject(object):
    """Proxy object that lazily constructs the object the first time it is
    actually used.

    Args:
        _callable: a callable that returns the object when called
    """

    def __init__(self, _callable):
        self._callable = _callable
        self._obj = None

    def __getattr__(self, attr):
        if self._obj is None:
            self._init()

        return getattr(self._obj, attr)

    def __dir__(self):
        if self._obj is None:
            self._init()

        return dir(self._obj)

    def _init(self):
        # Actually construct the object
        self._obj = self._callable()

        # Update this object's dict so that attribute references are efficient
        # (__getattr__ is only called on lookups that fail)
        self.__dict__.update(self._obj.__dict__)


def query_yes_no(question, default=None):
    """Asks a yes/no question via the command-line and returns the answer.

    This function is case insensitive and partial matches are allowed.

    Args:
        question: the question to ask
        default: the default answer, which can be "yes", "no", or None (a
            response is required). The default is None

    Returns:
        True/False whether the user replied "yes" or "no"

    Raises:
        ValueError: if the default value was invalid
    """
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


class CaptureStdout(object):
    """Class for temporarily capturing stdout.

    This class works by temporarily redirecting `sys.stdout` (and any stream
    handlers of the root logger that are streaming to `sys.stdout`) to a
    string buffer in between calls to `start()` and `stop()`.

    Example (suppressing stdout)::

        import eta.core.utils as etau

        print("foo")
        with etau.CaptureStdout():
            print("Hello, world!")

        print("bar")

    Example (capturing stdout)::

        import eta.core.utils as etau

        cap = etau.CaptureStdout()

        print("foo")
        with cap:
            print("Hello, world!")

        print("bar")
        print(cap.stdout)
    """

    def __init__(self):
        """Creates a CaptureStdout instance."""
        self._root_logger = logging.getLogger()
        self._orig_stdout = None
        self._cache_stdout = None
        self._handler_inds = None
        self._stdout_str = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    @property
    def is_capturing(self):
        """Whether stdout is currently being captured."""
        return self._cache_stdout is not None

    @property
    def stdout(self):
        """The stdout string captured by the last use of this instance, or None
        if no stdout has been captured.
        """
        return self._stdout_str

    def start(self):
        """Start capturing stdout."""
        if self.is_capturing:
            return

        self._stdout_str = None
        self._orig_stdout = sys.stdout
        self._cache_stdout = _StringIO()
        self._handler_inds = []

        # Update root logger handlers, if necessary
        for idx, handler in enumerate(self._root_logger.handlers):
            if isinstance(handler, logging.StreamHandler):
                if handler.stream == sys.stdout:
                    handler.stream = self._cache_stdout
                    self._handler_inds.append(idx)

        # Update `sys.stdout`
        sys.stdout.flush()
        sys.stdout = self._cache_stdout

    def stop(self):
        """Stop capturing stdout.

        Returns:
            a string containing the captured stdout
        """
        if not self.is_capturing:
            return ""

        self._stdout_str = self._cache_stdout.getvalue()
        self._cache_stdout.close()
        self._cache_stdout = None

        # Revert root logger handlers, if necessary
        for idx in self._handler_inds:
            self._root_logger.handlers[idx].stream = self._orig_stdout

        self._handler_inds = None

        # Revert `sys.stdout`
        sys.stdout = self._orig_stdout

        return self.stdout


class ProgressBar(object):
    """Class for printing a self-updating progress bar to stdout that tracks
    the progress of an iterative process towards completion.

    The simplest use of `ProgressBar` is to create an instance and then call it
    (`__call__`) with an iterable argument, which will pass through elements of
    the iterable when iterating over it, updating the progress bar each time an
    element is emitted.

    Alternatively, `ProgressBar`s can track the progress of a task towards a
    total iteration count, where the current iteration is updated manually by
    calling either `update()`, which will increment the iteration count by 1,
    or via `set_iteration()`, which lets you manually specify the new iteration
    status. By default, both methods will automatically redraw the bar. If
    manual control over drawing is required, you can pass `draw=False` to
    either method and then manually call `draw()` as desired.

    It is highly recommended that you always invoke `ProgressBar`s using their
    context manager interface via the `with` keyword, which will automatically
    handle calling `start()` and `close()` to appropriately initialize and
    finalize the task. Among other things, this ensures that stdout redirection
    is appropriately ended when an exception is encountered.

    When `start()` is called on the a `ProgressBar`, an internal timer is
    started to track the elapsed time of the task. In addition, stdout is
    automatically cached between calls to `draw()` and flushed each time
    `draw()` is called, which means that you can freely mix `print` statements
    into your task without interfering with the progress bar. When you are done
    tracking the task, call the `close()` method to finalize the progress bar.
    Both of these actions are automatically taken when the bar's context
    manager interface is invoked or when it is called with an iterable.

    If you want want to full manual control over the progress bar, call
    `start()` to start the task, call `pause()` before any `print` statements,
    and call `close()` when the task is finalized.

    `ProgressBar`s can optionally be configured to print any of the following
    statistics about the task:

    -   the elapsed time since the task was started
    -   the estimated time remaining until the task completes
    -   the current iteration rate of the task, in iterations per second
    -   customized status messages passed via the `suffix` argument

    Example (wrapping an iterator)::

        import time
        import eta.core.utils as etau

        with etau.ProgressBar() as pb:
            for _ in pb(range(100)):
                time.sleep(0.05)

    Example (with print statements interleaved)::

        import time
        import eta.core.utils as etau

        with etau.ProgressBar(100) as pb:
            while not pb.complete:
                if pb.iteration in {25, 50, 75}:
                    print("Progress = %.2f" % pb.progress)

                time.sleep(0.05)
                pb.update()

    Example (tracking a bit total)::

        import random
        import time
        import eta.core.utils as etau

        with etau.ProgressBar(1024 ** 2, use_bits=True) as pb:
            while not pb.complete:
                pb.update(random.randint(1, 10 * 1024))
                time.sleep(0.05)
    """

    def __init__(
        self,
        total=None,
        show_percentage=True,
        show_iteration=True,
        show_elapsed_time=True,
        show_remaining_time=True,
        show_iter_rate=True,
        iters_str="it",
        start_msg=None,
        complete_msg=None,
        use_bits=False,
        use_bytes=False,
        max_width=None,
        num_decimals=1,
        max_fps=10,
        quiet=False,
    ):
        """Creates a ProgressBar instance.

        Args:
            total: the total number of iterations to track, or an iterable that
                implements `len()` from which the total can be computed. If you
                intend to use this instance to track an iterable via the
                `__call__` syntax, then this argument can be omitted.
                Otherwise, if the provided value is non-numeric and doesn't
                implement `len()`, then a raw count with no progress bar will
                be displayed when this instance is used
            show_percentage: whether to show the percentage completion of the
                task. By default, this is True
            show_iteration: whether to show the current iteration count and the
                total (if available). By default, this is True
            show_elapsed_time: whether to print the elapsed time at the end of
                the progress bar. By default, this is False
            show_remaining_time: whether to print the estimated remaining time
                at the end of the progress bar. By default, this is False
            show_iter_rate: whether to show the current iterations per second
                being processed. By default, this is False
            iters_str: the string to print when `show_iter_rate == True`. The
                default is "it"
            start_msg: an optional message to log when the progress bar is
                started
            complete_msg: an optional message to log when the progress bar is
                complete when it is closed
            use_bits: whether to interpret `iteration` and `total` as numbers
                of bits when rendering iteration information. By default, this
                is False
            use_bytes: whether to interpret `iteration` and `total` as numbers
                of bytes when rendering iteration information. By default, this
                is False
            max_width: the maximum allowed with of the bar, in characters. By
                default, the bar is fitted to your Terminal window
            num_decimals: the number of decimal places to show when rendering
                times. The default is 1
            max_fps: the maximum allowed frames per second at which `draw()`
                will be executed. The default is 15
            quiet: whether to suppress printing of the bar
        """
        if not eta.config.show_progress_bars:
            quiet = True

        num_pct_decimals = 0

        self._total = self._get_total(total)
        self._iterator = None
        self._iteration = 0
        self._show_percentage = show_percentage
        self._show_iteration = show_iteration
        self._show_elapsed_time = show_elapsed_time
        self._show_remaining_time = show_remaining_time
        self._show_iter_rate = show_iter_rate
        self._use_bits = use_bits
        self._use_bytes = use_bytes
        self._iters_str = iters_str
        self._start_msg = start_msg
        self._complete_msg = complete_msg
        self._max_width = max_width
        self._has_dynamic_width = max_width is None and not quiet
        self._max_fps = max_fps
        self._timer = Timer(quiet=True)
        self._is_running = False
        self._last_draw_time = -1
        self._draw_times = deque([0], maxlen=10)
        self._draw_iters = deque([0], maxlen=10)
        self._num_decimals = num_decimals
        self._iter_fmt = None
        self._pct_fmt = " %%%d.%df%%%% " % (
            num_pct_decimals + 3 + bool(num_pct_decimals),
            num_pct_decimals,
        )
        self._max_len = 0
        self._spinner = it.cycle("|/-\\|/-\\")
        self._suffix = ""
        self._complete = False
        self._is_capturing_stdout = False
        self._cap_obj = None
        self._is_finalized = False
        self._final_elapsed_time = None
        self._time_remaining = None
        self._iter_rate = None
        self._quiet = quiet

        if self._has_dynamic_width:
            self._update_max_width()
            if hasattr(signal, "SIGWINCH"):
                try:
                    signal.signal(signal.SIGWINCH, self._update_max_width)
                except ValueError:
                    # for example, called from a non-main thread
                    pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close(*args)

    def __len__(self):
        return self.total

    def __call__(self, iterable):
        if self._total is None:
            try:
                self._total = len(iterable)
            except:
                self._total = None

        if self._total is None:
            self._show_remaining_time = False

        # When tracking iterators, the iteration isn't complete until the next
        # element is emitted, so progress should be delayed by 1
        self._iteration = -1

        self._iterator = iter(iterable)
        return self

    def __iter__(self):
        if not self.is_iterable:
            raise TypeError("This ProgressBar is not iterable")

        if not self.is_running and not self.is_finalized:
            self.start()

        return self

    def __next__(self):
        try:
            val = next(self._iterator)
        except StopIteration:
            # Update once more so progress reaches 100%
            self.update(draw=False)

            self.close(None, None, None)
            raise

        self.update()
        return val

    @property
    def is_iterable(self):
        """Whether the progress bar is tracking an iterable."""
        return self._iterator is not None

    @property
    def is_running(self):
        """Whether the progress bar is running, i.e., it is between `start()`
        and `close()` calls.
        """
        return self._is_running

    @property
    def is_finalized(self):
        """Whether the progress bar is finalized, i.e., `close()` has been
        called.
        """
        return self._is_finalized

    @property
    def is_capturing_stdout(self):
        """Whether stdout is being captured between calls to `draw()`."""
        return self._is_capturing_stdout

    @property
    def has_dynamic_width(self):
        """Whether the progress bar's width is adjusted dynamically based on
        the width of the terminal window.
        """
        return self._has_dynamic_width

    @property
    def has_total(self):
        """Whether the progress bar has a total iteration count."""
        return self._total is not None

    @property
    def total(self):
        """The total iterations, or None if not available."""
        return self._total

    @property
    def iteration(self):
        """The current iteration."""
        return self._iteration

    @property
    def progress(self):
        """The current progress, in [0, 1], or None if the progress bar has no
        total.
        """
        if not self.has_total:
            return None

        if self.total <= 0:
            return 1.0

        return self.iteration * 1.0 / self.total

    @property
    def complete(self):
        """Whether the task is 100%% complete, or None if the progress bar has
        no total.
        """
        if not self.has_total:
            return None

        return self.iteration >= self.total

    @property
    def elapsed_time(self):
        """The elapsed time since the task was started, or None if this
        progress bar is not timing.
        """
        if self.is_finalized:
            return self._final_elapsed_time

        if not self.is_running:
            return None

        return self._timer.elapsed_time

    @property
    def time_remaining(self):
        """The estimated time remaining for the task, or None if the progress
        bar has no total or is not timing.
        """
        return self._time_remaining

    @property
    def iter_rate(self):
        """The current iteration rate, in iterations per second, of the task,
        or None if the progress bar is not running.
        """
        return self._iter_rate

    @property
    def quiet(self):
        """Whether the progress bar is in quiet mode (no printing to stdout).
        """
        return self._quiet

    def start(self):
        """Starts the progress bar."""
        if self.is_running:
            return

        if self.is_finalized:
            raise Exception("Cannot start a finalized ProgressBar")

        self._cap_obj = CaptureStdout()
        if not self.quiet:
            if self._start_msg:
                logger.info(self._start_msg)

            self._is_capturing_stdout = True
            self._start_capture()

        self._timer.start()
        self._is_running = True

    def close(self, *args):
        """Closes the progress bar."""
        if self.is_finalized or not self.is_running:
            return

        self._flush_capture()
        self._is_capturing_stdout = False
        self._cap_obj = None
        self._final_elapsed_time = self.elapsed_time
        self._timer.stop()
        self._draw(force=True, last=True)
        self._is_running = False
        self._is_finalized = True

        if (
            not self.quiet
            and self._complete_msg  # have a complete message
            and (not args or args[0] is None)  # no error
            and (self.complete == True)  # progress bar completed
        ):
            logger.info(self._complete_msg)

        if self.has_dynamic_width and hasattr(signal, "SIGWINCH"):
            try:
                signal.signal(signal.SIGWINCH, signal.SIG_DFL)
            except ValueError:
                pass

    def update(self, count=1, suffix=None, draw=True):
        """Increments the current iteration count by the given value
        (default = 1).

        This method is shorthand for
        `self.set_iteration(self.iteration + count)`.

        Args:
            count: the iteration increment. The default is 1
            suffix: an optional suffix string to append to the progress bar. By
                default, the suffix is unchanged
            draw: whether to call `draw()` at the end of this method. By
                default, this is True
        """
        self.set_iteration(self.iteration + count, suffix=suffix, draw=draw)

    def set_iteration(self, iteration, suffix=None, draw=True):
        """Sets the current iteration.

        Args:
            iteration: the new iteration
            suffix: an optional suffix string to append to the progress bar.
                Once a custom suffix is provided, it will remain unchanged
                until you update it or remove it by passing `suffix == ""`
            draw: whether to call `draw()` at the end of this method. By
                default, this is True
        """
        if self.is_finalized:
            raise Exception(
                "The iteration of a finalized ProgressBar cannot be changed"
            )

        if self.has_total:
            iteration = min(iteration, self.total)

        self._iteration = max(0, iteration)

        if suffix is not None:
            self._suffix = suffix

        if draw:
            self.draw()

    def pause(self):
        """Pauses the progress bar so that other information can be printed.

        This function overwrites the current progress bar with whitespace and
        appends a carriage return so that any other information that is printed
        will overwrite the current progress bar.
        """
        sys.stdout.write("\r" + " " * self._max_len + "\r")

    def draw(self, force=False):
        """Draws the progress bar at its current progress.

        Args:
            force: whether to force the progress bar to be drawn. By default,
                it is only redrawn a maximum of `self.max_fps` times per second
        """
        self._draw(force=force)

    @staticmethod
    def _get_total(total):
        if is_numeric(total) or total is None:
            return total

        try:
            return len(total)
        except:
            return None

    def _start_capture(self):
        self._cap_obj.start()

    def _draw(self, force=False, last=False):
        if self.quiet:
            return

        elapsed_time = self.elapsed_time

        if (
            self.is_running
            and not force
            and (self._max_fps is not None)
            and (elapsed_time - self._last_draw_time) * self._max_fps < 1
        ):
            # Avoid rendering at greater than `max_fps`
            return

        self._last_draw_time = elapsed_time
        if self.iteration > self._draw_iters[-1]:
            self._draw_times.append(elapsed_time)
            self._draw_iters.append(self.iteration)

        if self.is_capturing_stdout:
            self._flush_capture()

        progress_str = self._render_progress(elapsed_time)

        if last:
            sys.stdout.write("\r")
            logger.info(progress_str)
        else:
            sys.stdout.write("\r" + progress_str)
            if self.is_capturing_stdout:
                self._start_capture()

        sys.stdout.flush()

    def _render_progress(self, elapsed_time):
        dw = 0

        #
        # Render suffix
        #

        if self._suffix:
            suffix_str = " " + self._suffix
            dw += len(suffix_str)
        else:
            suffix_str = ""

        #
        # Render stats
        #

        if elapsed_time is not None:
            # Update time remaining/iteration rate estimates
            self._update_estimates(elapsed_time)

            _stats = []

            # Elapsed time
            if self._show_elapsed_time:
                _max_len = 13 + self._num_decimals + bool(self._num_decimals)
                _msg = "%s elapsed" % to_human_time_str(
                    elapsed_time, short=True, decimals=self._num_decimals
                )
                _stats.append(_msg)
                dw += _max_len

            # Remaining time
            if self._show_remaining_time:
                _max_len = 15 + self._num_decimals + bool(self._num_decimals)
                if self.time_remaining is not None:
                    _tr_str = to_human_time_str(
                        self.time_remaining,
                        short=True,
                        decimals=self._num_decimals,
                    )
                else:
                    _tr_str = "?"

                _msg = "%s remaining" % _tr_str
                _stats.append(_msg)
                dw += _max_len

            # Iteration rate
            if self._show_iter_rate:
                if self._use_bits or self._use_bytes:
                    _max_len = 9
                    if self.iter_rate is not None:
                        if self._use_bits:
                            _br_str = to_human_bits_str(self.iter_rate)
                        else:
                            _br_str = to_human_bytes_str(self.iter_rate)
                    else:
                        _br_str = "?b" if self._use_bits else "?B"

                    _msg = "%s/s" % _br_str
                else:
                    _max_len = 8 + len(self._iters_str)
                    if self.iter_rate is not None:
                        _ir_str = to_human_decimal_str(self.iter_rate)
                    else:
                        _ir_str = "?"

                    _msg = "%s %s/s" % (_ir_str, self._iters_str)

                _stats.append(_msg)
                dw += _max_len

            # Render final stats block
            stats_str = " [%s]" % (", ".join(_stats))
            dw += 2 * len(_stats) + 3
        else:
            stats_str = ""

        #
        # Render percentage
        #

        if self._show_percentage and self.has_total:
            pct_str = self._pct_fmt % (100.0 * self.progress)
            dw += len(pct_str)
        else:
            pct_str = ""

        #
        # Render iteration count
        #

        if self._show_iteration:
            if self._iter_fmt is None:
                self._update_iter_fmt()

            if self._use_bits:
                _iter = to_human_bits_str(self.iteration)
            elif self._use_bytes:
                _iter = to_human_bytes_str(self.iteration)
            else:
                _iter = self.iteration

            iter_str = self._iter_fmt % _iter
            dw += len(iter_str)
        else:
            iter_str = ""

        #
        # Render bar
        #

        if self.has_total:
            bar_len = self._max_width - 3 - dw
            if bar_len >= 0:
                progress_len = int(bar_len * self.progress)
                bstr = "\u2588" * progress_len
                if progress_len < bar_len:
                    istr = next(self._spinner)
                    bstr += istr + "-" * max(0, bar_len - 1 - progress_len)

                bar_str = "|%s|" % bstr
            else:
                bar_str = ""
        else:
            bar_str = ""

        #
        # Combine bar and suffix
        #

        pstr = pct_str + bar_str + iter_str + suffix_str + stats_str + " "
        pstr_len = len(pstr)
        if pstr_len > self._max_width:
            pstr = pstr[: self._max_width]
            pstr_len = self._max_width

        self._max_len = min(max(pstr_len, self._max_len), self._max_width)
        pstr += " " * (self._max_len - pstr_len)

        # substitute any characters that can't be printed to stdout
        if sys.stdout.encoding:
            pstr = pstr.encode(sys.stdout.encoding, errors="replace").decode(
                sys.stdout.encoding
            )

        return pstr

    def _update_estimates(self, elapsed_time):
        # Estimate iteration rate
        try:
            self._iter_rate = (self._draw_iters[-1] - self._draw_iters[0]) / (
                self._draw_times[-1] - self._draw_times[0]
            )
        except ZeroDivisionError:
            self._iter_rate = None

        # Estimate time remaining
        if self.has_total and self.iteration > 0:
            _it = self.iteration
            _it_rem = self.total - self.iteration
            tr1 = elapsed_time * (_it_rem / _it)

            if self._iter_rate is not None:
                _prog = self.progress
                tr2 = _it_rem / self._iter_rate
                self._time_remaining = (1 - _prog) * tr1 + _prog * tr2
            else:
                self._time_remaining = tr1
        else:
            self._time_remaining = None

    def _flush_capture(self):
        if self._cap_obj is None or not self._cap_obj.is_capturing:
            return

        out = self._cap_obj.stop()
        self.pause()
        sys.stdout.write(out)
        sys.stdout.flush()

    def _update_iter_fmt(self):
        if self.has_total:
            if self._use_bits:
                self._iter_fmt = " %%8s/%s" % to_human_bits_str(self.total)
            elif self._use_bytes:
                self._iter_fmt = " %%8s/%s" % to_human_bytes_str(self.total)
            else:
                cap = get_int_pattern_with_capacity(
                    self.total, zero_padded=False
                )
                self._iter_fmt = " %s/%d" % (cap, self.total)
        else:
            if self._use_bits or self._use_bytes:
                self._iter_fmt = " %8s"
            else:
                self._iter_fmt = " %d"

    def _update_max_width(self, *args, **kwargs):
        self._max_width = get_terminal_size()[0]


def call(args, **kwargs):
    """Runs the command via `subprocess.call()`.

    stdout and stderr are streamed live during execution. `**kwargs` may be
    used to override this. Or, if you want to capture these streams, use
    `communicate()`.


    Args:
        args: the command specified as a ["list", "of", "strings"]
        **kwargs: keyword arguments to be passed through to
            `subprocess.call()`.

    Returns:
        True/False: if the command executed successfully
    """
    return subprocess.call(args, **kwargs) == 0


def communicate(args, decode=False):
    """Runs the command via `subprocess.communicate()`

    Args:
        args: the command specified as a ["list", "of", "strings"]
        decode: whether to decode the output bytes into utf-8 strings. By
            default, the raw bytes are returned

    Returns:
        True/False: if the command executed successfully
        out: the command's stdout
        err: the command's stderr
    """
    logger.debug("Executing '%s'", " ".join(args))
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if decode:
        out = out.decode()
        err = err.decode()

    return p.returncode == 0, out, err


def communicate_or_die(args, decode=False):
    """Wrapper around `communicate()` that raises an exception if any error
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
    """
    try:
        success, out, err = communicate(args, decode=decode)
        if not success:
            raise ExecutableRuntimeError(" ".join(args), err)

        return out
    except EnvironmentError as e:
        if e.errno == errno.ENOENT:
            raise ExecutableNotFoundError(exe=args[0])

        raise


def _run_system_os_cmd(args):
    try:
        communicate_or_die(args)
    except ExecutableRuntimeError as e:
        raise OSError(e)


class Timer(object):
    """Class for timing things that supports the context manager interface.

    Example::

        import time
        import eta.core.utils as etau

        with etau.Timer() as t:
            time.sleep(1.5)

    Args:
        quiet (False): whether to suppress timing message when timer exits
    """

    def __init__(self, quiet=False):
        self.quiet = quiet
        self._start_time = None
        self._stop_time = None
        self._is_running = None
        self._elapsed_time = None
        self.reset()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    @property
    def is_running(self):
        """Whether the timer is currently running."""
        return self._is_running

    @property
    def elapsed_time(self):
        """The number of elapsed seconds."""
        elapsed_time = self._elapsed_time
        if self.is_running:
            elapsed_time += self._get_current_time() - self._start_time

        return elapsed_time

    @property
    def elapsed_time_str(self):
        """The human-readable elapsed time string."""
        return to_human_time_str(self.elapsed_time)

    def start(self, reset=True):
        """Starts the timer.

        Args:
            reset: whether to reset the timer. By default, this is True
        """
        if reset:
            self.reset()

        self._start_time = self._get_current_time()
        self._is_running = True

    def stop(self):
        """Stops the timer."""
        self._stop_time = self._get_current_time()
        self._elapsed_time += self._stop_time - self._start_time
        self._is_running = False
        if not self.quiet:
            logger.info("Time elapsed: %s", self.elapsed_time_str)

    def reset(self):
        """Resets the timer."""
        self._elapsed_time = 0.0
        self._is_running = False
        self._start_time = None
        self._stop_time = None

    @staticmethod
    def _get_current_time():
        return timeit.default_timer()


def get_dir_size(dirpath):
    """Returns the size, in bytes, of the given directory.

    This method uses the system command `du -s <dirpath>`.

    Args:
        dirpath: the path to the directory

    Returns:
        the size, in bytes
    """
    if not os.path.isdir(dirpath):
        raise OSError("Directory '%s' does not exist" % dirpath)

    out = communicate_or_die(["du", "-sk", dirpath])  # -k means 1024B blocks
    return 1024 * int(out.split()[0].decode())


def guess_mime_type(filepath):
    """Guess the MIME type for the given file path.

    If no reasonable guess can be determined, `application/octet-stream` is
    returned.

    Args:
        filepath: path to the file

    Returns:
        the MIME type string
    """
    if filepath.startswith("http"):
        filepath = urlparse.urlparse(filepath).path

    mime_type, _ = mimetypes.guess_type(filepath)

    if not mime_type:
        return "application/octet-stream"

    return mime_type


def read_file(inpath, binary=False):
    """Reads the file from disk.

    Args:
        inpath: the path to the file to read
        binary: whether to read the file in binary mode. By default, this is
            False (text mode)

    Returns:
        the contents of the file as a string
    """
    mode = "rb" if binary else "rt"
    with open(inpath, mode) as f:
        return f.read()


def write_file(str_or_bytes, outpath):
    """Writes the given string/bytes to disk.

    If a string is provided, it is encoded via `.encode()`.

    Args:
        str_or_bytes: the string or bytes to write to disk
        outpath: the desired output filepath
    """
    ensure_basedir(outpath)
    if is_str(str_or_bytes):
        str_or_bytes = str_or_bytes.encode()

    with open(outpath, "wb") as f:
        f.write(str_or_bytes)


def copy_file(inpath, outpath, check_ext=False):
    """Copies the input file to the output location.

    The output location can be a filepath or a directory in which to write the
    file. The base output directory is created if necessary, and any existing
    file will be overwritten.

    Args:
        inpath: the input path
        outpath: the output location (file or directory)
        check_ext: whether to check if the extensions of the input and output
            paths match. Only applicable if the output path is not a directory

    Raises:
        OSError if the copy failed, or if `check_ext == True` and the input and
            output paths have different extensions
    """
    if check_ext and os.path.splitext(outpath)[1]:
        assert_same_extensions(inpath, outpath)

    ensure_basedir(outpath)
    shutil.copy(inpath, outpath)


def link_file(filepath, linkpath, check_ext=False):
    """Creates a hard link at the given location using the given file.

    The base output directory is created if necessary, and any existing file
    will be overwritten.

    Args:
        filepath: a file or directory
        linkpath: the desired symlink path
        check_ext: whether to check if the extensions (or lack thereof, for
            directories) of the input and output paths match

    Raises:
        OSError if the link failed or if `check_ext == True` and the input and
            output paths have different extensions
    """
    if check_ext:
        assert_same_extensions(filepath, linkpath)

    ensure_basedir(linkpath)
    if os.path.exists(linkpath):
        delete_file(linkpath)

    os.link(os.path.realpath(filepath), linkpath)


def symlink_file(filepath, linkpath, check_ext=False):
    """Creates a symlink at the given location that points to the given file.

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
    """
    if check_ext:
        assert_same_extensions(filepath, linkpath)

    ensure_basedir(linkpath)
    if os.path.exists(linkpath):
        delete_file(linkpath)

    os.symlink(os.path.realpath(filepath), linkpath)


def move_file(inpath, outpath, check_ext=False):
    """Moves the input file to the output location.

    The output location can be a filepath or a directory in which to move the
    file. The base output directory is created if necessary, and any existing
    file will be overwritten.

    Args:
        inpath: the input path
        outpath: the output location (file or directory)
        check_ext: whether to check if the extensions of the input and output
            paths match. Only applicable if the output path is not a directory

    Raises:
        OSError if the move failed, or if `check_ext == True` and the input and
            output paths have different extensions
    """
    if not os.path.splitext(outpath)[1]:
        # Output location is a directory
        ensure_dir(outpath)
    else:
        # Output location is a file
        if check_ext:
            assert_same_extensions(inpath, outpath)

        ensure_basedir(outpath)

    shutil.move(inpath, outpath)


def move_dir(indir, outdir):
    """Moves the input directory to the given output location.

    The base output directory is created, if necessary. Any existing directory
    will be deleted.

    Args:
        indir: the input directory
        outdir: the output directory to create

    Raises:
        OSError if the move failed
    """
    if os.path.isdir(outdir):
        delete_dir(outdir)

    ensure_basedir(outdir)
    shutil.move(indir, outdir)


def partition_files(indir, outdir=None, num_parts=None, dir_size=None):
    """Partitions the files in the input directory into the specified number
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
    """
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
    """Copies the input sequence to the output sequence.

    The base output directory is created if necessary, and any existing files
    will be overwritten.

    Args:
        inpatt: the input sequence
        outpatt: the output sequence
        check_ext: whether to check if the extensions of the input and output
            sequences match

    Raises:
        OSError if the copy failed or if `check_ext == True` and the input and
            output sequences have different extensions
    """
    if check_ext:
        assert_same_extensions(inpatt, outpatt)

    for idx in parse_pattern(inpatt):
        copy_file(inpatt % idx, outpatt % idx)


def link_sequence(inpatt, outpatt, check_ext=False):
    """Creates hard links at the given locations using the given sequence.

    The base output directory is created if necessary, and any existing files
    will be overwritten.

    Args:
        inpatt: the input sequence
        outpatt: the output sequence
        check_ext: whether to check if the extensions of the input and output
            sequences match

    Raises:
        OSError if the link failed or if `check_ext == True` and the input and
        output sequences have different extensions
    """
    if check_ext:
        assert_same_extensions(inpatt, outpatt)

    for idx in parse_pattern(inpatt):
        link_file(inpatt % idx, outpatt % idx)


def symlink_sequence(inpatt, outpatt, check_ext=False):
    """Creates symlinks at the given locations that point to the given
    sequence.

    The base output directory is created if necessary, and any existing files
    will be overwritten.

    Args:
        inpatt: the input sequence
        outpatt: the output sequence
        check_ext: whether to check if the extensions of the input and output
            sequences match

    Raises:
        OSError if the symlink failed or if `check_ext == True` and the input
            and output sequences have different extensions
    """
    if check_ext:
        assert_same_extensions(inpatt, outpatt)

    for idx in parse_pattern(inpatt):
        symlink_file(inpatt % idx, outpatt % idx)


def move_sequence(inpatt, outpatt, check_ext=False):
    """Moves the input sequence to the output sequence.

    The base output directory is created if necessary, and any existing files
    will be overwritten.

    Args:
        inpatt: the input sequence
        outpatt: the output sequence
        check_ext: whether to check if the extensions of the input and output
            sequences match

    Raises:
        OSError if the move failed or if `check_ext == True` and the input and
            output sequences have different extensions
    """
    if check_ext:
        assert_same_extensions(inpatt, outpatt)

    for idx in parse_pattern(inpatt):
        move_file(inpatt % idx, outpatt % idx)


def is_in_root_dir(path, rootdir):
    """Determines if the given path is a file or subdirectory (any levels deep)
    within the given root directory.

    Args:
        path: the input path (relative or absolute)
        rootdir: the root directory
    """
    path = os.path.abspath(path)
    rootdir = os.path.abspath(rootdir)
    return path.startswith(rootdir)


def copy_dir(indir, outdir):
    """Copies the input directory to the output directory.

    The base output directory is created if necessary, and any existing output
    directory will be deleted.

    Args:
        indir: the input directory
        outdir: the output directory

    Raises:
        OSError if the copy failed
    """
    if os.path.isdir(outdir):
        shutil.rmtree(outdir, ignore_errors=True)

    ensure_dir(outdir)

    for filepath in list_files(indir, include_hidden_files=True, sort=False):
        copy_file(
            os.path.join(indir, filepath), os.path.join(outdir, filepath)
        )

    for subdir in list_subdirs(indir):
        outsubdir = os.path.join(outdir, subdir)
        insubdir = os.path.join(indir, subdir)
        copy_dir(insubdir, outsubdir)


def delete_file(path):
    """Deletes the file at the given path and recursively deletes any empty
    directories from the resulting directory tree.

    Args:
        path: the filepath

    Raises:
        OSError if the deletion failed
    """
    os.remove(path)
    try:
        os.removedirs(os.path.dirname(path))
    except OSError:
        # found a non-empty directory or directory with no write access
        pass


def delete_dir(dir_):
    """Deletes the given directory and recursively deletes any empty
    directories from the resulting directory tree.

    Args:
        dir_: the directory path

    Raises:
        OSError if the deletion failed
    """
    dir_ = os.path.normpath(dir_)
    shutil.rmtree(dir_, ignore_errors=True)
    try:
        os.removedirs(os.path.dirname(dir_))
    except OSError:
        # found a non-empty directory or directory with no write access
        pass


def make_search_path(dirs):
    """Makes a search path for the given directories by doing the following:
        - converting all paths to absolute paths
        - removing directories that don't exist
        - removing duplicate directories

    The order of the original directories is preserved.

    Args:
        dirs: a list of relative or absolute directory paths

    Returns:
        a list of absolute paths with duplicates and non-existent directories
            removed
    """
    search_dirs = []
    for d in dirs:
        adir = os.path.abspath(d)
        if os.path.isdir(adir) and adir not in search_dirs:
            search_dirs.append(adir)

    if not search_dirs:
        logger.warning("Search path is empty")

    return search_dirs


def ensure_empty_dir(dirname, cleanup=False):
    """Ensures that the given directory exists and is empty.

    Args:
        dirname: the directory path
        cleanup: whether to delete any existing directory contents. By default,
            this is False

    Raises:
        ValueError: if the directory is not empty and `cleanup` is False
    """
    if os.path.isdir(dirname):
        if cleanup:
            delete_dir(dirname)
        elif os.listdir(dirname):
            raise ValueError("%s not empty" % dirname)

    ensure_dir(dirname)


def ensure_path(path):
    """Ensures that the given path is ready for writing by deleting any
    existing file and ensuring that the base directory exists.

    Args:
        path: the filepath
    """
    if os.path.isfile(path):
        delete_file(path)

    ensure_basedir(path)


def ensure_empty_file(path):
    """Ensures that an empty file exists at the given path.

    Any existing file is deleted.

    Args:
        path: the filepath
    """
    ensure_path(path)
    with open(path, "wt") as f:
        pass


def ensure_basedir(path):
    """Makes the base directory of the given path, if necessary.

    Args:
        path: the filepath
    """
    ensure_dir(os.path.dirname(path))


def ensure_dir(dirname):
    """Makes the given directory, if necessary.

    Args:
        dirname: the directory path
    """
    os.makedirs(dirname, exist_ok=True)


def has_extension(filename, *args):
    """Determines whether the filename has any of the given extensions.

    Args:
        filename: a file name
        *args: extensions like ".txt" or ".json"

    Returns:
        True/False
    """
    ext = os.path.splitext(filename)[1]
    return any(ext == a for a in args)


def have_same_extesions(*args):
    """Determines whether all of the input paths have the same extension.

    Args:
        *args: filepaths

    Returns:
        True/False
    """
    exts = [os.path.splitext(path)[1] for path in args]
    return exts[1:] == exts[:-1]


def assert_same_extensions(*args):
    """Asserts that all of the input paths have the same extension.

    Args:
        *args: filepaths

    Raises:
        OSError if all input paths did not have the same extension
    """
    if not have_same_extesions(*args):
        raise OSError("Expected %s to have the same extensions" % str(args))


def split_path(path):
    """Splits a path into a list of its individual parts.

    E.g. split_path("/path/to/file") = ["/", "path", "to", "file"]

    Taken from
    https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html

    Args:
        path: a path to a file or directory

    Returns:
        all_parts: the path split into its individual components (directory
            and file names)
    """
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


def make_unique_path(path, suffix=""):
    """Make a unique path based on the given path by possibly appending
    characters *before* the file extension.

    The returned path is::

        root, ext = os.path.splitext(path)
        outpath = root + suffix + ext

    where random chacters are appended to `suffix`, if necessary, until no
    existing path conflicts on disk.

    Args:
        path: a path
        suffix: an optional suffix. The default is ""

    Returns:
        the unique path
    """
    if suffix:
        root, ext = os.path.splitext(path)
        path = root + suffix + ext

    if not os.path.isfile(path):
        return path

    suffix += "_" + _get_random_characters(6)

    while os.path.isfile(root + suffix + ext):
        suffix += _get_random_characters(1)

    return root + suffix + ext


def _get_random_characters(n):
    return "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(n)
    )


_TIME_UNITS = [
    "ns",
    "us",
    "ms",
    " second",
    " minute",
    " hour",
    " day",
    " week",
    " month",
    " year",
]
_TIME_UNITS_SHORT = ["ns", "us", "ms", "s", "m", "h", "d", "w", "mo", "y"]
_TIME_CONVERSIONS = [
    1000,
    1000,
    1000,
    60,
    60,
    24,
    7,
    52 / 12,
    12,
    float("inf"),
]
_TIME_IN_SECS = [
    1e-9,
    1e-6,
    1e-3,
    1,
    60,
    3600,
    86400,
    606461.5384615385,
    2628000,
    31536000,
]
_TIME_PLURALS = [False, False, False, True, True, True, True, True, True, True]
_DECIMAL_UNITS = ["", "K", "M", "B", "T"]
_BYTES_UNITS = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
_BITS_UNITS = ["b", "Kb", "Mb", "Gb", "Tb", "Pb", "Eb", "Zb", "Yb"]


def to_time_str(num_seconds, decimals=0, fixed_width=False):
    """Converts the given number of seconds to a time string in "HH:MM:SS.XXX"
    format.

    By default, zero hours/minutes/milliseconds are omitted from the time
    string, i.e., the return format is `[[HH:]MM:]SS[.XXX]`.
    Use `fixed_width == True` to always return `HH:MM:SS.XXX`.

    Examples::

           0.001  =>  "00"
              60  =>  "01:00"
              65  =>  "01:05"
        60123123  =>  "16700:52:03"

    Examples (`decimals == 1`, `fixed_width == True`)::

           0.001  =>  "00:00:00.0"
              60  =>  "00:01:00.0"
              65  =>  "00:01:05.0"
        60123123  =>  "16700:52:03.0"

    Args:
        num_seconds: the number of seconds
        decimals: the desired number of millisecond decimals to show in the
            string. The default is 0
        fixed_width: whether to render the string with fixed width. See above
            for details. By default, this is False

    Returns:
        the time string in "HH:MM:SS.XXX" format
    """
    hh, val = divmod(num_seconds, 3600)
    mm, ss = divmod(val, 60)

    ssf = "%%0%d.%df" % (2 + bool(decimals) + decimals, decimals)

    if fixed_width:
        time_str = ("%02d:%02d:" + ssf) % (hh, mm, ss)
    else:
        hhs = "%02d:" % hh if hh > 0 else ""
        mms = "%02d:" % mm if max(hh, mm) > 0 else ""
        time_str = hhs + mms + (ssf % ss)

    return time_str


def from_time_str(time_str):
    """Parses the number of seconds from the given time string in
    [[HH:]MM:]SS[.XXX] format.

    Examples::

                 "00.0"  =>  0.0
              "01:00.0"  =>  60.0
              "01:05.0"  =>  65.0
        "16700:52:03.0"  =>  60123123.0

    Args:
        time_str: a time string in "HH:MM:SS.XXX" format

    Returns:
        the number of seconds
    """
    return sum(
        float(n) * m
        for n, m in zip(reversed(time_str.split(":")), (1, 60, 3600))
    )


def to_human_time_str(
    num_seconds, decimals=1, short=False, strip=False, max_unit=None
):
    """Converts the given number of seconds to a human-readable time string.

    When `short == False`, the units used are::

        ns, us, ms, second, minute, hour, day, week, month, year

    When `short == True`, the units used are::

        ns, us, ms, s, m, h, d, w, mo, y

    Examples::

           0.001  =>  "1ms"
              60  =>  "1 minute"
              65  =>  "1.1 minutes"
        60123123  =>  "1.9 years"

    Examples (short units)::

           0.001  =>  "1ms"
              60  =>  "1m"
              65  =>  "1.1m"
        60123123  =>  "1.9y"

    Args:
        num_seconds: the number of seconds
        decimals: the desired number of decimal points to show in the string.
            The default is 1
        short: whether to use abbreviated units. By default, this is False
        strip: whether to strip trailing zeros. By default, this is False
        max_unit: an optional max unit, e.g., "hour", beyond which to stop
            converting to larger units, e.g., "day". By default, no maximum
            unit is used

    Returns:
        a human-readable time string like "1.5 minutes" or "20.1 days"
    """
    if num_seconds == 0:
        return "0s" if short else "0 seconds"

    units = _TIME_UNITS_SHORT if short else _TIME_UNITS

    if max_unit and not any(u.strip() == max_unit for u in units):
        logger.warning("Unsupported max_unit = %s; ignoring", max_unit)
        max_unit = None

    num = 1e9 * num_seconds  # start with smallest unit
    for unit, conv, plural in zip(units, _TIME_CONVERSIONS, _TIME_PLURALS):
        if abs(num) < conv:
            break

        if max_unit and unit.strip() == max_unit:
            break

        num /= conv

    # Convert to string with the desired number of decimals, UNLESS those
    # decimals are zeros, in which case they are removed
    str_fmt = "%." + str(decimals) + "f"
    num_str = str_fmt % num
    if strip:
        num_str = num_str.rstrip("0").rstrip(".")

    # Add units
    time_str = num_str + unit

    # Handle pluralization
    if not short and plural and num_str != "1":
        time_str += "s"

    return time_str


def from_human_time_str(time_str):
    """Parses the number of seconds from the given human-readable time string.

    This function can parse any time string generated by `to_human_time_str()`,
    including those with full units::

        ns, us, ms, second, minute, hour, day, week, month, year

    and short units::

        ns, us, ms, s, m, h, d, w, mo, y

    Examples::

                "1ms"  =>  0.001
           "1 minute"  =>  60.0
        "1.1 minutes"  =>  66.0
          "1.9 years"  =>  59918400.0

    Examples (short units)::

         "1ms"  =>  0.001
          "1m"  =>  60.0
        "1.1m"  =>  66.0
        "1.9y"  =>  59918400.0

    Args:
        time_str: a human-readable time string

    Returns:
        the number of seconds
    """
    # Handle unit == "" outside loop
    for idx in reversed(range(len(_TIME_UNITS))):
        unit = _TIME_UNITS[idx].strip()
        if time_str.endswith(unit):
            return float(time_str[: -len(unit)]) * _TIME_IN_SECS[idx]

        can_be_plural = _TIME_PLURALS[idx]
        if can_be_plural and time_str.endswith(unit + "s"):
            return float(time_str[: -(len(unit) + 1)]) * _TIME_IN_SECS[idx]

        short_unit = _TIME_UNITS_SHORT[idx]
        if time_str.endswith(short_unit):
            try:
                return float(time_str[: -len(short_unit)]) * _TIME_IN_SECS[idx]
            except ValueError:
                if short_unit == "s":
                    continue  # may have found (ns, us, ms)

                raise

    return float(time_str)


def to_human_decimal_str(num, decimals=1, strip=False, max_unit=None):
    """Returns a human-readable string representation of the given decimal
    (base-10) number.

    The supported units are::

        "", K, M, B, T

    Examples::

            65  =>  "65"
        123456  =>  "123.5K"
           1e7  =>  "10M"

    Args:
        num: a number
        decimals: the desired number of digits after the decimal point to show.
            The default is 1
        strip: whether to strip trailing zeros. By default, this is False
        max_unit: an optional max unit, e.g., "M", beyond which to stop
            converting to larger units, e.g., "B". By default, no maximum unit
            is used

    Returns:
        a human-readable decimal string
    """
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
    num_str = str_fmt % num
    if strip:
        num_str = num_str.rstrip("0").rstrip(".")

    return num_str + unit


def from_human_decimal_str(num_str):
    """Parses the decimal number from the given human-readable decimal string.

    The supported units are::

        "", K, M, B, T

    Examples::

            "65"  =>  65.0
        "123.5K"  =>  123450.0
           "10M"  =>  1e7

    Args:
        num_str: a human-readable decimal string

    Returns:
        the decimal number
    """
    # Handle unit == "" outside loop
    for idx in reversed(range(len(_DECIMAL_UNITS))[1:]):
        unit = _DECIMAL_UNITS[idx]
        if num_str.endswith(unit):
            return float(num_str[: -len(unit)]) * (1000 ** idx)

    return float(num_str)


def to_human_bytes_str(num_bytes, decimals=1, strip=False, max_unit=None):
    """Returns a human-readable string representation of the given number of
    bytes.

    The supported units are::

        B, KB, MB, GB, TB, PB, EB, ZB, YB

    Examples::

              123  =>  "123B"
           123000  =>  "120.1KB"
        1024 ** 4  =>  "1TB"

    Args:
        num_bytes: a number of bytes
        decimals: the desired number of digits after the decimal point to show.
            The default is 1
        strip: whether to strip trailing zeros. By default, this is False
        max_unit: an optional max unit, e.g., "TB", beyond which to stop
            converting to larger units, e.g., "PB". By default, no maximum
            unit is used

    Returns:
        a human-readable bytes string
    """
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
    num_str = str_fmt % num_bytes
    if strip:
        num_str = num_str.rstrip("0").rstrip(".")

    return num_str + unit


def from_human_bytes_str(bytes_str):
    """Parses the number of bytes from the given human-readable bytes string.

    The supported units are::

        B, KB, MB, GB, TB, PB, EB, ZB, YB

    Examples::

           "123B"  =>  123
        "120.1KB"  =>  122982
            "1TB"  =>  1024 ** 4

    Args:
        bytes_str: a human-readable bytes string

    Returns:
        the number of bytes
    """
    for idx in reversed(range(len(_BYTES_UNITS))):
        unit = _BYTES_UNITS[idx]
        if bytes_str.endswith(unit):
            return int(float(bytes_str[: -len(unit)]) * 1024 ** idx)

    return int(bytes_str)


def to_human_bits_str(num_bits, decimals=1, strip=False, max_unit=None):
    """Returns a human-readable string representation of the given number of
    bits.

    The supported units are::

        b, Kb, Mb, Gb, Tb, Pb, Eb, Zb, Yb

    Examples::

              123  =>  "123b"
           123000  =>  "120.1Kb"
        1024 ** 4  =>  "1Tb"

    Args:
        num_bits: a number of bits
        decimals: the desired number of digits after the decimal point to show.
            The default is 1
        strip: whether to strip trailing zeros. By default, this is False
        max_unit: an optional max unit, e.g., "Tb", beyond which to stop
            converting to larger units, e.g., "Pb". By default, no maximum
            unit is used

    Returns:
        a human-readable bits string
    """
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
    num_str = str_fmt % num_bits
    if strip:
        num_str = num_str.rstrip("0").rstrip(".")

    return num_str + unit


def from_human_bits_str(bits_str):
    """Parses the number of bits from the given human-readable bits string.

    The supported units are::

        b, Kb, Mb, Gb, Tb, Pb, Eb, Zb, Yb

    Examples::

           "123b"  =>  123
        "120.1Kb"  =>  122982
            "1Tb"  =>  1024 ** 4

    Args:
        bits_str: a human-readable bits string

    Returns:
        the number of bits
    """
    for idx in reversed(range(len(_BITS_UNITS))):
        unit = _BITS_UNITS[idx]
        if bits_str.endswith(unit):
            return int(float(bits_str[: -len(unit)]) * 1024 ** idx)

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


def make_archive(dir_path, archive_path, cleanup=False):
    """Makes an archive containing the given directory.

    Supported formats include `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz`,
    and `.tbz`.

    Args:
        dir_path: the directory to archive
        archive_path: the path + filename of the archive to create
        cleanup: whether to delete the directory after tarring it. By default,
            this is False
    """
    outpath, format = _get_archive_format(archive_path)
    if format == "zip" and eta.is_python2():
        make_zip64(dir_path, archive_path)
        return

    rootdir, basedir = os.path.split(os.path.realpath(dir_path))
    shutil.make_archive(outpath, format, rootdir, basedir)
    if cleanup:
        delete_dir(dir_path)


def make_tar(dir_path, tar_path, cleanup=False):
    """Makes a tarfile containing the given directory.

    Supported formats include `.tar`, `.tar.gz`, `.tgz`, `.tar.bz`, and `.tbz`.

    Args:
        dir_path: the directory to tar
        tar_path: the path + filename of the .tar.gz file to create
        cleanup: whether to delete the directory after tarring it. By default,
            this is False
    """
    make_archive(dir_path, tar_path)
    if cleanup:
        delete_dir(dir_path)


def make_zip(dir_path, zip_path, cleanup=False):
    """Makes a zipfile containing the given directory.

    Python 2 users must use `make_zip64` when making large zip files.
    `shutil.make_archive` does not offer Zip64 in Python 2, and is therefore
    limited to 4GiB archives with less than 65,536 entries.

    Args:
        dir_path: the directory to zip
        zip_path: the path + filename of the zip file to create
        cleanup: whether to delete the directory after tarring it. By default,
            this is False
    """
    make_archive(dir_path, zip_path)
    if cleanup:
        delete_dir(dir_path)


def make_zip64(dir_path, zip_path, cleanup=False):
    """Makes a zip file containing the given directory in Zip64 format.

    Args:
        dir_path: the directory to zip
        zip_path: the path with extension of the zip file to create
        cleanup: whether to delete the directory after tarring it. By default,
            this is False
    """
    dir_path = os.path.realpath(dir_path)
    rootdir = os.path.dirname(dir_path)
    with zf.ZipFile(zip_path, "w", zf.ZIP_DEFLATED, allowZip64=True) as f:
        for root, _, filenames in os.walk(dir_path):
            base = os.path.relpath(root, rootdir)
            for name in filenames:
                src_path = os.path.join(root, name)
                dest_path = os.path.join(base, name)
                f.write(src_path, dest_path)

    if cleanup:
        delete_dir(dir_path)


def is_archive(filepath):
    """Determines whether the given filepath has an archive extension from the
    following list:

    `.zip`, `.rar`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz`, `.tbz`.

    Args:
        filepath: a filepath

    Returns:
        True/False
    """
    return filepath.endswith(
        (".zip", ".rar", ".tar", ".tar.gz", ".tgz", ".tar.bz", ".tbz")
    )


def split_archive(archive_path):
    """Splits the archive extension off of the given path.

    Similar to `os.path.splitext` but handles extensions like `.tar.gz`.

    Args:
        archive_path: the archive path

    Returns:
        a `(root, ext)` tuple
    """
    if archive_path.endswith((".tar.gz", ".tar.bz")):
        return archive_path[:-7], archive_path[-7:]

    return os.path.splitext(archive_path)


def extract_archive(archive_path, outdir=None, delete_archive=False):
    """Extracts the contents of an archive.

    The following formats are guaranteed to work:
    `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz`, `.tbz`.

    If an archive *not* in the above list is found, extraction will be
    attempted via the `patool` package, which supports many formats but may
    require that additional system packages be installed.

    Args:
        archive_path: the path to the archive file
        outdir: the directory into which to extract the archive contents. By
            default, the directory containing the archive is used
        delete_archive: whether to delete the archive after extraction. By
            default, this is False
    """
    if archive_path.endswith(".zip"):
        extract_zip(archive_path, outdir=outdir, delete_zip=delete_archive)
    elif archive_path.endswith(".rar"):
        extract_rar(archive_path, outdir=outdir, delete_rar=delete_archive)
    elif archive_path.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz", ".tbz")):
        extract_tar(archive_path, outdir=outdir, delete_tar=delete_archive)
    else:
        # Fallback to `patoolib`, which handles a lot of stuff, possibly
        # requiring the user to install system packages
        _extract_archive_patoolib(
            archive_path, outdir=outdir, delete_archive=delete_archive
        )


def extract_rar(rar_path, outdir=None, delete_rar=False):
    """Extracts the contents of a .rar file.

    This method will complain if you do not have a system package like `unrar`
    installed that can perform the actual extraction.

    Args:
        rar_path: the path to the RAR file
        outdir: the directory into which to extract the RAR contents. By
            default, the directory containing the RAR file is used
        delete_rar: whether to delete the RAR after extraction. By default,
            this is False
    """
    try:
        _extract_archive_patoolib(
            rar_path, outdir=outdir, delete_archive=delete_rar
        )
    except patoolib.util.PatoolError as e:
        message = (
            "Failed to extract RAR file '%s'. Extracting RAR files requires a "
            "system package like `unrar` to be installed on your machine, "
            "which you may need to install. Check the error message above for "
            "more information."
        ) % rar_path
        six.raise_from(IOError(message), e)


def extract_zip(zip_path, outdir=None, delete_zip=False):
    """Extracts the contents of a .zip file.

    Args:
        zip_path: the path to the zip file
        outdir: the directory into which to extract the zip contents. By
            default, the directory containing the zip file is used
        delete_zip: whether to delete the zip after extraction. By default,
            this is False
    """
    outdir = outdir or os.path.dirname(zip_path) or "."

    with zf.ZipFile(zip_path, "r", allowZip64=True) as f:
        f.extractall(outdir)

    if delete_zip:
        delete_file(zip_path)


def extract_tar(tar_path, outdir=None, delete_tar=False):
    """Extracts the contents of a tarfile.

    Supported formats include `.tar`, `.tar.gz`, `.tgz`, `.tar.bz`, and `.tbz`.

    Args:
        tar_path: the path to the tarfile
        outdir: the directory into which to extract the archive contents. By
            default, the directory containing the tar file is used
        delete_tar: whether to delete the tar archive after extraction. By
            default, this is False
    """
    if tar_path.endswith(".tar"):
        fmt = "r:"
    elif tar_path.endswith(".tar.gz") or tar_path.endswith(".tgz"):
        fmt = "r:gz"
    elif tar_path.endswith(".tar.bz") or tar_path.endswith(".tbz"):
        fmt = "r:bz2"
    else:
        raise ValueError(
            "Expected file '%s' to have extension .tar, .tar.gz, .tgz,"
            ".tar.bz, or .tbz in order to extract it" % tar_path
        )

    outdir = outdir or os.path.dirname(tar_path) or "."
    with tarfile.open(tar_path, fmt) as f:
        f.extractall(path=outdir)

    if delete_tar:
        delete_file(tar_path)


def _extract_archive_patoolib(archive_path, outdir=None, delete_archive=False):
    outdir = outdir or os.path.dirname(archive_path) or "."

    ensure_dir(outdir)

    patoolib.extract_archive(
        archive_path, outdir=outdir, verbosity=-1, interactive=False
    )

    if delete_archive:
        delete_file(archive_path)


def multiglob(*patterns, **kwargs):
    """Returns an iterable over the glob mathces for multiple patterns.

    Note that if a given file matches multiple patterns that you provided, it
    will appear multiple times in the output iterable.

    Examples::

        # Find all .py or .pyc files in a directory
        multiglob(".py", ".pyc", root="/path/to/dir/*")

        # Find all JSON files recursively in a given directory:
        multiglob(".json", root="/path/to/dir/**/*")

    Args:
        *patterns: the patterns to search for
        root: an optional root path to be applied to all patterns. This root is
            directly prepended to each pattern; `os.path.join` is NOT used

    Returns:
        an iteratable over the glob matches
    """
    root = kwargs.get("root", "")
    return it.chain.from_iterable(glob2.iglob(root + p) for p in patterns)


def list_files(
    dir_path,
    abs_paths=False,
    recursive=False,
    include_hidden_files=False,
    sort=True,
):
    """Lists the files in the given directory, sorted alphabetically and
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
    """
    if recursive:
        files = []
        for root, _, filenames in os.walk(dir_path):
            files.extend(
                [
                    os.path.relpath(os.path.join(root, f), dir_path)
                    for f in filenames
                    if not f.startswith(".")
                ]
            )
    else:
        files = [
            f
            for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
            and (not f.startswith(".") or include_hidden_files)
        ]

    if sort:
        files = sorted(files)

    if abs_paths:
        basedir = os.path.abspath(os.path.realpath(dir_path))
        files = [os.path.join(basedir, f) for f in files]

    return files


def list_subdirs(dir_path, abs_paths=False, recursive=False):
    """Lists the subdirectories in the given directory, sorted alphabetically
    and excluding hidden directories.

    Args:
        dir_path: the path to the directory to list
        abs_paths: whether to return the absolute paths to the dirs. By
            default, this is False
        recursive: whether to recursively traverse subdirectories. By default,
            this is False

    Returns:
        a sorted list of the non-hidden subdirectories in the directory
    """
    if recursive:
        dirs = []
        for root, dirnames, _ in os.walk(dir_path):
            dirs.extend(
                [
                    os.path.relpath(os.path.join(root, d), dir_path)
                    for d in dirnames
                    if not d.startswith(".")
                ]
            )
    else:
        dirs = [
            d
            for d in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, d))
            and not d.startswith(".")
        ]

    dirs = sorted(dirs)

    if abs_paths:
        basedir = os.path.abspath(os.path.realpath(dir_path))
        dirs = [os.path.join(basedir, d) for d in dirs]

    return dirs


def parse_pattern(patt):
    """Inspects the files matching the given numeric pattern and returns the
    numeric indicies of the sequence.

    Args:
        patt: a pattern with a one or more numeric sequences like
            "/path/to/frame-%05d.jpg" or `/path/to/clips/%02d-%d.mp4`

    Returns:
        a list (or list of tuples if the pattern contains multiple sequences)
            describing the numeric indices of the files matching the pattern.
            The indices are returned in alphabetical order of their
            corresponding files. If no matches were found, an empty list is
            returned
    """
    # Extract indices from exactly matching patterns
    inds = []
    for _, match, num_inds in _iter_pattern_matches(patt):
        idx = tuple(map(int, match.groups()))
        inds.append(idx[0] if num_inds == 1 else idx)

    return inds


def get_glob_matches(glob_patt):
    """Returns a list of file paths matching the given glob pattern.

    The matches are returned in sorted order.

    Args:
        glob_patt: a glob pattern like "/path/to/files-*.jpg" or
            "/path/to/files-*-*.jpg"

    Returns:
        a list of file paths that match `glob_patt`
    """
    return sorted(glob.glob(glob_patt))


def parse_glob_pattern(glob_patt):
    """Inspects the files matching the given glob pattern and returns a string
    pattern version of the glob along with the matching strings.

    Args:
        glob_patt: a glob pattern like "/path/to/files-*.jpg" or
            "/path/to/files-*-????.jpg"

    Returns:
        a tuple containing:
            - a string pattern version of the glob pattern with "%s" in place
                of each glob pattern (consecutive globs merged into one)
            - a list (or list of tuples if the string pattern contains multiple
                "%s") describing the string patterns matching the glob. If no
                matches were found, an empty list is returned
    """
    match_chunks = _get_match_chunks(glob_patt)

    matches = []
    for path in get_glob_matches(glob_patt):
        matches.append(_get_match_gaps(path, match_chunks))

    str_patt = "%s".join(match_chunks)
    return str_patt, matches


def glob_to_str_pattern(glob_patt):
    """Converts the glob pattern to a string pattern by replacing glob
    wildcards with "%s".

    Multiple consecutive glob wildcards are merged into single string patterns.

    Args:
        glob_patt: a glob pattern like "/path/to/files-*.jpg" or
            "/path/to/files-*-????.jpg"

    Returns:
        a string pattern like "/path/to/files-%s.jpg" or
            "/path/to/files-%s-%s.jpg"
    """
    return "%s".join(_get_match_chunks(glob_patt))


def _get_match_chunks(glob_patt):
    glob_chunks = re.split(r"(?<!\\)(\*|\?|\[.*\])", glob_patt)
    len_glob_chunks = len(glob_chunks)

    match_chunks = glob_chunks[:1]
    for idx in range(2, len_glob_chunks, 2):
        if glob_chunks[idx] or idx == len_glob_chunks - 1:
            match_chunks.append(glob_chunks[idx])

    return match_chunks


def _get_match_gaps(path, match_chunks):
    match = []

    len_path = len(path)
    idx = len(match_chunks[0])
    for chunk in match_chunks[1:]:
        last_idx = idx
        len_chunk = len(chunk)
        if not len_chunk:
            idx = len_path  # on empty match, consume rest of path

        while path[idx : (idx + len_chunk)] != chunk and idx < len_path:
            idx += 1

        match.append(path[last_idx:idx])
        idx += len_chunk

    return tuple(match)


def get_pattern_matches(patt):
    """Returns a list of file paths matching the given numeric pattern.

    Args:
        patt: a pattern with one or more numeric sequences like
            "/path/to/frame-%05d.jpg" or "/path/to/clips/%02d-%d.mp4"

    Returns:
        a list of file paths that match the pattern `patt`
    """
    return [path for path, _, _ in _iter_pattern_matches(patt)]


def fill_partial_pattern(patt, vals):
    """Partially fills a pattern with the given values.

    Only supports integer ("%05d", "%4d", or "%d") and string ("%s") patterns.

    Args:
        patt: a pattern with one or more numeric or string sequences like
            "/path/to/features/%05d.npy" or "/path/to/features/%s-%d.npy"
        vals: a tuple of values whose length matches the number of patterns in
            `patt`, with Nones as placeholders for patterns that should not be
            filled

    Returns:
        the partially filled pattern
    """
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
    glob_patt = re.sub(seq_exp, "*", _glob_escape(patt))
    files = get_glob_matches(glob_patt)

    # Create validation functions
    seq_patts = re.findall(seq_exp, patt)
    fcns = [parse_int_sprintf_pattern(sp) for sp in seq_patts]
    full_exp, num_inds = re.subn(seq_exp, r"(\\s*\\d+)", patt)

    # Iterate over exactly matching patterns and files
    for f in files:
        m = re.match(full_exp, f)
        if m and all(f(p) for f, p in zip(fcns, m.groups())):
            yield f, m, num_inds


def parse_bounds_from_pattern(patt):
    """Inspects the files satisfying the given pattern and returns the minimum
    and maximum indices satisfying it.

    Args:
        patt: a pattern with a single numeric sequence like
            "/path/to/frames/frame-%05d.jpg"

    Returns:
        a (first, last) tuple describing the first and last indices satisfying
            the pattern, or (None, None) if no matches were found
    """
    inds = parse_pattern(patt)
    if not inds or isinstance(inds[0], tuple):
        return None, None
    return min(inds), max(inds)


def parse_dir_pattern(dir_path):
    """Inspects the contents of the given directory, returning the numeric
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
                corresponding filenames. If no files were found, an empty list
                is returned
    """
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
    """Extracts the (first) numeric sequence index from the given pattern.

    Args:
        patt: a pattern like "/path/to/frames/frame-%05d.jpg"

    Returns:
        the numeric sequence string like "%05d", or None if no sequence was
            found
    """
    m = re.search("%[0-9]*d", patt)
    return m.group() if m else None


def parse_int_sprintf_pattern(patt):
    """Parses the integer sprintf pattern and returns a function that can
    detect whether a string matches the pattern.

    Args:
        patt: a sprintf pattern like "%05d", "%4d", or "%d"

    Returns:
        a function that returns True/False whether a given string matches the
            input numeric pattern
    """
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
    """Generates an n-lenth random key of lowercase characters and digits."""
    return "".join(
        random.SystemRandom().choice(string.ascii_lowercase + string.digits)
        for _ in range(n)
    )


def replace_strings(s, replacers):
    """Performs a sequence of find-replace operations on the given string.

    Args:
        s: the input string
        replacers: a list of (find, replace) strings

    Returns:
        a copy of the input strings with all of the find-and-replacements made
    """
    sout = s
    for sfind, srepl in replacers:
        sout = sout.replace(sfind, srepl)

    return sout


def escape_chars(s, chars):
    """Escapes any occurances of the the given characters in the given string.

    Args:
        s: a string
        chars: the iterable (e.g., string or list) of characters to escape

    Returns:
        the escaped string
    """
    # Must escape `]` and `-` because they have special meaning inside the
    # regex we're using to do the escaping
    chars = replace_strings(chars, [("]", "\\]"), ("-", "\\-")])

    return re.sub(r"([%s])" % "".join(chars), r"\\\1", s)


def remove_escape_chars(s, chars):
    """Removes escapes from any escaped occurances of the given characters in
    the given string.

    Args:
        s: a string
        chars: the iterable (e.g., string or list) of characters to unescape

    Returns:
        the unescaped string
    """
    return re.sub(r"\\([%s])" % "".join(chars), r"\1", s)


def join_dicts(*args):
    """Joins any number of dictionaries into a new single dictionary.

    Args:
        *args: one or more dictionaries

    Returns:
        a single dictionary containing all items.
    """
    d = {}
    for di in args:
        d.update(di)

    return d


def remove_none_values(d):
    """Returns a copy of the input dictionary with any keys with value None
    removed.

    Args:
        d: a dictionary

    Returns:
        a copy of the input dictionary with keys whose value was None ommitted
    """
    return {k: v for k, v in iteritems(d) if v is not None}


def find_duplicate_files(path_list, verbose=False):
    """Returns a list of lists of file paths from the input, that have
    identical contents to each other.

    Args:
        path_list: list of file paths in which to look for duplicate files
        verbose: if True, log progress

    Returns:
        duplicates: a list of lists, where each list contains a group of
            file paths that all have identical content. File paths in
            `path_list` that don't have any duplicates will not appear in
            the output.
    """
    if verbose:
        logger.info("Finding duplicates among %d files...", len(path_list))

    hash_buckets = _get_file_hash_buckets(path_list, verbose)

    duplicates = []
    for file_group in itervalues(hash_buckets):
        if len(file_group) >= 2:
            duplicates.extend(_find_duplicates_brute_force(file_group))

    if verbose:
        duplicate_count = sum(len(x) for x in duplicates) - len(duplicates)
        logger.info("Operation complete. Found %d duplicates", duplicate_count)

    return duplicates


def find_matching_file_pairs(path_list1, path_list2, verbose=False):
    """Returns a list of pairs of paths that have identical contents, where
    the paths in each pair aren't from the same path list.

    Args:
        path_list1: list of file paths
        path_list2: another list of file paths
        verbose: if True, log progress

    Returns:
        pairs: a list of pairs of file paths that have identical content,
            where one member of the pair is from `path_list1` and the other
            member is from `path_list2`
    """
    hash_buckets1 = _get_file_hash_buckets(path_list1, verbose)
    pairs = []
    for path in path_list2:
        with open(path, "rb") as f:
            content = f.read()
        candidate_matches = hash_buckets1.get(hash(content), [])
        for candidate_path in candidate_matches:
            if not _diff_paths(candidate_path, path, content2=content):
                pairs.append((candidate_path, path))

    return pairs


def _get_file_hash_buckets(path_list, verbose):
    hash_buckets = defaultdict(list)
    for idx, path in enumerate(path_list):
        if verbose and idx % 100 == 0:
            logger.info("\tHashing file %d/%d", idx, len(path_list))

        if not os.path.isfile(path):
            logger.warning(
                "'%s' is a directory or does not exist; skipping", path
            )
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
        if _diff_paths(candidate_file_path, path, content1=candidate_content):
            remaining_paths.append(path)
        else:
            candidate_duplicates.append(path)

    duplicates = []
    if len(candidate_duplicates) >= 2:
        duplicates.append(candidate_duplicates)

    duplicates.extend(_find_duplicates_brute_force(remaining_paths))

    return duplicates


def _diff_paths(path1, path2, content1=None, content2=None):
    """Returns whether or not the files at `path1` and `path2` are different
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
    """
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
    """Base class for file hashers."""

    EXT = ""

    def __init__(self, path):
        """Constructs a FileHasher instance based on the current version of
        the input file."""
        self.path = path
        self._new_hash = self.hash(path)
        self._cur_hash = self.read()

    @property
    def record_path(self):
        """The path to the hash record file."""
        return os.path.splitext(self.path)[0] + self.EXT

    @property
    def has_record(self):
        """True if the file has an existing hash record."""
        return self._cur_hash is not None

    @property
    def has_changed(self):
        """True if the file's current hash differs from it's last hash record.
        Always returns False if the file has no existing hash record.
        """
        return self.has_record and self._new_hash != self._cur_hash

    def read(self):
        """Returns the current hash record, or None if there is no record."""
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
        """Writes the current hash record."""
        with open(self.record_path, "wt") as f:
            f.write(self._new_hash)

    @staticmethod
    def hash(path):
        raise NotImplementedError("subclass must implement hash()")


class MD5FileHasher(FileHasher):
    """MD5 file hasher."""

    EXT = ".md5"

    @staticmethod
    def hash(path):
        """Computes the MD5 hash of the file contents."""
        with open(path, "rb") as f:
            return str(hashlib.md5(f.read()).hexdigest())


def make_temp_dir(basedir=None):
    """Makes a temporary directory.

    Args:
        basedir: an optional directory in which to create the new directory

    Returns:
        the path to the temporary directory
    """
    if not basedir:
        basedir = tempfile.gettempdir()

    ensure_dir(basedir)
    return tempfile.mkdtemp(dir=basedir)


class TempDir(object):
    """Context manager that creates and destroys a temporary directory."""

    def __init__(self, basedir=None):
        """Creates a TempDir instance.

        Args:
            basedir: an optional base directory in which to create the temp dir
        """
        self._basedir = basedir
        self._name = None

    def __enter__(self):
        self._name = make_temp_dir(basedir=self._basedir)
        return self._name

    def __exit__(self, *args):
        delete_dir(self._name)


class WorkingDir(object):
    """Context manager that temporarily changes working directories."""

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
    """Exception raised when an executable file is not found."""

    def __init__(self, message=None, exe=None):
        if exe is not None:
            message = (
                "The requested operation failed because the `%s` executable "
                "could not be found. You may need to install it on your "
                "machine."
            ) % exe

        super(ExecutableNotFoundError, self).__init__(message)


class ExecutableRuntimeError(Exception):
    """Exception raised when an executable call throws a runtime error."""

    def __init__(self, cmd, err):
        message = "Command '%s' failed with error:\n%s" % (cmd, err)
        super(ExecutableRuntimeError, self).__init__(message)


def validate_type(obj, expected_type):
    """Validates an object's type against an expected type.

    Args:
        obj: the python object to validate
        expected_type: the type that `obj` must be (via `isinstance`)

    Raises:
        TypeError: if `obj` is not of `expected_type`
    """
    if not isinstance(obj, expected_type):
        raise TypeError(
            "Unexpected argument type:\n\tExpected: %s\n\tActual: %s"
            % (get_class_name(expected_type), get_class_name(obj))
        )


def get_terminal_size():
    """Gets the size of your current Terminal window.

    Returns:
        the (width, height) of the Terminal
    """
    try:
        try:
            return tuple(os.get_terminal_size())
        except AttributeError:
            # Fallback for Python 2.X
            # https://stackoverflow.com/a/3010495
            import fcntl, termios, struct

            h, w, hp, wp = struct.unpack(
                "HHHH",
                fcntl.ioctl(
                    0, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0)
                ),
            )
            return w, h
    except OSError as e:
        if e.errno in (
            getattr(errno, "ENOTTY", None),
            getattr(errno, "ENXIO", None),
            getattr(errno, "EBADF", None),
        ):
            return (80, 24)

        raise


def save_window_snapshot(window_name, filepath):
    """Take a screenshot of the window with the given name and saves it to an
    image on disk.

    Args:
        window_name: the name of the window
        filepath: the path to write the snapshot image
    """
    ensure_basedir(filepath)
    _run_system_os_cmd(["import", "-window", window_name, filepath])
