"""
Core infrastructure for managing models across local and remote storage.

See `docs/models_dev_guide.md` for detailed information about the design of
the ETA model management system.

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

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from collections import defaultdict
from distutils.version import LooseVersion
import logging
import os
import sys

import eta
import eta.constants as etac
from eta.core.config import Config, Configurable
from eta.core.serial import Serializable
import eta.core.utils as etau
import eta.core.web as etaw

etas = etau.lazy_import("eta.core.storage")


logger = logging.getLogger(__name__)


MODELS_MANIFEST_JSON = "manifest.json"


def list_models(downloaded_only=False):
    """Returns a list of all models on the models search path.

    Args:
        downloaded_only: whether to only include models that are currently
            downloaded. By default, this is False

    Returns:
        a list of model names (with "@<ver>" strings, if any)
    """
    models = _list_models(downloaded_only=downloaded_only)[0]
    return sorted(list(models.keys()))


def list_models_in_directory(models_dir, downloaded_only=False):
    """Returns a list of all models in the given directory.

    Args:
        models_dir: the models directory
        downloaded_only: whether to only include models that are currently
            downloaded. By default, this is False

    Returns:
        a list of model names (with "@<ver>" strings, if any)

    Raises:
        ModelError: if the directory was not a valid models directory
    """
    manifest = ModelsManifest.from_dir(models_dir)
    return sorted(
        [
            model.name
            for model in manifest
            if not downloaded_only or model.is_in_dir(models_dir)
        ]
    )


def get_model(name):
    """Gets the Model instance for the given model, which must appear in a
    ModelsManifest in one of the `eta.config.models_dirs` directories.

    Args:
        name: the name of the model, which can have "@<ver>" appended to refer
            to a specific version of the model. If no version is specified, the
            latest version of the model is assumed

    Returns:
        the Model instance for the specified model

    Raises:
        ModelError: if the model could not be found
    """
    return _find_model(name)[0]


def find_model(name):
    """Finds the given model, which must appear in a ModelsManifest in one of
    the `eta.config.models_dirs` directories.

    Note that the model might not actually exist at the returned model path.
    To download it, use `download_model()`.

    Args:
        name: the name of the model, which can have "@<ver>" appended to refer
            to a specific version of the model. If no version is specified, the
            latest version of the model is assumed

    Returns:
        the full path to the model file (which might not exist if it hasn't
            been downloaded yet)

    Raises:
        ModelError: if the model could not be found
    """
    model, models_dir, _ = _find_model(name)
    return model.get_path_in_dir(models_dir)


def find_all_models(downloaded_only=False):
    """Finds all models on the models search path.

    Args:
        downloaded_only: whether to only include models that are currently
            downloaded. By default, this is False

    Returns:
        a dictionary mapping model names (with "@<ver>" strings, if any) to
            full paths to the model files
    """
    models = _list_models(downloaded_only=downloaded_only)[0]
    return {
        name: md[0].get_path_in_dir(md[1]) for name, md in iteritems(models)
    }


def is_model_downloaded(name):
    """Determines whether the given model is downloaded.

    Args:
        name: the name of the model, which can have "@<ver>" appended to refer
            to a specific version of the model. If no version is specified, the
            latest version of the model is assumed

    Returns:
        True/False whether the model is downloaded

    Raises:
        ModelError: if the model could not be found
    """
    model, models_dir, _ = _find_model(name)
    return model.is_in_dir(models_dir)


def download_model(name, force=False):
    """Downloads the given model, if necessary.

    If the download is forced, the local copy of the model will be overwitten
    if it exists.

    Args:
        name: the name of the model, which can have "@<ver>" appended to refer
            to a specific version of the model. If no version is specified, the
            latest version of the model is assumed
        force: whether to force download the model. If True, the model is
            always downloaded. If False, the model is only downloaded if
            necessary. The default is False

    Returns:
        the path to the downloaded model

    Raises:
        ModelError: if the model could not be found
    """
    model, models_dir, _ = _find_model(name)
    model_path = model.get_path_in_dir(models_dir)
    model.manager.download_model(model_path, force=force)
    return model_path


def flush_model(name):
    """Deletes the local copy of the given model, if necessary.

    The models is not removed from its associated manifest and can be
    downloaded again at any time.

    Args:
        name: the name of the model, which can have "@<ver>" appended to refer
            to a specific version of the model. If no version is specified, the
            latest version of the model is assumed

    Raises:
        ModelError: if the model could not be found
    """
    model, models_dir, _ = _find_model(name)
    model.flush_model_from_dir(models_dir)


def flush_old_models():
    """Deletes local copies of any old models, i.e. models for which the number
    of versions stored on disk exceeds `eta.config.max_model_versions_to_keep`.

    The models are not removed from their associated manifests and can be
    downloaded again at any time.
    """
    max_vers = eta.config.max_model_versions_to_keep
    if max_vers < 0:
        # No flushing required
        return

    # Get downloaded models
    downloaded_models = _list_models(downloaded_only=True)[0]

    # Group by base name
    bmodels = defaultdict(list)
    for model, mdir in itervalues(downloaded_models):
        bmodels[model.base_name].append((model, mdir))

    # Sort by version (newest first)
    bmodels = {
        k: sorted(v, reverse=True, key=lambda vi: vi[0].comp_version)
        for k, v in iteritems(bmodels)
    }

    # Flush old models
    for base_name, models_list in iteritems(bmodels):
        num_to_flush = len(models_list) - max_vers
        if num_to_flush > 0:
            logger.info(
                "*** Flushing %d old version(s) of model '%s'",
                num_to_flush,
                base_name,
            )
            for model, models_dir in reversed(models_list[max_vers:]):
                model.flush_model_from_dir(models_dir)


def flush_models_directory(models_dir):
    """Deletes the local copies of all models in the given models directory.

    The models are not removed from their associated manifests and can be
    downloaded again at any time.

    Args:
        models_dir: the models directory

    Raises:
        ModelError: if the directory contains no models manifest
    """
    _warn_if_not_on_search_path(models_dir)
    for model in ModelsManifest.from_dir(models_dir):
        model.flush_model_from_dir(models_dir)


def flush_all_models():
    """Deletes all local copies of all models on the models search path.

    The models are not removed from their associated manifests and can be
    downloaded again at any time.
    """
    for models_dir in _get_models_search_path():
        flush_models_directory(models_dir)


def init_models_dir(new_models_dir):
    """Initializes the given directory as a models directory by creating an
    empty models manifest file for it.

    The directory is created if necessary.

    Note that the directory is not automatically added to your models search
    path, so models in this directory will not be findable until you update
    your ETA config.

    Args:
        new_models_dir: the directory to initialize

    Raises:
        ModelError: if the models directory is already initialized
    """
    if ModelsManifest.dir_has_manifest(new_models_dir):
        raise ModelError(
            "Directory '%s' already has a models manifest" % new_models_dir
        )

    logger.info("Initializing new models directory '%s'", new_models_dir)
    manifest = ModelsManifest()
    manifest.write_to_dir(new_models_dir)


def publish_public_model(
    name,
    google_drive_id,
    description=None,
    base_filename=None,
    models_dir=None,
):
    """Publishes a new model to the public ETA model registry.

    This function assumes that the model has been uploaded to Google Drive by
    an ETA administrator and that you have the ID of the file to provide here.

    This function performs the following actions:
        - recommends a base filename and models directory, if necessary
        - performs a dry run of the model registration process to check for
            potential problems
        - registers the model in the manifest of its models directory

    The keyword arguments `base_filename` and `models_dir` configure the
    filename (version information is added automatically) and models directory,
    respectively, to use when downloading the model. If you are uploading a
    new version of an existing model, these arguments can be omitted and the
    same values are inherited from the most recent version of the model. If
    this is a brand new model, the `base_filename` is required, and if a
    `models_dir` is not provided, the first directory on your models path will
    be used by default.

    If the specified models directory does not have a models manifest file, one
    is created. Note that new models directories are not automatically added to
    your models search path, so models in a newly initialized directory will
    not be findable until you add the directory to your models search path.

    Args:
        name: a name for the model, which can optionally have "@<ver>" appended
            to assign a version to the model
        google_drive_id: the ID of the model file in Google Drive
        description: an optional description for the model
        base_filename: an optional base filename to use when writing the model
            to disk. By default, a value is inferred as explained above
        models_dir: an optional directory in which to register the model. By
            default, a value is inferred as explained above

    Raises:
        ModelError: if the publishing failed for any reason
    """
    # Recommend paths if necessary
    base_filename, models_dir = recommend_paths_for_model(
        name, base_filename=base_filename, models_dir=models_dir
    )

    # Perform a dry run of the model registration
    register_model_dry_run(name, base_filename, models_dir)

    # Construct model manager instance for model
    config = ETAModelManagerConfig({"google_drive_id": google_drive_id})
    manager = ETAModelManager(config)

    # Register model
    register_model(
        name, base_filename, models_dir, manager, description=description
    )


def recommend_paths_for_model(
    name, model_path=None, base_filename=None, models_dir=None
):
    """Recommends a base filename and models directory for the given model,
    if possible, using the provided information to inform the recommendation.

    The recommendations are made using the first applicable option below:
        (a) if `base_filename` or `models_dir` is provided, that value is used
        (b) if an older version of the model exists, the `base_filename` and
            `models_dir` values are inherited from that model as necesasry
        (c) if `model_path` is provided, `base_filename` and `models_dir` are
            set to the filename and base directory of the model path, as
            necessary
        (d) `models_dir` is set to the first directory on the models path, if
            any
        (d) None is returned

    Args:
        name: the model name, which can optionally have "@<ver>" appended
            to assign a version to the model
        model_path: an optional path to the model on disk. If provided and
            this path may be used to recommended values as described above
        base_filename: an optional base filename to use when writing the model
            to disk. If provided, this value is directly returned. If not
            provided, a value is recommended as explained above
        models_dir: an optional directory in which to register the model. If
            provided, this value is directly returned. If not provided, a value
            is recommended as explained above

    Returns:
        base_filename: the recommended base filename for the model, or None if
            no recommendation could be made
        models_dir: the recommended models directory to store the model, or
            None if no recommendation could be made
    """
    try:
        base_name = Model.parse_name(name)[0]
        model, _rec_models_dir, _ = _find_model(base_name)
        _rec_base_filename = model.base_filename
    except ModelError:
        _rec_base_filename = None
        _rec_models_dir = None

    # Recommend base filename
    if not base_filename:
        if _rec_base_filename:
            base_filename = _rec_base_filename
            logger.info(
                "Found a previous model version '%s'; recommending the same "
                "base filename: '%s'",
                model.name,
                base_filename,
            )
        elif model_path:
            base_filename = os.path.basename(model_path)
            logger.info(
                "No previous model version found; recommending the base "
                "filename of the input model path: '%s'",
                base_filename,
            )
        else:
            logger.info("Unable to recommended a base filename...")
            base_filename = None

    # Recommend models directory
    if not models_dir:
        if _rec_models_dir:
            models_dir = _rec_models_dir
            logger.info(
                "Found a previous model version '%s'; recommending the same "
                "model directory: '%s'",
                model.name,
                models_dir,
            )
        elif model_path:
            models_dir = os.path.dirname(model_path)
            logger.info(
                "No previous model version found; recommending the parent "
                "directory of the model path as the models directory: '%s'",
                models_dir,
            )
        elif eta.config.models_dirs:
            models_dir = eta.config.models_dirs[0]
            logger.info(
                "No models directory was specified; recommending the first "
                "directory on the models search path: '%s'",
                models_dir,
            )
        else:
            logger.info("Unable to recommended a models directory...")
            models_dir = None

    return base_filename, models_dir


def register_model_dry_run(name, base_filename, models_dir):
    """Performs a dry-run of the model registration process to ensure that no
    errors will happen when a real model registration is performed.

    *** No files are modified by this function. ***

    This function performs the following actions:
        - verifies that the proposed model name is valid
        - verifies that no models exist with the given name
        - verifies that no filename conflicts will occur in the proposed
            model directory

    Args:
        name: the proposed name for the model, which can optionally have
            "@<ver>" appended to assign a version to the model
        base_filename: the proposed base filename (e.g. "model.npz") to use
            when storing this model locally on disk
        models_dir: the proposed directory in which to register the model

    Returns:
        the path to the proposed model as it will appear when downloaded

    Raises:
        ModelError: if the dry run failed for any reason
    """
    # Verify name
    logger.info("Verifying that model name '%s' is valid", name)
    base_name, version = Model.parse_name(name)
    model = Model(base_name, base_filename, None, version=version)

    # Verify novelty
    logger.info("Verifying that model '%s' does not yet exist", name)
    models = _list_models()[0]
    base_names = [mm[0].base_name for mm in itervalues(models)]
    if name in models:
        raise ModelError("Model '%s' already exists" % name)

    if name in base_names:
        raise ModelError(
            "A versioned model with base name '%s' already exists, and "
            "publishing a versionless model with the same name as a versioned "
            "model can lead to unexpected behavior. Please choose another "
            "model name." % name
        )

    if base_name in models:
        raise ModelError(
            "A versionless model with name '%s' already exists, and "
            "publishing a versioned model with the same name as a versionless "
            "model can lead to unexpected behavior. Please choose another "
            "model name." % base_name
        )

    # Verify no filename conflicts
    if ModelsManifest.dir_has_manifest(models_dir):
        logger.info(
            "Verifying that no filename conflicts exist in models "
            "directory '%s'",
            models_dir,
        )
        manifest = ModelsManifest.from_dir(models_dir)
        manifest.add_model(model)

    return model.get_path_in_dir(models_dir)


def register_model(name, base_filename, models_dir, manager, description=None):
    """Registers a new model in the given models directory.

    If the directory does not have a models manifest file, one is created.
    Note that new models directories are not automatically added to your
    models search path, so models in a newly initialized directory will not be
    findable until you add the directory to your models search path.

    Note that this function does not upload the model to remote storage, it
    simply registers the model in the ETA system. To publish the model to
    remote storage, use the relevant ModelManager.

    Args:
        name: a name for the model, which can optionally have "@<ver>" appended
            to assign a version to the model
        base_filename: the base filename (e.g. "model.npz") to use when storing
            this model locally on disk
        models_dir: the directory in which to register the model
        manager: the ModelManager instance for the model
        description: an optional description for the model

    Raises:
        ModelError: if the registration failed for any reason
    """
    _warn_if_not_on_search_path(models_dir)

    # Create model
    logger.info("Creating a new model '%s'", name)
    base_name, version = Model.parse_name(name)
    date_added = etau.get_localtime()
    model = Model(
        base_name,
        base_filename,
        manager,
        version=version,
        description=description,
        date_added=date_added,
    )

    # Initialize models directory, if necessary
    if not ModelsManifest.dir_has_manifest(models_dir):
        init_models_dir(models_dir)

    # Add model to manifest
    manifest = ModelsManifest.from_dir(models_dir)
    logger.info("Adding model '%s' to manifest in '%s'", name, models_dir)
    manifest.add_model(model)
    manifest.write_to_dir(models_dir)


def delete_model(name, force=False):
    """Permanently deletes the given model from local and remote storage.

    CAUTION: this cannot be undone!

    Args:
        name: the name of the model, which can have "@<ver>" appended to refer
            to a specific version of the model. If no version is specified, the
            latest version of the model is assumed
        force: whether to force delete the remote model (True) or display a
            confirmation message that the user must approve before deleting
            the remote model (False). The default is False

    Raises:
        ModelError: if the model could not be found
    """
    # Flush model locally
    flush_model(name)

    model, models_dir, manifest = _find_model(name)
    if force or etau.query_yes_no(
        "Are you sure you want to permanently delete this model from "
        "remote storage? This cannot be undone!",
        default="no",
    ):
        # Flush model remotely
        logger.info("Deleting model '%s' from remote storage", name)
        model.manager.delete_model()

        # Delete model from manifest
        manifest_path = manifest.make_manifest_path(models_dir)
        logger.info(
            "Removing model '%s' from manifest '%s'", name, manifest_path
        )
        manifest.remove_model(model.name)
        manifest.write_json(manifest_path)
    else:
        logger.info("Remote deletion of model '%s' aborted", name)


def _find_model(name):
    if Model.has_version_str(name):
        return _find_exact_model(name)

    return _find_latest_model(name)


def _find_exact_model(name):
    models, manifests = _list_models()
    if name not in models:
        raise ModelError("No model with name '%s' was found" % name)

    model, mdir = models[name]
    return model, mdir, manifests[mdir]


def _find_latest_model(base_name):
    _model = None
    _mdir = None

    models, manifests = _list_models()
    for model, mdir in itervalues(models):
        if model.base_name == base_name:
            if _model is None or model.comp_version > _model.comp_version:
                _model = model
                _mdir = mdir

    if _model is None:
        raise ModelError("No models found with base name '%s'" % base_name)

    if _model.has_version:
        logger.debug(
            "Found version %s of model '%s'", _model.version, base_name
        )

    return _model, _mdir, manifests[_mdir]


def _list_models(downloaded_only=False):
    models = {}
    manifests = {}
    for mdir in _get_models_search_path():
        manifest = ModelsManifest.from_dir(mdir)
        manifests[mdir] = manifest

        for model in manifest:
            if model.name in models:
                raise ModelError(
                    "Found two '%s' models. Names must be unique" % model.name
                )

            if not downloaded_only or model.is_in_dir(mdir):
                models[model.name] = (model, mdir)

    return models, manifests


def _get_models_search_path():
    mdirs = []
    for mdir in etau.make_search_path(eta.config.models_dirs):
        if ModelsManifest.dir_has_manifest(mdir):
            mdirs.append(mdir)
        else:
            logger.debug(
                "Directory '%s' is on the models search path but has no "
                "manifest; omitting from search path",
                mdir,
            )

    return mdirs


def _warn_if_not_on_search_path(models_dir):
    mdir = os.path.abspath(models_dir)
    if mdir not in _get_models_search_path():
        logger.warning(
            "Directory '%s' is not on the ETA models search path", models_dir
        )


class ModelRequirements(Serializable):
    """Requirements for running a model.

    Example requirements::

        {
            "packages": [
                "numpy==1.14.0"
            ],
            "cpu": {
                "support": true,
                "packages": [
                    "tensorflow>=1.14,<2"
                ]
            },
            "gpu": {
                "support": false,
                "cuda_version": ">=9",
                "cudnn_version": ">=7.5",
                "packages": [
                    "tensorflow-gpu>=1.14,<2"
                ]
            }
        }

    Attributes:
        packages: (optional) a list of `setuptools`-style package requirements
            in order to use the model
        cpu: (optional) a CPU requirements dict
        gpu: (optional) a GPU requirements dict
    """

    def __init__(self, packages=None, cpu=None, gpu=None):
        self.packages = packages
        self.cpu = cpu
        self.gpu = gpu

    @property
    def supports_cpu(self):
        """Whether the model supports CPU (True), or not (False), or unknown
        (None).
        """
        if self.cpu is None:
            return None

        return self.cpu.get("support", None)

    @property
    def supports_gpu(self):
        """Whether the model supports GPU (True), or not (False), or unknown
        (None).
        """
        if self.gpu is None:
            return None

        return self.gpu.get("support", None)

    @property
    def cpu_packages(self):
        """A list of `setuptools`-style package requirements in order to use
        the model on CPU, or ``None`` otherwise.
        """
        if self.cpu is None:
            return None

        return self.cpu.get("packages", None)

    @property
    def gpu_packages(self):
        """A list of `setuptools`-style package requirements in order to use
        the model on GPU, or ``None`` otherwise.
        """
        if self.gpu is None:
            return None

        return self.gpu.get("packages", None)

    def install_base_requirements(self, error_level=0, error_suffix=None):
        """Installs any base package requirements for the model.

        Args:
            error_level: the error level to use, defined as:

                0: raise error if a package install fails
                1: log warning if a package install fails
                2: ignore package install fails
            error_suffix: an optional message to append to the error if the
                installation fails and ``error_level == 0``
        """
        if self.packages is None:
            return

        for requirement_str in self.packages:
            etau.install_package(
                requirement_str,
                error_level=error_level,
                error_suffix=error_suffix,
            )

    def install_cpu_requirements(self, error_level=0, error_suffix=None):
        """Installs any CPU package requirements for the model.

        Args:
            error_level: the error level to use, defined as:

                0: raise error if a package install fails
                1: log warning if a package install fails
                2: ignore package install fails
            error_suffix: an optional message to append to the error if the
                installation fails and ``error_level == 0``
        """
        if self.cpu_packages is None:
            return

        for requirement_str in self.cpu_packages:
            etau.install_package(
                requirement_str,
                error_level=error_level,
                error_suffix=error_suffix,
            )

    def install_gpu_requirements(self, error_level=0, error_suffix=None):
        """Installs any GPU package requirements for the model.

        Args:
            error_level: the error level to use, defined as:

                0: raise error if a package install fails
                1: log warning if a package install fails
                2: ignore package install fails
            error_suffix: an optional message to append to the error if the
                installation fails and ``error_level == 0``
        """
        if self.gpu_packages is None:
            return

        for requirement_str in self.gpu_packages:
            etau.install_package(
                requirement_str,
                error_level=error_level,
                error_suffix=error_suffix,
            )

    def ensure_base_requirements(
        self, error_level=0, error_suffix=None, log_success=False
    ):
        """Ensures that any base package requirements for the model are
        satisfied.

        Args:
            error_level: the error level to use, defined as:

                0: raise error if a requirement is not satisfied
                1: log warning if a requirement is not satisifed
                2: ignore unsatisifed requirements
            error_suffix: an optional message to append to the error if a
                requirement is not satisifed and ``error_level == 0``
            log_success: whether to generate a log message when a requirement
                is satisifed
        """
        if self.packages is None or error_level >= 2:
            return

        for requirement_str in self.packages:
            etau.ensure_package(
                requirement_str,
                error_level=error_level,
                error_suffix=error_suffix,
                log_success=log_success,
            )

    def ensure_cpu_requirements(
        self, error_level=0, error_suffix=None, log_success=False
    ):
        """Ensures that any CPU package requirements for the model are
        satisfied.

        Args:
            error_level: the error level to use, defined as:

                0: raise error if a requirement is not satisfied
                1: log warning if a requirement is not satisifed
                2: ignore unsatisifed requirements
            error_suffix: an optional message to append to the error if a
                requirement is not satisifed and ``error_level == 0``
            log_success: whether to generate a log message when a requirement
                is satisifed
        """
        if self.cpu_packages is None or error_level >= 2:
            return

        for requirement_str in self.cpu_packages:
            etau.ensure_package(
                requirement_str,
                error_level=error_level,
                error_suffix=error_suffix,
                log_success=log_success,
            )

    def ensure_gpu_requirements(
        self, error_level=0, error_suffix=None, log_success=False
    ):
        """Ensures that any GPU package requirements for the model are
        satisfied.

        Args:
            error_level: the error level to use, defined as:

                0: raise error if a requirement is not satisfied
                1: log warning if a requirement is not satisifed
                2: ignore unsatisifed requirements
            error_suffix: an optional message to append to the error if a
                requirement is not satisifed and ``error_level == 0``
            log_success: whether to generate a log message when a requirement
                is satisifed
        """
        if self.gpu_packages is None or error_level >= 2:
            return

        self._ensure_cuda(error_level, error_suffix)

        for requirement_str in self.gpu_packages:
            etau.ensure_package(
                requirement_str,
                error_level=error_level,
                error_suffix=error_suffix,
                log_success=log_success,
            )

    def _ensure_cuda(self, error_level, error_suffix):
        if self.gpu is None or error_level >= 2:
            return

        cuda_version = self.gpu.get("cuda_version", None)
        if cuda_version is not None:
            etau.ensure_cuda_version(
                cuda_version,
                error_level=error_level,
                error_suffix=error_suffix,
            )

        cudnn_version = self.gpu.get("cudnn_version", None)
        if cudnn_version is not None:
            etau.ensure_cudnn_version(
                cudnn_version,
                error_level=error_level,
                error_suffix=error_suffix,
            )

    def attributes(self):
        """Returns the list of class attributes that will be serialized.

        Returns:
            a list of attributes
        """
        _attrs = []
        if self.packages is not None:
            _attrs.append("packages")
        if self.cpu is not None:
            _attrs.append("cpu")
        if self.gpu is not None:
            _attrs.append("gpu")
        return _attrs

    @classmethod
    def from_dict(cls, d):
        """Creates a ModelRequirements from a JSON dict.

        Args:
            d: a JSON dict

        Returns:
            a ModelRequirements instance
        """
        packages = d.get("packages", None)
        cpu = d.get("cpu", None)
        gpu = d.get("gpu", None)
        return cls(packages=packages, cpu=cpu, gpu=gpu)


class Model(Serializable):
    """Class that describes a model.

    Attributes:
        base_name: the base name of the model (no version info)
        base_filename: the base filename of the model (no version info)
        manager: the ModelManager instance that describes the remote storage
            location of the models_dir
        version: the version of the model (if any)
        description: the description of the model (if any)
        source: the source of the model (if any)
        size_bytes: the size of the model on disk (if any)
        default_deployment_config_dict: a dictionary representation of an
            `eta.core.learning.ModelConfig` describing the recommended settings
            for deploying the model
        requirements: the ModelRequirements for the model (if any)
        tags: a list of tags for the model (if any)
        date_added: the datetime that the model was added (if any)
    """

    # An optional error message to append to any error messages resulting from
    # unsatisifed requirements
    _REQUIREMENT_ERROR_SUFFIX = None

    def __init__(
        self,
        base_name,
        base_filename,
        manager,
        version=None,
        description=None,
        source=None,
        size_bytes=None,
        default_deployment_config_dict=None,
        requirements=None,
        tags=None,
        date_added=None,
    ):
        """Creates a Model instance.

        Args:
            base_name: the base name of the model
            base_filename: the base filename for the model
            manager: the ModelManager for the model
            version: (optional) the model version
            description: (optional) the description of the model
            source: (optional) the source of the model
            size_bytes: (optional) the size of the model on disk
            default_deployment_config_dict: (optional) a dictionary
                representation of an `eta.core.learning.ModelConfig` describing
                the recommended settings for deploying the model
            requirements: (optional) a ModelRequirements for the model
            tags: (optional) a list of tags for the model
            date_added: (optional) the datetime that the model was created
        """
        self.base_name = base_name
        self.base_filename = base_filename
        self.manager = manager
        self.version = version or None
        self.description = description
        self.source = source
        self.size_bytes = size_bytes
        self.default_deployment_config_dict = default_deployment_config_dict
        self.requirements = requirements
        self.tags = tags
        self.date_added = date_added

    @property
    def name(self):
        """The version-aware name of the model."""
        if not self.has_version:
            return self.base_name

        base, ext = os.path.splitext(self.base_name)
        return base + "@" + self.version + ext

    @property
    def filename(self):
        """The version-aware filename of the model."""
        if not self.has_version:
            return self.base_filename

        base, ext = os.path.splitext(self.base_filename)
        return base + "-v" + self.version + ext

    @property
    def has_version(self):
        """Determines whether the model has a version."""
        return self.version is not None

    @property
    def comp_version(self):
        """The version of this model expressed as a
        `distutils.version.LooseVersion` intended for comparison operations.

        Models with no version are given a version of 0.0.0.
        """
        return LooseVersion(self.version or "0.0.0")

    @property
    def has_requirements(self):
        """Whether this model has requirements in order to be used."""
        return self.requirements is not None

    @property
    def supports_cpu(self):
        """Whether the model supports CPU (True), or not (False), or unknown
        (None).
        """
        if not self.has_requirements:
            return None

        return self.requirements.supports_cpu

    @property
    def supports_gpu(self):
        """Whether the model supports GPU (True), or not (False), or unknown
        (None).
        """
        if not self.has_requirements:
            return None

        return self.requirements.supports_gpu

    @property
    def has_tags(self):
        """Whether this model has tags."""
        return self.tags is not None

    def has_tag(self, tag):
        """Whether this model has the given tag.

        Args:
            tag: a tag

        Returns:
            True/False
        """
        if not self.has_tags:
            return False

        return tag in self.tags

    def install_requirements(self, error_level=0):
        """Installs any necessary requirement(s) for this model.

        Args:
            error_level: the error level to use, defined as:

                0: raise error if an install fails
                1: log warning if an install fails
                2: ignore install fails
        """
        if not self.has_requirements:
            return

        self.requirements.install_base_requirements(
            error_level=error_level,
            error_suffix=self._REQUIREMENT_ERROR_SUFFIX,
        )

        found_gpu = self._ensure_environment(error_level)

        if found_gpu:
            self.requirements.install_gpu_requirements(
                error_level=error_level,
                error_suffix=self._REQUIREMENT_ERROR_SUFFIX,
            )
        else:
            self.requirements.install_cpu_requirements(
                error_level=error_level,
                error_suffix=self._REQUIREMENT_ERROR_SUFFIX,
            )

    def ensure_requirements(self, error_level=0, log_success=False):
        """Ensures that any requirement(s) for this model are satisfied.

        Args:
            error_level: the error level to use, defined as:

                0: raise error if a requirement is not satisfied
                1: log warning if a requirement is not satisifed
                2: ignore unsatisifed requirements

            log_success: whether to generate a log message when a requirement
                is satisifed
        """
        if not self.has_requirements or error_level >= 2:
            return

        self.requirements.ensure_base_requirements(
            error_level=error_level,
            error_suffix=self._REQUIREMENT_ERROR_SUFFIX,
            log_success=log_success,
        )

        found_gpu = self._ensure_environment(error_level)

        if found_gpu:
            self.requirements.ensure_gpu_requirements(
                error_level=error_level,
                error_suffix=self._REQUIREMENT_ERROR_SUFFIX,
                log_success=log_success,
            )
        else:
            self.requirements.ensure_cpu_requirements(
                error_level=error_level,
                error_suffix=self._REQUIREMENT_ERROR_SUFFIX,
                log_success=log_success,
            )

    def _ensure_environment(self, error_level):
        if not self.has_requirements:
            return

        found_gpu = etau.has_gpu()

        if found_gpu:
            if self.requirements.supports_gpu == False:  # False, not None
                error_msg = (
                    "Model '%s' requires GPU but no GPU was found" % self.name
                )

                if self._REQUIREMENT_ERROR_SUFFIX:
                    error_msg += "\n\n" + self._REQUIREMENT_ERROR_SUFFIX

                etau.handle_error(ModelError(error_msg), error_level)
        else:
            if self.requirements.supports_cpu == False:  # False not None
                error_msg = (
                    "Model '%s' does not support CPU and no GPU was found"
                    % self.name
                )

                if self._REQUIREMENT_ERROR_SUFFIX:
                    error_msg += "\n\n" + self._REQUIREMENT_ERROR_SUFFIX

                etau.handle_error(ModelError(error_msg), error_level)

        return found_gpu

    def get_path_in_dir(self, models_dir):
        """Gets the model path for the model in the given models directory.

        Args:
            models_dir: the models directory

        Returns:
            the model path
        """
        return os.path.join(models_dir, self.filename)

    def is_in_dir(self, models_dir):
        """Determines whether a copy of the model exists in the given models
        directory.

        Args:
            models_dir: the models directory

        Returns:
            True/False
        """
        model_path = self.get_path_in_dir(models_dir)
        return self.is_model_downloaded(model_path)

    def is_model_downloaded(self, model_path):
        """Determines whether the model is downloaded to the given location.

        If `model_path` is an archive, this method will also return `True` if a
        directory with the same basename as `model_path` exists.

        Args:
            model_path: the path on disk for the model

        Returns:
            True/False
        """
        return self.manager.is_model_downloaded(model_path)

    def flush_model(self, model_path):
        """Flushes the copy of the model at the given local path, if necessary.

        Args:
            model_path: the path on disk for the model
        """
        self.manager.flush_model(model_path)

    def flush_model_from_dir(self, models_dir):
        """Flushes the copy of the model in the given models directory, if
        necessary.

        Args:
            models_dir: the models directory
        """
        model_path = self.get_path_in_dir(models_dir)
        self.flush_model(model_path)

    @staticmethod
    def parse_name(name):
        """Parses the model name, returning the base name and the version,
        if any.

        Args:
            name: the name of the model, which can have "@<ver>" appended to
                refer to a specific version of the model

        Returns:
            base_name: the base name of the model
            version: the version of the model, or None if no version was found

        Raises:
            ModelError: if the model name was invalid
        """
        chunks = name.split("@")
        if len(chunks) == 1:
            return name, None

        if chunks[1] == "" or len(chunks) > 2:
            raise ModelError("Invalid model name '%s'" % name)

        return chunks[0], chunks[1]

    @staticmethod
    def has_version_str(name):
        """Determines whether the given model name has a version string.

        Args:
            name: the model name

        Returns:
            True/False
        """
        return bool(Model.parse_name(name)[1])

    def attributes(self):
        """Returns a list of class attributes to be serialized.

        Returns:
            a list of class attributes
        """
        return [
            "base_name",
            "base_filename",
            "version",
            "description",
            "source",
            "size_bytes",
            "manager",
            "default_deployment_config_dict",
            "requirements",
            "tags",
            "date_added",
        ]

    @classmethod
    def from_dict(cls, d):
        """Constructs a Model from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a Model instance
        """
        model_manager = ModelManager.from_dict(d["manager"])

        requirements = d.get("requirements", None)
        if requirements is not None:
            requirements = ModelRequirements.from_dict(requirements)

        tags = d.get("tags", None)

        date_added = etau.parse_isotime(d.get("date_added"))

        return cls(
            d["base_name"],
            d["base_filename"],
            model_manager,
            version=d.get("version", None),
            description=d.get("description", None),
            source=d.get("source", None),
            size_bytes=d.get("size_bytes", None),
            default_deployment_config_dict=d.get(
                "default_deployment_config_dict", None
            ),
            requirements=requirements,
            tags=tags,
            date_added=date_added,
        )


class ModelsManifest(Serializable):
    """Class that describes the contents of a models directory."""

    _MODEL_CLS = Model

    def __init__(self, models=None):
        """Creates a ModelsManifest instance.

        Args:
            models: a list of Model instances
        """
        self.models = models or []

    def __iter__(self):
        return iter(self.models)

    def add_model(self, model):
        """Adds the given model to the manifest.

        Args:
            model: a Model instance

        Raises:
            ModelError: if the model conflicts with an existing model in the
                manifest
        """
        if self.has_model_with_name(model.name):
            raise ModelError(
                "Manifest already contains model called '%s'" % model.name
            )

        if self.has_model_with_filename(model.filename):
            raise ModelError(
                "Manifest already contains model with filename '%s'"
                % (model.filename)
            )

        if self.has_model_with_name(model.base_name):
            raise ModelError(
                "Manifest already contains a versionless model called '%s', "
                "so a versioned model is not allowed" % model.base_name
            )

        self.models.append(model)

    def remove_model(self, name):
        """Removes the model with the given name from the ModelsManifest.

        Args:
            name: the name of the model

        Raises:
            ModelError: if the model was not found
        """
        if not self.has_model_with_name(name):
            raise ModelError("Manifest does not contain model '%s'" % name)

        self.models = [model for model in self.models if model.name != name]

    def merge(self, models_manifest):
        """Merges the models manifest into this one.

        Args:
            models_manifest: a ModelsManifest
        """
        for model in models_manifest:
            self.add_model(model)

    def get_model_with_name(self, name):
        """Gets the model with the given name.

        Args:
            name: the name of the model

        Returns:
            the Model instance

        Raises:
            ModelError: if the model was not found
        """
        for model in self.models:
            if name == model.name:
                return model

        raise ModelError("Manifest does not contain model '%s'" % name)

    def get_latest_model_with_base_name(self, base_name):
        """Gets the Model instance for the latest version of the model with the
        given base name.

        Args:
            base_name: the base name of the model

        Returns:
            the Model instance

        Raises:
            ModelError: if the model was not found
        """
        _model = None
        for model in self.models:
            if base_name == model.base_name:
                if _model is None or model.comp_version > _model.comp_version:
                    _model = model

        if _model is None:
            raise ModelError(
                "Manifest does not contain model '%s'" % base_name
            )

        return _model

    def has_model_with_name(self, name):
        """Determines whether this manifest contains the model with the
        given name.

        Args:
            name: the model name

        Returns:
            True/False
        """
        return any(name == model.name for model in self.models)

    def has_model_with_filename(self, filename):
        """Determines whether this manifest contains a model with the given
        filename.

        Args:
            filename: the filename

        Returns:
            True/False
        """
        return any(filename == model.filename for model in self.models)

    @staticmethod
    def make_manifest_path(models_dir):
        """Makes the manifest path for the given models directory.

        Args:
            models_dir: the models directory

        Returns:
             the manifest path
        """
        return os.path.join(models_dir, MODELS_MANIFEST_JSON)

    @staticmethod
    def dir_has_manifest(models_dir):
        """Determines whether the given directory has a models manifest.

        Args:
            models_dir: the models directory

        Returns:
            True/False
        """
        return os.path.isfile(ModelsManifest.make_manifest_path(models_dir))

    def write_to_dir(self, models_dir):
        """Writes the ModelsManifest to the given models directory.

        Args:
            models_dir: the models directory
        """
        self.write_json(self.make_manifest_path(models_dir))

    @classmethod
    def from_dir(cls, models_dir):
        """Loads the ModelsManifest from the given models directory.

        Args:
            models_dir: the models directory

        Returns:
            a ModelsManifest
        """
        if not cls.dir_has_manifest(models_dir):
            raise ModelError(
                "Directory '%s' has no models manifest" % models_dir
            )

        return cls.from_json(cls.make_manifest_path(models_dir))

    @classmethod
    def from_dict(cls, d):
        """Constructs a ModelsManifest from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a ModelsManifest
        """
        return cls(models=[cls._MODEL_CLS.from_dict(md) for md in d["models"]])


class ModelManager(Configurable, Serializable):
    """Base class for model managers.

    Attributes:
        type: the fully-qualified name of the ModelManager subclass
        config: the Config instance for the ModelManager subclass
    """

    def __init__(self, config):
        """Initializes a ModelManager instance.

        Args:
            config: a Config for the ModelManager subclass
        """
        self.validate(config)
        self.type = etau.get_class_name(self)
        self.config = config

    @staticmethod
    def upload_model(model_path, *args, **kwargs):
        raise NotImplementedError("subclass must implement upload_model()")

    def is_model_downloaded(self, model_path):
        """Determines whether the model is downloaded to the given location.

        If `model_path` is an archive, this method will also return `True` if a
        directory with the same basename as `model_path` exists.

        Args:
            model_path: the path on disk for the model

        Returns:
            True/False
        """
        if etau.is_archive(model_path):
            archive_dir = etau.split_archive(model_path)[0]
            if os.path.isdir(archive_dir):
                # Extracted archive already exists
                return True

        return os.path.isfile(model_path)

    def download_model(self, model_path, force=False):
        """Downloads the model to the given local path.

        If the download is forced, any existing model is overwritten. If the
        download is not forced, the model will only be downloaded if it does
        not already exist locally.

        Args:
            model_path: the path to which to download the model
            force: whether to force download the model. If True, the model is
                always downloaded. If False, the model is only downloaded if
                necessary. The default is False

        Raises:
            ModelError: if model downloading is not currently allowed
        """
        if not force and self.is_model_downloaded(model_path):
            return

        if not eta.config.allow_model_downloads:
            raise ModelError(
                "Model downloading is currently disabled. Modify your ETA "
                "config to change this setting."
            )

        etau.ensure_basedir(model_path)
        self._download_model(model_path)

        if self.config.extract_archive:
            if self.config.delete_archive is not None:
                delete_archive = self.config.delete_archive
            else:
                delete_archive = False

            logger.info("Extracting archive '%s'", model_path)
            etau.extract_archive(model_path, delete_archive=delete_archive)

    def flush_model(self, model_path):
        """Flushes the copy of the model at the given local path, if necessary.

        Args:
            model_path: the path on disk for the model
        """
        if etau.is_archive(model_path):
            archive_dir = etau.split_archive(model_path)[0]
            if os.path.isdir(archive_dir):
                etau.delete_dir(archive_dir)

        if os.path.isfile(model_path):
            etau.delete_file(model_path)

    def delete_model(self):
        """Deletes the model from remote storage."""
        raise NotImplementedError("subclass must implement delete_model()")

    def _download_model(self, model_path):
        """Subclass implementation of downloading the model to the given path.

        Args:
            model_path: the path to which to download the model
        """
        raise NotImplementedError("subclass must implement _download_model()")

    def attributes(self):
        """Returns a list of attributes to be serialized.

        Returns:
            a list of attributes
        """
        return ["type", "config"]

    @classmethod
    def from_dict(cls, d):
        """Builds the ModelManager subclass from a JSON dictionary.

        Args:
            d: a JSON dictionary

        Returns:
            a ModelManager instance
        """
        manager_cls, config_cls = cls.parse(d["type"])
        return manager_cls(config_cls.from_dict(d["config"]))


class ModelManagerConfig(Config):
    """Base configuration settings for a ModelManager instance.

    All ModelManagerConfig subclasses must call `super().__init__()`.

    Attributes:
        extract_archive: whether to extract the downloaded model, which is
            assumed to be an archive
        delete_archive: whether to delete the archive after extracting it, if
            applicable
    """

    def __init__(self, d):
        self.extract_archive = self.parse_bool(
            d, "extract_archive", default=None
        )
        self.delete_archive = self.parse_bool(
            d, "delete_archive", default=None
        )

    def attributes(self):
        """Returns a list of attributes to be serialized.

        Returns:
            a list of attributes
        """
        # Omit attributes with no value, for clarity
        return [a for a in vars(self) if getattr(self, a) is not None]


class ETAModelManagerConfig(ModelManagerConfig):
    """Configuration settings for an ETAModelManager instance.

    Attributes:
        url: the URL of the file
        google_drive_id: the ID of the file in Google Drive
        extract_archive: whether to extract the downloaded model, which is
            assumed to be an archive
        delete_archive: whether to delete the archive after extracting it, if
            applicable
    """

    def __init__(self, d):
        super(ETAModelManagerConfig, self).__init__(d)

        self.url = self.parse_string(d, "url", default=None)
        self.google_drive_id = self.parse_string(
            d, "google_drive_id", default=None
        )


class ETAModelManager(ModelManager):
    """Class that manages public models for the ETA repository."""

    @staticmethod
    def upload_model(model_path, *args, **kwargs):
        raise NotImplementedError(
            "ETA models must be uploaded by a Voxel51 administrator. "
            "Please contact %s for more information." % etac.AUTHOR_EMAIL
        )

    def _download_model(self, model_path):
        if self.config.google_drive_id:
            gid = self.config.google_drive_id
            try:
                logger.info(
                    "Downloading model from Google Drive ID '%s' to '%s'",
                    gid,
                    model_path,
                )
                etaw.download_google_drive_file(gid, path=model_path)
            except etaw.WebSessionError as e:
                logger.error("***** FAILED TO DOWNLOAD '%s' *****", gid)
                logger.error(e, exc_info=sys.exc_info())

                #
                # Fallback to downloading via GoogleDriveStorageClient, which
                # will only work for folks with credentials to the Drive
                #
                logger.warning("Trying GoogleDriveStorageClient instead...")
                client = etas.GoogleDriveStorageClient()
                client.download(gid, model_path)
                logger.warning("Bandage applied")

        elif self.config.url:
            url = self.config.url
            logger.info("Downloading model from '%s' to '%s'", url, model_path)
            etaw.download_file(url, path=model_path)
        else:
            raise ModelError(
                "Invalid ETAModelManagerConfig '%s'" % str(self.config)
            )

    def delete_model(self):
        raise NotImplementedError(
            "ETA models must be deleted by a Voxel51 administrator. "
            "Please contact %s for more information." % etac.AUTHOR_EMAIL
        )


class ModelError(Exception):
    """Exception raised when an invalid model is encountered."""

    pass
