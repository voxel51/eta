'''
Core infrastructure for managing models across local and remote storage.

@todo explain model storage architecture.

Copyright 2017-2018, Voxel51, LLC
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
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import dill as pickle
from distutils.version import LooseVersion
import logging
import os

import numpy as np

import eta
import eta.constants as etac
from eta.core.config import Config, Configurable
from eta.core.serial import Serializable
import eta.core.utils as etau
import eta.core.web as etaw


logger = logging.getLogger(__name__)


MODELS_MANIFEST_JSON = "manifest.json"


def find_model(name):
    '''Finds the given model, which must appear in a ModelsManifest in one of
    the `eta.config.models_dirs` directories.

    Args:
        name: the name of the model, which can have "@<ver>" appended to refer
            to a specific version of the model. If no version is specified, the
            latest version of the model is assumed

    Returns:
        the full path to the model file

    Raises:
        ModelError: if the model could not be found
    '''
    model, models_dir, _ = _find_model(name)
    return model.get_path_in_dir(models_dir)


def is_model_downloaded(name):
    '''Determines whether the given model is downloaded.

    Args:
        name: the name of the model, which can have "@<ver>" appended to refer
            to a specific version of the model. If no version is specified, the
            latest version of the model is assumed

    Returns:
        True/False whether the model is downloaded

    Raises:
        ModelError: if the model could not be found
    '''
    model, models_dir, _ = _find_model(name)
    return model.is_in_dir(models_dir)


def download_model_if_necessary(name):
    '''Downloads the given model if necessary.

    Args:
        name: the name of the model, which can have "@<ver>" appended to refer
            to a specific version of the model. If no version is specified, the
            latest version of the model is assumed

    Returns:
        the path to the downloaded model

    Raises:
        ModelError: if the model could not be found
    '''
    model, models_dir, _ = _find_model(name)
    model_path = model.get_path_in_dir(models_dir)
    model.manager.download_if_necessary(model_path)
    return model_path


def flush_model(name):
    '''Deletes the local copy of the given model, if necessary.

    Args:
        name: the name of the model, which can have "@<ver>" appended to refer
            to a specific version of the model. If no version is specified, the
            latest version of the model is assumed

    Raises:
        ModelError: if the model could not be found
    '''
    model, models_dir, _ = _find_model(name)
    if model.is_in_dir(models_dir):
        _delete_model_from_dir(model, models_dir)


def flush_models_directory(models_dir):
    '''Deletes the local copies of all models in the given models directory.

    Args:
        models_dir: the models directory

    Raises:
        ModelError: if the directory contains no models manifest
    '''
    mdir = os.path.abspath(models_dir)
    mdirs = etau.make_search_path(eta.config.models_dirs)
    if mdir not in mdirs:
        logger.warning(
            "Directory '%s' is not on the ETA models path", models_dir)

    if not ModelsManifest.dir_has_manifest(models_dir):
        raise ModelError(
            "Directory '%s' has no models manifest file", models_dir)

    for model in ModelsManifest.from_dir(models_dir).models:
        if model.is_in_dir(models_dir):
            _delete_model_from_dir(model, models_dir)


def flush_all_models():
    '''Deletes all local copies of all models on the models search path.'''
    for models_dir in etau.make_search_path(eta.config.models_dirs):
        flush_models_directory(models_dir)


def delete_model(name, force=False):
    '''Permanently deletes the given model from local and remote storage.

    Args:
        name: the name of the model, which can have "@<ver>" appended to refer
            to a specific version of the model. If no version is specified, the
            latest version of the model is assumed
        force: whether to force delete the remote model (True) or display a
            confirmation message that the user must approve before deleting
            the remote model (False). The default is False

    Raises:
        ModelError: if the model could not be found
    '''
    # Flush model locally
    flush_model(name)

    model, models_dir, manifest = _find_model(name)
    if force or etau.query_yes_no(
            "Are you sure you want to permanently delete this model from "
            "remote storage? This cannot be undone!", default="no"):
        # Flush model remotely
        logger.info("Deleting model '%s' from remote storage", name)
        model.manager.delete_model()

        # Delete model from manifest
        manifest_path = manifest.make_manifest_path(models_dir)
        logger.info(
            "Removing model '%s' from manifest '%s'", name, manifest_path)
        manifest.remove_model(model.name)
        manifest.write_json(manifest_path)
    else:
        logger.info("Remote deletion of model '%s' aborted", name)


def _find_model(name):
    if "@" in name:
        return _find_exact_model(name)
    return _find_latest_model(name)


def _find_exact_model(name):
    models, manifests = _load_models()
    if name not in models:
        raise ModelError("No model with name '%s' was found" % name)

    model, mdir = models[name]
    return model, mdir, manifests[mdir]


def _find_latest_model(base_name):
    _model = None
    _mdir = None

    models, manifests = _load_models()
    for model, mdir in itervalues(models):
        if model.base_name == base_name:
            if _model is None or model.comp_version > _model.comp_version:
                _model = model
                _mdir = mdir

    if _model is None:
        raise ModelError("No model with base name '%s' was found" % base_name)
    if _model.has_version:
        logger.info(
            "Found version %s of model '%s'", _model.version, base_name)

    return _model, _mdir, manifests[_mdir]


def _load_models(local_only=False):
    models = {}
    manifests = {}

    mdirs = etau.make_search_path(eta.config.models_dirs)
    for mdir in mdirs:
        manifest = ModelsManifest.from_dir(mdir)
        manifests[mdir] = manifest

        for model in manifest:
            if model.name in models:
                raise ModelError(
                    "Found two '%s' models. Names must be unique" % model.name)
            if not local_only or model.is_in_dir(mdir):
                models[model.name] = (model, mdir)

    return models, manifests


def _delete_model_from_dir(model, models_dir):
    model_path = model.get_path_in_dir(models_dir)
    logger.info(
        "Deleting local copy of model '%s' from '%s'", model.name, model_path)
    os.remove(model_path)


class ModelsManifest(Serializable):
    '''Class that describes the contents of a models directory.'''

    def __init__(self, models=None):
        '''Creates a ModelsManifest instance.

        Args:
            models: a list of Model instances
        '''
        self.models = models or []

    def __iter__(self):
        return iter(self.models)

    def add_model(self, model):
        '''Adds the given model to the manifest.

        Args:
            model: a Model instance

        Raises:
            ModelError: if the model conflicts with an existing model in the
                manifest
        '''
        if self.has_model_with_name(model.name):
            raise ModelError(
                "Manifest already contains model '%s'" % model.name)
        if self.has_model_with_filename(model.filename):
            raise ModelError(
                "Manifest already contains model with filename '%s'" % (
                    model.filename))
        self.models.append(model)

    def remove_model(self, name):
        '''Removes the model with the given name from the ModelsManifest.

        Args:
            name: the name of the model

        Raises:
            ModelError: if the model was not found
        '''
        if not self.has_model_with_name(name):
            raise ModelError(
                "Manifest does not contain model '%s'" % name)
        self.models = [model for model in self.models if model.name != name]

    def get_model_with_name(self, name):
        '''Gets the model with the given name.

        Args:
            name: the name of the model

        Returns:
            the Model instance

        Raises:
            ModelError: if the model was not found
        '''
        for model in self.models:
            if name == model.name:
                return model

        raise ModelError("Manifest does not contain model '%s'" % name)

    def get_latest_model_with_base_name(self, base_name):
        '''Gets the Model instance for the latest version of the model with the
        given base name.

        Args:
            base_name: the base name of the model

        Returns:
            the Model instance

        Raises:
            ModelError: if the model was not found
        '''
        _model = None
        for model in self.models:
            if base_name == model.base_name:
                if _model is None or model.comp_version > _model.comp_version:
                    _model = model

        if _model is None:
            raise ModelError(
                "Manifest does not contain model '%s'" % base_name)

        return _model

    def has_model_with_name(self, name):
        '''Determines whether this manifest contains the model with the
        given name.
        '''
        return any(name == model.name for model in self.models)

    def has_model_with_base_name(self, base_name):
        '''Determines whether this manifest contains a model with the given
        base name.'''
        return any(base_name == model.base_name for model in self.models)

    def has_model_with_filename(self, filename):
        '''Determines whether this manifest contains a model with the given
        filename.
        '''
        return any(filename == model.filename for model in self.models)

    @staticmethod
    def make_manifest_path(models_dir):
        '''Makes the manifest path for the given models directory.'''
        return os.path.join(models_dir, MODELS_MANIFEST_JSON)

    @staticmethod
    def dir_has_manifest(models_dir):
        '''Determines whether the given directory has a models manifest.'''
        return os.path.isfile(ModelsManifest.make_manifest_path(models_dir))

    @classmethod
    def from_dir(cls, models_dir):
        '''Loads the ModelsManifest from the given models directory. If no
        manifest is found, an empty one is created and '''
        manifest_path = cls.make_manifest_path(models_dir)
        if not cls.dir_has_manifest(models_dir):
            logger.warning(
                "No models manifest found at '%s'; creating an empty "
                "manifest now", manifest_path)
            manifest = cls()
            manifest.write_json(manifest_path)
        else:
            manifest = cls.from_json(manifest_path)
        return manifest

    @classmethod
    def from_dict(cls, d):
        '''Constructs a ModelsManifest from a JSON dictionary.'''
        return cls(models=[Model.from_dict(md) for md in d["models"]])


class Model(Serializable):
    '''Class that describes a model.

    Attributes:
        base_name: the base name of the model (no version info)
        base_filename: the base filename of the model (no version info)
        manager: the ModelManager instance that describes the remote storage
            location of the model
        date_created: the date that the model was created
        version: the version of the model, or None if it has no version
    '''

    def __init__(
            self, base_name, base_filename, manager, date_created,
            version=None):
        '''Creates a Model instance.

        Args:
            base_name: the base name of the model
            base_filename: the base filename for the model
            manager: the ModelManager for the model
            date_created: the date that the model was created
            version: (optional) the model version
        '''
        self.base_name = base_name
        self.base_filename = base_filename
        self.manager = manager
        self.date_created = date_created
        self.version = version or None

    @property
    def name(self):
        '''The version-aware name of the model.'''
        if not self.has_version:
            return self.base_name
        base, ext = os.path.splitext(self.base_name)
        return base + "@" + self.version + ext

    @property
    def filename(self):
        '''The version-aware filename of the model.'''
        if not self.has_version:
            return self.base_filename
        base, ext = os.path.splitext(self.base_filename)
        return base + "-" + self.version + ext

    @property
    def has_version(self):
        '''Determines whether the model has a version.'''
        return self.version is not None

    @property
    def comp_version(self):
        '''The version of this model expressed as a
        `distutils.version.LooseVersion` intended for comparison operations.

        Models with no version are given a version of 0.0.0.
        '''
        return LooseVersion(self.version or "0.0.0")

    def get_path_in_dir(self, models_dir):
        '''Gets the model path for the model in the given models directory.'''
        return os.path.join(models_dir, self.filename)

    def is_in_dir(self, models_dir):
        '''Determines whether a copy of the model exists in the given models
        directory.
        '''
        return os.path.isfiled(self.get_path_in_dir(models_dir))

    @classmethod
    def from_dict(cls, d):
        '''Constructs a Model from a JSON dictionary.'''
        return cls(
            d["base_name"], d["base_filename"],
            ModelManager.from_dict(d["manager"]), d["date_created"],
            d["version"])


class ModelWeights(object):
    '''Base class for classes that encapsulate read-only model weights.

    This class can load the model locally or from remote storage as needed.

    Subclasses must implement the `_load` method to perform the actual loading.

    Attributes:
        model_name: the name of the model
        model_path: the local path to the model on disk
    '''

    def __init__(self, model_name):
        '''Initializes a ModelWeights instance.

        Args:
            model_name: the model to load

        Raises:
            ModelError: if the model was not found
        '''
        self.model_name = model_name
        self.model_path = find_model(self.model_name)

    def load(self):
        '''Loads the model weights, downloading them from remote storage if
        necessary.

        Returns:
            the model weights
        '''
        download_model_if_necessary(self.model_name)
        return self._load()

    def _load(self):
        raise NotImplementedError("subclass must implement _load()")


class NpzModelWeights(ModelWeights, dict):
    '''A read-only model weights class that provides a dictionary interface to
    access the underlying weights, which must be stored as an .npz file on
    disk.
    '''

    def _load(self):
        self.update(np.load(self.model_path))
        return self


class PklModelWeights(ModelWeights):
    '''A read-only model weights class that can load a model stored as a .pkl
    file on disk.
    '''

    def _load(self):
        with open(self.model_path, "rb") as f:
            return pickle.load(f)


class ModelManager(Configurable, Serializable):
    '''Base class for model managers.'''

    def __init__(self, type_, config_):
        self.validate(config_)
        self.type = type_
        self.config = config_

    @staticmethod
    def register_model(model_path, name, version=None, models_dir=None):
        raise NotImplementedError(
            "subclass must implement register_new_model()")

    def delete_model(self):
        raise NotImplementedError(
            "subclass must implement delete_model()")

    def download_if_necessary(self, local_path):
        raise NotImplementedError(
            "subclass must implement download_if_necessary()")

    @classmethod
    def from_dict(cls, d):
        '''Builds the ModelManager subclass from a JSON dictionary.'''
        manager_cls, config_cls = cls.parse(d["type"])
        return manager_cls(d["type"], config_cls.from_dict(d["config"]))


class ETAModelManagerConfig(Config):
    '''Configuration settings for an ETAModelManager instance.

    Exactly one of the attributes should be set.

    Attributes:
        url: the URL of the file
        google_drive_id: the ID of the file in Google Drive
    '''

    def __init__(self, d):
        self.url = self.parse_string(d, "url", default=None)
        self.google_drive_id = self.parse_string(
            d, "google_drive_id", default=None)


class ETAModelManager(ModelManager):
    '''Class that manages public models for the ETA repository.'''

    @staticmethod
    def register_model(model_path, name, version=None, models_dir=None):
        raise NotImplementedError(
            "ETA models must be registered by a Voxel51 administrator. "
            "Please contact %s for more information." % etac.CONTACT)

    def delete_model(self):
        raise NotImplementedError(
            "ETA models must be deleted by a Voxel51 administrator. "
            "Please contact %s for more information." % etac.CONTACT)

    def download_if_necessary(self, local_path):
        '''Downloads the model to the given local path, if necessary.

        Args:
            local_path: the path to which to download the model

        Raises:
            ModelError: if the configuration was invalid
        '''
        if os.path.isfile(local_path):
            return

        if self.config.google_drive_id:
            gid = self.config.google_drive_id
            logger.info(
                "Downloading model from Google Drive ID '%s' to '%s'",
                gid, local_path)
            etaw.download_google_drive_file(gid, path=local_path)
        elif self.config.url:
            url = self.config.url
            logger.info(
                "Downloading model from '%s' to '%s'", url, local_path)
            etaw.download_file(url, path=local_path)
        else:
            raise ModelError(
                "Invalid ETAModelManagerConfig '%s'" % str(self.config))


class ModelError(Exception):
    '''Exception raised when an invalid model is encountered.'''
    pass
