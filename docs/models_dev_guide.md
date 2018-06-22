# Models Developer's Guide

This document describes the design of the model management system in ETA and
best practices for using it.


## Motivation

The ETA library relies on a number of large models in order to power its
analytics. Models are trained on different machines, by different developers,
and constantly need to be shared. We frequently generate new versions of a
given model over time as we improve the implementation of the analytic and
train it on evolving datasets. In addition, models are valuable, so they need
to be archived and backed up to ensure that they remain secure and readily
accessible.

The ETA library provides a simple, yet powerful interface that satisifies
these constraints.


## Design

The `eta.core.models` module implements the core interface for managing models
in ETA. The underlying design principle of the interface is that all models
are stored in cloud storage and can be easily downloaded to a local ETA
installation as needed using a simple version-aware naming system.

### Model versions

Each model is identified by a unique string of the form `<name>@<version>`,
where `<name>` is the canonical name of the model and `<version>` is an
optional version of the model.

If a model is registered without a version, it can be referenced simply by
its `<name>`. If a model is registered with a version, it can be referenced
explicitly using the `<name>@<version>` syntax. The `<name>` syntax can
also be used with versioned models, in which case the latest version of the
model (i.e. the largest version number) will be referenced.

#### Models search path

The `models_dirs` field of the ETA config defines the list of directories to
use when performing all models-related operations. If you add a new models
directory to your path, you can use the `eta.core.models.init_models_dir`
function to initialize it for use with the model management interface.

#### Model manifest files

Each directory on the model search path contains a `manifest.json` file that
describes the models that have been registered in that directory. The manifest
files are lightweight and are maintained in version control in the repository.

The manifest file contains the names and versions of each model registered in
its parent directory; it also contain the necessary information to locate
each model in cloud storage. When a model is downloaded, it will be stored
in the same directory as its corresponding manifest file.

The following is an example contents of a `manifest.json` file:

```json
{
    "models": [
        {
            "base_name": "VGG-16",
            "base_filename": "vgg16_weights.npz",
            "version": null,
            "manager": {
                "config": {
                    "google_drive_id": "0B7phNvpRqNdpT0lZU1NOWXIzRlE"
                },
                "type": "eta.core.models.ETAModelManager"
            },
            "date_created": "2018-06-20 17:35:53"
        },
        ...
    ]
}
```

#### Model managers

The `eta.core.models.ModelManager` class defines the base implementation of
a model manager, which is a class that can upload, download, and delete a model
from remote storage. Each entry in a manifest file contains a `manager` field
that describes the model manager instance for that model.

It is straightforward to implement subclasses of `ModelManager` to support
custom cloud (or other remote) storage solutions based on your needs.

All ETA models are currently stored in Google Drive in a publicly accessible
folder. The `eta.core.models.ETAModelManager` class provides the underlying
functionality used to download these models.

> Note: For security reasons, the ETA model manager is unable to upload and
> delete models. These tasks must be performed manually by a Voxel51
> administrator.

#### Model weights

We use the term "model weights" to refer broadly to the actual data contained
in a model, regardless of its concrete form (weights of a network, parameters,
coefficients, etc). Obscure model formats can always be read manually from disk
after they are downloaded, but the `eta.core.models.ModelWeights` class defines
the interface for classes that can load models in certain "known" formats.
Subclasses of `ModelWeights` can load models automatically based on their
`name` either by reading it from local storage if it exists or by automatically
downloading the model from remote storage. It is straightforward to add new
subclasses of `ModelWeights` to support new model formats.

For example, the `eta.core.models.NpzModelWeights` provides a dictionary
interface to access weights stored as an `.npz` file on disk:

```py
import eta.core.models as etam

# Loads the model by name
# Automatically downloads the model from the cloud, if necessary
weights = NpzModelWeights(name)

# Dictionary-based access to the weights
weights["layer-1"]
```


## Basic Usage

The following sections provide a brief overview of the basic usage of the
model management system. They assume that the following import has been issued:

```py
import eta.core.models as etam
```

#### List available models

To generate a list of available models, run:

```py
models = etam.list_models()
```

The above function lists the names of all models registered in manifests on
the current model search path, regardless of whether they are currently
downloaded to your machine.

#### Downloading a model

To determine if a model is currently downloaded locally, run:

```py
is_downloaded = etam.is_model_downloaded(name)
```

To download a model from the cloud, run:

```py
etam.download_model(name)
```

Recall that model versions can be easily configured in the `name` argument.
For example:

```py
etam.list_models()
# ["model@0.1", "model@1.0", "model@2.0"]

etam.download_model("model@0.1")
# model@0.1 is downloaded

etam.download_model("model")
# model@2.0 is downloaded (the latest version)
```

#### Locating and loading models

You can obtain the local path to a model by running:

```py
model_path = etam.find_model(name)
```

Note that the model will not yet exist at the returned model path if you
haven't downloaded it.

Although you are free to manually load a model from its `model_path`, the ETA
library provides several subclasses of `eta.core.models.ModelWeights` that can
automatically load a model given only its `name`. See the
[Model weights](#model-weights) section for more information.

#### Publishing a new model

To publish a new model, you can follow the general receipt:

```py
#
# Full workflow for publishing a new model
#

# Recommend paths for the given model by looking for older versions
# of the model in the model manifests
base_filename, models_dir = etam.recommend_paths_for_model(name)

# Perform a dry run of the model registration to check for any errors
# before uploading the model to the cloud
etam.register_model_dry_run(name, base_filename, models_dir)

# Upload the file to remote storage
# Then instantiate the relevant ModelManager describing its location
manager = ...

# Register the model
etam.register_model(name, base_filename, models_dir, manager)
```

The above process can be completely automated for custom cloud storage
applications, including the uploading of the model to cloud storage and the
generation of the relevant model manager instance describing the model.
To do this, one should implement a new subclass of
`eta.core.models.ModelManager`.

> Note: Only Voxel51 administrators can upload models to cloud storage, so
> one must submit a request to have the ETAModelManager instance generated for
> your model

#### Flushing local models

Over time you may accumulate too many local copies of models. To clear your
local disk space, you can flush local models in various ways:

```py
# Flush a specific model from local storage
etam.flush_model(model)

# Flush an entire models directory
etam.flush_models_directory(models_dir)

# Flush all local models
etam.flush_all_models()
```

Remember that flushing is a reversible operation; you can always re-download
all models from cloud storage at any time.

#### Managing the lifecycle of versioned models

The ETA config provides a `max_model_versions_to_keep` attribute that allows
you to configure the maximum number of versions of a given model to keep on
disk. If this limit is execeeded, the oldest versions of a model are deleted
until the limit is satisifed again.

The models framework automatically checks this condition and performs the
necessary cleanup after each time a model is downloaded. However, you can
manually invoke the cleanup at any time by running:

```py
etam.flush_old_models()
```

#### Permanently deleting a model

To permanently delete a model, run:

```py
etam.delete_model(name)
```

The above command deletes the model from both local and cloud storage, and it
deletes the associated entry from the `manifest.json` file.
