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

The ETA library provides a simple, yet powerful interface that satisifies these
constraints.

## Design

The `eta.core.models` module implements the core interface for managing models
in ETA. The underlying design principle of the interface is that all models are
stored in cloud storage and can be easily downloaded to a local ETA
installation as needed using a simple version-aware naming system.

### Model versions

Each model is identified by a unique string of the form `<name>@<version>`,
where `<name>` is the canonical name of the model and `<version>` is an
optional version of the model.

If a model is registered without a version, it can be referenced simply by its
`<name>`. If a model is registered with a version, it can be referenced
explicitly using the `<name>@<version>` syntax. The `<name>` syntax can also be
used with versioned models, in which case the latest version of the model (i.e.
the largest version number) will be referenced.

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
its parent directory; it also contain the necessary information to locate each
model in cloud storage. When a model is downloaded, it will be stored in the
same directory as its corresponding manifest file.

The following is an example contents of a `manifest.json` file:

```json
{
    "models": [
        {
            "base_name": "ssd-resnet50-v1-fpn-coco",
            "base_filename": "ssd-resnet50-v1-fpn-coco.pb",
            "version": null,
            "description": "A FPN Single Shot Detector with ResNet50 backbone trained on COCO",
            "manager": {
                "type": "eta.core.models.ETAModelManager",
                "config": {
                    "google_drive_id": "1ZvVQTuDIexyfntq8ajBVp5aSndHW6wQk"
                }
            },
            "default_deployment_config_dict": {
                "type": "eta.detectors.TFModelsDetector",
                "config": {
                    "model_name": "ssd-resnet50-v1-fpn-coco",
                    "labels_path": "{{eta}}/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt"
                }
            },
            "date_added": "2019-04-11 12:01:29"
        },
        ...
    ]
}
```

The `manager` and `default_deployment_config_dict` entries of the manifest are
described in the following sections.

#### Model managers

The `eta.core.models.ModelManager` class defines the base implementation of a
model manager, which is a class that can upload, download, and delete a model
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

### Default deployment configs

The `default_deployment_config_dict` value in each models manifest entry
defines an `eta.core.learning.ModelConfig` instance that specifies how to load
and perform inference with the model using its default settings.

The `ModelConfig` specifies the `type` of model (e.g.,
`eta.detectors.TFModelsDetector` in the above example) and the `config` field
provides a valid config instance (e.g., `eta.detectors.TFModelsDetectorConfig`)
for instantiating the model.

For example, the following code loads the `ssd-resnet50-v1-fpn-coco` with its
default settings and performs inference on an image:

```py
import eta.core.image as etai
import eta.core.learning as etal

model = etal.load_default_deployment_model("ssd-resnet50-v1-fpn-coco")

image = etai.read("/path/to/image.jpg")

with model:
    objects = model.detect(img)

print(objects)
```

Example output:

```json
{
    "objects": [
        {
            "label": "vehicle",
            "confidence": 0.95,
            "bounding_box": {
                "bottom_right": {
                    "y": 0.5722222222222222,
                    "x": 0.09765625
                },
                "top_left": {
                    "y": 0.4708333333333333,
                    "x": 0.0
                }
            },
            "attrs": {}
        }
    ]
}
```

## Basic Usage

The following sections provide a brief overview of the basic usage of the model
management system. They assume that the following import has been issued:

```py
import eta.core.models as etam
```

#### List available models

To generate a list of available models, run:

```py
models = etam.list_models()
```

The above function lists the names of all models registered in manifests on the
current model search path, regardless of whether they are currently downloaded
to your machine.

#### Downloading a model

To determine if a model is currently downloaded locally, run:

```py
is_downloaded = etam.is_model_downloaded(name)
```

To download a model from the cloud, run:

```py
etam.download_model(name)
```

The above function downloads the model (if necessary) from cloud storage and
stores it in the models directory whose `manifest.json` file contains the model
definition.

Recall that model versions can be easily configured in the `name` argument. For
example:

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

By default, all ETA models are stored in a Google Drive folder with permissions
set to be publicly readable. You can publish a new model to this public
registry as follows:

```py
#
# Publishing a new public ETA model
#
# Assumes the model has been uploaded to Google Drive by an ETA administrator
# and you have the ID of the model in Google Drive
#

# The name for your model
name = "your-model@1.0"

# The ID of your model in Google Drive
google_drive_id = "XXXXXXXX"

# A short description of your model
description = "A short description of your model"

#
# The base filename (no version information, which is added automatically) and
# models directory, respectively, to use when downloading the model. If you
# are uploading a new version of an existing model, these arguments can be set
# to None and the same values are inherited from the most recent version of the
# model. Otherwise, if no models directory is provided, the first directory on
# your models search path will be used by default.
#
base_filename = "your-model.pb"
models_dir = "/path/to/models/dir"

# Publish the model
etam.publish_public_model(
    name, google_drive_id, description=description, base_filename=None,
    models_dir=None)
```

The above publishing process adds an entry to the `manifest.json` file for your
new model in the models directory that you specified (or the first directory in
`eta.config.models_dir` if none is specified). The new model will now be
findable by all of the model-related functionality in `eta.core.models` as long
as the models directory you chose above remains on your models search path.

When a published model is downloaded from cloud storage, it will be written to
the models directly specified above (i.e. the directory whose `manifest.json`
file contains the model definition) and an appropriate filename will be dervied
from the base filename and version number of the model.

Internally, the models an `eta.core.models.ETAModelManager` to manage access to
the model in the public Google Drive folder. However, the publishing process
can be easily extended to custom cloud storage solutions. To do so, one should
implement a new subclass of `eta.core.models.ModelManager` and use it together
with the `eta.core.models.register_model` function to implement a custom
publishing workflow.

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

You can enforce this limit at any time by running:

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

## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
