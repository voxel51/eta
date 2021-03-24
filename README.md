<div align="center">

<h1>
    ETA: Extensible Toolkit for Analytics
</h1>

**An open and extensible computer vision, machine learning and video analytics
infrastructure.**

[![PyPI python](https://img.shields.io/pypi/pyversions/voxel51-eta)](https://pypi.org/project/voxel51-eta)
[![PyPI version](https://badge.fury.io/py/voxel51-eta.svg)](https://pypi.org/project/voxel51-eta)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Twitter](https://img.shields.io/twitter/follow/Voxel51?style=social)](https://twitter.com/voxel51)

<img src="https://user-images.githubusercontent.com/25985824/78944107-2d766c80-7a8b-11ea-8863-fcb4897eecb5.png" alt="eta-infrastructure.png" width="75%"/>

</div>

## Requirements

ETA is very portable:

-   Installable on Mac or Linux
-   Supports Python 2.7 and Python 3.6 or later
-   Supports TensorFlow 1.X and 2.X
-   Supports OpenCV 2.4+ and OpenCV 3.0+
-   Supports CPU-only and GPU-enabled installations
-   Supports CUDA 8, 9 and 10 for GPU installations

## Installation

You can install the latest release of ETA via `pip`:

```shell
pip install voxel51-eta
```

This will perform a [lite installation of ETA](#lite-installation). If you use
an ETA feature that requires additional dependencies (e.g., `ffmpeg` or
`tensorflow`), you will be prompted to install the relevant packages.

## Docker Installation

If you prefer to operate via Docker, see the
[Docker Build Guide](https://github.com/voxel51/eta/blob/develop/docs/docker_build_guide.md)
for simple instructions for building a Docker image with an ETA environment
installed.

## Installation from source

#### Step 0: Setup your Python environment

It is assumed that you already have
[Python installed](https://www.python.org/downloads) on your machine.

> **IMPORTANT:** ETA assumes that the version of Python that you intend to use
> is accessible via `python` and `pip` on your path. In particular, for Python
> 3 users, this means that you may need to alias `python3` and `pip3` to
> `python` and `pip`, respectively.

We strongly recommend that you install ETA
[in a virtual environment](https://github.com/voxel51/eta/blob/develop/docs/virtualenv_guide.md)
to maintain a clean workspace.

#### Step 1: Clone the repository

```shell
git clone https://github.com/voxel51/eta
cd eta
```

#### Step 2: Run the install script

```shell
bash install.bash
```

Note that the install script supports flags that control things like (on macOS)
whether `port` or `brew` is used to install packages. Run
`bash install.bash -h` for more information.

For Linux installs, the script inspects your system to see if CUDA is installed
via the `lspci` command. If CUDA is available, TensorFlow is installed with GPU
support.

The table below lists the version of TensorFlow that will be installed by the
installer, as recommended by the
[tested build configurations](https://www.tensorflow.org/install/source#tested_build_configurations):

| CUDA Version Found | TensorFlow Version Installed |
| ------------------ | ---------------------------- |
| CUDA 8             | `tensorflow-gpu~=1.4`        |
| CUDA 9             | `tensorflow-gpu~=1.12`       |
| CUDA 10            | `tensorflow-gpu~=1.15`       |
| Other CUDA         | `tensorflow-gpu~=1.15`       |
| No CUDA            | `tensorflow~=1.15`           |

> Note that ETA also supports TensorFlow 2.X. The only problems you may face
> when using ETA with TensorFlow 2 are when trying to run inference with
> [ETA models](https://github.com/voxel51/eta/blob/develop/eta/models/manifest.json)
> that only support TensorFlow 1. A notable case here are TF-slim models. In
> such cases, you should see an informative error message alerting you of the
> requirement mismatch.

### Lite installation

Some ETA users are only interested in using the core ETA library defined in the
`eta.core` package. In such cases, you can perform a lite installation using
the `-l` flag of the install script:

```shell
bash install.bash -l
```

Lite installation omits submodules and other large dependencies that are not
required in order for the core library to function. If you use an ETA feature
that requires additional dependencies (e.g., `ffmpeg` or `tensorflow`), you
will be prompted to install the relevant packages.

### Developer installation

If you are interested in contributing to ETA or generating its documentation
from source, you should perform a developer installation using the `-d` flag of
the install script:

```shell
bash install.bash -d
```

## Setting up your execution environment

When the root `eta` package is imported, it tries to read the `eta/config.json`
file to configure various package-level constants. Many advanced ETA features
such as pipeline building, model management, etc. require a properly configured
environment to function.

To setup your environment, create a copy the example configuration file:

```shell
cp config-example.json eta/config.json
```

If desired, you can edit your config file to customize the various paths,
change default constants, add environment variables, customize your default
`PYTHONPATH`, and so on. You can also add additional paths to the
`module_dirs`, `pipeline_dirs`, and `models_dirs` sections to expose custom
modules, pipelines, and models to your system.

Note that, when the config file is loaded, any `{{eta}}` patterns in directory
paths are replaced with the absolute path to the `eta/` directory on your
machine.

The default config includes the `modules/`, `pipelines/`, and `models/`
directories on your module, pipeline, and models search paths, respectively.
These directories contain the necessary information to run the standard
analytics exposed by the ETA library. In addition, the relative paths
`./modules/`, `./pipelines/`, and `./models/` are added to their respective
paths to support the typical directory structure that we adopt for our custom
projects.

### CLI

Installing ETA automatically installs `eta`, a command-line interface (CLI) for
interacting with the ETA Library. This utility provides access to many useful
features of ETA, including building and running pipelines, downloading models,
and interacting with remote storage.

To explore the CLI, type `eta --help`, and see the
[CLI Guide](https://github.com/voxel51/eta/blob/develop/docs/cli_guide.md) for
complete information.

## Quickstart

Get your feet wet with ETA by running some of examples in the
[examples folder](https://github.com/voxel51/eta/tree/develop/eta/examples).

Also, see the [docs folder](https://github.com/voxel51/eta/tree/develop/docs)
for more documentation about the various components of the ETA library.

## Organization

The ETA package is organized as described below. For more information about the
design and function of the various ETA components, read the documentation in
the [docs folder](https://github.com/voxel51/eta/tree/develop/docs).

| Directory         | Description                                                                                                                                                                                                                                                                                  |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eta/classifiers` | wrappers for performing inference with common classifiers                                                                                                                                                                                                                                    |
| `eta/core`        | the core ETA library, which includes utilities for working with images, videos, embeddings, and much more                                                                                                                                                                                    |
| `eta/detectors`   | wrappers for performing inference with common detectors                                                                                                                                                                                                                                      |
| `eta/docs`        | documentation about the ETA library                                                                                                                                                                                                                                                          |
| `eta/examples`    | examples of using the ETA library                                                                                                                                                                                                                                                            |
| `eta/models`      | library of ML models. The `manifest.json` file in this folder enumerates the models, which are downloaded to this folder as needed. See the [Models developer's guide](https://github.com/voxel51/eta/blob/develop/docs/models_dev_guide.md) for more information about ETA's model registry |
| `eta/modules`     | library of video processing/analytics modules. See the [Module developer's guide](https://github.com/voxel51/eta/blob/develop/docs/modules_dev_guide.md) for more information about ETA modules                                                                                              |
| `eta/pipelines`   | library of video processing/analytics pipelines. See the [Pipeline developer's guide](https://github.com/voxel51/eta/blob/develop/docs/pipelines_dev_guide.md) for more information about ETA pipelines                                                                                      |
| `eta/resources`   | resources such as media, templates, etc                                                                                                                                                                                                                                                      |
| `eta/segmenters`  | wrappers for performing inference with common semantic segmenters                                                                                                                                                                                                                            |
| `eta/tensorflow`  | third-party TensorFlow repositories that ETA builds upon                                                                                                                                                                                                                                     |

## Generating Documentation

This project uses
[Sphinx-Napoleon](https://pypi.python.org/pypi/sphinxcontrib-napoleon) to
generate its documentation from source.

To generate the documentation, you must install the developer dependencies by
running the `install.bash` script with the `-d` flag.

Then you can generate the docs by running:

```shell
bash sphinx/generate_docs.bash
```

To view the documentation, open the `sphinx/build/html/index.html` file in your
browser.

## Uninstallation

```shell
pip uninstall voxel51-eta
```

## Acknowledgements

This project was gratefully supported by the
[NIST Public Safety Innovation Accelerator Program](https://www.nist.gov/news-events/news/2017/06/nist-awards-385-million-accelerate-public-safety-communications).

## Citation

If you use ETA in your research, feel free to cite the project (but only if you
love it 😊):

```bibtex
@article{moore2017eta,
  title={ETA: Extensible Toolkit for Analytics},
  author={Moore, B. E. and Corso, J. J.},
  journal={GitHub. Note: https://github.com/voxel51/eta},
  year={2017}
}
```
