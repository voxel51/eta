# ETA: Extensible Toolkit for Analytics

An open and extensible computer vision, machine learning and video analytics
infrastructure.

This project is supported by the [NIST Public Safety Innovation Accelerator
Program](
https://www.nist.gov/news-events/news/2017/06/nist-awards-385-million-accelerate-public-safety-communications).

<img
    src="https://drive.google.com/uc?id=14ZclqNXJXSct6O0sqcUoxFpzt_CnZuGP"
    alt="eta-infrastructure.png"
    width="75%"
/>


## Requirements

The ETA package requires the following external dependencies:
- [OpenCV](https://opencv.org)
- [TensorFlow](https://www.tensorflow.org/)
- [ffmpeg](https://www.ffmpeg.org)
- [ImageMagick](https://www.imagemagick.org/script/index.php)
- [tensorflow/models](https://github.com/tensorflow/models)

ETA is very portable:
- Installable on Mac or Linux
- Supports Python 2.7.X and Python 3.6.X
- Supports OpenCV 2.4+ and OpenCV 3.0+
- Supports CPU-only and GPU-enabled installations
- Supports CUDA 8, 9 and 10 (for GPU installations)


## Installation

0. _(Optional but highly recommended)_ Activate a
[virtual environment](docs/virtualenv_guide.md)

1. Clone the repository:

```shell
git clone https://github.com/voxel51/eta
cd eta
```

2. Run the install script:

```shell
bash install.bash
```

Depending on your Python environment, you may need to run the script with
sudo privileges. Note that the install script supports flags that control
things like (on macOS) whether `port` or `brew` is used to install packages.
Run `bash install.bash -h` for more information.

ETA assumes that the version of Python that you intend to use is available on
your system path via the `python` command, and it will install packages via the
`pip` executable on your path. In particular, for Python 3.X users, this means
that you may need to alias `python3` and `pip3` to `python` and `pip`,
respectively. If you find this annoying, let us again mention
[virtual environments](docs/virtualenv_guide.md).

For Linux installs, the script inspects your system to see if CUDA is installed
via the `lspci` command. If CUDA is available, TensorFlow is installed with GPU
support. The table below lists the version of TensorFlow that will be
installed:

| CUDA Version Found | TensorFlow Version Installed |
| ------------------ | ---------------------------- |
| CUDA 8 | `tensorflow-gpu==1.4` |
| CUDA 9 | `tensorflow-gpu==1.4` |
| CUDA 10 | `tensorflow-gpu==1.4` |
| Other CUDA | the latest available `tensorflow-gpu` |
| No CUDA | `tensorflow==1.12.0`

Note that ETA is installed in editable mode via `pip install -e .`, so don't
delete the directory after installation!

### Lite installation

Some ETA users are only interested in using the core ETA library defined in
the `eta.core` package. In such cases, you can perform a lite installation
using the `-l` flag of the install script:

```shell
bash install.bash -l
```

Lite installation omits submodules and other large dependencies that are not
required in order for the core library to function.


### Setting up your execution environment

When the root `eta` package is imported, it tries to read the `eta/config.json`
file to configure various package-level constants. Many advanced ETA features
such as pipeline building, model management, etc. require a properly configured
environment to function.

To setup your environment, create a copy the example configuration file

```shell
cp config-example.json config.json
```

If desired, you can edit your config file to customize the various paths,
change default constants, add environment variables, customize your default
`PYTHONPATH`, and so on. You can also add additional paths to the
`module_dirs`, `pipeline_dirs`, and `models_dirs` sections to expose custom
modules, pipelines, and models to your system.

Note that, when the config file is loaded, any `{{eta}}` patterns in directory
paths are replaced with the absolute path to the ETA repository on your
machine.

The default config includes the `eta/modules`, `eta/pipelines`, and
`eta/models` directories on your module, pipeline, and models search paths,
respectively. These directories contain the necessary information to run the
standard analytics exposed by the ETA library. In addition, the relative paths
`./modules`, `./pipelines`, and `./models` are added to their respective paths
to support the typical directory structure that we adopt for our custom
projects.

### CLI

Installing ETA automatically installs `eta`, a command-line interface (CLI) for
interacting with the ETA Library. This utility provides access to many useful
features of ETA, including building and running pipelines, downloading models,
and interacting with remote storage.

To explore the CLI, type `eta --help`, and see the
[CLI Guide](docs/cli_guide.md) for complete information.


## Quickstart

Get your feet wet with ETA by running some of examples in the
[examples folder](https://github.com/voxel51/eta/tree/develop/examples).

Also, see the [docs folder](https://github.com/voxel51/eta/tree/develop/docs)
for more documentation about the various components of the ETA library.


## Organization

The ETA package is organized as described below. For more information about the
design and function of the various ETA components, read the documentation in
the [docs folder](https://github.com/voxel51/eta/tree/develop/docs).

- `eta/classifiers/`: interfaces for common classifiers

- `eta/core/`: the core ETA library, which includes utilities for working
with images, videos, embeddings, and much more.

- `eta/detectors/`: interfaces for common detectors

- `eta/docs/`: documentation about the ETA library

- `eta/examples/`: examples of using the ETA library

- `eta/models/`: library of ML models. The `manifest.json` file in this
folder enumerates the models, which are downloaded to this folder as needed.
See the [Models developer's guide](https://github.com/voxel51/eta/blob/develop/docs/models_dev_guide.md)
for more information about ETA's model registry.

- `eta/modules/`: library of video processing/analytics modules. See the
[Module developer's guide](https://github.com/voxel51/eta/blob/develop/docs/modules_dev_guide.md)
for more information about ETA modules.

- `eta/pipelines/`: library of video processing/analytics pipelines. See the
[Pipeline developer's guide](https://github.com/voxel51/eta/blob/develop/docs/pipelines_dev_guide.md)
for more information about ETA pipelines.

- `eta/resources/`: resources such as media, templates, etc.

In addition, ETA makes use of the following external dependencies:

- `tensorflow/`: Third-party TensorFlow repositories that ETA builds upon


## Uninstallation

```shell
pip uninstall eta
```


## Copyright

Copyright 2017-2020, Voxel51, Inc.<br>
voxel51.com
