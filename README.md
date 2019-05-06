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
- Supports both Python 2.7.X and Python 3.6.X
- Supports both OpenCV 2.4+ and OpenCV 3.0+
- Supports both CPU and GPU execution
- Supports both CUDA 8 and CUDA 9 (for GPU installations)


## Installation

0. Activate a [virtual environment](docs/virtualenv_guide.md) (optional but
highly recommended)

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

The script inspects your system to see if CUDA is installed, and, if it is,
TensorFlow is installed with GPU support. In particular, if CUDA 9 is found,
the latest version of the `tensorflow-gpu` package is installed, and if CUDA 8
is found, `tensorflow-gpu 1.4` is installed.

Note that ETA is installed in editable via `pip install -e .`, so don't delete
the directory after installation!


## Setting up your execution environment

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


## The `eta` command-line utility

When you installed ETA, an `eta` command-line utility was added to your path.
This utility provides access to many usefuel features of ETA, including
building pipelines from requests, running pipelines, and generating module
metadata files.

To learn more about the supported operations and their syntaxes, type

```
eta --help
```


## Testing your installation

To test your installation, run the following commands:

```shell
# Download example data
python examples/download_data.py

# Perform some image manipulation
python examples/demo_images.py

# Run a pre-configued video processing pipeline
eta run examples/demo_video/pipeline.json

# Build and run pipelines from requests using the `eta` command-line tool
eta build -r examples/demo_video_formatter/request.json --run-now
eta build -r examples/demo_video_clipper/request.json --run-now

#
# Run an object detection pipeline and visualize the results
#
# Note: A 120MB Faster R-CNN ResNet-50 model will be downloaded from the web
#   and stored in `eta/models` the first time you run this
#
cd examples/demo_object_detector
eta build -r detect-people.json --run-now
eta build -r detect-vehicles.json --run-now
open out/people-annotated.mp4
open out/vehicles-annotated.mp4

#
# Run an image detection + classification pipeline and visualize the results
#
# Note: A 550MB VGG-16 weights file will be downloaded from the web and stored
#   in `eta/models` the first time you run this
#
cd examples/demo_cats
eta build -r detect-classify-cats.json --run-now
# view the contents of out/cats

#
# Example image/video embedding
#
cd examples/demo_embed_vgg16
# Example image embedding
python embed_image.py
# Another example image embedding
python embed_image_direct.py
# Example video embedding
python embed_video.py
# Example use of the embed_vgg16 module
bash embed_vgg16_module.bash
# Example embedding pipeline
eta run embed_vgg16_pipeline-config.json
```


## Uninstallation

```shell
pip uninstall eta
```


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
