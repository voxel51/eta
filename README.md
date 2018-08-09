# ETA: Extensible Toolkit for Analytics

An open and extensible video analytics infrastructure.

This project is supported by the [NIST Public Safety Innovation Accelerator
Program](https://www.nist.gov/news-events/news/2017/06/nist-awards-385-million-accelerate-public-safety-communications).


## Installation

First, clone the repository

```shell
git clone https://github.com/voxel51/eta
cd eta/
```

The ETA package requires various Python packages, as well as
[OpenCV](http://opencv.org),
[ffmpeg](https://www.ffmpeg.org), and
[ImageMagick](https://www.imagemagick.org/script/index.php).

> ETA supports both Python 2.7 and Python 3.0 or later.

> ETA supports both OpenCV 2.4 or later and OpenCV 3.0 or later.

To install the external dependencies, run the install script

```shell
bash install_externals.bash
```

Depending on your Python environment, you may need to run the script with
sudo privileges. Note that the install script supports flags that control
things like (on macOS) whether `port` or `brew` is used to install packages.

Next, if you want to simply install the toolkit and use it, run

```shell
pip install .
```

Now `import eta` will work from anywhere, and you can delete the directory you
just cloned.

If you want to install the project in development mode, run

```shell
pip install -e .
```

Now `import eta` will still work from anywhere, and any changes you make in
your local copy of `eta/` will take effect immediately.


## Setting up your execution environment

When the root `eta` package is imported, it tries to read the `eta/config.json`
file to configure various package-level constants. Many advanced ETA features
such as pipeline building, model management, etc. require a properly configured
environment to function.

To setup your environment, copy the example configuration file

```shell
cp config-example.json config.json
```

and then edit your config file to provide the paths to the relevant
directories in your installation. If you do not require a customized
installation, the example configuration file contains the pattern `{{eta}}`
that you can perform a quick find-and-replace on to populate the config:

```shell
# on a mac
sed -i '' -e "s|{{eta}}|$(pwd)|g" config.json

# on most linux distributions
sed -i -e "s|{{eta}}|$(pwd)|g" config.json
```

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
bash examples/download_data.bash

# Perform some image manipulation
python examples/demo_images.py

# Run a pre-configued video processing pipeline
python eta/core/pipeline.py examples/demo_video/pipeline.json

# Build and run pipelines from requests using the `eta` command-line tool
eta build -r examples/demo_video_formatter/request.json --run-now
eta build -r examples/demo_video_clipper/request.json --run-now

#
# Example image/video embedding
#
# NOTE: A 550MB VGG16 weights file will be downloaded from the web and stored
#       in eta/models/ the first time you run one of these!
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
bash embed_vgg16_pipeline.bash
```

## Using virtual environments

You can use [virtual environments](https://virtualenv.pypa.io/en/stable) to
maintain a separate Python working environment for ETA that operates
independently of other packages and Python applications on your machine. See
`docs/virtualenv_guide.md` for more details and setup instructions.


## Uninstallation

```shell
pip uninstall eta
```
