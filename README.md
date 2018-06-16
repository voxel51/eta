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
# See install options
bash install_externals.bash -h

# Install default OpenCV release (3.3.0)
bash install_externals.bash

# Install specific OpenCV release
bash install_externals.bash -v 2.4.13.3
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

When the root `eta` package is imported, it (tries to) read the
`eta/config.json` file to configure various package-level constants.

> If you do not create this file, `import eta` will still work and you will
> still be able to use most of the ETA library. However, if you want to use
> advanced features such as pipeline building, you will need to properly
> configure your `eta/config.json` file.

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

The default config includes the `eta/modules` and `eta/pipelines` directories
in your module and pipeline config search paths, respectively. In addition,
it includes the relative paths `./modules` and `./pipelines` to support the
typical directory structure that we adopt for our individual projects.


## Installing the command-line utility

You can install the ETA command-line utility by simply placing the `eta/eta`
executable on your system path.

For example, you could make a symlink to an appropriate `bin` directory on your
path:

```shell
# Choose a different target directory if you prefer
ln -s "$(pwd)/eta/eta" /usr/local/bin/eta
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

# Build and run a video formatting pipeline from a request using
# the `eta` command-line tool
eta build -r examples/demo_video_formatter/request.json --run-now

#
# Example image/video embedding
#
# NOTE: A 550MB VGG16 weights file will be downloaded from the web and stored
#       in eta/cache/ the first time you run one of these!
#
cd examples/embed_vgg16
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
