# ETA: Extensible Toolkit for Analytics

An open and extensible video analytics infrastructure.

This project is supported by the [NIST Public Safety Innovation Accelerator
Program](https://www.nist.gov/news-events/news/2017/06/nist-awards-385-million-accelerate-public-safety-communications).


## Installation

> ETA is currently built on Python 2.7.

Clone the repository

```shell
git clone https://github.com/voxel51/eta
cd eta/
```

If you want to simply install the toolkit and use it, run

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


## Testing your installation

To test your installation, run the following commands:

```shell
# Example image manipulation
python examples/demo_images.py

# Example video processing pipeline
python eta/core/pipeline.py examples/demo_video/pipeline-config.json

# Example image/video embedding
#
# NOTE: A 550MB VGG16 weights file will be downloaded from the web and stored
#       in eta/cache/ the first time you run one of these!
#
cd examples/embed_vgg16
# Example image embedding
python embed_image.py
# Example video embedding
python embed_video.py
# Example use of the embed_vgg16 module
bash embed_vgg16_module.bash
# Example embedding pipeline
bash embed_vgg16_pipeline.bash
```


## External dependencies

The ETA package requires various Python packages, as well as
[OpenCV](http://opencv.org),
[ffmpeg](https://www.ffmpeg.org), and
[ImageMagick](https://www.imagemagick.org/script/index.php).

Currently ETA supports both `OpenCV 2.4` or later and `OpenCV 3.0` or later.

To install the external dependencies, run the install script

```shell
# Install default OpenCV release (3.3.0)
bash install_externals.bash

# Install specific OpenCV release
bash install_externals.bash 2.4.13.3
```

After installation, ensure that the binaries installed by the above script
are on your system `PATH` in your execution environment.

## Uninstallation

```shell
pip uninstall eta
```
