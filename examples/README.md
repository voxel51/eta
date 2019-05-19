# ETA Examples

This directory contains various examples of using the ETA library.


## Setup

Download the example data:

```shell
python download_data.py
```


## Basics

Perform some image manipulation:

```shell
python demo_images/demo_images.py
```

Run a pre-configued video processing pipeline:

```shell
eta run demo_video/pipeline.json
```

Build and run pipelines from requests:

```shell
eta build -r demo_video_formatter/request.json --run-now
eta build -r demo_video_clipper/request.json --run-now
```


## Embeddings

Examples of image/video embedding:

Note that a 550MB VGG-16 weights file will be downloaded from the web and
stored in `eta/models` the first time you run this code.

```shell
cd demo_embed_vgg16

# Image embedding using `eta.core.vgg16.VGG16` directly
python embed_image_direct.py

# Image embedding using `eta.core.vgg16.VGG16Featurizer`
python embed_image.py

# Video embedding using `eta.core.features.VideoFramesFeaturizer`
python embed_video.py

# Video embedding using the `embed_vgg16` module
bash embed_vgg16_module.bash

# Video embedding pipeline
eta run embed_vgg16_pipeline-config.json

cd ..
```


## Object detection

The following code runs an object detection pipeline.

See [the README](demo_object_detector/README.md) for more information.

Note that a 120MB Faster R-CNN ResNet-50 model will be downloaded from the web
and stored in `eta/models` the first time you run this.

```shell
cd demo_object_detector
eta build -r detect-people.json --run-now
eta build -r detect-vehicles.json --run-now
cd ..
```

Open the `out/people-annotated.mp4` and `out/vehicles-annotated.mp4` in your
video player to inspect the output of the pipelines.


## Image detection and classification

The following code runs a pipeline that performs image detection and
classification to detect cats in a directory of images.

See [the README](demo_cats/README.md) for more information.

Note that a 550MB VGG-16 weights file will be downloaded from the web and
stored in `eta/models` the first time you run this code (if you have not
already run an example that uses VGG-16).

```shell
cd demo_cats
eta build -r detect-classify-cats.json --run-now
cd ..
```

View the images in the `demo_cats/out/cats` directory to inspect the output of
the pipeline.


## Cleanup

To cleanup the outputs of the examples, run:

```shell
bash clean.bash
```


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
