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
python demo_image_manipulation/demo_images.py
```

Run a pre-configued video processing pipeline:

```shell
eta run demo_video_pipeline/pipeline.json
```

Build and run pipelines from requests:

```shell
eta build -r demo_video_formatter/request.json --run-now
eta build -r demo_video_clipper/request.json --run-now
```


## Image classification

The following code runs a pipeline that performs image classification on
a directory of images.

See [this README](demo_image_classifier/README.md) for more information,
including instructions for running a variety of network architectures besides
MobileNet v2.

```shell
cd demo_image_classifier
eta build -r classify-images-tfslim-template.json --patterns {{model}}=mobilenet-v2-imagenet --run-now
cd ..
```

View the images in the `demo_image_classifier/out/images-mobilenet-v2-imagenet`
directory to inspect the output of the pipeline.


## Object detection

The following code runs an object detection pipeline on video.

See [this README](demo_object_detector/README.md) for more information.

```shell
cd demo_object_detector
eta build -r detect-people.json --run-now
eta build -r detect-vehicles.json --run-now
cd ..
```

Open the `demo_object_detector/out/people-annotated.mp4` and
`demo_object_detector/out/vehicles-annotated.mp4` in your video player to
inspect the output of the pipelines.


## Instance segmentation

The following code runs an instance segmentation pipeline on video.

See [this README](demo_instance_segmentation/README.md) for more information.

```shell
cd demo_instance_segmentation
eta build -r segment-people.json --run-now
eta build -r segment-vehicles.json --run-now
cd ..
```

Open the `demo_instance_segmentation/out/people-annotated.mp4` and
`demo_instance_segmentation/out/vehicles-annotated.mp4` in your video player to
inspect the output of the pipelines.


## Image detection and classification

The following code runs a pipeline that performs image detection and
classification to detect cats in a directory of images.

See [this README](demo_cats/README.md) for more information.

```shell
cd demo_cats
eta build -r detect-classify-cats.json --run-now
cd ..
```

View the images in the `demo_cats/out/cats-annotated` directory to inspect the
output of the pipeline.


## Embeddings

Examples of image/video embedding:

```shell
cd demo_embed_vgg16

# Image embedding using `eta.core.vgg16.VGG16` directly
python embed_image_direct.py

# Image embedding using `eta.core.vgg16.VGG16Featurizer`
python embed_image.py

# Video embedding using `eta.core.features.CachingVideoFeaturizer`
python embed_video.py

# Video embedding using the `embed_vgg16` module
bash embed_vgg16_module.bash

# Video embedding pipeline
eta run embed_vgg16_pipeline-config.json

cd ..
```


## Cleanup

To cleanup the example outputs, run:

```shell
bash clean.bash
```


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
