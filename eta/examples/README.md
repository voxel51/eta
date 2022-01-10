# ETA Examples

This directory contains various examples of using the ETA library. See the
[docs folder](https://github.com/voxel51/eta/tree/develop/docs) for more
documentation about the various components of the ETA library that are used
here.

## Setup

Download the example data:

```shell
python download_data.py
```

## Basics

Build and run pipelines from requests:

```shell
eta build -r demo_video_formatter/request.json --run-now
eta build -r demo_video_clipper/request.json --run-now
```

Run a pre-configued instance of a pipeline:

```shell
eta run demo_video_pipeline/pipeline.json
```

Note that the best practice is to formally define pipelines and then build and
run instances of them via pipeline requests (as demonstrated in the
`demo_video_formatter` and `demo_video_clipper` examples), rather than manually
instantiating your own pipelines (as demonstrated in the `demo_video_pipeline`
example).

## Image classification

The following code runs a pipeline that performs image classification on a
directory of images.

See [this README](demo_image_classifier/README.md) for more information,
including instructions for running a variety of network architectures besides
MobileNet v2.

```shell
cd demo_image_classifier

eta build -r classify-images-tfslim-template.json \
    --patterns {{model}}=mobilenet-v2-imagenet --run-now

cd ..
```

View the images in the `demo_image_classifier/out/images-mobilenet-v2-imagenet`
directory to inspect the output of the pipeline.

## Object detection

The following code runs two object detection pipelines on video.

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

The following code runs two instance segmentation pipelines on video.

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

## Semantic segmentation

The following code runs a semantic segmentation pipeline on video.

See [this README](demo_semantic_segmentation/README.md) for more information.

```shell
cd demo_semantic_segmentation

eta build -r segment-frames.json --run-now

cd ..
```

Open the `demo_semantic_segmentation/out/people-annotated.mp4` in your video
player to inspect the output of the pipeline.

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

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
