# VGG-16 Embedding Examples

This directory contains example scripts that demonstrate the VGG-16 embedding
capabilities supported in ETA.

## Contents

-   `embed_image.py`: a simple example showing how to embed an image into the
    VGG-16 feature space using a `VGG16Featurizer`
-   `embed_image_direct.py`: another example showing how to manually embed an
    image into the VGG-16 feature space using the `VGG16` class itself
-   `embed_video.py`: an example of using `CachingVideoFeaturizer` to embed
    each frame of a video via `VGG16Featurizer`
-   `embed_vgg16_module-config.json`: an example module config file to execute
    the `embed_vgg16` ETA module
-   `embed_vgg16_module.bash`: a bash script to run the `embed_vgg16` module
    with the above config
-   `embed_vgg16_pipeine-config.json`: a pipeline config file that defines a
    simple one-step pipeline that runs the `embed_vgg16` ETA module on a video

## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
