# Cat Detection and Classification Demo

This demo runs an object detection and classification pipeline that uses a
pre-trained `eta.detectors.TFModelsDetector` model to detect cats in a
directory of images and then uses a pre-trained
`eta.classifiers.TFSlimClassifier` to predict the breed of cat from the entire
image.

## Instructions

To run the `detect_and_classify_images` pipeline, simply execute the following:

```
eta build -r detect-classify-cats.json --run-now
```

To visualize the results, open the images in `out/cats-annotated` in your image
viewer of choice.

## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
