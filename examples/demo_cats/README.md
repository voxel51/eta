# Cat Detection + Classification Demo

This demo shows how to run an object detection and classification pipeline that
uses a pre-trained `eta.detectors.TFModelsDetector` model to detect cats
in a directory of images and also uses a pre-trained
`eta.classifiers.VGG16Classifier` to predict the breed of cat from the entire
image.


## Instructions

To run an `detect_and_classify_images` pipeline, simply execute the following:

```
# detect cats
eta build -r detect-classify-cats.json --run-now
```

To visualize the results, open the images in `out/cats-annotated` in your
favorite image viewer:


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
