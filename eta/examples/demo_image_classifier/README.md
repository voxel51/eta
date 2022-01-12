# Image Classification Demo

This demo runs an image classification pipeline that uses various pre-trained
classifiers to predict labels from the ImageNet dataset for a directory of
images.

## Instructions

To run the `image_classifier` pipeline, simply execute one of the following:

```shell
# eta.classifiers.TFSlimClassifier
eta build -r classify-images-tfslim-template.json --patterns {{model}}=mobilenet-v2-imagenet --run-now
eta build -r classify-images-tfslim-template.json --patterns {{model}}=resnet-v1-50-imagenet --run-now
eta build -r classify-images-tfslim-template.json --patterns {{model}}=resnet-v2-50-imagenet --run-now
eta build -r classify-images-tfslim-template.json --patterns {{model}}=inception-resnet-v2-imagenet --run-now
eta build -r classify-images-tfslim-template.json --patterns {{model}}=inception-v4-imagenet --run-now

# eta.classifiers.VGG16Classifier
eta build -r classify-images-vgg16.json --run-now
```

To visualize the results, open the images in the `out/images-XXX` folder
corresponding to the model that you ran in your image viewer of choice.

## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
