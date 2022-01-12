# Instance Segmentation Demo

This demo runs an instance detection pipeline that uses a pre-trained
`eta.detectors.TFModelsInstanceSegmenter` model to detect and segment vehicles
and peoples in videos.

## Instructions

To run the `object_detector` pipeline, simply execute the following commands:

```
# segment people
eta build -r segment-people.json --run-now

# segment vehicles
eta build -r segment-vehicles.json --run-now
```

To visualize the results, view `out/people-annotated.mp4` and
`out/vehicles-annotated.mp4` in your video player of choice.

## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
