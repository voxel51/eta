# Object Detector Demo

This demo runs an object detection pipeline that uses a pre-trained
`eta.detectors.TFModelsDetector` model to detect vehicles and peoples in
videos.

## Instructions

To run the `object_detector` pipeline, simply execute the following commands:

```
# detect people
eta build -r detect-people.json --run-now

# detect vehicles
eta build -r detect-vehicles.json --run-now
```

To visualize the results, view `out/people-annotated.mp4` and
`out/vehicles-annotated.mp4` in your video player of choice.

## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
