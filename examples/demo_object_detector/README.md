# Object Detector Demo

This demo shows how to run an object detection pipeline that uses a
pre-trained `eta.detectors.tfmodels.TFModelsDetector` model to detect objects
in the [COCO dataset](http://cocodataset.org).


## Instructions

To run an `object_detector` pipeline using the default `TFModelsDetector`,
simply run one of the following commands:

```
# detect people
eta build -r detect-people.json --run-now

# detect vehicles
eta build -r detect-vehicles.json --run-now
```

To visualize the detections, run:

```
# visualize people
open out/people-annotated.mp4

# visualize vehicles
open out/vehicles-annotated.mp4
```


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
