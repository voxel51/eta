# TensorFlow Models Object Detection Interface

Interface to the [TensorFlow Object Detection Models Library](
https://github.com/tensorflow/models/tree/master/research/object_detection).


## Example Usage

To run an `object_detector` pipeline using the default `TFModelsDetector`,
simply run one of the following commands:

# @todo add data download instructions

```
# detect people
eta build -r requests/detect-people.json --run-now

# detect vehicles
eta build -r requests/detect-vehicles.json --run-now
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
