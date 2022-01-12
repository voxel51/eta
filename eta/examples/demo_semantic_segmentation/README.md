# Semantic Segmentation Demo

This demo runs a semantic segmentation pipeline that uses a pre-trained
`eta.segmenters.TFSemanticSegmenter` model to segment the frames of videos.

## Instructions

To run the `semantic_segmenter` pipeline, simply execute the following
commands:

```
eta build -r segment-frames.json --run-now
```

To visualize the results, view `out/people-annotated.mp4` in your video player
of choice.

## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
