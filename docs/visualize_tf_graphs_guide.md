# Visualizing TensorFlow Graphs Guide

This document describes how to visualize the architecture of a frozen
TensorFlow graph using TensorBoard.

See [this document](export_tf_graphs_guide.md) for more information about
exporting TensorFlow graphs for inference.


## Visualizing a frozen TF graph

Run the commands below to launch a tensorboard session for a frozen graph of
your choice:

```py
import eta.core.tfutils as etat

# The frozen inference graph to load
model_path = "/path/to/frozen_inference_graph.pb"

# An optional log directory in which to write the TensorBoard files. By
# default, a temp directory is created
log_dir = None

etat.visualize_frozen_graph(model_path, log_dir=log_dir, port=8000)
```

The command will print the URL that you should open in your browser to view
TensorBoard.


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
