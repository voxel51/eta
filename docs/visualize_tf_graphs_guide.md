# Visualizing TensorFlow Graphs Guide

This document describes how to visualize the architecture of a frozen
TensorFlow graph using TensorBoard.

See [this document](export_tf_graphs_guide.md) for more information about
exporting TensorFlow graphs for inference.


## Visualizing a frozen TF graph

Run the commands below to launch a tensorboard session for a frozen graph of
your choice:

```shell
# The frozen inference graph to load
MODEL_PATH=models/frozen_inference_graph.pb

# The path to the root directory of your TF installation
TFDIR=$(dirname "$(python -c "import tensorflow as tf; print(tf.__file__)")")

# Load the frozen graph in `TFDIR`
LOGDIR=/tmp/tflogdir
python "${TFDIR}/python/tools/import_pb_to_tensorboard.py" \
    --model_dir ${MODEL_PATH} \
    --log_dir ${LOGDIR}

# Launch tensorboard on `TFDIR`
tensorboard --port 8000 --logdir=${LOGDIR}
```

View TensorBoard by opening [http://localhost:8000](http://localhost:8000) in
your browser.


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
