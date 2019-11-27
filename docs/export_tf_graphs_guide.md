# Exporting TensorFlow Graphs Guide

This document describes how to export frozen TensorFlow graphs for inference.


## Exporting pre-trained models from the TF-Slim Model Zoo

This section describes how to export a pre-trained model from the
[TensorFlow-Slim Model Zoo](https://github.com/tensorflow/models/tree/master/research/slim).

First, choose a model of interest from the zoo:

```shell
#
# Choose any model from
# https://github.com/tensorflow/models/tree/master/research/slim
#
MODEL=resnet_v2_50_2017_04_14
MODEL_NAME=resnet_v2_50
OUTPUT_NODE=resnet_v2_50/predictions/Reshape_1
```

download it:

```
mkdir -p tmp
wget -P tmp http://download.tensorflow.org/models/${MODEL}.tar.gz
tar -xf tmp/${MODEL}.tar.gz -C tmp
```

and extract the training checkpoint:

```shell
mv tmp/*.ckpt ${MODEL_NAME}.ckpt
rm -r tmp
```

Next, export the inference graph using the `export_inference_graph.py` tool
provided in the TF-Slim library:

```shell
TF_SLIM_DIR = $(eta constants TF_SLIM_DIR)

python "${TF_SLIM_DIR}/export_inference_graph.py" \
    --alsologtostderr \
    --model_name=${MODEL_NAME} \
    --output_file=${MODEL_NAME}_graph.pb
```

Finally, freeze the graph using the `freeze_graph` tool from the TF repository:

```shell
bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=${MODEL_NAME}_graph.pb \
  --input_checkpoint=${MODEL_NAME}.ckpt \
  --input_binary=true \
  --output_graph=${MODEL_NAME}.pb \
  --output_node_names=${OUTPUT_NODE}
```

The last step assumes that you have built the `freeze_graph` tool
from the [TensorFlow Repository](https://github.com/tensorflow/tensorflow):

```shell
bazel build tensorflow/python/tools:freeze_graph
```


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
