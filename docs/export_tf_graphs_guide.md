# Exporting TensorFlow Graphs Guide

This document describes how to export frozen TensorFlow graphs for inference.

See [this document](visualize_tf_graphs_guide.md) for more information about
visualizing the architecture of graphs that you have exported.


## Exporting a generic training checkpoint

Training a generic TF model yields a model directory with contents similar to:

```
.
├── model.ckpt.meta
├── model.ckpt.data-?????-of-?????
└── model.ckpt.index
```

Run the commands below to export the checkpoint from a `model_dir` whose
contents are as shown above as a `frozen_model.pb` file in that directory:

```py
import eta.core.tfutils as etat

# The model directory to process
model_dir = "/path/to/model/dir"

# The names of the output nodes to export
output_node_names = ["output", "nodes"]

model_path = etat.export_frozen_graph(model_dir, output_node_names)
```


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
```

download it:

```
mkdir -p tmp
wget -P tmp http://download.tensorflow.org/models/${MODEL}.tar.gz
tar -xf tmp/${MODEL}.tar.gz -C tmp
```

and extract the training checkpoint:

```shell
mv tmp/*.ckpt ${MODEL}.ckpt
rm -r tmp
```

Finally, export the frozen inference graph using the
`eta.classifiers.tfslim_classifiers.export_frozen_inference_graph()` method:

```py
import eta.classifiers.tfslim_classifiers as etat

checkpoint_path = "resnet_v2_50_2017_04_14.ckpt"
network_name = "resnet_v2_50"
output_path = "resnet_v2_50_2017_04_14.pb"

etat.export_frozen_inference_graph(
    checkpoint_path, network_name, output_path, num_classes=1001)
```


## Exporting pre-trained models from the TF-Models Detection Zoo

This section describes how to export a pre-trained model from the
[TensorFlow Detection Model Zoo](
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

First, choose a model of interest from the zoo:

```shell
#
# Choose any model from
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
#
MODEL=ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
```

download it:

```
mkdir -p models
wget -P models/ http://download.tensorflow.org/models/object_detection/${MODEL}.tar.gz
```

and extract the training checkpoint file(s) and the pipeline config:

```shell
tar -xf models/${MODEL}.tar.gz -C models/
mv models/${MODEL}/model.ckpt* models/
mv models/${MODEL}/pipeline.config models/
rm models/${MODEL}.tar.gz
rm -r models/${MODEL}
```

Finally, export the frozen inference graph using the
`eta.detectors.tfmodels_detectors.export_frozen_inference_graph()` method:

```py
import eta.detectors.tfmodels_detectors as etat

checkpoint_path = "models/model.ckpt"
pipeline_config_path = "models/pipeline.config"
output_dir = "out/"

etat.export_frozen_inference_graph(
    checkpoint_path, pipeline_config_path, output_dir)
```


## Copyright

Copyright 2017-2022, Voxel51, Inc.<br>
voxel51.com
