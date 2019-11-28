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


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
