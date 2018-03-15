# ETA Module Developer's Guide

This document describes best practices for contributing modules to ETA. See
`core_dev_guide.md` for instructions on contributing to the core ETA
infrastructure, and see `pipelines_dev_guide.md` for more information about
ETA pipelines and how to define, build, and run them.


## What are ETA Modules?

Modules are the workhorses of the ETA system. They perform the actual
computations required to process videos and generate specific features or
analytics. Abstractly, the ETA system is a graph whose nodes are modules. In
order to compute a desired analytic, the ETA system chooses a path through the
graph and configures a pipeline of modules to run that traverses the chosen
path through the graph.

Concretely, ETA modules are simply _executables_ that take JSON files as input
and write output data to disk. The input JSON configures a module for execution
by telling it what parameter settings to use, what data to read as input, and
where to write its output data. Modules must also provide a metadata JSON file
that tells the ETA system what parameters they expect and the input/output
data formats they use. We chose this generic modular architecture for ETA
because we want to support as many third-party modules as possible!

This repository contains many modules implemented in Python using the core ETA
libraries, and this is how most ETA modules will be contributed. However, in
principle modules can be implemented in any language as long as they provide a
valid metadata JSON file. In fact, some modules may live outside this
repository on your local machine, in the cloud, or behind a pay-wall on a
vendor's cloud.


## Module Metadata JSON Files

Every ETA module must provide a metadata JSON file describing the inputs and
outputs of the module. The metadata file contains all the necessary information
to generate configuration JSON files that are passed to a module during
execution. Indeed, the ETA system automatically generates configuration files
whenever it builds and executes pipelines.

The following JSON gives an example of the metadata file for a simple object
detector module:

```json
{
    "info": {
        "name": "simple-object-detector",
        "type": "eta.core.types.Module",
        "version": "0.1.0",
        "description": "A simple object detector",
        "exe": "simple_object_detector.py"
    },
    "inputs": [
        {
            "name": "raw_video_path",
            "type": "eta.core.types.Video",
            "description": "The input video",
            "required": true
        }
    ],
    "outputs": [
        {
            "name": "objects_json_path",
            "type": "eta.core.types.Frame",
            "description": "The output path for the objects JSON file",
            "required": true
        },
        {
            "name": "annotated_frames_path",
            "type": "eta.core.types.ImageSequence",
            "description": "The output path for the annotated frames",
            "required": false
        }
    ],
    "parameters": [
        {
            "name": "labels",
            "type": "eta.core.types.Array",
            "description": "The classes of objects to detect",
            "required": true
        },
        {
            "name": "weights",
            "type": "eta.core.types.Weights",
            "description": "The weights for the network",
            "required": true,
            "default": "/path/to/weights.npz"
        }
    ]
}
```

When discussing module metadata files, we refer to each JSON object `{}` as a
**spec** (specification) because it specifies the semantics of a certain
entity, and we refer to the keys of a JSON object (e.g., "info") as **fields**.

The module metadata file contains the following top-level fields:

- `info`: a spec containing basic information about the module

- `inputs`: a list of specs describing each input data-related field that
    the module expects in its configuration file

- `outputs`: a list of specs describing each output data-related field that
    the module expects in its configuration file

- `parameters`: a list of specs describing additional parameters that the
    module expects in its configuration file

The `info` spec contains the following fields:

- `name`: the name of the module

- `type`: the type of the module from `eta.core.types`, i.e., what computation
    it performs. Must be a valid module type exposed by the ETA library

- `version`: the current module version

- `description`: a short free-text description of the module purpose and
    implementation

- `exe`: the name of the module executable file

The remaining specs describe the fields in the module's configuration files.
Each spec has the fields:

- `name`: the name of the field

- `type`: the type of the data referenced by this field. Must be a valid
    data type exposed by the ETA library

- `description`: a short free-text description of the field

- `required`: whether the field is required or optional for the module to
    function

- `default`: (parameters only) an optional default value for the parameter


## Visualizing Modules

The ETA system provides the ability to visualize modules as block diagrams
using the [blockdiag](https://pypi.python.org/pypi/blockdiag) package.

For example, the block diagram file for the simple object detector module
described above can be generated by executing:

```python
import eta.core.module as etam

# Load the pipeline
module = etam.load_metadata("simple_object_detector")

# Render the block diagram
module.render("module_block_diagram.svg")
```

The above code generates a `module_block_diagram.svg` vector graphics image of
the module block diagram. It also generates the following intermediate
`module_block_diagram.diag` file describing the network architecture:

```
blockdiag {

  // module
  simple-object-detector [width = 187, shape = box, height = 60];

  // inputs
  raw_video_path [width = 204, shape = endpoint, height = 40];

  // outputs
  objects_json_path [width = 234, shape = endpoint, height = 40];
  annotated_frames_path [width = 274, shape = endpoint, height = 40];

  // parameters
  labels [width = 40, shape = beginpoint, rotate = 270, height = 124];
  weights [width = 40, shape = beginpoint, rotate = 270, height = 134];

  // I/O connections
  raw_video_path -> simple-object-detector;
  simple-object-detector -> objects_json_path;
  simple-object-detector -> annotated_frames_path;

  // parameter connections
  group {
    color = "#EE7531";
    orientation = portrait;
    labels -> simple-object-detector;
    weights -> simple-object-detector;
  }
}
```


## Module Configuration Files

The module metadata file contains all the information necessary for the ETA
system to automatically generate module configuration JSON files for a module.

The precise syntax of the configuration files is still under development,
but the current format is:

```json
{
    "data": [
        {<inputs1>, <outputs1>},
        {<inputs2>, <outputs2>},
        ...
    ],
    "param1": <val1>,
    "param2": <val2>,
    ...
}
```

The `data` field contains a list of specs, each of which contains the input
and output fields specified by the module's metadata file. This field expects
a list so that multiple datasets can be processed in a single module
execution, if desired. The remaining fields contain the parameters specified
by the module's metadata file.

For example, a valid configuration file for the object detector defined by the
above metadata file is:

```json
{
    "data": [
        {
            "raw_video_path": "/path/to/vid1.mov",
            "objects_json_path": "/path/to/obj1/%05d.json",
            "annotated_frames_path": "/path/to/ann1/%05d.png"
        },
        {
            "raw_video_path": "/path/to/vid2.mp4",
            "objects_json_path": "/path/to/obj2/%05d.json"
        }
    ],
    "labels": ["car", "truck", "bus"],
    "weights": "/path/to/better/weights.npz"
}
```

Note that the `annotated_frames_path` is omitted from the second data spec,
which is allowed since the field was optional (a default value was
provided in the metadata file).


## Building Standalone Modules

Since ETA modules are independent programs or scripts, they can be implemented
in any language as long as they provide a valid metadata file and write their
outputs in ETA-supported formats. Thus, even if developers don't intend to use
ETA libraries to build their modules, they must be familiar with the ETA
supported data types.


## Building Modules using ETA

ETA provides a core library that can be leveraged to easily define new
analytics modules. This is the most common method for creating new modules.

This section summarizes the key features of the ETA module creation syntax.

#### Conventions for implementing modules using ETA

Here is a not-exactly-python-code template example for a module in ETA.

```python
#!/usr/bin/env python
'''
{{Description Of The Module}}

Copyright 2017, Voxel51, LLC
voxel51.com
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging
import sys

import eta.core.module as etam


logger = logging.getLogger(__name__)


class {{ModuleTemplate}}Config(etam.BaseModuleConfig):
    '''Module configuration settings.'''

    def __init__(self, d):
        super({{ModuleTemplate}}Config, self).__init__(d)


def run(config_path, pipeline_config_path=None):
    '''Run the module.

    Args:
        config_path: path to {{ModuleTemplate}}Config file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    config = {{ModuleTemplate}}Config.from_json(config_path)
    etam.setup(config, pipeline_config_path=pipeline_config_path)


if __name__ == "__main__":
    run(*sys.argv[1:])
```

Modules include the following, in this order from top to bottom:

- A full docstring at the top of the file to describe the module capabilities

- The `__future__` imports, pragmas, and other definitions to alow for cross
    cross version python support (see `python23_guide.md`)

- Imports organized according to our `style_guide.md`, namely standard library
    imports, third-party library imports, and application-specific imports,
    alphabetized within each group and with a single blank line between each

- Logging setup

- Definitions of module configuration classes

- A `run` fuction that defines the main driver of the module

- Any additional methods needed by the main driver

- The canonical `if __name__ == "__main__":` startup statement


#### Parsing module configuration files

The following snippet shows a canonical definition of a module configuration in
ETA:

```python
import eta.core.module as etam

class ExampleConfig(etam.BaseModuleConfig):
    '''An example config class.'''

    def __init__(self, d):
        super({{ExampleConfig}}, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)


class DataConfig(Config):
    '''An example data config class.'''

    def __init__(self, d):
        self.input_path = self.parse_string(d, "input_path")
        self.output_path = self.parse_string(d, "output_path")
        self.parameter = self.parse_number(d, "parameter", default=1.0)
```

The snippet defines a configuration class called `ExampleConfig` which contains
a single `data` field that contains an array of `DataConfig` instances.

Note that the `DataConfig` class derives from the `eta.core.config.Config`
class, which implements the basic semantics of configuration classes.  Further,
note that the `ExampleConfig` class derives from
`eta.core.module.BaseModuleConfig` which implements additional configuration
functionality for modules.  The `Config.parse_*` methods are
used to define the names and data types of the JSON fields.

Fields defined with no `default` keyword are *mandatory*, and fields with a
`default` keyword are *optional*.

The following JSON file is a valid `ExampleConfig` configuration file:

```json
{
    "data": [
        {
            "input_path": "/path/to/input.mp4",
            "output_path": "/path/to/output.mp4"
        }
    ]
}
```

Note that the `parameter` field is omitted, which is allowed since a default
value was specified in the `DataConfig` class.

To load the configuration file into an `ExampleConfig` instance, simply do:

```python
# Load config from JSON
example_config = ExampleConfig.from_json(path)
```

The `from_json` method, which is inherited from the super class
`eta.core.serial.Serializable`, reads the JSON dictionary and passes it to the
`ExampleConfig` constructor.


#### Defining new data types

In ETA, data is usually written to disk in JSON format. The following snippet
demonstrates how to define a custom data type in ETA:

```python
from eta.core.serial import Serializable

class Point(Serializable):
    '''An (x, y) point.'''

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def from_dict(cls, d):
        '''Constructs a Point from a JSON dictionary.'''
        return cls(d["x"], d["y"])
```

Note that `Point` derives from the `eta.core.serial.Serializable` class, which
implements the semantics of classes that are meant to be read/written to JSON.

To write a `Point` instance to a JSON file, simply do:

```python
from eta.core import utils

# Serialize Point to JSON
point = Point(0, 1)
utils.write_json(point, path)
```

This code produces the JSON file:

```json
{
    "x": 0,
    "y": 1
}
```

To load the `Point` from JSON, simply do:

```python
# Load Point from JSON
point = Point.from_json(path)
```

The `eta.core.serial.Serializable` class provides the `from_json` method, which
internally calls `Point.from_dict` to parse the JSON data.


#### Building objects from configuration files

Often in ETA, one may want to define a class that can be initialized from a
configuration file. Furthermore, one may have multiple classes that all derive
from a common base class (e.g., different types of filters to apply to an
image), and one wants to select and configure a particular type of filter from
a configuration file. The `eta.core.config.Configurable` class is provided
to facilitate these use cases.

Consider the following definitions:

```python
import math
from eta.core.config import Config, Configurable

class ShapeConfig(Config):
    '''Parses a Shape config.'''

    def __init__(self, d):
        self.type = self.parse_string(d, "type")
        self._shape, config_cls = Configurable.parse(__name__, self.type)
        self.config = self.parse_object(d, "config", config_cls)

    def build(self):
        '''Builds the Shape instance specified by the config.'''
        return self._shape(self.config)


class Shape(Configurable):
    '''Base class for shapes.'''

    def area(self):
        raise NotImplementedError("subclass must implement area()")


class CircleConfig(Config):
    '''Configuration settings for a circle.'''

    def __init__(self, d):
        self.radius = self.parse_number(d, "radius")


class Circle(Shape):
    '''Class representing a circle.'''

    def __init__(self, config):
        self.validate(config)
        self.config = config

    def area(self):
        return math.pi * self.config.radius ** 2.0
```

The above code define a `Shape` base class and a `Circle` class that
derives from it. Note that the `Shape` class derives from `Configurable`, since
we intend to instantiate shapes from configuration files.

Along with these classes, a `ShapeConfig` class is defined that uses the
`Configurable.parse` method to dynamically load the shape's type from the
configuration file and then build its config instance. The class also provides
a `build` method to instantiate the shape from its config. Finally, the
`CircleConfig` class is provided to specify the semantics of a `Circle`
configuration file.

We can use the above definitions to parse configuration files like this:

```json
{
    "type": "Circle",
    "config": {
        "radius": 1.0
    }
}
```

To load the configuration file and build the shape, simply do:

```python
# Load Circle configured in JSON
shape_config = ShapeConfig.from_json(path)
circle = shape_config.build()

area = circle.area()  # pi
```
