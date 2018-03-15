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
execution. The ETA system automatically generates these configuration files
whenever it builds and executes pipelines.

The following shows an example metadata file `simple_object_detector.json` for
a simple object detector module:

```json
{
    "info": {
        "name": "simple_object_detector",
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

- `name`: the name of the module. By convention, this name should match the
    name of the module metadata file without the extension

- `type`: the type of the module, i.e., what computation it performs. Must be a
    valid module type exposed by the ETA library, i.e. a subclass of
    `eta.core.types.Module`

- `version`: the current module version

- `description`: a short free-text description of the module purpose and
    implementation

- `exe`: the name of the module executable file

The remaining specs describe the fields in the module's configuration files.
Each spec has the fields:

- `name`: the name of the field

- `type`: the type of the field, which must be a valid type exposed by the ETA
    library. Module inputs must have a type that is a subclass of
    `eta.core.types.ConcreteData` or `eta.core.types.AbstractData`. Module
    outputs must have a type that is a subclass of
    `eta.core.types.ConcreteData`. Module parameters can have types that are
    subclasses of `eta.core.types.ConcreteData` or `eta.core.types.Builtin`

- `description`: a short free-text description of the field

- `required`: whether the field is required or optional for the module to
    function

- `default`: (parameters only) an optional default value for the parameter. If
    a parameter is required but has a default value, it may be omitted from the
    module configuration file


#### Exposing a new module

In order for ETA to use a module, its metadata JSON file must be placed in a
directory where the ETA system can find it. The `module_dirs` field in the
ETA-wide `config.json` file defines a list of directories for which all JSON
files contained in them are assumed to be module metadata files.

> To add a new module directory to the ETA path, either append it to the
> `module_dirs` list in the ETA-wide `config.json` file or add it to the
> `ETA_MODULE_DIRS` environment variable during execution.


## Types in the ETA System

Because the ETA module system is generic and supports third-party modules that
may be written in languages other than Python or otherwise developed
independently from the ETA codebase and exposed only through executable files
(perhaps even remotely via a REST API), the ETA library exposes a
**type system** in the `eta.core.types` module that defines a common framework
that modules must use to define the semantics of their fields (inputs, outputs,
and parameters).

The ETA types must be used by all module metadata files whether or not the
module is built using the ETA library. In particular, if a third-party module
introduces a new type of field (e.g., a new output JSON format or a new type of
parameter), a corresponding class must be added to the `eta.core.types` module
describing the type so that ETA that can understand the semantics of the module
and properly configure it and invoke it in the midst of a pipeline.

The `eta.core.types` module defines four top-level categories of types:
pipelines, modules, builtins, and data. The following sections provide a brief
overview of these basic types and describe their use.

> Types may be defined in modules other than `eta.core.types` if necessary
> (e.g. on a project-specific basis), but these types must still inherit from
> the base type `eta.core.types.Type` and from the relevant base module,
> pipeline, builtin, or data types as appropriate.


#### Pipelines

All ETA pipelines must be declared with a `type` in their pipeline metadata
file that is a subclass of `eta.core.types.Pipeline`. Pipeline types allow
developers to declare the purpose of their pipelines and allow the ETA system
to classify and organize the available pipelines by purpose. See
`piplines_dev_guide.md` for more information about pipeline types, which are
beyond the scope of this guide.


#### Modules

All ETA modules must be declared with a `type` in their module metadata file
that is a subclass of `eta.core.types.Module`. Module types allow developers
to declare the purpose of their module and allows the ETA system to classify
and organize the available modules in an informative fashion for end-users.

> Currently only the base module type `eta.core.types.Module` is available, so
> all modules must declare this as their type. As the ETA system grows, more
> fine-grained module types will be added to make the module taxonomy more
> descriptive and useful for building custom pipelines.


#### Builtins

Builtins are literal types that are extracted directly from JSON files.
All builtin types must be subclasses of `eta.core.types.Builtin`. There is a
builtin type corresponding to each of the main types of data that can be stored
in JSON files:

- `eta.core.types.Null`: A JSON null value. `None` in Python

- `eta.core.types.Boolean`: A JSON boolean value. A `bool` in Python

- `eta.core.types.String`: A JSON string. A `str` in Python

- `eta.core.types.Number`: A numeric value

- `eta.core.types.Array`: A JSON array. A `list` in Python

- `eta.core.types.Object`: An object in JSON. A dict in Python

In addition, more specific types can be defined that are subclasses of the
above base types. For example, the following class is a subclass of
`eta.core.types.Array`:

- `eta.core.types.StringArray`: An array of strings in JSON. A list of strings
    in Python.

There are also a number of subclasses of `eta.core.types.Object` that define
custom object types, which are typically used to define module parameters that
are more sophisticated than simple JSON primitives:

- `eta.core.types.Point`: An (x, y) coordinate point defined by "x" and "y"
    coordinates, which must be nonnegative. Typically, Points represent
    coordinates of pixels in images. For example:
    ```json
    {
        "x": 0,
        "y": 128
    }
    ```

In the context of module metadata files, only _parameters_ can have types that
are subclasses of `Builtin`, since module inputs must be read from disk and
module outputs must be written to disk (i.e. they must be `Data` types, as
described below).

All `Builtin` subclasses must implement a static `is_valid_value(val)` method
that verifies that `val` is a valid value for that type.


#### Data

Data are types that are stored on disk and referenced by a filepath. All data
types must be subclasses of `eta.core.types.Data`, and they must implement a
static `is_valid_path(path)` method that validates whether the given `path` is
a valid filepath for that type.

There are two primary classes of data:

- `eta.core.types.ConcreteData`: the base type for concrete data types, which
    represent well-defined data types that can be written to disk

- `eta.core.types.AbstractData`: the base type for abstract data types, which
    define base data types that encapsulate one or more `ConcreteData` types

Concrete data types must implement a static `gen_path(basedir, params)` method,
which is used to automatically generate filepaths. In this method, `basedir`
is the base output directory where the data will be written, and `params` is an
instance of the `eta.core.types.ConcreteDataParams` class, which contains a
dictionary of configuration settings that the `ConreteData` subclass can use to
properly generate output paths. This automatic path generation capability is
utilized during the pipeline building process when populating module configs
based on a pipeline request.

Abstract data types allow the ETA type system to express that multiple concrete
data types are interchangable instantiations of a single concept. Since
abstract types do not refer to a unique concrete data type, they do not provide
`gen_path` methods. An example of an abstract data type is:

- `eta.core.types.Video`: the abstract data type representing a single video

The abstract video type currently has two concrete implementations in ETA:

- `eta.core.types.VideoFile`: a video represented as a single encoded video
    file, e.g. `"/path/to/video.mp4"`

- `eta.core.types.ImageSequence`: a video represented as a sequence of images
    with one numeric parameter, e.g. `"/path/to/video/%05d.png"`

In the context of module metadata files, module inputs can have types that are
subclasses of `ConcreteData` or `AbstractData`. A module input declared with an
abstract data type promises that it can understand any concrete subclass of
that type.  Module inputs inherit data paths from their incoming connections,
so they do not require the ability to generate paths automatically during
pipeline building. Conversely, module outputs must be subclasses of
`ConcreteData` because they must be able to generate their output paths during
pipeline building. Module parameters may have types that are subclasses of
`ConcreteData` (e.g. a model weights file) or they may have `Builtin` types.

An important class of concrete data types in ETA are JSON files:

- `eta.core.types.JSONFile`: a JSON file on disk

JSON files are important because they are the primary way that modules write
their analytic outputs to disk. As such, there are many subclasses of JSON
that describe the various JSON formats used by modules. For example:

- `eta.core.types.Frame`: A type describing detected objects in a frame. This
    type is implemented in ETA by the `eta.core.objects.Frame` class

- `eta.core.types.EventDetection`: A per-frame binary event detection. This
    type is implemented in ETA by the `eta.core.events.EventDetection` class

Whenver a new JSON data type is used by an ETA module, a corresponding class
must be added to the `eta.core.types` module to define this type so that other
modules can declare their compatibility with this JSON format.


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

The above code generates this
<a href="https://drive.google.com/uc?id=1v3CLijGzcXawzR8L44bhr_lC5B7aYPzv">
module_block_diagram.svg
</a>
image that depicts the module as a block diagram. Behind the scenes, it first
generates the following intermediate `module_block_diagram.diag` file
describing the module definition in a format understood by the `blockdiag`
package:

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

Module configuration files define the necessary information for the ETA system
to configure the parameters of a module and execute it on some input data to
generate some output data.

The general format of a module configuration file is:

```json
{
    "data": [
        {
            <inputs>,
            <outputs>
        },
        ...
    ],
    "parameters": {
        <parameters>
    },
    "base": {
        <base-settings>
    }
}
```

The `data` field contains a list of specs, each of which contains a valid set
of input and output fields specifiying where to read input data and write
output data when the module is executed. This field expects a list so that
multiple datasets can be processed in a single module execution, if desired.
The possible fields that can be listed in `<inputs>` and `<outputs>` in the
above JSON are defined by the `inputs` and `outputs` fields of the module's
metadata JSON file. In particular, each spec in the `data` field must contain
all inputs and outputs that are marked as _required_ in the module metadata
file and may also contain any inputs and ouptuts that are optional.

The `parameters` field defines the parameter values to use when executing the
module. The possible fields that can be listed in `<parameters>` in the
above JSON are defined by the `parameters` field of the module's metadata JSON
file. In particular, each spec in the `parameters` field must contain all
parameters that are _required_ and have _no default value_ in the module
metadata file and may also contain any optional parameters.
Again, the particular parameters supported by the module are defined by
the module's metadata JSON file, and all required parameters and zero or more
optional parameters must be specified.

Finally, the `base` field defines module configuration fields that all ETA
modules must support. Note that the `base` field is not mentioned in the
module metadata file because it contains generic fields that are the same for
all modules. Indeed, the `base` field is an instance of the
`eta.core.module.BaseModuleConfigSettings` class, which defines the following
fields:

- `logging_config`: an `eta.core.log.LoggingConfig` instance that configures
    the logging behavior of the module during execution

Importantly, all ETA modules must obey the convention that the `base` field and
all of its sub-fields are _optional_; i.e., modules must internally provide
default values for these parameters and module configuration files need not
specify these values. The ETA library automates these boilerplate constructs
via the `BaseModuleConfig` and the `setup()` methods from the `eta.core.module`
module.


#### Example module configuration file

The following JSON depicts a valid module configuration file for the simple
object detector module defined earlier:

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
    "parameters": {
        "labels": ["car", "truck", "bus"]
    }
}
```

Note that the `annotated_frames_path` is omitted from the second data spec,
which is allowed since the field is optional. Similarly, the `weights`
parameter is omitted, which is allowed because a default value was provided for
that parameter in the metadata file. The top-level `base` field was omitted
entirely, which is allowed since this field is always optional.


## Module Execution Syntax

Recall that ETA modules are simply _executables_ that take JSON files as input
and write output data to disk. All ETA modules must follow the following
command-line syntax, regardless of whether they are implemented using the ETA
library:

```shell
<module-name> MODULE_CONFIG_PATH [PIPELINE_CONFIG_PATH]
```

Here, `<module-name>` is the name of the module and `MODULE_CONFIG_PATH` is
the path to a valid module configuration JSON file for the module as described
in the previous section.

Modules executables must support an optional `PIPELINE_CONFIG_PATH` argument
that specifies the path to a _pipeline configuration JSON file_, which is
supplied when a module is executed in the context of a pipeline.
Pipeline configuration JSON files may set/override zero or more base module
configuration settings defined by `eta.core.module.BaseModuleConfigSettings`,
so ETA modules must check for and appropriately handle these fields.

> Pipeline configuration JSON files also contain various pipeline-level fields
> that are not relevant to modules and should be ignored.


## Building Standalone Modules

Since ETA modules are simply executables, they can be implemented in any
language as long as they provide a valid metadata file, follow the ETA module
execution syntax, and
module execution syntax.

However, even if developers don't intend to use ETA libraries to build their
modules, they must be familiar with the ETA supported data types in order to
properly generate their module metadata files, and they must be familiar with
the `eta.core.module.BaseModuleConfig` class in order to appropriately
implement the generic module configuration settings that all modules must
support.


## Building Modules Using ETA

ETA provides a core library that can be leveraged to easily define new modules.
This section provides numerous examples describing the key features of the
module implementation utilities in ETA.


#### Module template

The following liberally-documented Python code describes the template that most
modules provided in the ETA repository follow.

> The `{{}}` blocks denote placeholders that are replaced in practice by the
> appropriate strings for the module being written.

```python
#!/usr/bin/env python
# ETA modules are simply executables, so the module definition must start with
# a shebang line declaring the python interpreter to use during execution

# All modules should provide a docstring that describes the purpose of the
module.
'''
{{Description of the module}}

Copyright 2017-2018, Voxel51, LLC
voxel51.com
'''
#
# Inputs in ETA are separated into groups and sorted alphabetically within
# each group. See `style_guide.md` for more information
#
# The first block of `__future__` imports allow for cross-version Python
# support. See `python23_guide.md` for more information
#

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


# By convention, all ETA modules should use a logger whose name is `__name__`
# to log all messages
logger = logging.getLogger(__name__)


# Defines the module configuration file.
# Inherits from `BaseModuleConfig`, which handles the parsing of the base
# module settings automatically
class {{ModuleName}}Config(etam.BaseModuleConfig):
    '''Module configuration settings.'''

    def __init__(self, d):
        # Call the `BaseModuleConfig` constructor, which parses the optional
        # `base` field
        super({{ModuleName}}Config, self).__init__(d)

        # Parse the `data` field, which is defined by an array of `DataConfig`
        # instances
        self.data = self.parse_object_array(d, "data", DataConfig)

        # Parse the `parameters` field, which is defined by a
        # `ParametersConfig` instance
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


# Parses the inputs and outputs for the module as defined in its module
# metadata file
class DataConfig(Config):
    '''Data configuration settings.'''

    def __init__(self, d):
        # Template for parsing an input field
        self.{{input}} = self.parse_{{input_type}}(d, "{{input}}")

        # Template for parsing an output field
        self.{{output}} = self.parse_{{output_type}}(d, "{{output}}")


# Parses the parameters for the module as defined in its module metadata file
class ParametersConfig(Config):
    '''Parameter configuration settings.'''

    def __init__(self, d):
        # Template for parsing a parameter with a default value
        self.{{parameter}} = self.parse_{{parameter_type}}(
            d, "{{parameter}}", default={{parameter_default_value}})


# By convention, all modules in the ETA library define a `run()` method that
# parses the command-line arguments, performs base module setup, and then calls
# another method that implements the actual module-specific actions
def run(config_path, pipeline_config_path=None):
    '''Run the {{module_name}} module.

    Args:
        config_path: path to a {{ModuleName}}Config file
        pipeline_config_path: optional path to a PipelineConfig file
    '''
    # Load the module config
    config = {{ModuleName}}Config.from_json(config_path)

    # Perform base module setup via the `eta.core.module.setup()` method
    # provided by the ETA library
    etam.setup(config, pipeline_config_path=pipeline_config_path)

    # Now pass the `config` instance to another private method to perform the
    # actual module computations
    # ...


if __name__ == "__main__":
    # Pass the command-line arguments to the `run()` method for parsing and
    # processing
    run(*sys.argv[1:])
```


#### Concrete module configuration example

The following snippet shows a concrete example of a module configuration
definition using the ETA library:

```python
import eta.core.module as etam

class ExampleModuleConfig(etam.BaseModuleConfig):
    '''An example config class.'''

    def __init__(self, d):
        super(ExampleModuleConfig, self).__init__(d)
        self.data = self.parse_object_array(d, "data", DataConfig)
        self.parameters = self.parse_object(d, "parameters", ParametersConfig)


class DataConfig(Config):
    '''Data configuration settings.'''

    def __init__(self, d):
        self.input_path = self.parse_string(d, "input_path")
        self.output_path = self.parse_string(d, "output_path")


class ParametersConfig(Config):
    '''Parameter configuration settings.'''

    def __init__(self, d):
        self.parameter = self.parse_number(d, "parameter", default=1.0)
```

The snippet defines a configuration class called `ExampleModuleConfig` that
defines a `data` field that contains an array of `DataConfig` instances and
a `parameters` field that contains an instance of `ParametersConfig`. Also,
`ExampleModuleConfig` derives from `eta.core.module.BaseModuleConfig`, which
which handles the parsing of the optional `base` field containing any base
module settings.

The `eta.core.config.Config.parse_*` methods are used to parse the JSON fields
according to their declared types. Fields defined with no `default` keyword
are _required_, and fields with a `default` keyword are _optional_.

The following JSON file is a valid `ExampleModuleConfig` configuration file:

```json
{
    "data": [
        {
            "input_path": "/path/to/input.mp4",
            "output_path": "/path/to/output.mp4"
        }
    ],
    "parameters": {}
}
```

Note that the `parameter` field is omitted from the `parameters` spec, which
is allowed since a default value was specified in the `ParametersConfig` class.
The top-level `base` field is also omitted entirely, which is allowed since
the `eta.core.module.BaseModuleConfig` class supports a default value for it.

To load the configuration file into an `ExampleModuleConfig` instance,
simply do:

```python
# Load config from JSON
example_config = ExampleModuleConfig.from_json(path)
```

The `from_json` method, which is inherited from the super class
`eta.core.serial.Serializable`, reads the JSON dictionary and passes it to the
`ExampleModuleConfig` constructor.


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
