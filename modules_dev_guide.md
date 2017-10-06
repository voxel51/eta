# ETA Module Developer's Guide

This document describes best practices for contributing modules to ETA.

## What are ETA modules?

Modules in ETA are simply executables that take JSON *configuration* files as
input and write output data to disk.

Configuration JSON files tell the module what parameters to use, what data to
read as input, and where to write output data.

## Module metadata JSON files

Every ETA module `my-module` must provide a metadata file `my-module.json`
describing the inputs and outputs of the module.

The basic syntax is:

```json
{
    "_comment": "@todo: define this!"
}
```

## Standalone modules

Since ETA modules are just executables, they can be implemented in any language
as long as they provide a valid metadata JSON file.

## Modules built using ETA

ETA provides a core library that can be used to define new analytics modules.
This section summarizes the key features of the ETA module creation syntax.

#### Parsing module configuration files

The following snippet shows a canonical definition of a module configuration in
ETA:

```python
from eta.core.config import Config

class ExampleConfig(Config):
    '''An example config class.'''

    def __init__(self, d):
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

Note that the `ExampleConfig` and `DataConfig` classes derive from the
`eta.core.config.Config` class, which implements the basic semantics of
configuration classes. In particular, the `Config` class provides `parse_*`
static methods, which are used to define the names and data types of the JSON
fields.

Fields with no `default=` keyword value are *mandatory*, and fields with a
`default=` keyword are *optional*.

The following JSON file defines a valid `ExampleConfig` instance:

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

To load the JSON file into an `ExampleConfig` instance, simply do:

```python
# Load config from JSON
example_config = ExampleConfig.from_json(path)
```

The `from_json` method, which is inherited from the super class
`eta.core.serial.Serializable`, reads the JSON dictionary and passes it to the
`ExampleConfig` constructor.

#### Defining custom output data types

The following snippet demonstrates how to define a custom data type in ETA:

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
import eta.core.utils as utils

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
