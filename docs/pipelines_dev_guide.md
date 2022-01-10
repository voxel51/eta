# Pipeline Developer's Guide

This document describes how to add new analytics pipelines to ETA. See
`modules_dev_guide.md` for more information about adding new modules to ETA,
and see `core_dev_guide.md` for instructions on contributing to the core ETA
infrastructure.


## What are ETA Pipelines?

Pipelines are the mechanisms by which analytics capabilities are exposed to the
users of the ETA system. Pipelines define the pre-packaged computations that
users can run on their data, and they expose the relevant inputs, outputs, and
tunable parameters of the analytic that the user can provide, access, and
customize.

Each ETA pipeline is represented internally as a graph whose nodes are ETA
modules and whose edges define the flow of data between the modules. As such,
the ETA pipeline system is general purpose and highly customizable. The ETA
repository defines a collection of pre-configured pipelines that arrange the
builtin ETA modules in graphs to implement various useful video analytics.

New pipelines can be easily added to the ETA system by writing a simple JSON
configuration file whose syntax is described in the next section.


## Pipeline Metadata JSON Files

Every ETA pipeline must provide a metadata JSON file describing the inputs,
outputs, modules (nodes), and data flow (edges) of the computational graph.

The pipeline metadata file contains all the necessary information to
instantiate the pipeline and module configuration files that are required
under-the-hood to run an ETA pipeline on data. The pipeline and associated
metadata files of the constituent modules define a template that is populated
by the pipeline builder for each new piece of input data.

The following JSON gives an example of the metadata file for a simple video
formatting pipeline with one module:

```json
{
    "info": {
        "name": "video_formatter",
        "type": "eta.core.types.Pipeline",
        "version": "0.1.0",
        "description": "A pipeline for formatting video files"
    },
    "inputs": ["video"],
    "outputs": ["formatted_video"],
    "modules": {
        "format_videos": {
            "name": "format_videos",
            "tunable_parameters": [
                "fps", "size", "scale", "max_fps", "max_size"
            ],
            "set_parameters": {}
        }
    },
    "connections": [
        {
            "source": "INPUT.video",
            "sink": "format_videos.input_path"
        },
        {
            "source": "format_videos.output_video_path",
            "sink": "OUTPUT.formatted_video"
        }
    ]
}
```

When discussing pipeline metadata files, we refer to each JSON object `{}` as a
**spec** (specification) because it specifies the semantics of a certain
entity, and we refer to the keys of a JSON object (e.g., `info`) as **fields.**

The pipeline metadata file contains the following top-level fields:

- `info`: a spec containing basic information about the module
- `inputs`: a list defining the names of the pipeline inputs
- `outputs`: a list defining the names of the pipeline outputs
- `modules`: a list of specs describing the modules (nodes) in the pipeline
- `connections`: a list of specs describing the connections (edges) between
    modules in the pipeline

The `info` spec contains the following fields:

- `name`: the name of the pipeline
- `type`: the type of the pipeline, i.e., what computation it performs. Must
    be a valid pipeline type exposed by the ETA library, i.e. a subclass of
    `eta.core.types.Pipeline`
- `version`: the current pipeline version
- `description`: a free-text description of the pipeline purpose and
  implementation

The `inputs` field defines the names of the inputs exposed by the pipeline.

The `outputs` field defines the names of the outputs exposed by the pipeline.

The `modules` field contains a list of module specs with the following fields:

- `name`: the name of the module to include
- `tunable_parameters`: a list of module parameters that are exposed to the
    end-user for tuning
- `set_parameters`: a dictionary whose keys are module parameters and whose
    values are values to assign to those parameters

The `connections` field contains a list of connection (edge) specs with the
following fields:

- `source`: the source (starting point) of the edge. The syntax for a source is
    `<module>.<node>`. Alternatively, the special module `INPUT` can be
    used to refer to a pipeline input
- `sink`: the sink (stopping point) of the edge. The syntax for a sink is
    `<module>.<node>`. Alternatively, the special module `OUTPUT` can be
    used to refer to a pipeline output

The pipeline metadata file defines the connectivity of the computation graph,
and the pipeline builder uses this information to instantiate the
necessary configuration files to run a pipeline on new input data.


#### Exposing a new pipeline

In order for ETA to use a pipeline, its metadata JSON file must be placed in a
directory where the ETA system can find it. The `pipeline_dirs` field in the
ETA-wide `config.json` file defines a list of directories for which all JSON
files contained in them are assumed to be pipeline metadata files.

> To add a new pipeline directory to the ETA path, either append it to the
> `pipeline_dirs` list in the ETA-wide `config.json` file or add it to the
> `ETA_PIPELINE_DIRS` environment variable during execution.


## Types in the ETA System

Because the ETA system is generic and supports third-party pipelines that may
be written in languages other than Python or otherwise developed independently
from the ETA codebase and exposed only through executable files (perhaps even
remotely via a REST API), the ETA library exposes a **type system** in the
`eta.core.types` module that defines a common framework that pipelines must
use to define their semantics.

The ETA types must be used by all pipeline metadata files whether or not the
pipeline is built using the ETA library. In particular, if a third-party
pipeline performs a new type of operation, a corresponding class must be added
to the `eta.core.types` module describing the associated type so that ETA can
understand the semantics of the pipeline. This information is used when
presenting user-facing information about available pipelines.

The `eta.core.types` module defines four top-level categories of types:
pipelines, modules, builtins, and data. See `modules_dev_guide.md` for more
information about module-specific types, which are beyond the scope of this
guide.

> Note: types may be defined in modules other than `eta.core.types` if
> necessary (e.g. on a project-specific basis), but these types must still
> inherit from the base type `eta.core.types.Type`. More specifically, for
> pipelines, all new types must be subclasses of `eta.core.types.Pipeline`.


#### Pipelines

All ETA pipelines must be declared with a `type` in their pipeline metadata
file that is a subclass of `eta.core.types.Pipeline`. Pipeline types allow
developers to declare the purpose of their pipelines and allow the ETA system
to classify and organize the available pipelines by function.

> Currently only the base pipeline type `eta.core.types.Pipeline` is available,
> so all pipelines must declare this as their type. As the ETA system grows,
> more fine-grained pipeline types will be added to make the pipeline taxonomy
> more descriptive for the end-user.


## Building and Running Pipelines

Once a pipeline metadata file exists describing the architecture of the
pipeline, it is straightforward to run a pipeline on new input data using the
`eta.core.builder` module. The following subsections describe the process.


#### Pipeline build requests

Pipeline build requests provide a JSON format that can be used to request that
a given pipeline be run on some given input data(s). The general syntax for
pipeline build requests is:

```json
{
    "pipeline": <pipeline-name>,
    "inputs": {
        <inputs>
    },
    "parameters": {
        <parameters>
    }
}
```

Here, `<pipeline-name>` specifies the name of the pipeline to run, and
`<inputs>` and `<parameters>` define the data to pass into the pipeline and
the parameter values to use during processing, respectively.

A pipeline build request is valid only if all of the following conditions are
met:

- The pipeline name must be the name of a valid pipeline metadata file exposed
    by the ETA system
- All required pipeline inputs (as defined by the pipeline metadata file) are
    provided and have valid values
- All required pipeline parameters (as defined by the pipeline metadata file)
    are provided and have valid values

For example, the following JSON defines a valid pipeline build request for the
video formatting pipeline whose metadata file was given earlier:

```json
{
    "pipeline": "video_formatter",
    "inputs": {
        "video": "examples/data/water.mp4"
    },
    "parameters": {
        "format_videos.scale": 0.5,
        "format_videos.fps": 1
    }
}
```

Note that the `<module>.<parameter>` notion is used in the above JSON to
set the parameters of the relevant modules in the pipeline.

Pipeline build requests are defined in the ETA library by the
`eta.core.builder.PipelineBuildRequestConfig` class, and they can be loaded
from JSON via the simple pattern:

```python
import eta.core.builder as etab

# Load a pipeline request from JSON
request = etab.PipelineBuildRequest.from_json("/path/to/pipeline-request.json")
```


#### Building a pipeline instance

Given a valid pipeline build request, it is straightforward to build a pipeline
that can execute it using the `eta.core.builder.PipelineBuilder` class. The
class constructor accepts an `eta.core.builder.PipelineBuildRequest` instance
and provides a `build()` method that instantiates the module and pipeline
configuration files. In other words,

```python
import eta.core.builder as etab

# Build the pipeline defined by the given PipelineBuildRequest instance
builder = etab.PipelineBuilder(request)
builder.build()
```

When the `build()` method is called, a pipeline config file and the associated
module config files necessary to run the pipeline on the given input data with
the given parameters are automatically generated and written to disk. The
outputs of each module are written to a specified output directory with a
folder structure based on the names of the constituent modules. Both the
config directory and the base output directory can be configured as desired.

> To change the directory where pipeline configs are written, either modify
> the `config_dir` field in the ETA-wide `config.json` file or set the
> `ETA_CONFIG_DIR` environment variable during execution.

> To change the directory where pipeline outputs are written, either modify
> the `output_dir` field in the ETA-wide `config.json` file or set the
> `ETA_OUTPUT_DIR` environment variable during execution.


#### Full pipeline building example

Combining the above steps, the following Python code shows how to build and run
a pipeline from a build request:

```python
import eta.core.builder as etab
import eta.core.pipeline as etap

# Load the pipeline request
request = etab.PipelineBuildRequest.from_json("/path/to/pipeline-request.json")

# Build the pipeline
builder = etab.PipelineBuilder(request)
builder.build()

# Run the pipeline
etap.run(builder.pipeline_config_path)
```

Alternatively, one can build and run a pipeline from the command-line using the
`eta` executable:

```shell
# Build and run the pipeline
eta build -r "/path/to/pipeline-request.json" --run-now
```


## Visualizing Pipelines

The ETA system provides the ability to visualize pipelines as block diagrams
using the [blockdiag](https://pypi.python.org/pypi/blockdiag) package.

For example, the block diagram file for the simple video formatter described
above can be generated by executing:

```python
import eta.core.pipeline as etap

# Load the pipeline metadata
pipeline = etap.load_metadata("video_formatter")

# Render the block diagram
pipeline.render("pipeline_block_diagram.svg")
```

The above code generates the following image that depicts the pipeline as a
block diagram:

[![pipeline\_block\_diagram.png](
https://drive.google.com/uc?id=1GQGAnDAi3ZZCCsIP8xNP7ul8gJFlvYtk)](
https://drive.google.com/uc?id=1ArnECNoNFm_f9--vxWVtn80RQqDQgtKH)

Behind the scenes, an intermediate `pipeline_block_diagram.diag` file is
generated that describes the pipeline in a format understood by the `blockdiag`
package.


## Copyright

Copyright 2017-2022, Voxel51, Inc.<br>
voxel51.com
