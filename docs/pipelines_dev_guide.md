# ETA Pipeline Developer's Guide

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
modules and whose edges define the flow of data between the modules. Thus the
ETA pipeline system is general purpose and highly customizable. The ETA
repository defines a collection of pre-configured pipelines that combine the
builtin ETA modules in many ways to useful video analytics capabilities.\

New pipelines can be easily added to the ETA system by writing a simple JSON
configuration file whose syntax is described in the next section.


## Pipeline Metadata JSON Files

Every ETA pipeline must provide a metadata JSON file describing the inputs,
outputs, modules (nodes), and data flow (edges) of the computational graph.

The metadata file contains all the necessary information to instantiate the
pipeline and module configuration files that are required under-the-hood to run
an ETA pipeline on data. The pipeline and associated metadata files of the
constituent modules define a template that is populated by the pipeline builder
for each new piece of input data.

The following JSON gives an example of the metadata file for a simple video
formatting pipeline:

```json
{
    "info": {
        "name": "video-formatter",
        "type": "eta.core.types.Pipeline",
        "version": "0.1.0",
        "description": "A pipeline for formatting video files"
    },
    "inputs": ["video", "event_detection"],
    "outputs": ["sampled_frames"],
    "modules": [
        {
            "name": "resize_videos",
            "set_parameters": {},
            "tunable_parameters": ["size", "scale"]
        },
        {
            "name": "clip_videos",
            "set_parameters": {},
            "tunable_parameters": ["frames"]
        },
        {
            "name": "sample_videos",
            "set_parameters": {},
            "tunable_parameters": ["fps"]
        }
    ],
    "connections": [
        {
            "source": "INPUT.video",
            "sink": "resize_videos.input_path"
        },
        {
            "source": "INPUT.event_detection",
            "sink": "clip_videos.events_json_path"
        },
        {
            "source": "resize_videos.output_path",
            "sink": "clip_videos.input_path"
        },
        {
            "source": "clip_videos.output_path",
            "sink": "sample_videos.input_path"
        },
        {
            "source": "sample_videos.output_path",
            "sink": "OUTPUT.sampled_frames"
        }
    ]
}
```

When discussing pipeline metadata files, we refer to each JSON object `{}` as a
**spec** (specification) because it specifies the semantics of a certain
entity, and we refer to the keys of a JSON object (e.g., "info") as **fields**.

The pipeline metadata file contains the following top-level fields:

- `info`: a spec containing basic information about the module

- `inputs`: a list defining the names of the pipeline inputs

- `inputs`: a list defining the names of the pipeline outputs

- `modules`: a list of specs describing the modules (nodes) in the pipeline

- `connections`: a list of specs describing the connections (edges) between
    modules in the pipeline

The `info` spec contains the following fields:

- `name`: the name of the pipeline

- `type`: the type of the pipeline, i.e., what computation it performs. Must
    be a valid pipeline type exposed by the ETA library

- `version`: the current pipeline version

- `description`: a short free-text description of the pipeline purpose and
    implementation

The `modules` field contains a list of module specs with the following fields:

- `name`: the name of the module to include

- `set_parameters`: a dictionary whose keys are module parameters and whose
    values are values to assign to those parameters

- `tunable_parameters': a list of module parameters that are exposed to the
    end-user for tuning

The `connections` field contains a list of connection (edge) specs with the
following fields:

- `source`: the source (starting point) of the edge. The syntax for a source is
    `"<module>.<field>"`. Alternatively, the special module `"INPUT"` can be
    used to refer to a pipeline input

- `sink`: the sink (stopping point) of the edge. The syntax for a sink is
    `"<module>.<field>"`. Alternatively, the special module `"OUTPUT"` can be
    used to refer to a pipeline output

The pipeline metadata file defines the connectivity of the computation graph.
In practice, the pipeline builder uses this information to instantiate the
necessary configuration files to run a pipeline on new input data.


## Visualizing Pipelines

The ETA system provides the ability to visualize pipelines as block diagrams
using the [blockdiag](https://pypi.python.org/pypi/blockdiag) package.

For example, the block diagram file for the simple video formatter described
above can be generated by executing:

```python
import eta.core.pipeline as etap

# Load the pipeline
pipeline = etap.load_metadata("video_formatter")

# Render the block diagram
pipeline.render("pipeline_block_diagram.svg")
```

The above code generates a `block_diagram.svg` vector graphics image of the
block diagram. It also generates the following intermediate
`block_diagram.diag` file describing the network architecture:

```
blockdiag {

  // inputs
  video [width = 114, shape = cloud, height = 40];
  event_detection [width = 214, shape = cloud, height = 40];

  // outputs
  sampled_frames [width = 204, shape = cloud, height = 40];

  // connections
  video -> 1.input_path;
  event_detection -> 2.events_json_path;
  1.output_path -> 2.input_path;
  2.output_path -> 3.input_path;
  3.output_path -> sampled_frames;

  // modules
  group {
    color = "#AAAAAA";

    // module
    1.resize-videos [width = 110, shape = box, height = 60];

    // inputs
    1.input_path [width = 164, shape = endpoint, height = 40];

    // outputs
    1.output_path [width = 174, shape = endpoint, height = 40];

    // parameters
    1.size [width = 40, shape = beginpoint, rotate = 270, height = 104];
    1.scale [width = 40, shape = beginpoint, rotate = 270, height = 114];
    1.scale_str [width = 40, shape = beginpoint, rotate = 270, height = 154];
    1.ffmpeg_out_opts [width = 40, shape = beginpoint, rotate = 270, height = 214];

    // I/O connections
    1.input_path -> 1.resize-videos;
    1.resize-videos -> 1.output_path;

    // parameter connections
    group {
      color = "#EE7531";
      orientation = portrait;
      1.size -> 1.resize-videos;
      1.scale -> 1.resize-videos;
      1.scale_str -> 1.resize-videos;
      1.ffmpeg_out_opts -> 1.resize-videos;
    }
  }
  group {
    color = "#AAAAAA";

    // module
    2.clip-videos [width = 94, shape = box, height = 60];

    // inputs
    2.input_path [width = 164, shape = endpoint, height = 40];
    2.events_json_path [width = 224, shape = endpoint, height = 40];

    // outputs
    2.output_path [width = 174, shape = endpoint, height = 40];

    // parameters
    2.frames [width = 40, shape = beginpoint, rotate = 270, height = 124];

    // I/O connections
    2.input_path -> 2.clip-videos;
    2.events_json_path -> 2.clip-videos;
    2.clip-videos -> 2.output_path;

    // parameter connections
    group {
      color = "#EE7531";
      orientation = portrait;
      2.frames -> 2.clip-videos;
    }
  }
  group {
    color = "#AAAAAA";

    // module
    3.sample-videos [width = 110, shape = box, height = 60];

    // inputs
    3.input_path [width = 164, shape = endpoint, height = 40];
    3.clips_path [width = 164, shape = endpoint, height = 40];

    // outputs
    3.output_path [width = 174, shape = endpoint, height = 40];

    // parameters
    3.fps [width = 40, shape = beginpoint, rotate = 270, height = 94];

    // I/O connections
    3.input_path -> 3.sample-videos;
    3.clips_path -> 3.sample-videos;
    3.sample-videos -> 3.output_path;

    // parameter connections
    group {
      color = "#EE7531";
      orientation = portrait;
      3.fps -> 3.sample-videos;
    }
  }
}
```
