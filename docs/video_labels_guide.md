# VideoLabels Developer's Guide

The `eta.core.video.VideoLabels` class is the primary data structure in ETA for
storing information about the semantic content of video.

`VideoLabels` instances have a top-level `filename` field that contains the
filename of the video, a top-level `attrs` field that contains an
`eta.core.data.AttributeContainer` instance containing video-level attributes,
and a `frames` field that contains a dictionary keyed by frame number. In turn,
each frame contains an `attrs` field that stores an
`eta.core.data.AttributeContainer` instance containing frame-level attributes
describing the scene and an `objects` field that stores an
`eta.core.objects.DetectedObjectContainer` instance describing the objects that
were detected in the frame.

The `eta.core.video.VideoLabelsSchema` is a related class that describes the
taxonomy of video-level, frame-level, and object-level attributes that may be
generated when processing a video.

`VideoLabelsSchema` files have an `attrs` field that contains an
`eta.core.data.AttributesSchema` instance that describes the schema of each
video-level attribute that may be generated. They also contain a `frames`
field that contains an `eta.core.data.AttributesSchema` instance that describes
the schema of each frame-level attribute that may be generated. Finally, they
contain an `objects` field that maps the set of object labels that may be
detected to `eta.core.data.AttributesSchema` instances that describe each of
the object-level attributes that may be generated for each object.

Finally, the `eta.core.video.VideoSetLabels` class provides the ability to
store a collection of `VideoLabels` instances describing multiple videos.
`VideoSetLabels` instances can have their own global schema that controls the
content that can be contained in any video in the collection.


## Example Use

The following code demonstrates a toy use of the `VideoLabels` class to store
metadata about a video.

```py
import eta.core.data as etad
import eta.core.geometry as etag
import eta.core.video as etav
import eta.core.objects as etao

# Video attributes
vattr1 = etad.CategoricalAttribute("weather", "rain", confidence=0.95)
vattr2 = etad.NumericAttribute("fps", 30)
vattr3 = etad.BooleanAttribute("daytime", True)

# Frame attributes
fattr1 = etad.CategoricalAttribute("scene", "intersection", confidence=0.9)
fattr2 = etad.NumericAttribute("quality", 0.5)
fattr3 = etad.BooleanAttribute("on_road", True)

# Create some objects
tl = etag.RelativePoint(0, 0)
br = etag.RelativePoint(1, 1)
bb = etag.BoundingBox(tl, br)
obj1 = etao.DetectedObject("car", bb, confidence=0.9, index=1)
obj1.add_attribute(etad.CategoricalAttribute("make", "Honda"))
obj2 = etao.DetectedObject("person", bb, index=2)
obj2.add_attribute(etad.NumericAttribute("age", 42, confidence=0.99))

#
# Populate VideoLabels
#

video_labels = etav.VideoLabels()

vattrs = etad.AttributeContainer()
vattrs.add(vattr2)
vattrs.add(vattr3)
video_labels.add_video_attribute(vattr1)
video_labels.add_video_attributes(vattrs)

# Add a frame
frame_labels = etav.VideoFrameLabels(1)
frame_labels.add_attribute(fattr1)
frame_labels.add_attribute(fattr2)
frame_labels.add_object(obj1)
video_labels.add_frame(frame_labels)

# Add another frame a different way
video_labels.add_frame_attribute(fattr3, 2)
video_labels.add_object(obj2, 2)

# View the labels
print(video_labels)
```

```json
{
    "attrs": {
        "attrs": [
            {
                "type": "eta.core.data.CategoricalAttribute",
                "name": "weather",
                "value": "rain",
                "confidence": 0.95
            },
            {
                "type": "eta.core.data.NumericAttribute",
                "name": "fps",
                "value": 30.0
            },
            {
                "type": "eta.core.data.BooleanAttribute",
                "name": "daytime",
                "value": true
            }
        ]
    },
    "frames": {
        "1": {
            "frame_number": 1,
            "attrs": {
                "attrs": [
                    {
                        "type": "eta.core.data.CategoricalAttribute",
                        "name": "scene",
                        "value": "intersection",
                        "confidence": 0.9
                    },
                    {
                        "type": "eta.core.data.NumericAttribute",
                        "name": "quality",
                        "value": 0.5
                    }
                ]
            },
            "objects": {
                "objects": [
                    {
                        "label": "car",
                        "bounding_box": {
                            "bottom_right": {
                                "y": 1.0,
                                "x": 1.0
                            },
                            "top_left": {
                                "y": 0.0,
                                "x": 0.0
                            }
                        },
                        "confidence": 0.9,
                        "index": 1,
                        "attrs": {
                            "attrs": [
                                {
                                    "type": "eta.core.data.CategoricalAttribute",
                                    "name": "make",
                                    "value": "Honda"
                                }
                            ]
                        }
                    }
                ]
            }
        },
        "2": {
            "frame_number": 2,
            "attrs": {
                "attrs": [
                    {
                        "type": "eta.core.data.BooleanAttribute",
                        "name": "on_road",
                        "value": true
                    }
                ]
            },
            "objects": {
                "objects": [
                    {
                        "label": "person",
                        "bounding_box": {
                            "bottom_right": {
                                "y": 1.0,
                                "x": 1.0
                            },
                            "top_left": {
                                "y": 0.0,
                                "x": 0.0
                            }
                        },
                        "index": 2,
                        "frame_number": 2,
                        "attrs": {
                            "attrs": [
                                {
                                    "type": "eta.core.data.NumericAttribute",
                                    "name": "age",
                                    "value": 42.0,
                                    "confidence": 0.99
                                }
                            ]
                        }
                    }
                ]
            }
        }
    }
}
```

To view the active (current) schema of the labels:

```python
print(video_labels.get_active_schema())
```

```json
{
    "schema": {
        "attrs": {
            "schema": {
                "weather": {
                    "type": "eta.core.data.CategoricalAttribute",
                    "name": "weather",
                    "categories": [
                        "rain"
                    ]
                },
                "daytime": {
                    "type": "eta.core.data.BooleanAttribute",
                    "name": "daytime"
                },
                "fps": {
                    "range": [
                        30.0,
                        30.0
                    ],
                    "type": "eta.core.data.NumericAttribute",
                    "name": "fps"
                }
            }
        },
        "frames": {
            "schema": {
                "quality": {
                    "range": [
                        0.5,
                        0.5
                    ],
                    "type": "eta.core.data.NumericAttribute",
                    "name": "quality"
                },
                "scene": {
                    "type": "eta.core.data.CategoricalAttribute",
                    "name": "scene",
                    "categories": [
                        "intersection"
                    ]
                },
                "on_road": {
                    "type": "eta.core.data.BooleanAttribute",
                    "name": "on_road"
                }
            }
        },
        "objects": {
            "car": {
                "schema": {
                    "make": {
                        "type": "eta.core.data.CategoricalAttribute",
                        "name": "make",
                        "categories": [
                            "Honda"
                        ]
                    }
                }
            },
            "person": {
                "schema": {
                    "age": {
                        "range": [
                            42.0,
                            42.0
                        ],
                        "type": "eta.core.data.NumericAttribute",
                        "name": "age"
                    }
                }
            }
        }
    }
}
```

Now freeze the schema so no labels of new previously unseen types or values can
be added.

```py
# Test schema
video_labels.freeze_schema()
```

To demonstrate that the schema is frozen, try violating it:

```py
video_labels.add_video_attribute(etad.NumericAttribute("weather", 100.0))
# AttributeSchemaError: Expected attribute 'weather' to have type 'eta.core.data.CategoricalAttribute'; found 'eta.core.data.NumericAttribute'

video_labels.add_video_attribute(etad.CategoricalAttribute("weather", "bad"))
# AttributeContainerSchemaError: Value 'bad' of attribute 'weather' is not allowed by the schema
```
