# ImageLabels Developer's Guide

The `eta.core.image.ImageLabels` class is the primary data structure in ETA for
storing information about the semantic content of images.

`ImageLabels` instances have a top-level `filename` field that contains the
filename of the image, a top-level `attrs` field that contains an
`eta.core.data.AttributeContainer` instance containing image-level attributes,
and an `objects` field that stores an
`eta.core.objects.DetectedObjectContainer` instance describing the objects that
were detected in the image.

The `eta.core.image.ImageLabelsSchema` is a related class that describes the
taxonomy of image-level and object-level attributes that may be generated when
processing an image.

`ImageLabelsSchema` files have an `attrs` field that contains an
`eta.core.data.AttributesSchema` instance that describes the schema of each
image-level attribute that may be generated. They also contain an `objects`
field that maps the set of object labels that may be detected to
`eta.core.data.AttributesSchema` instances that describe each of the
object-level attributes that may be generated for each object.

Finally, the `eta.core.image.ImageSetLabels` class provides the ability to
store a collection of `ImageLabels` instances describing multiple images.
`ImageSetLabels` instances can have their own global schema that controls the
content that can be contained in any image in the collection.


## Example Use

The following code demonstrates a toy use of the `ImageLabels` and
`ImageSetLabels` classes to store metadata about a small collection of images.

```py
import eta.core.data as etad
import eta.core.image as etai
import eta.core.geometry as etag
import eta.core.objects as etao

# Attributes
iattr1 = etad.CategoricalAttribute("scene", "intersection", confidence=0.9)
iattr2 = etad.NumericAttribute("quality", 0.5)
iattr3 = etad.BooleanAttribute("on_road", True)

# Objects
tl = etag.RelativePoint(0, 0)
br = etag.RelativePoint(1, 1)
bb = etag.BoundingBox(tl, br)
obj1 = etao.DetectedObject("car", bb, confidence=0.9, index=1)
obj1.add_attribute(etad.CategoricalAttribute("make", "Honda"))
obj2 = etao.DetectedObject("person", bb, index=2)
obj2.add_attribute(etad.NumericAttribute("age", 42, confidence=0.99))

# Test ImageLabels and ImageSetLabels
image_set_labels = etai.ImageSetLabels()

image_labels1 = etai.ImageLabels()
image_labels1a = etai.ImageLabels()
image_labels1b = etai.ImageLabels()

image_labels1.add_image_attribute(iattr1)
image_labels1a.add_image_attribute(iattr2)
image_labels1b.add_object(obj1)
image_labels1.merge_labels(image_labels1a)
image_labels1.merge_labels(image_labels1b)

image_labels2 = etai.ImageLabels()
image_labels2.add_image_attribute(iattr3)
image_labels2.add_object(obj2)

image_set_labels2 = etai.ImageSetLabels()
image_set_labels2.add_image_labels(image_labels2)

image_set_labels.add_image_labels(image_labels1)
image_set_labels.merge_image_set_labels(image_set_labels2)

# View the labels
print(image_set_labels)
```

```json
{
    "images": [
        {
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
        {
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
    ]
}
```

To view the active (current) schema of the labels:

```python
print(image_set_labels.get_active_schema())
```

```json
{
    "schema": {
        "attrs": {
            "schema": {
                "quality": {
                    "range": [0.5, 0.5],
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
                        "categories": ["Honda"]
                    }
                }
            },
            "person": {
                "schema": {
                    "age": {
                        "range": [42.0, 42.0],
                        "type": "eta.core.data.NumericAttribute",
                        "name": "age"
                    }
                }
            }
        }
    }
}
```

Now freeze the schema so labels of previously unseen types or values cannot be
added.

```py
image_set_labels.freeze_schema()
```

To demonstrate that the schema is frozen, try violating it:

```py
_labels = etai.ImageLabels()
_labels.add_image_attribute(etad.CategoricalAttribute("weather", "bad"))
image_set_labels.add_image_labels(_labels)
# AttributeContainerSchemaError: Attribute 'weather' is not allowed by the schema

_labels = etai.ImageLabels()
_labels.add_image_attribute(etad.NumericAttribute("quality", 100.0))
image_set_labels.add_image_labels(_labels)
# AttributeContainerSchemaError: Value '100.0' of attribute 'quality' is not allowed by the schema
```
