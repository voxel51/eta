import eta.core.annotations as etaa
from eta.detectors import TFModelsSegmenter, TFModelsSegmenterConfig
import eta.core.image as etai


TEST_IMAGE_PATH = "/path/to/image.jpg"
ANNO_IMAGE_PATH = "/path/for/annotated-image.jpg"
MODEL_NAME = "mask-rcnn-resnet101-atrous-coco"
CONFIDENCE_THRESH = 0.3

# Load model
config = TFModelsSegmenterConfig({
    "model_name": MODEL_NAME,
    "confidence_thresh": CONFIDENCE_THRESH,
})
segmenter = TFModelsSegmenter(config)

# Perform detection
img = etai.read(TEST_IMAGE_PATH)
with segmenter:
    objects = segmenter.detect(img)

# Visualize detections
image_labels = etai.ImageLabels(objects=objects)
img_anno = etaa.annotate_image(img, image_labels)
etai.write(img_anno, ANNO_IMAGE_PATH)
