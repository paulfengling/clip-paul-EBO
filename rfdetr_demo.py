from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv
from PIL import Image
import numpy as np
import cv2 as cv
from rfdetr import RFDETRNano
image = Image.open("dog-2.jpeg")
model = RFDETRNano()
model.optimize_for_inference()
predictions = model.predict(image, confidence=0.5)[0]

detections = sv.Detections.from_inference(predictions)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]
print(labels)
annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)


# Convert PIL Image to numpy array
numpy_image = np.array(annotated_image)

# Convert RGB to BGR (OpenCV uses BGR by default)
opencv_image = cv.cvtColor(numpy_image, cv.COLOR_RGB2BGR)

# Now you can use it with OpenCV
cv.imshow('OpenCV Image', opencv_image)
cv.waitKey(0)
cv.destroyAllWindows()