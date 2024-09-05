from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

class IDDetector:
    def __init__(self, model_path):
        self.detector = YOLO(model_path)

    def detect(self, image: Image.Image):
        results = self.detector(image)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist() 
        detected_class = names_dict[np.argmax(probs)]  
        
        return detected_class

class IDSegmenter:
    def __init__(self, model_path):
        self.segmenter = YOLO(model_path)  # Load the YOLOv8 model

    def segment(self, image: Image.Image):
        results = self.segmenter(image)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        highest_confidence_boxes = {}
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if class_id not in highest_confidence_boxes or confidence > highest_confidence_boxes[class_id]['confidence']:
                highest_confidence_boxes[class_id] = {
                    'box': box,
                    'confidence': confidence,
                    'class_id': class_id  # Store the class ID
                }

        segmented_images = []
        class_names = []  # To store class names

        for entry in highest_confidence_boxes.values():
            box = entry['box']
            x1, y1, x2, y2 = box.astype(int)
            segmented_image = Image.fromarray(cv2.cvtColor(cv2.imread(image)[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            segmented_images.append(segmented_image)
            
            # Append the class name using the model's class mapping
            class_names.append(self.segmenter.names[int(entry['class_id'])])  # Map class ID to name

        return segmented_images, class_names
