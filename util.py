from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2


id_detector=YOLO('best-cls.pt')
id_segmentation=YOLO('best-seg.pt')

def detect_id(image: Image.Image):
    results = id_detector(image)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist() 
    detected_class = names_dict[np.argmax(probs)]  
    
    return detected_class
def segment_id(image: Image.Image):
    
    results = id_segmentation(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    highest_confidence_boxes = {}
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
   
        if class_id not in highest_confidence_boxes or confidence > highest_confidence_boxes[class_id]['confidence']:
            highest_confidence_boxes[class_id] = {
                'box': box,
                'confidence': confidence
        }


    segmented_images = []
    for entry in highest_confidence_boxes.values():
        box = entry['box']
        x1, y1, x2, y2 = box.astype(int)
        segmented_image = Image.fromarray(cv2.cvtColor(cv2.imread(image)[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        segmented_images.append(segmented_image)
        

    return segmented_images