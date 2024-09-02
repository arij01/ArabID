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
        self.segmenter = YOLO(model_path)

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
                    'confidence': confidence
                }

        segmented_images = []
        for entry in highest_confidence_boxes.values():
            box = entry['box']
            x1, y1, x2, y2 = box.astype(int)
            segmented_image = Image.fromarray(cv2.cvtColor(cv2.imread(image)[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            segmented_images.append(segmented_image)

        return segmented_images

class PreprocessImage:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = self._load_image()

    def _load_image(self) -> np.ndarray:
        """Load the image from the file path."""
        return cv2.imread(self.image_path)

    def to_grayscale(self) -> np.ndarray:
        """Convert the image to grayscale."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian Blur to the image."""
        return cv2.GaussianBlur(image, (5, 5), 0)

    def apply_thresholding(self, image: np.ndarray) -> np.ndarray:
        """Apply thresholding to convert the image to binary."""
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def preprocess(self) -> np.ndarray:
        """Apply the full preprocessing pipeline."""
        gray_image = self.to_grayscale()
        blurred_image = self.apply_gaussian_blur(gray_image)
        preprocessed_image = self.apply_thresholding(blurred_image)
        return preprocessed_image


