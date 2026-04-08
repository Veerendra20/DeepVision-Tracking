from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Any
import config

class YOLODetector:
    def __init__(self, model_path: str = config.YOLO_MODEL):
        """
        Initialize the YOLOv8 detector.
        :param model_path: Path to the YOLOv8 model file.
        """
        self.model = YOLO(model_path)
        self.classes = self.model.names

    def detect(self, frame: np.ndarray, conf_threshold: float = config.DEFAULT_CONFIDENCE) -> List[List[float]]:
        """
        Detect humans in a frame.
        :param frame: Input image/frame.
        :param conf_threshold: Confidence threshold for detection.
        :return: List of detections [[x1, y1, x2, y2, confidence, class_id], ...]
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # Filter for 'person' class (usually class 0 in COCO)
                if self.classes[cls] == 'person' and conf >= conf_threshold:
                    detections.append([x1, y1, x2, y2, conf, cls])
                    
        return detections
