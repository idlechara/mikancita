"""Cat detection module using YOLO model."""

import os
from typing import Tuple, Optional, Any
import numpy as np
from ultralytics import YOLO


class CatDetector:
    """Handles cat detection using YOLO model."""
    
    def __init__(self, model_path: str):
        """Initialize the cat detector with a YOLO model."""
        # Disable verbose output from YOLO
        os.environ["YOLO_VERBOSE"] = "False"
        self.model = YOLO(model_path)
        
    def detect(self, frame: np.ndarray) -> Any:
        """Run object detection on a frame."""
        return self.model(frame, verbose=False)
    
    def get_cat_box(self, results: Any) -> Optional[Tuple[int, int, int, int]]:
        """Extract cat bounding box from detection results.
        
        Returns:
            Tuple (x1, y1, x2, y2) if cat found, None otherwise
        """
        for r in results:
            boxes = r.boxes
            for i, class_id in enumerate(boxes.cls):
                if self.model.names[int(class_id)].lower() == 'cat':
                    box = boxes.xyxy[i].cpu().numpy().astype(int)
                    return tuple(box)  # (x1, y1, x2, y2)
        return None