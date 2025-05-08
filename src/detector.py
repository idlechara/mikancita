"""Cat detection module using YOLO model."""

import os
from typing import Tuple, Optional, Any
import numpy as np
import cv2
from ultralytics import YOLO


class CatDetector:
    """Handles cat detection using YOLO model."""
    
    def __init__(self, model_path: str):
        """Initialize the cat detector with a YOLO model."""
        os.environ["YOLO_VERBOSE"] = "False"
        self.model = YOLO(model_path)
        self.mask = None
        self.fade_factor = 0.3  # Opacity for non-masked areas
        self.display_frame = None
        
    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        """Set a binary mask for filtering detection area."""
        self.mask = mask
        
    def set_fade_factor(self, fade_factor: float) -> None:
        """Set the fade factor for non-masked areas."""
        self.fade_factor = max(0.0, min(1.0, fade_factor))
        
    def load_mask(self, mask_path: str) -> None:
        """Load a mask from file."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            self.mask = mask
        else:
            print(f"Warning: Failed to load mask from {mask_path}")
        
    def detect(self, frame: np.ndarray) -> Any:
        """Run object detection on a frame."""
        self.display_frame = frame.copy()
        
        if self.mask is not None:
            # Ensure mask size matches frame
            if self.mask.shape[:2] != frame.shape[:2]:
                self.mask = cv2.resize(self.mask, (frame.shape[1], frame.shape[0]))
            
            # Create detection frame with masked areas
            detection_frame = np.zeros_like(frame)
            mask_3ch = np.stack([self.mask] * 3, axis=2)
            
            # Apply the mask - only copy pixels where mask is non-zero
            np.copyto(detection_frame, frame, where=mask_3ch > 0)
            
            # Create faded version for display
            faded_area = (frame.astype(np.float32) * self.fade_factor).astype(np.uint8)
            np.copyto(self.display_frame, faded_area, where=mask_3ch == 0)
            
            # Run detection on the masked frame
            return self.model(detection_frame, verbose=False)
        else:
            # No mask, use original frame
            return self.model(frame, verbose=False)
    
    def get_display_frame(self) -> np.ndarray:
        """Get the frame with faded non-masked areas for display."""
        if self.display_frame is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return self.display_frame
    
    def get_cat_box(self, results: Any) -> Optional[Tuple[int, int, int, int]]:
        """Extract cat bounding box from detection results."""
        for r in results:
            boxes = r.boxes
            for i, class_id in enumerate(boxes.cls):
                if self.model.names[int(class_id)].lower() == 'cat':
                    box = boxes.xyxy[i].cpu().numpy().astype(int)
                    return tuple(box)  # (x1, y1, x2, y2)
        return None
    
    def get_cat_box_with_confidence(self, results: Any) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Extract cat bounding box and confidence from detection results."""
        for r in results:
            boxes = r.boxes
            for i, class_id in enumerate(boxes.cls):
                if self.model.names[int(class_id)].lower() == 'cat':
                    box = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    return tuple(box), conf
        return None, 0.0