"""Module for creating and managing detection area masks."""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional


class MaskManager:
    """Handles creation and management of detection masks."""
    
    def __init__(self):
        """Initialize the mask manager."""
        self.mask = None
        self.drawing = False
        self.points = []
        self.temp_point = None
        self.reference_frame = None
    
    def create_interactive_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Launch an interactive tool to create a mask."""
        frame_size = (frame.shape[1], frame.shape[0])
        self.reference_frame = frame.copy()
        self.mask = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint8)
        
        window_name = "Create Detection Mask - Press 'c' to clear, 's' to save, 'q' to quit"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_mouse)
        
        while True:
            display_frame = self.reference_frame.copy()
            
            # Draw the current polygon points
            if len(self.points) > 0:
                # Draw all points
                for point in self.points:
                    cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
                
                # Draw lines between points
                for i in range(len(self.points) - 1):
                    cv2.line(display_frame, self.points[i], self.points[i+1], (0, 255, 0), 2)
                
                # Draw line from last point to first point if we have at least 3 points
                if len(self.points) >= 3:
                    cv2.line(display_frame, self.points[-1], self.points[0], (0, 255, 0), 2)
                
                # Draw line from last point to current mouse position
                if self.temp_point is not None:
                    cv2.line(display_frame, self.points[-1], self.temp_point, (0, 255, 0), 2)
            
            # Show mask overlay
            if self.mask is not None and np.any(self.mask):
                mask_overlay = np.zeros_like(display_frame)
                mask_overlay[self.mask > 0] = [0, 200, 0]  # Green color where mask is active
                alpha = 0.3  # transparency factor
                display_frame = cv2.addWeighted(display_frame, 1, mask_overlay, alpha, 0)
            
            # Show instructions
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_frame, "Click to add points, double-click to complete polygon", 
                       (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'c' to clear, 's' to save, 'q' to quit", 
                       (10, 60), font, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.mask = None
                break
            elif key == ord('c'):
                self.points = []
                self.mask = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint8)
            elif key == ord('s'):
                if len(self.points) >= 3:
                    self._complete_polygon()
                    break
                else:
                    print("Need at least 3 points to create a mask")
        
        cv2.destroyWindow(window_name)
        return self.mask
    
    def _on_mouse(self, event, x, y, flags, param):
        """Mouse callback function for interactive mask creation."""
        if event == cv2.EVENT_MOUSEMOVE:
            if len(self.points) > 0:
                self.temp_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            if len(self.points) >= 3:
                self._complete_polygon()
    
    def _complete_polygon(self):
        """Complete the polygon and create the mask."""
        if len(self.points) >= 3:
            points_array = np.array([self.points], dtype=np.int32)
            cv2.fillPoly(self.mask, points_array, 255)
    
    def save_mask(self, mask: np.ndarray, filepath: str) -> bool:
        """Save a mask to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        return cv2.imwrite(filepath, mask)
    
    def load_mask(self, filepath: str) -> Optional[np.ndarray]:
        """Load a mask from disk."""
        if not os.path.exists(filepath):
            print(f"Mask file not found: {filepath}")
            return None
        
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask from {filepath}")
            return None
        
        # Ensure it's binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask