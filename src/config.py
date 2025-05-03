"""Configuration settings for the cat detection application."""
from enum import Enum


class RecorderMode(Enum):
    """Recording modes for cat capture."""
    VIDEO = "video"  # Record as video file
    PHOTOS = "photos"  # Save individual photos


class Config:
    """Configuration settings for the application."""
    # Camera settings
    WEBCAM_WIDTH = 1280
    WEBCAM_HEIGHT = 720
    
    # Detection settings
    CAT_MARGIN_PERCENT = 0.1  # 10% margin around cat bounding box
    CAT_ABSENCE_THRESHOLD = 0.5  # Seconds before considering cat gone
    
    # Output settings
    OUTPUT_DIR = "cat_captures"
    VIDEO_FORMAT = "avi"
    VIDEO_CODEC = "XVID"
    DEFAULT_FPS = 15.0
    PHOTO_FORMAT = "jpg"
    PHOTO_QUALITY = 95  # JPEG quality (0-100)
    
    # Video source
    VIDEO_SOURCE = 2  # Default webcam (0 for built-in, 1 for external)
    
    # Model settings
    YOLO_MODEL_PATH = "yolo11n.pt"
    
    # Recorder settings
    DEFAULT_RECORDER_MODE = RecorderMode.PHOTOS  # Change to RecorderMode.PHOTOS to default to photos