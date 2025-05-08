"""Configuration settings for the cat detection application."""
import os
import json
from enum import Enum


class RecorderMode(Enum):
    """Recording modes for cat capture."""
    VIDEO = "video"
    PHOTOS = "photos"


class VideoSourceType(Enum):
    """Video source types for capturing."""
    WEBCAM = "webcam"
    RTMP = "rtmp"


class Config:
    """Configuration settings for the application."""
    # Camera settings
    WEBCAM_WIDTH = 1280
    WEBCAM_HEIGHT = 720
    
    # Detection settings
    CAT_MARGIN_PERCENT = 0.1
    CAT_ABSENCE_THRESHOLD = 0.5
    
    # Mask settings
    USE_DETECTION_MASK = False
    MASK_PATH = None
    MASK_OPACITY = 0.5
    
    # Output settings
    OUTPUT_DIR = "cat_captures"
    VIDEO_FORMAT = "avi"
    VIDEO_CODEC = "XVID"
    DEFAULT_FPS = 15.0
    PHOTO_FORMAT = "jpg"
    PHOTO_QUALITY = 95
    
    # Video source settings
    VIDEO_SOURCE_TYPE = VideoSourceType.WEBCAM
    VIDEO_SOURCE = 0  # Webcam index or RTMP URL
    RTMP_RECONNECT_ATTEMPTS = 3  # Number of times to attempt reconnection
    RTMP_RECONNECT_DELAY = 5     # Seconds to wait between reconnection attempts
    
    # Model settings
    YOLO_MODEL_PATH = "yolo11n.pt"
    
    # Recorder settings
    DEFAULT_RECORDER_MODE = RecorderMode.PHOTOS
    
    # Config file path
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_config.json")
    
    @classmethod
    def save_user_config(cls):
        """Save user configuration settings to a file."""
        config_data = {
            "USE_DETECTION_MASK": cls.USE_DETECTION_MASK,
            "MASK_PATH": cls.MASK_PATH,
            "DEFAULT_RECORDER_MODE": cls.DEFAULT_RECORDER_MODE.value,
            "VIDEO_SOURCE_TYPE": cls.VIDEO_SOURCE_TYPE.value,
            "VIDEO_SOURCE": cls.VIDEO_SOURCE
        }
        
        try:
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=4)
                print(f"Configuration saved to {cls.CONFIG_FILE}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    @classmethod
    def load_user_config(cls):
        """Load user configuration settings from file."""
        if not os.path.exists(cls.CONFIG_FILE):
            print("No saved configuration found, using defaults")
            return
        
        try:
            with open(cls.CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
                
                if "USE_DETECTION_MASK" in config_data:
                    cls.USE_DETECTION_MASK = config_data["USE_DETECTION_MASK"]
                
                if "MASK_PATH" in config_data and config_data["MASK_PATH"]:
                    cls.MASK_PATH = config_data["MASK_PATH"]
                
                if "DEFAULT_RECORDER_MODE" in config_data:
                    mode_str = config_data["DEFAULT_RECORDER_MODE"]
                    cls.DEFAULT_RECORDER_MODE = (
                        RecorderMode.VIDEO if mode_str == "video" else RecorderMode.PHOTOS
                    )
                
                if "VIDEO_SOURCE_TYPE" in config_data:
                    source_type = config_data["VIDEO_SOURCE_TYPE"]
                    cls.VIDEO_SOURCE_TYPE = (
                        VideoSourceType.RTMP if source_type == "rtmp" else VideoSourceType.WEBCAM
                    )
                
                if "VIDEO_SOURCE" in config_data:
                    cls.VIDEO_SOURCE = config_data["VIDEO_SOURCE"]
                
                print(f"Configuration loaded from {cls.CONFIG_FILE}")
        except Exception as e:
            print(f"Error loading configuration: {e}")