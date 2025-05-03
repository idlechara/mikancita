#!/usr/bin/env python3
"""Main entry point for the cat detection application."""

import argparse
import os
import sys

# Add the parent directory to the path so we can import modules correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitor import CatMonitor
from src.config import RecorderMode, Config
from src.init import export_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cat Detection and Recording")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["video", "photos"], 
        default="photos",
        help="Recording mode: 'video' to save videos, 'photos' to save individual images"
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Enable detection mask for specifying regions of interest"
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        help="Path to a previously saved mask file (png format)"
    )
    return parser.parse_args()


def check_and_initialize_model():
    """Check if the model exists and initialize it if needed."""
    model_path = Config.YOLO_MODEL_PATH
    ncnn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolo11n_ncnn_model")
    
    # Check if the model exists
    if not os.path.exists(model_path) or not os.path.exists(ncnn_dir):
        print(f"Model not found. Initializing model...")
        export_model()
        print(f"Model initialization complete.")


def main():
    """Run the cat detection application."""
    # Load user configuration
    Config.load_user_config()
    
    # Check and initialize the model if needed
    check_and_initialize_model()
    
    args = parse_args()
    
    # Command line args override saved config
    mode = RecorderMode.VIDEO if args.mode == "video" else RecorderMode.PHOTOS
    
    # Set mask configuration
    if args.mask:
        Config.USE_DETECTION_MASK = True
    if args.mask_path:
        Config.MASK_PATH = os.path.abspath(args.mask_path)
        Config.USE_DETECTION_MASK = True
    
    # Initialize and run the monitor
    monitor = CatMonitor(recorder_mode=mode)
    monitor.run()


if __name__ == "__main__":
    main()