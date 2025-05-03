"""Recording module for capturing cat videos and photos."""

import os
import time
import cv2
import numpy as np
import shutil
from typing import Tuple, List, Optional, Union
from datetime import datetime
from config import Config, RecorderMode


class CatRecorder:
    """Records and saves videos or photos of detected cats."""
    
    def __init__(self, output_dir: str, video_format: str, codec: str, mode: RecorderMode = Config.DEFAULT_RECORDER_MODE):
        """Initialize the cat recorder.
        
        Args:
            output_dir: Directory to save recordings
            video_format: Format for video files (e.g., 'avi')
            codec: Video codec to use
            mode: Recording mode (video or photos)
        """
        self.output_dir = output_dir
        self.video_format = video_format
        self.codec = codec
        self.mode = mode
        self.state = {
            "is_recording": False,
            "frames": [],
            "frame_size": None,
            "session_path": None,  # Video file path or photo directory path
            "start_time": None,
            "end_time": None,
            "frame_count": 0  # Counter for photo filenames
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start(self) -> None:
        """Start a new recording session."""
        # Generate timestamp for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.mode == RecorderMode.VIDEO:
            # Create video file path
            session_path = f"{self.output_dir}/cat_{timestamp}.{self.video_format}"
            print(f"Started collecting frames for video: {session_path}")
        else:
            # Create directory for photos
            session_path = f"{self.output_dir}/cat_photos_{timestamp}"
            os.makedirs(session_path, exist_ok=True)
            print(f"Started collecting individual photos in: {session_path}")
        
        # Initialize recording state
        self.state["frames"] = [] if self.mode == RecorderMode.VIDEO else None
        self.state["frame_size"] = None
        self.state["session_path"] = session_path
        self.state["is_recording"] = True
        self.state["start_time"] = time.time()
        self.state["end_time"] = None
        self.state["frame_count"] = 0
    
    def add_frame(self, frame: np.ndarray, cat_box: Tuple[int, int, int, int]) -> None:
        """Add a frame with cat to the current recording."""
        if not self.state["is_recording"]:
            return
            
        # Extract the region with the cat + margin
        cropped_frame = self._crop_frame_to_cat(frame, cat_box)
        
        # Ensure consistent frame size for videos
        if self.state["frame_size"] is None:
            self.state["frame_size"] = (cropped_frame.shape[1], cropped_frame.shape[0])
        elif self.mode == RecorderMode.VIDEO:
            cropped_frame = cv2.resize(cropped_frame, self.state["frame_size"])
        
        # Handle the frame based on mode
        if self.mode == RecorderMode.VIDEO:
            # Store the frame in memory for later video creation
            self.state["frames"].append(cropped_frame)
        else:
            # Save the photo immediately
            self._save_photo(cropped_frame)
            
    def stop(self) -> None:
        """Stop recording and finalize output."""
        if not self.state["is_recording"]:
            return
        
        # Record end time if not already set
        if self.state["end_time"] is None:
            self.state["end_time"] = time.time()
        
        # Calculate visibility duration
        real_duration = self.state["end_time"] - self.state["start_time"]
        print(f"Cat was visible for {real_duration:.2f} seconds")
        
        if self.mode == RecorderMode.VIDEO:
            # Finalize video
            if self.state["frames"] and len(self.state["frames"]) > 0:
                frame_count = len(self.state["frames"])
                print(f"Creating video from {frame_count} frames...")
                
                # Calculate FPS to match real-time duration
                fps = frame_count / real_duration if real_duration > 0 else Config.DEFAULT_FPS
                print(f"Setting video to {fps:.1f} FPS to match real-time duration")
                
                # Create the video
                self._create_video(fps, real_duration)
            else:
                print("No frames collected, video not created")
        else:
            # Finalize photos collection
            photo_count = self.state["frame_count"]
            if photo_count > 0:
                print(f"Saved {photo_count} photos in: {self.state['session_path']}")
            else:
                print("No photos captured")
                # Remove empty directory
                if os.path.exists(self.state["session_path"]):
                    try:
                        shutil.rmtree(self.state["session_path"])
                        print(f"Removed empty directory: {self.state['session_path']}")
                    except OSError:
                        pass
        
        # Reset state
        self._reset_state()
    
    def set_mode(self, mode: RecorderMode) -> None:
        """Change the recording mode.
        
        Args:
            mode: New recording mode (VIDEO or PHOTOS)
        """
        if self.state["is_recording"]:
            print("Cannot change mode while recording is in progress")
            return
            
        self.mode = mode
        print(f"Recorder mode set to: {mode.value}")
    
    def is_recording(self) -> bool:
        """Check if recording is in progress."""
        return self.state["is_recording"]
    
    def set_end_time(self, end_time: float) -> None:
        """Set the end time of the recording."""
        self.state["end_time"] = end_time
    
    def _crop_frame_to_cat(self, frame: np.ndarray, cat_box: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop the frame to the cat's bounding box with margin."""
        x1, y1, x2, y2 = cat_box
        
        # Calculate margin (10% of width/height)
        w, h = x2 - x1, y2 - y1
        margin_w = int(w * Config.CAT_MARGIN_PERCENT)
        margin_h = int(h * Config.CAT_MARGIN_PERCENT)
        
        # Ensure margins don't go outside the frame
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(frame_w, x2 + margin_w)
        y2 = min(frame_h, y2 + margin_h)
        
        # Crop the frame
        return frame[y1:y2, x1:x2]
    
    def _create_video(self, fps: float, real_duration: float) -> None:
        """Create a video file from collected frames."""
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        out = cv2.VideoWriter(
            self.state["session_path"],
            fourcc,
            fps,
            self.state["frame_size"]
        )
        
        # Write all frames
        for frame in self.state["frames"]:
            out.write(frame)
        
        # Finalize video
        out.release()
        
        # Report results
        print(f"Finished saving video to {self.state['session_path']}")
        print(f"Video duration: {len(self.state['frames'])/fps:.2f} seconds (should match {real_duration:.2f} seconds)")
    
    def _save_photo(self, frame: np.ndarray) -> None:
        """Save a single photo to the photos directory."""
        # Create a sequential filename
        filename = f"cat_{self.state['frame_count']:04d}.{Config.PHOTO_FORMAT}"
        filepath = os.path.join(self.state["session_path"], filename)
        
        # Save the image
        success = cv2.imwrite(
            filepath, 
            frame, 
            [cv2.IMWRITE_JPEG_QUALITY, Config.PHOTO_QUALITY] if Config.PHOTO_FORMAT.lower() == 'jpg' else None
        )
        
        if success:
            self.state["frame_count"] += 1
    
    def _reset_state(self) -> None:
        """Reset the recording state."""
        self.state["frames"] = [] if self.mode == RecorderMode.VIDEO else None
        self.state["frame_size"] = None
        self.state["session_path"] = None
        self.state["is_recording"] = False
        self.state["start_time"] = None
        self.state["end_time"] = None
        self.state["frame_count"] = 0