"""Recording module for capturing cat videos and photos."""

import os
import time
import cv2
import numpy as np
import shutil
import yaml
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime
from config import Config, RecorderMode


class CatRecorder:
    """Records and saves videos or photos of detected cats."""
    
    def __init__(self, output_dir: str, video_format: str, codec: str, mode: RecorderMode = Config.DEFAULT_RECORDER_MODE):
        """Initialize the cat recorder."""
        self.output_dir = output_dir
        self.video_format = video_format
        self.codec = codec
        self.mode = mode
        self.state = {
            "is_recording": False,
            "frames": [],
            "frame_size": None,
            "session_path": None,
            "start_time": None,
            "end_time": None,
            "frame_count": 0,
            "confidences": [],
            "frame_sizes": [],
            "timestamp": None,
            "metadata": {}
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start(self) -> None:
        """Start a new recording session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state["timestamp"] = timestamp
        
        if self.mode == RecorderMode.VIDEO:
            session_path = f"{self.output_dir}/cat_{timestamp}.{self.video_format}"
            print(f"Started collecting frames for video: {session_path}")
        else:
            session_path = f"{self.output_dir}/cat_photos_{timestamp}"
            os.makedirs(session_path, exist_ok=True)
            print(f"Started collecting individual photos in: {session_path}")
        
        self.state["frames"] = [] if self.mode == RecorderMode.VIDEO else None
        self.state["frame_size"] = None
        self.state["session_path"] = session_path
        self.state["is_recording"] = True
        self.state["start_time"] = time.time()
        self.state["end_time"] = None
        self.state["frame_count"] = 0
        self.state["confidences"] = []
        self.state["frame_sizes"] = []
        self.state["metadata"] = {}
    
    def add_frame(self, frame: np.ndarray, cat_box: Tuple[int, int, int, int], confidence: float = 0.0) -> None:
        """Add a frame with cat to the current recording."""
        if not self.state["is_recording"]:
            return
            
        cropped_frame = self._crop_frame_to_cat(frame, cat_box)
        
        if self.state["frame_size"] is None:
            self.state["frame_size"] = (cropped_frame.shape[1], cropped_frame.shape[0])
        elif self.mode == RecorderMode.VIDEO:
            cropped_frame = cv2.resize(cropped_frame, self.state["frame_size"])
        
        if self.mode == RecorderMode.VIDEO:
            self.state["frames"].append(cropped_frame)
            self.state["confidences"].append(confidence)
            self.state["frame_sizes"].append(cropped_frame.nbytes)
        else:
            self._save_photo(cropped_frame, confidence)
            
    def stop(self) -> None:
        """Stop recording and finalize output."""
        if not self.state["is_recording"]:
            return
        
        if self.state["end_time"] is None:
            self.state["end_time"] = time.time()
        
        real_duration = self.state["end_time"] - self.state["start_time"]
        print(f"Cat was visible for {real_duration:.2f} seconds")
        
        if self.mode == RecorderMode.VIDEO:
            if self.state["frames"] and len(self.state["frames"]) > 0:
                frame_count = len(self.state["frames"])
                print(f"Creating video from {frame_count} frames...")
                
                fps = frame_count / real_duration if real_duration > 0 else Config.DEFAULT_FPS
                print(f"Setting video to {fps:.1f} FPS to match real-time duration")
                
                self._create_video(fps, real_duration)
                self._save_video_metadata(frame_count, fps, real_duration)
            else:
                print("No frames collected, video not created")
        else:
            photo_count = self.state["frame_count"]
            if photo_count > 0:
                print(f"Saved {photo_count} photos in: {self.state['session_path']}")
                self._save_photos_metadata(photo_count, real_duration)
            else:
                print("No photos captured")
                if os.path.exists(self.state["session_path"]):
                    try:
                        shutil.rmtree(self.state["session_path"])
                        print(f"Removed empty directory: {self.state['session_path']}")
                    except OSError:
                        pass
        
        self._reset_state()
    
    def set_mode(self, mode: RecorderMode) -> None:
        """Change the recording mode."""
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
        
        # Calculate margin (based on config percentage)
        w, h = x2 - x1, y2 - y1
        margin_w = int(w * Config.CAT_MARGIN_PERCENT)
        margin_h = int(h * Config.CAT_MARGIN_PERCENT)
        
        # Ensure margins don't go outside the frame
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(frame_w, x2 + margin_w)
        y2 = min(frame_h, y2 + margin_h)
        
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
        
        for frame in self.state["frames"]:
            out.write(frame)
        
        out.release()
        print(f"Finished saving video to {self.state['session_path']}")
        print(f"Video duration: {len(self.state['frames'])/fps:.2f} seconds (should match {real_duration:.2f} seconds)")
    
    def _save_photo(self, frame: np.ndarray, confidence: float = 0.0) -> None:
        """Save a single photo to the photos directory."""
        filename = f"cat_{self.state['frame_count']:04d}.{Config.PHOTO_FORMAT}"
        filepath = os.path.join(self.state["session_path"], filename)
        
        success = cv2.imwrite(
            filepath, 
            frame, 
            [cv2.IMWRITE_JPEG_QUALITY, Config.PHOTO_QUALITY] if Config.PHOTO_FORMAT.lower() == 'jpg' else None
        )
        
        if success:
            self.state["confidences"].append(confidence)
            self.state["frame_sizes"].append(frame.nbytes)
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
        self.state["confidences"] = []
        self.state["frame_sizes"] = []
    
    def _get_frame_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about recorded frames."""
        if not self.state["confidences"]:
            return {}
            
        middle_idx = len(self.state["confidences"]) // 2
        
        if self.state["frame_sizes"]:
            largest_idx = self.state["frame_sizes"].index(max(self.state["frame_sizes"]))
        else:
            largest_idx = 0
            
        highest_conf_idx = self.state["confidences"].index(max(self.state["confidences"]))
        
        return {
            "middle_frame_index": middle_idx,
            "middle_frame_confidence": self.state["confidences"][middle_idx],
            "largest_frame_index": largest_idx,
            "largest_frame_confidence": self.state["confidences"][largest_idx],
            "highest_confidence_frame_index": highest_conf_idx,
            "highest_confidence_value": self.state["confidences"][highest_conf_idx],
        }
        
    def _save_video_metadata(self, frame_count: int, fps: float, real_duration: float) -> None:
        """Save metadata for the recorded video to a YAML file."""
        frame_stats = self._get_frame_statistics()
        
        start_dt = datetime.fromtimestamp(self.state["start_time"])
        end_dt = datetime.fromtimestamp(self.state["end_time"])
        
        metadata = {
            "date_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "duration": real_duration,
            "frame_count": frame_count,
            "fps": fps,
            "timestamp": self.state["timestamp"],
        }
        
        metadata.update(frame_stats)
        metadata["frame_confidences"] = self.state["confidences"]
        
        metadata_file = f"{self.state['session_path']}.yaml"
        
        with open(metadata_file, "w") as yaml_file:
            yaml.dump(metadata, yaml_file, default_flow_style=False, sort_keys=False)
        
        print(f"Metadata saved to: {metadata_file}")
    
    def _save_photos_metadata(self, photo_count: int, real_duration: float) -> None:
        """Save metadata for the captured photos to a YAML file."""
        frame_stats = self._get_frame_statistics()
        
        start_dt = datetime.fromtimestamp(self.state["start_time"])
        end_dt = datetime.fromtimestamp(self.state["end_time"])
        
        metadata = {
            "date_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "duration": real_duration,
            "photo_count": photo_count,
            "timestamp": self.state["timestamp"],
        }
        
        metadata.update(frame_stats)
        metadata["frame_confidences"] = self.state["confidences"]
        
        metadata_file = f"{self.state['session_path']}/metadata.yaml"
        
        with open(metadata_file, "w") as yaml_file:
            yaml.dump(metadata, yaml_file, default_flow_style=False, sort_keys=False)
        
        print(f"Metadata saved to: {metadata_file}")