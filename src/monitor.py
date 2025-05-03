"""Main cat monitoring application module."""

import cv2
from typing import Dict, Any, Optional, Tuple
import numpy as np

from config import Config, RecorderMode
from detector import CatDetector
from recorder import CatRecorder
from tracker import CatTracker


class CatMonitor:
    """Main application for monitoring cats."""
    
    def __init__(self, recorder_mode: RecorderMode = Config.DEFAULT_RECORDER_MODE):
        """Initialize the cat monitor application.
        
        Args:
            recorder_mode: Mode for recording (VIDEO or PHOTOS)
        """
        self.cap = self._setup_webcam()
        self.detector = CatDetector(Config.YOLO_MODEL_PATH)
        self.recorder = CatRecorder(
            Config.OUTPUT_DIR, 
            Config.VIDEO_FORMAT, 
            Config.VIDEO_CODEC,
            mode=recorder_mode
        )
        self.tracker = CatTracker(Config.CAT_ABSENCE_THRESHOLD)
        
        # Print current mode
        print(f"Cat recorder mode: {recorder_mode.value}")
    
    def run(self) -> None:
        """Run the main detection and recording loop."""
        try:
            self._main_loop()
        finally:
            # Clean up resources
            if self.tracker.is_detected():
                self._report_cat_duration()
            
            if self.recorder.is_recording():
                self.recorder.stop()
            
            self.cap.release()
            cv2.destroyAllWindows()
    
    def _main_loop(self) -> None:
        """Main processing loop."""
        while True:
            # Get frame
            success, frame = self.cap.read()
            if not success:
                break

            # Detect objects
            results = self.detector.detect(frame)
            
            # Check for cat
            cat_box = self.detector.get_cat_box(results)
            
            # Update cat tracking state
            events = self.tracker.update(cat_box is not None)
            
            # Handle tracking events
            self._handle_events(events, frame, cat_box)
            
            # Display frame
            self._show_frame(results, frame)
            
            # Check for key presses
            key = cv2.waitKey(1)
            if key == ord("q"):
                # Exit if 'q' is pressed
                break
            elif key == ord("m"):
                # Toggle mode if 'm' is pressed (only between recordings)
                self._toggle_recorder_mode()
    
    def _toggle_recorder_mode(self) -> None:
        """Toggle between video and photo modes."""
        if self.recorder.is_recording():
            print("Cannot change mode while recording is in progress")
            return
            
        # Switch mode
        new_mode = (
            RecorderMode.PHOTOS 
            if self.recorder.mode == RecorderMode.VIDEO 
            else RecorderMode.VIDEO
        )
        self.recorder.set_mode(new_mode)
    
    def _handle_events(self, events: Dict[str, Any], frame: np.ndarray, cat_box: Optional[Tuple[int, int, int, int]]) -> None:
        """Handle cat tracking events."""
        if events["appeared"]:
            # Cat just appeared
            print("Cat detected!")
            self.recorder.start()
        
        if cat_box is not None and self.tracker.is_detected():
            # Record frame with cat
            self.recorder.add_frame(frame, cat_box)
        
        if events["disappeared"]:
            # Cat has been gone for the threshold time
            print(f"Cat was on camera for {events['duration']:.2f} seconds")
            
            # Set recorder end time to when cat first disappeared
            away_since = self.tracker.get_away_since()
            if away_since is not None:
                self.recorder.set_end_time(away_since)
            
            # Stop recording
            self.recorder.stop()
    
    def _report_cat_duration(self) -> None:
        """Report how long the cat was visible when exiting."""
        duration = self.tracker.get_detection_time()
        if duration is not None:
            print(f"Cat was on camera for {duration:.2f} seconds")
    
    def _show_frame(self, results: Any, frame: np.ndarray) -> None:
        """Display the frame with detection results."""
        annotated_frame = results[0].plot()
        
        # Add mode indicator to the frame
        mode_text = f"Mode: {'PHOTOS' if self.recorder.mode == RecorderMode.PHOTOS else 'VIDEO'}"
        cv2.putText(
            annotated_frame,
            mode_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Add key help
        cv2.putText(
            annotated_frame,
            "q: quit, m: change mode",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.imshow("Cat Detection", annotated_frame)
    
    def _setup_webcam(self) -> cv2.VideoCapture:
        """Initialize and configure the webcam."""
        cap = cv2.VideoCapture(Config.VIDEO_SOURCE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.WEBCAM_HEIGHT)
        
        # Set pixel format to improve performance
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Don't convert to RGB internally
        
        # Set FPS to help maintain consistent frame rate
        cap.set(cv2.CAP_PROP_FPS, Config.DEFAULT_FPS)
        
        return cap