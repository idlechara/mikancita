"""Main cat monitoring application module."""

import cv2
import os
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np

from config import Config, RecorderMode, VideoSourceType
from detector import CatDetector
from recorder import CatRecorder
from tracker import CatTracker
from mask import MaskManager


class CatMonitor:
    """Main application for monitoring cats."""
    
    def __init__(self, recorder_mode: RecorderMode = Config.DEFAULT_RECORDER_MODE):
        """Initialize the cat monitor application.
        
        Args:
            recorder_mode: Mode for recording (VIDEO or PHOTOS)
        """
        self.cap = self._setup_video_source()
        self.detector = CatDetector(Config.YOLO_MODEL_PATH)
        self.recorder = CatRecorder(
            Config.OUTPUT_DIR, 
            Config.VIDEO_FORMAT, 
            Config.VIDEO_CODEC,
            mode=recorder_mode
        )
        self.tracker = CatTracker(Config.CAT_ABSENCE_THRESHOLD)
        self.mask_manager = MaskManager()
        
        # Load or create mask if needed
        if Config.USE_DETECTION_MASK:
            self._setup_mask()
        
        # Print current mode
        print(f"Cat recorder mode: {recorder_mode.value}")
        if Config.USE_DETECTION_MASK:
            print("Detection mask is enabled")
    
    def _setup_mask(self) -> None:
        """Load or create a detection mask."""
        # Create mask directory if needed
        mask_dir = os.path.join(os.path.dirname(Config.OUTPUT_DIR), "masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        # Set default mask path if not specified
        if Config.MASK_PATH is None:
            Config.MASK_PATH = os.path.join(mask_dir, "detection_mask.png")
        
        # Try to load existing mask
        if os.path.exists(Config.MASK_PATH):
            mask = self.mask_manager.load_mask(Config.MASK_PATH)
            if mask is not None:
                print(f"Loaded detection mask from {Config.MASK_PATH}")
                self.detector.set_mask(mask)
                return
        
        # Create new mask if loading failed
        print("Creating new detection mask...")
        _, frame = self.cap.read()
        if frame is not None:
            mask = self.mask_manager.create_interactive_mask(frame)
            if mask is not None:
                if self.mask_manager.save_mask(mask, Config.MASK_PATH):
                    print(f"Saved detection mask to {Config.MASK_PATH}")
                self.detector.set_mask(mask)
            else:
                print("Mask creation canceled, proceeding without a mask")
    
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
        reconnect_attempts = 0
        
        while True:
            # Get and process frame
            success, frame = self.cap.read()
            
            # Handle potential stream disconnection
            if not success:
                # For RTMP, try to reconnect
                if Config.VIDEO_SOURCE_TYPE == VideoSourceType.RTMP and reconnect_attempts < Config.RTMP_RECONNECT_ATTEMPTS:
                    reconnect_attempts += 1
                    print(f"Stream disconnected, attempting to reconnect ({reconnect_attempts}/{Config.RTMP_RECONNECT_ATTEMPTS})...")
                    
                    # Release current capture and try to reconnect
                    self.cap.release()
                    time.sleep(Config.RTMP_RECONNECT_DELAY)
                    self.cap = cv2.VideoCapture(Config.VIDEO_SOURCE)
                    
                    if self.cap.isOpened():
                        print("Successfully reconnected to stream")
                        reconnect_attempts = 0
                        continue
                else:
                    # Either webcam disconnected or too many reconnection attempts
                    print("Video source disconnected. Exiting...")
                    break
            else:
                # Reset reconnection attempts on successful frame read
                reconnect_attempts = 0

            # Detect cats and handle events
            results = self.detector.detect(frame)
            cat_box, confidence = self.detector.get_cat_box_with_confidence(results)
            events = self.tracker.update(cat_box is not None)
            self._handle_events(events, frame, cat_box, confidence)
            
            # Display frame and handle user input
            self._show_frame(results, frame)
            if self._handle_key_press(frame):
                break
    
    def _handle_key_press(self, frame: np.ndarray) -> bool:
        """Handle key press events for controlling the application.
        
        Returns:
            True if application should exit, False otherwise
        """
        key = cv2.waitKey(1)
        if key == ord("q"):
            # Exit if 'q' is pressed
            return True
        elif key == ord("m"):
            # Toggle mode if 'm' is pressed (only between recordings)
            self._toggle_recorder_mode()
        elif key == ord("k"):
            # Toggle mask mode
            self._toggle_mask_mode(frame)
        
        return False
    
    def _toggle_recorder_mode(self) -> None:
        """Toggle between video and photo recording modes."""
        if self.recorder.is_recording():
            print("Cannot change mode while recording is in progress")
            return
        
        # Switch to opposite mode
        new_mode = (
            RecorderMode.PHOTOS if self.recorder.mode == RecorderMode.VIDEO else RecorderMode.VIDEO
        )
        
        # Update recorder and config
        self.recorder.set_mode(new_mode)
        Config.DEFAULT_RECORDER_MODE = new_mode
        Config.save_user_config()
    
    def _toggle_mask_mode(self, frame: np.ndarray) -> None:
        """Toggle mask creation mode."""
        # Don't allow toggling during recording
        if self.recorder.is_recording():
            print("Cannot change mask while recording is in progress")
            return
        
        # Create mask directory if needed
        mask_dir = os.path.join(os.path.dirname(Config.OUTPUT_DIR), "masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        # Set default mask path if not specified
        if Config.MASK_PATH is None:
            Config.MASK_PATH = os.path.join(mask_dir, "detection_mask.png")

        # Toggle mask state
        if self.detector.mask is None:
            self._show_mask_options(Config.MASK_PATH, frame)
        else:
            # Mask is active, disable it
            self.detector.set_mask(None)
            print("Detection mask disabled")
            Config.USE_DETECTION_MASK = False
            Config.save_user_config()
    
    def _show_mask_options(self, mask_path: str, frame: np.ndarray) -> None:
        """Show mask options dialog and handle selection."""
        options_window = "Mask Options"
        options_frame = np.zeros((300, 600, 3), dtype=np.uint8)
        
        # Add options text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(options_frame, "Mask Options:", (20, 40), font, 0.8, (255, 255, 255), 2)
        cv2.putText(options_frame, "c - Create new mask", (40, 100), font, 0.7, (255, 255, 255), 2)
        
        # Only show load option if a mask file exists
        if os.path.exists(mask_path):
            cv2.putText(options_frame, "l - Load saved mask", (40, 150), font, 0.7, (255, 255, 255), 2)
            cv2.putText(options_frame, "r - Remove saved mask", (40, 200), font, 0.7, (255, 255, 255), 2)
        
        cv2.putText(options_frame, "q - Cancel", (40, 250), font, 0.7, (255, 255, 255), 2)
        
        # Show options window
        cv2.imshow(options_window, options_frame)
        
        # Handle option selection
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('c'):
                # Create a new mask
                cv2.destroyWindow(options_window)
                self._create_new_mask(frame)
                break
            elif key == ord('l') and os.path.exists(mask_path):
                # Load saved mask
                mask = self.mask_manager.load_mask(mask_path)
                if mask is not None:
                    self.detector.set_mask(mask)
                    print(f"Loaded detection mask from {mask_path}")
                    Config.USE_DETECTION_MASK = True
                    Config.save_user_config()
                cv2.destroyWindow(options_window)
                break
            elif key == ord('r') and os.path.exists(mask_path):
                # Remove saved mask
                try:
                    os.remove(mask_path)
                    print(f"Removed mask file: {mask_path}")
                    Config.USE_DETECTION_MASK = False
                    Config.save_user_config()
                except Exception as e:
                    print(f"Error removing mask file: {e}")
                cv2.destroyWindow(options_window)
                break
            elif key == ord('q'):
                # Cancel
                cv2.destroyWindow(options_window)
                break
    
    def _create_new_mask(self, frame: np.ndarray) -> None:
        """Create a new detection mask using the current frame as reference."""
        # Create a new mask
        mask = self.mask_manager.create_interactive_mask(frame)
        
        if mask is not None:
            # Save the mask
            if self.mask_manager.save_mask(mask, Config.MASK_PATH):
                print(f"Saved detection mask to {Config.MASK_PATH}")
            
            # Set the mask in the detector
            self.detector.set_mask(mask)
            print("Detection mask enabled")
            
            # Save this setting to Config
            Config.USE_DETECTION_MASK = True
            Config.save_user_config()
        else:
            print("Mask creation canceled")
    
    def _report_cat_duration(self) -> None:
        """Report how long the cat was visible when exiting."""
        duration = self.tracker.get_detection_time()
        if duration is not None:
            print(f"Cat was on camera for {duration:.2f} seconds")
    
    def _show_frame(self, results: Any, frame: np.ndarray) -> None:
        """Display the frame with detection results."""
        # Get the appropriate display frame based on mask status
        if self.detector.mask is not None:
            display_frame = self.detector.get_display_frame()
            annotated_frame = results[0].plot(img=display_frame)
        else:
            annotated_frame = results[0].plot()
        
        # Add overlay text with status information
        self._add_status_overlay(annotated_frame)
        
        cv2.imshow("Cat Detection", annotated_frame)
    
    def _add_status_overlay(self, frame: np.ndarray) -> None:
        """Add status text overlay to the frame."""
        # Add mode indicator
        mode_text = f"Mode: {'PHOTOS' if self.recorder.mode == RecorderMode.PHOTOS else 'VIDEO'}"
        cv2.putText(
            frame, mode_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        # Add mask status
        mask_text = "Mask: ON" if self.detector.mask is not None else "Mask: OFF"
        cv2.putText(
            frame, mask_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        # Add source type
        source_type = "RTMP" if Config.VIDEO_SOURCE_TYPE == VideoSourceType.RTMP else "Webcam"
        cv2.putText(
            frame, f"Source: {source_type}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        # Add key help
        cv2.putText(
            frame, "q: quit, m: change mode, k: edit mask", (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    
    def _setup_video_source(self) -> cv2.VideoCapture:
        """Initialize and configure the video source (webcam or RTMP)."""
        source = Config.VIDEO_SOURCE
        source_type = Config.VIDEO_SOURCE_TYPE
        
        # Print information about the video source
        if source_type == VideoSourceType.WEBCAM:
            print(f"Using webcam index: {source}")
        else:
            print(f"Connecting to RTMP stream: {source}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(source)
        
        # Configure properties based on source type
        if source_type == VideoSourceType.WEBCAM:
            # For webcams, set resolution and FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.WEBCAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.WEBCAM_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, Config.DEFAULT_FPS)
        else:
            # For RTMP, handle potential connection issues
            attempts = 0
            while not cap.isOpened() and attempts < Config.RTMP_RECONNECT_ATTEMPTS:
                print(f"Failed to connect to RTMP stream, retrying ({attempts+1}/{Config.RTMP_RECONNECT_ATTEMPTS})...")
                cap = cv2.VideoCapture(source)
                attempts += 1
                time.sleep(Config.RTMP_RECONNECT_DELAY)
            
            if not cap.isOpened():
                print("WARNING: Failed to connect to RTMP stream after multiple attempts.")
                print("Falling back to webcam...")
                Config.VIDEO_SOURCE_TYPE = VideoSourceType.WEBCAM
                Config.VIDEO_SOURCE = 0
                Config.save_user_config()
                return self._setup_video_source()
            
            print("Successfully connected to RTMP stream")
        
        return cap
    
    def _handle_events(self, events: Dict[str, Any], frame: np.ndarray, cat_box: Optional[Tuple[int, int, int, int]], confidence: float = 0.0) -> None:
        """Handle cat tracking events."""
        # Handle cat appearance
        if events["appeared"]:
            print("Cat detected!")
            self.recorder.start()
        
        # Add frame to recording if cat is visible
        if cat_box is not None and self.tracker.is_detected():
            self.recorder.add_frame(frame, cat_box, confidence)
        
        # Handle cat disappearance
        if events["disappeared"]:
            print(f"Cat was on camera for {events['duration']:.2f} seconds")
            
            # Set recorder end time to when cat first disappeared
            if away_since := self.tracker.get_away_since():
                self.recorder.set_end_time(away_since)
            
            # Stop recording
            self.recorder.stop()