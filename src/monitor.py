"""Main cat monitoring application module."""

import cv2
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np

from config import Config, RecorderMode
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
        self.cap = self._setup_webcam()
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
        mask = None
        
        # Create mask directory if needed
        mask_dir = os.path.join(os.path.dirname(Config.OUTPUT_DIR), "masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        # Default mask path if not specified
        if Config.MASK_PATH is None:
            Config.MASK_PATH = os.path.join(mask_dir, "detection_mask.png")
        
        # Try to load existing mask
        if os.path.exists(Config.MASK_PATH):
            mask = self.mask_manager.load_mask(Config.MASK_PATH)
            if mask is not None:
                print(f"Loaded detection mask from {Config.MASK_PATH}")
                self.detector.set_mask(mask)
                return
        
        # Create mask interactively if loading failed
        print("Creating new detection mask...")
        # Get the frame size
        _, frame = self.cap.read()
        if frame is not None:
            # Create mask using the current frame as reference
            mask = self.mask_manager.create_interactive_mask(frame)
            
            if mask is not None:
                # Save the mask
                if self.mask_manager.save_mask(mask, Config.MASK_PATH):
                    print(f"Saved detection mask to {Config.MASK_PATH}")
                
                # Set the mask in the detector
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
        while True:
            # Get frame
            success, frame = self.cap.read()
            if not success:
                break

            # Detect objects
            results = self.detector.detect(frame)
            
            # Check for cat with confidence score
            cat_box, confidence = self.detector.get_cat_box_with_confidence(results)
            
            # Update cat tracking state
            events = self.tracker.update(cat_box is not None)
            
            # Handle tracking events
            self._handle_events(events, frame, cat_box, confidence)
            
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
            elif key == ord("k"):
                # Toggle mask mode
                self._toggle_mask_mode(frame)
    
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

        # Check current mask state
        if self.detector.mask is None:
            # No mask currently active - display options
            options_window = "Mask Options"
            options_frame = np.zeros((300, 600, 3), dtype=np.uint8)
            
            # Add options text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(options_frame, "Mask Options:", (20, 40), font, 0.8, (255, 255, 255), 2)
            cv2.putText(options_frame, "c - Create new mask", (40, 100), font, 0.7, (255, 255, 255), 2)
            
            # Only show load option if a mask file exists
            if os.path.exists(Config.MASK_PATH):
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
                elif key == ord('l') and os.path.exists(Config.MASK_PATH):
                    # Load saved mask
                    mask = self.mask_manager.load_mask(Config.MASK_PATH)
                    if mask is not None:
                        self.detector.set_mask(mask)
                        print(f"Loaded detection mask from {Config.MASK_PATH}")
                        # Save this setting to Config
                        Config.USE_DETECTION_MASK = True
                        Config.save_user_config()
                    cv2.destroyWindow(options_window)
                    break
                elif key == ord('r') and os.path.exists(Config.MASK_PATH):
                    # Remove saved mask
                    try:
                        os.remove(Config.MASK_PATH)
                        print(f"Removed mask file: {Config.MASK_PATH}")
                        # Clear config
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
        else:
            # Mask is active, disable it
            self.detector.set_mask(None)
            print("Detection mask disabled")
            # Save this setting to Config
            Config.USE_DETECTION_MASK = False
    
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
        # Use the display frame from detector if mask is active, otherwise use results plot
        if self.detector.mask is not None:
            # Get the frame with properly faded non-masked areas
            display_frame = self.detector.get_display_frame()
            
            # Plot detection results on the display frame (not the original frame)
            annotated_frame = results[0].plot(img=display_frame)
        else:
            # No mask, just use the normal results plotting
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
        
        # Add mask status if applicable
        if self.detector.mask is not None:
            mask_text = "Mask: ON"
            
            # We no longer need to overlay the mask here since the display frame already has it
        else:
            mask_text = "Mask: OFF"
        
        cv2.putText(
            annotated_frame,
            mask_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Add key help
        cv2.putText(
            annotated_frame,
            "q: quit, m: change mode, k: edit mask",
            (10, 90),
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
        
        # Set FPS to help maintain consistent frame rate
        cap.set(cv2.CAP_PROP_FPS, Config.DEFAULT_FPS)
        
        return cap
    
    def _handle_events(self, events: Dict[str, Any], frame: np.ndarray, cat_box: Optional[Tuple[int, int, int, int]], confidence: float = 0.0) -> None:
        """Handle cat tracking events.
        
        Args:
            events: Dictionary of tracking events (appeared, disappeared, etc.)
            frame: Current video frame
            cat_box: Bounding box of detected cat (x1, y1, x2, y2) or None
            confidence: Detection confidence score (0.0-1.0)
        """
        if events["appeared"]:
            # Cat just appeared
            print("Cat detected!")
            self.recorder.start()
        
        if cat_box is not None and self.tracker.is_detected():
            # Record frame with cat - use the original frame without mask overlay
            # Pass the confidence score to the recorder
            self.recorder.add_frame(frame, cat_box, confidence)
        
        if events["disappeared"]:
            # Cat has been gone for the threshold time
            print(f"Cat was on camera for {events['duration']:.2f} seconds")
            
            # Set recorder end time to when cat first disappeared
            away_since = self.tracker.get_away_since()
            if away_since is not None:
                self.recorder.set_end_time(away_since)
            
            # Stop recording
            self.recorder.stop()