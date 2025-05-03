"""Cat tracking module for monitoring cat presence and absence."""

import time
from typing import Dict, Any, Optional


class CatTracker:
    """Tracks cat presence and absence."""
    
    def __init__(self, absence_threshold: float):
        """Initialize cat tracker with timeout threshold."""
        self.absence_threshold = absence_threshold
        self.state = {
            "is_detected": False,
            "start_time": None,
            "away_since": None
        }
    
    def update(self, cat_detected: bool) -> Dict[str, Any]:
        """Update tracking state based on current detection."""
        events = {
            "appeared": False,
            "disappeared": False,
            "duration": None
        }
        
        if cat_detected:
            # Cat is currently visible
            self.state["away_since"] = None
            
            if not self.state["is_detected"]:
                # Cat just appeared
                self.state["is_detected"] = True
                self.state["start_time"] = time.time()
                events["appeared"] = True
        else:
            # Cat is not visible in this frame
            if self.state["is_detected"]:
                if self.state["away_since"] is None:
                    # Cat just disappeared, start tracking absence
                    self.state["away_since"] = time.time()
                elif time.time() - self.state["away_since"] >= self.absence_threshold:
                    # Cat has been away for threshold time
                    events["disappeared"] = True
                    events["duration"] = self.state["away_since"] - self.state["start_time"]
                    self._reset()
        
        return events
    
    def get_detection_time(self) -> Optional[float]:
        """Get how long the cat has been detected."""
        if self.state["is_detected"] and self.state["start_time"] is not None:
            return time.time() - self.state["start_time"]
        return None
    
    def is_detected(self) -> bool:
        """Check if cat is currently detected."""
        return self.state["is_detected"]
    
    def get_start_time(self) -> Optional[float]:
        """Get time when cat first appeared."""
        return self.state["start_time"]
    
    def get_away_since(self) -> Optional[float]:
        """Get time when cat started being away."""
        return self.state["away_since"]
    
    def _reset(self) -> None:
        """Reset tracking state."""
        self.state["is_detected"] = False
        self.state["start_time"] = None
        self.state["away_since"] = None