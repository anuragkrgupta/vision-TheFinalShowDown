import json
import time
from pathlib import Path

class EventCooldownManager:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "detection_config.json"
            
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.cooldown_seconds = self.config.get("cooldown_seconds", 5.0)
        
        # Dictionary mapping (class_name, zone, proximity) -> last_emitted_timestamp
        self.last_seen = {}
        
    def filter(self, detections, current_time=None):
        """
        Filters a list of spatially analyzed detections, returning only those
        that have surpassed their cooldown window.
        """
        if current_time is None:
            current_time = time.time()
            
        emitted_events = []
        
        for det in detections:
            class_name = det["class_name"]
            zone = det["zone"]
            proximity = det["proximity"]
            
            event_key = (class_name, zone, proximity)
            
            if event_key not in self.last_seen or (current_time - self.last_seen[event_key]) >= self.cooldown_seconds:
                # Update the timestamp
                self.last_seen[event_key] = current_time
                emitted_events.append(det)
                
        return emitted_events
