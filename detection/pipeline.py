import cv2
import time
from collections import deque
from pathlib import Path
from detection.detector import NavigationDetector
from detection.spatial_analyzer import SpatialAnalyzer
from detection.cooldown import EventCooldownManager


class TemporalSmoother:
    """
    Implements N-of-M smoothing.
    For a given class, it must be detected in M out of the last N frames to be considered 'active'.
    """
    def __init__(self, n_frames=3, m_required=2):
        self.n_frames = n_frames
        self.m_required = m_required
        self.history = deque(maxlen=n_frames)

    def update(self, detected_classes):
        """
        detected_classes: set of class names detected in the current frame
        Returns: set of classes that pass the N-of-M threshold
        """
        self.history.append(detected_classes)
        
        active_classes = set()
        # Count occurrences of each class in history
        counts = {}
        for frame_classes in self.history:
            for cls in frame_classes:
                counts[cls] = counts.get(cls, 0) + 1
                
        for cls, count in counts.items():
            if count >= self.m_required:
                active_classes.add(cls)
                
        return active_classes

class DetectionPipeline:
    def __init__(self, detector=None):
        self.detector = detector or NavigationDetector()
        
        smoothing_cfg = self.detector.config.get("temporal_smoothing", {})
        n = smoothing_cfg.get("n_frames", 3)
        m = smoothing_cfg.get("m_required", 2)
        self.smoother = TemporalSmoother(n_frames=n, m_required=m)
        
        self.spatial_analyzer = SpatialAnalyzer()
        self.cooldown_manager = EventCooldownManager()
        

    def process_image(self, image_path):
        """Processes a static image and returns raw detections."""
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image at {image_path}")
        return self.detector.detect(frame)

    def process_frame(self, frame):
        """
        Runs the full pipeline on a single frame:
        Inference -> Spatial -> Temporal Smoothing -> Cooldown
        Returns the final events to be announced, and the spatial detections for drawing.
        """
        raw_detections = self.detector.detect(frame)
        spatial_detections = self.spatial_analyzer.analyze(raw_detections)
        
        # Build set of unique event keys for the smoother
        # Key format: "class_name|zone|proximity"
        current_keys = set(f"{d['class_name']}|{d['zone']}|{d['proximity']}" for d in spatial_detections)
        
        # Get smoothed active keys
        active_keys = self.smoother.update(current_keys)
        
        # Filter spatial detections to only those that are temporally active
        active_detections = []
        seen_keys = set()
        for d in spatial_detections:
            key = f"{d['class_name']}|{d['zone']}|{d['proximity']}"
            # Only keep one instance of each key per frame to avoid duplicate announcements
            if key in active_keys and key not in seen_keys:
                active_detections.append(d)
                seen_keys.add(key)
                
        # Filter through cooldown
        emitted_events = self.cooldown_manager.filter(active_detections)
        
        return emitted_events, spatial_detections


