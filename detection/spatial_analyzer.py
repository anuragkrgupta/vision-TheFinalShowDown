import json
from pathlib import Path

class SpatialAnalyzer:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "detection_config.json"
            
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        zone_cfg = self.config.get("zone_boundaries", {})
        self.left_to_center = zone_cfg.get("left_to_center", 0.33)
        self.center_to_right = zone_cfg.get("center_to_right", 0.66)
        
        prox_cfg = self.config.get("proximity_proxy_thresholds", {})
        self.near_area_ratio = prox_cfg.get("near_area_ratio", 0.30)
        self.mid_area_ratio = prox_cfg.get("mid_area_ratio", 0.10)
        
    def analyze(self, detections):
        """
        Takes a list of detections and appends 'zone' and 'proximity' keys.
        """
        analyzed = []
        for det in detections:
            # bbox_norm is [x1, y1, x2, y2] normalized between 0.0 and 1.0
            x1, y1, x2, y2 = det["bbox_norm"]
            
            # Calculate center X
            center_x = (x1 + x2) / 2.0
            
            # Determine Zone
            if center_x < self.left_to_center:
                zone = "Left"
            elif center_x > self.center_to_right:
                zone = "Right"
            else:
                zone = "Center"
                
            # Calculate Area proxy
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Determine Proximity
            if area >= self.near_area_ratio:
                proximity = "Near"
            elif area >= self.mid_area_ratio:
                proximity = "Mid"
            else:
                proximity = "Far"
                
            # Create a new dict with the added spatial data
            new_det = dict(det)
            new_det["zone"] = zone
            new_det["proximity"] = proximity
            new_det["area"] = area
            
            analyzed.append(new_det)
            
        return analyzed
