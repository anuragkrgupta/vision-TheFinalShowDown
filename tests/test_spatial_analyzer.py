import pytest
from detection.spatial_analyzer import SpatialAnalyzer

def test_spatial_analyzer_zones():
    # Use a dummy config to avoid loading the actual file in tests
    analyzer = SpatialAnalyzer()
    analyzer.left_to_center = 0.33
    analyzer.center_to_right = 0.66
    
    # Bbox norm is [x1, y1, x2, y2]
    # Center X = (x1 + x2) / 2
    
    detections = [
        {"bbox_norm": [0.0, 0.0, 0.2, 0.2]}, # Center 0.1 -> Left
        {"bbox_norm": [0.4, 0.0, 0.6, 0.2]}, # Center 0.5 -> Center
        {"bbox_norm": [0.8, 0.0, 1.0, 0.2]}  # Center 0.9 -> Right
    ]
    
    analyzed = analyzer.analyze(detections)
    
    assert analyzed[0]["zone"] == "Left"
    assert analyzed[1]["zone"] == "Center"
    assert analyzed[2]["zone"] == "Right"

def test_spatial_analyzer_proximity():
    analyzer = SpatialAnalyzer()
    analyzer.near_area_ratio = 0.30
    analyzer.mid_area_ratio = 0.10
    
    # Area = (x2 - x1) * (y2 - y1)
    
    detections = [
        {"bbox_norm": [0.0, 0.0, 0.7, 0.7]}, # Area 0.49 (> 0.30) -> Near
        {"bbox_norm": [0.0, 0.0, 0.4, 0.4]}, # Area 0.16 (> 0.10) -> Mid
        {"bbox_norm": [0.0, 0.0, 0.2, 0.2]}  # Area 0.04 (< 0.10) -> Far
    ]
    
    analyzed = analyzer.analyze(detections)
    
    assert analyzed[0]["proximity"] == "Near"
    assert analyzed[1]["proximity"] == "Mid"
    assert analyzed[2]["proximity"] == "Far"
