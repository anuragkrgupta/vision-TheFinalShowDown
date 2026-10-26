import pytest
from detection.cooldown import EventCooldownManager

def test_cooldown_manager_filters_repeats():
    manager = EventCooldownManager()
    manager.cooldown_seconds = 5.0
    
    # Same event key
    det1 = {"class_name": "person", "zone": "Center", "proximity": "Near"}
    
    # Time 0.0: should emit
    emitted = manager.filter([det1], current_time=0.0)
    assert len(emitted) == 1
    
    # Time 2.0: should drop (2.0 < 5.0)
    emitted = manager.filter([det1], current_time=2.0)
    assert len(emitted) == 0
    
    # Time 6.0: should emit (6.0 > 5.0)
    emitted = manager.filter([det1], current_time=6.0)
    assert len(emitted) == 1

def test_cooldown_manager_allows_zone_changes():
    manager = EventCooldownManager()
    manager.cooldown_seconds = 5.0
    
    det_center = {"class_name": "person", "zone": "Center", "proximity": "Near"}
    det_left = {"class_name": "person", "zone": "Left", "proximity": "Near"}
    
    # Time 0.0: Person in Center -> emits
    emitted = manager.filter([det_center], current_time=0.0)
    assert len(emitted) == 1
    
    # Time 2.0: Person moves to Left -> emits immediately because zone changed
    emitted = manager.filter([det_left], current_time=2.0)
    assert len(emitted) == 1
    
    # Time 3.0: Person still in Left -> dropped
    emitted = manager.filter([det_left], current_time=3.0)
    assert len(emitted) == 0
