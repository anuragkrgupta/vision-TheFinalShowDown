import pytest
import time
from audio.voice_engine import VoiceEvent
import queue

def test_priority_queue_ordering():
    q = queue.PriorityQueue()
    
    # Left zone event (Priority 1)
    e1 = VoiceEvent(priority=1, timestamp=1.0, text="Left, person", class_name="person", zone="Left")
    # Right zone event (Priority 1, happened later)
    e2 = VoiceEvent(priority=1, timestamp=2.0, text="Right, car", class_name="car", zone="Right")
    # Center zone event (Priority 0, happened latest)
    e3 = VoiceEvent(priority=0, timestamp=3.0, text="Center, bicycle", class_name="bicycle", zone="Center")
    
    # Put in out-of-order
    q.put(e2)
    q.put(e1)
    q.put(e3)
    
    # The PriorityQueue should yield lowest priority number first (0)
    # If priorities are equal, it sorts by the next attribute, which is timestamp (lower timestamp first)
    
    first = q.get()
    assert first.zone == "Center"
    assert first.priority == 0
    
    second = q.get()
    assert second.zone == "Left"
    assert second.priority == 1
    assert second.timestamp == 1.0
    
    third = q.get()
    assert third.zone == "Right"
    assert third.priority == 1
    assert third.timestamp == 2.0
