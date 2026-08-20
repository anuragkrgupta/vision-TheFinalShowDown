import pyttsx3
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass(order=True)
class VoiceEvent:
    priority: int
    timestamp: float
    text: str = field(compare=False)
    class_name: str = field(compare=False)
    zone: str = field(compare=False)
    distance_band: Optional[str] = field(default=None, compare=False)

class AudioOrchestrator:
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.engine = pyttsx3.init()
        # Speed up speech rate slightly for better responsiveness
        rate = self.engine.getProperty('rate')
        self.engine.setProperty('rate', rate + 25)
        
        self.running = True
        self.current_priority = None
        
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        while self.running:
            try:
                # Block until an event is available
                event = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            if not self.running:
                break
                
            self.current_priority = event.priority
            
            # Speak the text (blocking call)
            self.engine.say(event.text)
            self.engine.runAndWait()
            
            self.current_priority = None
            self.queue.task_done()

    def enqueue(self, event: VoiceEvent):
        """
        Pushes a new event to the queue. If it's a higher priority event 
        than the currently playing one, attempt to interrupt it.
        (Lower integer = higher priority)
        """
        # If we have a higher priority event and something lower is playing
        if self.current_priority is not None and event.priority < self.current_priority:
            # Try to interrupt the current speech
            try:
                self.engine.stop()
            except Exception as e:
                print(f"Warning: Failed to interrupt pyttsx3 engine: {e}")
                
            # Clear all pending lower-priority events from the queue
            self._clear_queue(min_priority_to_keep=event.priority)
            
        self.queue.put(event)

    def _clear_queue(self, min_priority_to_keep=None):
        """Empties the current queue of items with lower priority."""
        temp_list = []
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                if min_priority_to_keep is not None and item.priority <= min_priority_to_keep:
                    temp_list.append(item)
                self.queue.task_done()
            except queue.Empty:
                break
                
        # Put back the high priority ones
        for item in temp_list:
            self.queue.put(item)

    def stop(self):
        self.running = False
        try:
            self.engine.stop()
        except:
            pass
        self.worker_thread.join(timeout=1.0)
