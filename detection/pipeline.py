import cv2
import time
from collections import deque
from pathlib import Path
from detection.detector import NavigationDetector

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

    def process_image(self, image_path):
        """Processes a static image and returns raw detections."""
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image at {image_path}")
        return self.detector.detect(frame)

    def process_video_offline(self, video_path, visualize=False):
        """
        Processes a video frame-by-frame as fast as possible to test temporal logic.
        Returns a timeline of active classes.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {video_path}")
            
        timeline = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            detections = self.detector.detect(frame)
            current_classes = set(d["class_name"] for d in detections)
            active_classes = self.smoother.update(current_classes)
            
            timeline.append({
                "frame": frame_idx,
                "raw_detections": detections,
                "smoothed_active_classes": list(active_classes)
            })
            
            if visualize:
                # Draw bounding boxes
                for d in detections:
                    x1, y1, x2, y2 = map(int, d["bbox"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, d["class_name"], (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imshow("Offline Video Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            frame_idx += 1
            
        cap.release()
        if visualize:
            cv2.destroyAllWindows()
            
        return timeline

import threading

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera {src}")
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        
    def start(self):
        self.thread.start()
        return self
        
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.ret = ret
            self.frame = frame
            
    def read(self):
        return self.ret, self.frame
        
    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

    def __del__(self):
        if hasattr(self, 'running') and self.running:
            self.stop()

    # In DetectionPipeline class, add this method:
def process_live_stream(pipeline_self, duration_minutes=None, visualize=True, log_file=None):
    """
    Runs real-time inference on the webcam.
    Maintains target_fps. Logs FPS and Memory if log_file is provided.
    """
    import psutil
    import csv
    import datetime
    
    stream = CameraStream().start()
    time.sleep(1.0) # warm up
    
    target_fps = pipeline_self.detector.config.get("target_fps", 5.0)
    target_frame_time = 1.0 / target_fps if target_fps > 0 else 0
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60) if duration_minutes else None
    
    csv_file = None
    writer = None
    if log_file:
        csv_file = open(log_file, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["Timestamp", "Actual_FPS", "Memory_MB", "Active_Classes"])

    frame_count = 0
    last_fps_time = time.time()
    current_fps = 0.0

    try:
        while True:
            loop_start = time.time()
            
            if end_time and loop_start > end_time:
                print("Duration reached. Stopping live stream.")
                break
                
            ret, frame = stream.read()
            if not ret or frame is None:
                print("Failed to grab frame.")
                break
                
            # Copy frame to avoid thread race condition during drawing
            disp_frame = frame.copy()
            
            # Inference
            detections = pipeline_self.detector.detect(disp_frame)
            current_classes = set(d["class_name"] for d in detections)
            active_classes = pipeline_self.smoother.update(current_classes)
            
            # Calculate FPS
            frame_count += 1
            if loop_start - last_fps_time >= 1.0:
                current_fps = frame_count / (loop_start - last_fps_time)
                frame_count = 0
                last_fps_time = loop_start
                
                if log_file and writer:
                    mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                    timestamp = datetime.datetime.now().isoformat()
                    writer.writerow([timestamp, f"{current_fps:.2f}", f"{mem_mb:.2f}", list(active_classes)])
                    csv_file.flush()
            
            if visualize:
                # Draw info
                cv2.putText(disp_frame, f"FPS: {current_fps:.1f} / Target: {target_fps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(disp_frame, f"Active: {list(active_classes)}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                for d in detections:
                    x1, y1, x2, y2 = map(int, d["bbox"])
                    cv2.rectangle(disp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(disp_frame, f"{d['class_name']} {d['confidence']:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                cv2.imshow("Live Detection", disp_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            # Enforce target FPS
            loop_end = time.time()
            elapsed = loop_end - loop_start
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("Live stream manually interrupted.")
    finally:
        stream.stop()
        if visualize:
            cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()

DetectionPipeline.process_live_stream = process_live_stream
