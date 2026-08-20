import os
import sys
from detection.pipeline import DetectionPipeline
from detection.detector import NavigationDetector

def main():
    video_path = "tests/fixtures/videos/stock-footage-india-busy-road-with-traffic-pothole-and-sanitation-worker-cleaning.webm"
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return
        
    print(f"Testing pipeline on {video_path}...")
    pipeline = DetectionPipeline()
    
    # We will process offline to get a timeline of events
    timeline = pipeline.process_video_offline(video_path, visualize=False)
    
    # Analyze the timeline to summarize what was detected
    detected_classes = set()
    total_events = 0
    class_counts = {}
    
    for frame_data in timeline:
        events = frame_data["emitted_events"]
        for e in events:
            cls_name = e["class_name"]
            detected_classes.add(cls_name)
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            total_events += 1
            
    print("\n--- Test Results ---")
    print(f"Total frames processed: {len(timeline)}")
    print(f"Total auditory events emitted: {total_events}")
    print(f"Unique object types detected: {', '.join(detected_classes) if detected_classes else 'None'}")
    
    print("\n--- Event Frequency by Class ---")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{cls_name.capitalize()}: {count} announcements")
    
    print("\nSample of emitted events:")
    sample_count = 0
    for frame_data in timeline:
        for e in frame_data["emitted_events"]:
            print(f"Frame {frame_data['frame']}: {e['zone']}, {e['class_name']} (Prox: {e['proximity']})")
            sample_count += 1
            if sample_count >= 15:
                break
        if sample_count >= 15:
            break

if __name__ == "__main__":
    main()
