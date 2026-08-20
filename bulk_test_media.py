import os
import glob
from pathlib import Path
from detection.pipeline import DetectionPipeline

def main():
    print("Initializing Multi-Model Pipeline for Bulk Testing...")
    pipeline = DetectionPipeline()
    
    # Paths
    photos_dir = Path("tests/fixtures/photos")
    videos_dir = Path("tests/fixtures/videos")
    
    # 1. Test all Photos
    print("\n" + "="*50)
    print("📸 TESTING PHOTOS")
    print("="*50)
    photo_extensions = ['*.jpg', '*.jpeg', '*.png']
    photo_files = []
    for ext in photo_extensions:
        photo_files.extend(photos_dir.glob(ext))
        
    for photo_path in photo_files:
        try:
            # process_image bypasses temporal smoothing (since it's 1 frame)
            raw_detections = pipeline.process_image(str(photo_path))
            
            # Count classes detected
            counts = {}
            for d in raw_detections:
                cls_name = d['class_name']
                counts[cls_name] = counts.get(cls_name, 0) + 1
                
            detected_str = ", ".join([f"{k} ({v})" for k, v in counts.items()])
            if not detected_str:
                detected_str = "No objects detected"
                
            print(f"[{photo_path.name[:30]:<30}] -> {detected_str}")
        except Exception as e:
            print(f"[{photo_path.name[:30]:<30}] -> ERROR: {e}")

    # 2. Test all Videos
    print("\n" + "="*50)
    print("🎥 TESTING VIDEOS")
    print("="*50)
    video_extensions = ['*.mp4', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_dir.glob(ext))
        
    for video_path in video_files:
        try:
            print(f"\nProcessing Video: {video_path.name}")
            timeline = pipeline.process_video_offline(str(video_path), visualize=False)
            
            total_events = 0
            class_counts = {}
            
            for frame_data in timeline:
                for e in frame_data["emitted_events"]:
                    cls_name = e["class_name"]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    total_events += 1
                    
            print(f"  Frames processed: {len(timeline)}")
            print(f"  Total Voice Alerts: {total_events}")
            
            if class_counts:
                print("  Alerts by Class:")
                for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {cls_name.capitalize()}: {count}")
            else:
                print("  No voice alerts generated.")
                
        except Exception as e:
            print(f"  ERROR processing video: {e}")

if __name__ == "__main__":
    main()
