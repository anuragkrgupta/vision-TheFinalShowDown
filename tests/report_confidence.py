import json
from pathlib import Path
from collections import defaultdict
from detection.pipeline import DetectionPipeline
import cv2

def main():
    base_dir = Path(__file__).parent / "fixtures"
    photos_dir = base_dir / "photos"
    videos_dir = base_dir / "videos"
    
    pipeline = DetectionPipeline()
    
    print("# Detection Confidence Report")
    print("\n## Photos")
    
    for img_path in photos_dir.glob("*.*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        try:
            detections = pipeline.process_image(img_path)
            print(f"\n### `{img_path.name}`")
            if not detections:
                print("- *No navigation classes detected*")
            for d in detections:
                print(f"- **{d['class_name']}**: {d['confidence']:.2f}")
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    print("\n## Videos")
    
    for video_path in videos_dir.glob("*.mp4"):
        print(f"\n### `{video_path.name}`")
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print("- *Failed to open video*")
                continue
                
            class_confidences = defaultdict(list)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Sample every 10th frame to speed up reporting
                if frame_count % 10 == 0:
                    detections = pipeline.detector.detect(frame)
                    for d in detections:
                        class_confidences[d['class_name']].append(d['confidence'])
                frame_count += 1
                
            cap.release()
            
            if not class_confidences:
                print("- *No navigation classes detected*")
            else:
                print("| Class | Avg Confidence | Max Confidence | Frames Detected |")
                print("|-------|----------------|----------------|-----------------|")
                for cls_name, confs in class_confidences.items():
                    avg_conf = sum(confs) / len(confs)
                    max_conf = max(confs)
                    print(f"| {cls_name} | {avg_conf:.2f} | {max_conf:.2f} | {len(confs)} |")
                    
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")

if __name__ == "__main__":
    main()
