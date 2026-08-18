import json
import cv2
from pathlib import Path
from detection.pipeline import DetectionPipeline

def main():
    base_dir = Path(__file__).parent / "fixtures"
    photos_dir = base_dir / "photos"
    results_path = base_dir / "expected_results.json"
    
    pipeline = DetectionPipeline()
    expected_results = {}
    
    # Process all image files
    for img_path in photos_dir.glob("*.*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        print(f"Processing {img_path.name}...")
        try:
            detections = pipeline.process_image(img_path)
            # Simplify detections for expected results (just store class names)
            classes_detected = sorted(list(set(d["class_name"] for d in detections)))
            expected_results[img_path.name] = {
                "expected_classes": classes_detected
            }
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            
    with open(results_path, "w") as f:
        json.dump(expected_results, f, indent=4)
        
    print(f"Baseline saved to {results_path}")

if __name__ == "__main__":
    main()
