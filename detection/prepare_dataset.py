import os
import cv2
import json
from pathlib import Path
from ultralytics import YOLO

def load_config():
    config_path = Path(__file__).parent.parent / "config" / "detection_config.json"
    with open(config_path, "r") as f:
        return json.load(f)

def prepare_dataset(video_dir, output_dir, frame_interval=10):
    config = load_config()
    target_classes = config.get("target_classes", [])
    
    if not target_classes:
        print("No navigation classes defined in config.")
        return
        
    print(f"Loading large teacher model (yolov8x.pt)...")
    teacher_model = YOLO('yolov8x.pt')
    
    # Map model's class names to our new custom 0-indexed IDs
    model_names = teacher_model.names
    name_to_original_id = {v: k for k, v in model_names.items()}
    
    target_class_ids = []
    for cls_name in target_classes:
        if cls_name in name_to_original_id:
            target_class_ids.append(name_to_original_id[cls_name])
            
    # Our new custom IDs
    custom_class_map = {cls_name: i for i, cls_name in enumerate(target_classes)}
    
    images_dir = Path(output_dir) / "images" / "raw"
    labels_dir = Path(output_dir) / "labels" / "raw"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target classes: {custom_class_map}")
    
    video_paths = list(Path(video_dir).glob("*.mp4"))
    
    total_frames_extracted = 0
    total_labels_created = 0
    
    for video_path in video_paths:
        print(f"Processing video: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open {video_path.name}")
            continue
            
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # Run inference
                results = teacher_model.predict(frame, classes=target_class_ids, verbose=False)
                
                # We only save frames that actually contain target objects to avoid a completely empty dataset
                # although some empty frames are good for negative mining, we will keep it simple and only save active frames
                boxes = results[0].boxes
                
                if len(boxes) > 0:
                    img_height, img_width = frame.shape[:2]
                    
                    img_filename = f"{video_path.stem}_f{frame_idx}.jpg"
                    lbl_filename = f"{video_path.stem}_f{frame_idx}.txt"
                    
                    img_save_path = images_dir / img_filename
                    lbl_save_path = labels_dir / lbl_filename
                    
                    valid_detections = 0
                    labels_content = []
                    
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = model_names[cls_id]
                        
                        if cls_name in custom_class_map:
                            custom_id = custom_class_map[cls_name]
                            
                            # YOLO format: normalized x_center, y_center, width, height
                            x_c, y_c, w, h = box.xywhn[0].tolist()
                            labels_content.append(f"{custom_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                            valid_detections += 1
                            
                    if valid_detections > 0:
                        cv2.imwrite(str(img_save_path), frame)
                        with open(lbl_save_path, "w") as f:
                            f.write("\n".join(labels_content))
                        
                        total_frames_extracted += 1
                        total_labels_created += valid_detections
            
            frame_idx += 1
            
        cap.release()
        
    print(f"Finished. Extracted {total_frames_extracted} frames with {total_labels_created} labels.")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    vid_dir = base_dir / "tests" / "fixtures" / "videos"
    out_dir = base_dir / "dataset"
    prepare_dataset(vid_dir, out_dir, frame_interval=10)
