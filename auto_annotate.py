import os
import glob
import random
import shutil
from ultralytics import YOLO

def main():
    print("Starting auto-annotation and splitting process...")
    
    raw_images_dir = os.path.join("dataset", "images", "raw")
    images = glob.glob(os.path.join(raw_images_dir, "*.jpg"))
    print(f"Found {len(images)} raw images to annotate.")
    
    if len(images) == 0:
        print("No images found. Exiting.")
        return
        
    # Load a pre-trained model to generate pseudo-labels
    # Using the standard YOLOv8n which is very fast and accurate for COCO classes
    model = YOLO("yolov8n.pt")
    
    # We will save the labels to a temporary raw labels directory
    raw_labels_dir = os.path.join("dataset", "labels", "raw")
    os.makedirs(raw_labels_dir, exist_ok=True)
    
    print("Generating YOLO annotations...")
    # Run inference on all images, save txt labels
    for img_path in images:
        basename = os.path.basename(img_path)
        name, _ = os.path.splitext(basename)
        txt_path = os.path.join(raw_labels_dir, f"{name}.txt")
        
        # Only predict if label doesn't exist to allow resuming
        if not os.path.exists(txt_path):
            results = model.predict(img_path, save_txt=True, save_conf=False, exist_ok=True, project="runs/detect", name="auto_annotate", verbose=False)
            
            # Ultralytics saves labels deep in the project folder, let's move it to our raw_labels_dir
            # If no objects were detected, ultralytics might not create a txt file
            pred_txt = os.path.join("runs", "detect", "auto_annotate", "labels", f"{name}.txt")
            if os.path.exists(pred_txt):
                shutil.move(pred_txt, txt_path)
            else:
                # Create empty file for background image
                open(txt_path, 'w').close()
                
    print("Annotation complete. Now splitting data 85/15...")
    
    # Define split directories
    dirs = {
        'images_train': os.path.join("dataset", "images", "train"),
        'images_val': os.path.join("dataset", "images", "val"),
        'labels_train': os.path.join("dataset", "labels", "train"),
        'labels_val': os.path.join("dataset", "labels", "val")
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    random.shuffle(images)
    split_idx = int(len(images) * 0.85)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    def move_files(img_list, split_name):
        for img_path in img_list:
            basename = os.path.basename(img_path)
            name, _ = os.path.splitext(basename)
            
            src_label = os.path.join(raw_labels_dir, f"{name}.txt")
            dst_img = os.path.join(dirs[f'images_{split_name}'], basename)
            dst_label = os.path.join(dirs[f'labels_{split_name}'], f"{name}.txt")
            
            shutil.move(img_path, dst_img)
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
                
    move_files(train_images, 'train')
    move_files(val_images, 'val')
    
    # Create dataset yaml
    abs_path = os.path.abspath('dataset').replace('\\', '/')
    yaml_content = f"""path: {abs_path}
train: images/train
val: images/val

# Standard COCO classes as we used yolov8n.pt for annotation
nc: 80
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
"""
    yaml_path = os.path.join("dataset", "custom_dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"Successfully processed {len(images)} images.")
    print("Data splitting and formatting complete!")

if __name__ == "__main__":
    main()
