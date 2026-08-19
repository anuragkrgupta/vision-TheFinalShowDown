import os
import shutil
import random
import json
import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent.parent / "config" / "detection_config.json"
    with open(config_path, "r") as f:
        return json.load(f)

def split_dataset(dataset_dir, train_ratio=0.8):
    base_path = Path(dataset_dir)
    raw_images_dir = base_path / "images" / "raw"
    raw_labels_dir = base_path / "labels" / "raw"
    
    if not raw_images_dir.exists() or not raw_labels_dir.exists():
        print("Raw dataset directories not found. Run prepare_dataset.py first.")
        return
        
    images = list(raw_images_dir.glob("*.jpg"))
    if not images:
        print("No images found in raw directory.")
        return
        
    # Shuffle
    random.seed(42)
    random.shuffle(images)
    
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Create target dirs
    for split in ["train", "val"]:
        (base_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    def move_files(file_list, split):
        for img_path in file_list:
            lbl_path = raw_labels_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.move(str(img_path), str(base_path / "images" / split / img_path.name))
                shutil.move(str(lbl_path), str(base_path / "labels" / split / lbl_path.name))
                
    print(f"Splitting dataset: {len(train_images)} train, {len(val_images)} val.")
    move_files(train_images, "train")
    move_files(val_images, "val")
    
    # Generate YAML
    config = load_config()
    target_classes = config.get("target_classes", [])
    
    # We must use absolute paths for Ultralytics YOLO to avoid path resolution errors
    yaml_data = {
        "path": str(base_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(target_classes)}
    }
    
    yaml_path = base_path / "custom_dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f"Dataset YAML generated at {yaml_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    out_dir = base_dir / "dataset"
    split_dataset(out_dir)
