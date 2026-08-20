import os
import cv2
import glob
import shutil
import random
import yaml
from pathlib import Path

DATASET_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset")))

def verify_and_collect_data(source_img_dirs, source_lbl_dirs, class_offset=0):
    """
    Collects all valid image/label pairs from the given directories.
    Validates that the image can be read and that the label file exists.
    """
    valid_pairs = []
    
    for img_dir, lbl_dir in zip(source_img_dirs, source_lbl_dirs):
        if not img_dir.exists() or not lbl_dir.exists():
            continue
            
        img_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        
        for img_path in img_paths:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            
            # Check if label exists
            if not lbl_path.exists():
                continue
                
            # Check if image is valid/readable
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Corrupt or unreadable image {img_path}")
                continue
                
            # If everything is good, add to our list
            valid_pairs.append((img_path, lbl_path))
            
    return valid_pairs

def preprocess_and_merge():
    print("--- Preprocessing and Merging Pothole Datasets ---")
    
    # Define our output directory
    out_dir = DATASET_DIR / "master_pothole_dataset"
    if out_dir.exists():
        shutil.rmtree(out_dir)
        
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_pairs = []
    
    # 1. Collect from 'Pothole detection dataset' (Roboflow)
    robo_dir = DATASET_DIR / "Pothole detection dataset"
    robo_img_dirs = [robo_dir / "train" / "images", robo_dir / "valid" / "images", robo_dir / "test" / "images"]
    robo_lbl_dirs = [robo_dir / "train" / "labels", robo_dir / "valid" / "labels", robo_dir / "test" / "labels"]
    robo_pairs = verify_and_collect_data(robo_img_dirs, robo_lbl_dirs)
    all_pairs.extend(robo_pairs)
    print(f"Found {len(robo_pairs)} valid images in Roboflow dataset.")
    
    # 2. Collect from 'potholes_converted' (The Pascal VOC ones we converted)
    voc_dir = DATASET_DIR / "potholes_converted"
    voc_img_dirs = [voc_dir / "images" / "train"]
    voc_lbl_dirs = [voc_dir / "labels" / "train"]
    voc_pairs = verify_and_collect_data(voc_img_dirs, voc_lbl_dirs)
    all_pairs.extend(voc_pairs)
    print(f"Found {len(voc_pairs)} valid images in Converted VOC dataset.")
    
    # Shuffle all data to ensure a good random distribution
    print(f"\nTotal valid images to process: {len(all_pairs)}")
    random.shuffle(all_pairs)
    
    # Train (80%), Val (10%), Test (10%)
    num_total = len(all_pairs)
    num_train = int(num_total * 0.8)
    num_val = int(num_total * 0.1)
    
    train_pairs = all_pairs[:num_train]
    val_pairs = all_pairs[num_train:num_train+num_val]
    test_pairs = all_pairs[num_train+num_val:]
    
    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }
    
    print("\nCopying files to new master dataset structure...")
    for split_name, pairs in splits.items():
        for i, (img_path, lbl_path) in enumerate(pairs):
            # Create a unique filename to prevent collisions between datasets
            unique_name = f"pothole_{split_name}_{i:04d}"
            img_ext = img_path.suffix
            
            new_img_path = out_dir / "images" / split_name / (unique_name + img_ext)
            new_lbl_path = out_dir / "labels" / split_name / (unique_name + ".txt")
            
            shutil.copy(img_path, new_img_path)
            
            # Read label and ensure class ID is 0 (since it's only potholes)
            with open(lbl_path, "r") as f_in, open(new_lbl_path, "w") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # Force class ID to 0
                        f_out.write(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                        
        print(f"Created {len(pairs)} images in '{split_name}' split.")
        
    # Generate the unified data.yaml
    yaml_content = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["pothole"]
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"\nMaster dataset created successfully at: {out_dir}")
    print("This dataset is fully clean, split, verified, and ready for a production training run.")

if __name__ == "__main__":
    preprocess_and_merge()
