import os
import glob
import cv2

def clean_and_verify_dataset(base_dir):
    print(f"Starting data cleaning and verification for {base_dir}...")
    
    splits = ['train', 'valid', 'test']
    corrupt_images = 0
    missing_labels = 0
    total_images = 0
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(images_dir):
            continue
            
        os.makedirs(labels_dir, exist_ok=True)
        
        image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                      glob.glob(os.path.join(images_dir, "*.png")) + \
                      glob.glob(os.path.join(images_dir, "*.jpeg"))
                      
        total_images += len(image_files)
        
        for img_path in image_files:
            # 1. Verify Image is readable
            img = cv2.imread(img_path)
            if img is None:
                print(f"Corrupt image found and removed: {img_path}")
                os.remove(img_path)
                corrupt_images += 1
                
                # Try to remove associated label if it exists
                basename = os.path.basename(img_path)
                name, _ = os.path.splitext(basename)
                label_path = os.path.join(labels_dir, name + ".txt")
                if os.path.exists(label_path):
                    os.remove(label_path)
                continue
                
            # 2. Verify Label exists, if not create empty (background image)
            basename = os.path.basename(img_path)
            name, _ = os.path.splitext(basename)
            label_path = os.path.join(labels_dir, name + ".txt")
            
            if not os.path.exists(label_path):
                open(label_path, 'w').close()
                missing_labels += 1

    print("\n--- Cleaning Summary ---")
    print(f"Total Images Processed: {total_images}")
    print(f"Corrupt Images Removed: {corrupt_images}")
    print(f"Missing Labels Created (Background): {missing_labels}")
    
    # 3. Rewrite data.yaml with absolute path and correct relative paths
    yaml_path = os.path.join(base_dir, "data.yaml")
    abs_path = os.path.abspath(base_dir).replace('\\', '/')
    
    yaml_content = f"""path: {abs_path}
train: train/images
val: valid/images
test: test/images

nc: 10
names: 
- door
- cabinetDoor
- refrigeratorDoor
- window
- chair
- table
- cabinet
- couch
- openedDoor
- pole
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"Updated {yaml_path} with absolute paths.")
    print("Data cleaning and preprocessing complete!")

if __name__ == "__main__":
    dataset_path = os.path.join("dataset", "Indoor Objects Detection")
    clean_and_verify_dataset(dataset_path)
