import os
import glob
import random
import shutil

def main():
    src_dir = "dataset/Door, Windows and Stairs Dataset/images"
    dst_dir = "dataset/hazards_dataset"
    
    print("Preparing Hazards Dataset...")
    
    # Create YOLO directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dst_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, 'labels', split), exist_ok=True)
        
    # Get all images
    all_images = glob.glob(os.path.join(src_dir, "*.jpg"))
    random.shuffle(all_images)
    
    # Split 85% train, 15% val
    split_idx = int(len(all_images) * 0.85)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    def copy_files(img_list, split_name):
        for img_path in img_list:
            basename = os.path.basename(img_path)
            txt_name = basename.replace('.jpg', '.txt')
            txt_path = os.path.join(src_dir, txt_name)
            
            # Copy image
            shutil.copy(img_path, os.path.join(dst_dir, 'images', split_name, basename))
            
            # Copy label if exists, else create empty label file (background image)
            if os.path.exists(txt_path):
                shutil.copy(txt_path, os.path.join(dst_dir, 'labels', split_name, txt_name))
            else:
                open(os.path.join(dst_dir, 'labels', split_name, txt_name), 'w').close()
                
    print(f"Copying {len(train_images)} training files...")
    copy_files(train_images, 'train')
    
    print(f"Copying {len(val_images)} validation files...")
    copy_files(val_images, 'val')
    
    # Create data.yaml
    yaml_content = f"""
train: images/train
val: images/val

nc: 3
names: ['door', 'window', 'stairs']
"""
    yaml_path = os.path.join(dst_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
        
    print(f"Successfully formatted dataset to {dst_dir}")
    print(f"Created {yaml_path}")

if __name__ == '__main__':
    main()
