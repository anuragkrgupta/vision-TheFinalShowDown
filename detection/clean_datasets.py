import os
import glob
import xml.etree.ElementTree as ET
import shutil
import yaml

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

def convert_voc_to_yolo():
    print("--- Converting Pascal VOC Pothole Dataset to YOLO ---")
    source_dir = os.path.join(DATASET_DIR, "potholes_and_road_damage_with_annotations", "potholes")
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} not found. Skipping.")
        return
        
    out_dir = os.path.join(DATASET_DIR, "potholes_converted")
    img_out = os.path.join(out_dir, "images", "train")
    lbl_out = os.path.join(out_dir, "labels", "train")
    
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    
    # Class map (only pothole in this dataset)
    classes = {"pothole": 0}
    
    xml_files = glob.glob(os.path.join(source_dir, "*.xml"))
    converted = 0
    
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        base_name = os.path.basename(xml_file).replace(".xml", "")
        img_file = os.path.join(source_dir, base_name + ".jpg")
        
        if not os.path.exists(img_file):
            continue
            
        txt_out_file = os.path.join(lbl_out, base_name + ".txt")
        
        with open(txt_out_file, "w") as out_f:
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in classes:
                    classes[cls_name] = len(classes)
                cls_id = classes[cls_name]
                
                xmlbox = obj.find("bndbox")
                xmin = float(xmlbox.find("xmin").text)
                xmax = float(xmlbox.find("xmax").text)
                ymin = float(xmlbox.find("ymin").text)
                ymax = float(xmlbox.find("ymax").text)
                
                # Normalize
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                w = (xmax - xmin) / float(width)
                h = (ymax - ymin) / float(height)
                
                out_f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                
        # Copy image
        shutil.copy(img_file, os.path.join(img_out, base_name + ".jpg"))
        converted += 1
        
    # Generate data.yaml for this new dataset
    yaml_content = {
        "train": "images/train",
        "val": "images/train",  # Just use train for val if un-split
        "nc": len(classes),
        "names": list(classes.keys())
    }
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        yaml.dump(yaml_content, f)
        
    print(f"Converted {converted} XML files. Saved to {out_dir}\n")

def fix_yamls():
    print("--- Fixing paths in existing YAMLs ---")
    
    # 1. Indoor Dataset
    indoor_yaml = os.path.join(DATASET_DIR, "Indoor_Object_Detection", "data.yaml")
    if os.path.exists(indoor_yaml):
        with open(indoor_yaml, 'r') as f:
            data = yaml.safe_load(f)
        if data:
            data['train'] = "train/images"
            data['val'] = "valid/images"
            data['test'] = "test/images"
            with open(indoor_yaml, 'w') as f:
                yaml.dump(data, f)
            print("Fixed Indoor_Object_Detection/data.yaml")

    # 2. Pothole Roboflow Dataset
    pothole_yaml = os.path.join(DATASET_DIR, "Pothole detection dataset", "data.yaml")
    if os.path.exists(pothole_yaml):
        with open(pothole_yaml, 'r') as f:
            data = yaml.safe_load(f)
        if data:
            data['train'] = "train/images"
            data['val'] = "valid/images"
            data['test'] = "test/images"
            with open(pothole_yaml, 'w') as f:
                yaml.dump(data, f)
            print("Fixed Pothole detection dataset/data.yaml")
            
    # 3. Car Dataset
    car_yaml = os.path.join(DATASET_DIR, "car", "data.yaml")
    if os.path.exists(car_yaml):
        with open(car_yaml, 'r') as f:
            data = yaml.safe_load(f)
        if data:
            data['train'] = "train/images"
            data['val'] = "valid/images"
            data['test'] = "test/images"
            with open(car_yaml, 'w') as f:
                yaml.dump(data, f)
            print("Fixed car/data.yaml")

def generate_traffic_yaml():
    print("--- Generating data.yaml for traffic dataset ---")
    traffic_dir = os.path.join(DATASET_DIR, "traffic dataset")
    if os.path.exists(traffic_dir):
        classes_file = os.path.join(traffic_dir, "classes.txt")
        classes = []
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
        
        if classes:
            yaml_content = {
                "train": "train/images",
                "val": "val/images",
                "nc": len(classes),
                "names": classes
            }
            with open(os.path.join(traffic_dir, "data.yaml"), "w") as f:
                yaml.dump(yaml_content, f)
            print("Generated traffic dataset/data.yaml\n")

if __name__ == "__main__":
    convert_voc_to_yolo()
    fix_yamls()
    generate_traffic_yaml()
    print("All dataset prep completed.")
