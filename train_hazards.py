from ultralytics import YOLO
import os

def main():
    print("Initializing YOLOv8n for Hazards Dataset (Stairs, Poles, Curbs)...")
    model = YOLO("yolov8n.pt")
    
    # Path to the upcoming data.yaml
    yaml_path = os.path.abspath(os.path.join("dataset", "hazards_dataset", "data.yaml"))
    
    if not os.path.exists(yaml_path):
        print(f"ERROR: Dataset not found at {yaml_path}.")
        print("Please export your CVAT annotations in YOLOv8 format and place them in dataset/hazards_dataset/")
        return
        
    print(f"Starting production training on {yaml_path}")
    
    # Train the model prioritizing Recall
    results = model.train(
        data=yaml_path,
        epochs=100,       
        imgsz=640,
        batch=4,          
        patience=15,      
        device=0,         
        project="runs/detect",
        name="hazards_model",
        exist_ok=True,
        workers=2,
        # Setting a higher confidence loss weight to prioritize recall
        # Alternatively, we just evaluate heavily on recall later.
    )
    
    print("Training complete! Model saved to runs/detect/hazards_model/weights/best.pt")

if __name__ == '__main__':
    main()
