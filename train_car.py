from ultralytics import YOLO
import os

def main():
    print("Initializing YOLOv8n for Traffic Signs & Lights Dataset (dataset/car)...")
    
    # Start from standard YOLO weights
    model = YOLO("yolov8n.pt")
    
    yaml_path = os.path.abspath(os.path.join("dataset", "car", "data.yaml"))
    
    if not os.path.exists(yaml_path):
        print(f"ERROR: Dataset not found at {yaml_path}.")
        return
        
    print(f"Starting training on {yaml_path}")
    
    results = model.train(
        data=yaml_path,
        epochs=50,       
        imgsz=640,
        batch=8,          
        patience=10,      
        device=0,         
        project="runs/detect",
        name="traffic_signs_model",
        exist_ok=True,
        workers=2,
    )
    
    print("Training complete! Model saved to runs/detect/traffic_signs_model/weights/best.pt")

if __name__ == '__main__':
    main()
