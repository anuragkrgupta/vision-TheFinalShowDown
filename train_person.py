from ultralytics import YOLO
import os

def main():
    print("Initializing YOLOv8n for Person Dataset fine-tuning...")
    # Start from COCO weights because it already knows "person" well, 
    # but we will fine-tune it to our specific dataset (likely different camera angles)
    model = YOLO("yolov8n.pt")
    
    yaml_path = os.path.abspath(os.path.join("dataset", "Person", "person", "data.yaml"))
    
    if not os.path.exists(yaml_path):
        print(f"ERROR: Dataset not found at {yaml_path}.")
        return
        
    print(f"Starting training on {yaml_path}")
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=50,       
        imgsz=640,
        batch=8,          
        patience=10,      
        device=0,         
        project="runs/detect",
        name="custom_person_model",
        exist_ok=True,
        workers=2,
    )
    
    print("Training complete! Model saved to runs/detect/custom_person_model/weights/best.pt")

if __name__ == '__main__':
    main()
