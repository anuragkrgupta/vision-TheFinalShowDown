from ultralytics import YOLO
import os

def main():
    print("Initializing YOLOv8n for Pothole Detection...")
    model = YOLO("yolov8n.pt")
    
    # Path to the data.yaml
    yaml_path = os.path.abspath(os.path.join("dataset", "Pothole detection dataset", "data.yaml"))
    
    print(f"Starting training on {yaml_path}")
    print("Running a short training run (10 epochs) for testing...")
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=10,        # Short run for testing
        imgsz=640,
        batch=4,          # Small batch size to avoid OOM on 4GB GPU
        device=0,         # Use GPU
        project="runs/detect",
        name="pothole_model",
        exist_ok=True
    )
    
    print("Training complete! Model saved to runs/detect/pothole_model/weights/best.pt")

if __name__ == '__main__':
    main()
