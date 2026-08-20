from ultralytics import YOLO
import os

def main():
    print("Initializing YOLOv8n for Master Pothole Dataset...")
    model = YOLO("yolov8n.pt")
    
    # Path to the unified data.yaml
    yaml_path = os.path.abspath(os.path.join("dataset", "master_pothole_dataset", "data.yaml"))
    
    print(f"Starting production training on {yaml_path}")
    print("Running for 50 epochs (early stopping enabled)...")
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=50,        # Increased epochs for a production run
        imgsz=640,
        batch=4,          # Kept at 4 to prevent OOM errors on 4GB VRAM
        patience=10,      # Early stopping if no improvement after 10 epochs
        device=0,         # Use GPU
        project="runs/detect",
        name="master_pothole_model",
        exist_ok=True,
        workers=2         # Limit dataloader workers to avoid CPU bottlenecking
    )
    
    print("Training complete! Model saved to runs/detect/master_pothole_model/weights/best.pt")

if __name__ == '__main__':
    main()
