from ultralytics import YOLO
import shutil
from pathlib import Path

def train_model():
    base_dir = Path(__file__).parent.parent
    yaml_path = base_dir / "dataset" / "custom_dataset.yaml"
    
    if not yaml_path.exists():
        print(f"Dataset YAML not found at {yaml_path}. Run split_dataset.py first.")
        return
        
    print("Loading base model yolov8n.pt...")
    model = YOLO("yolov8n.pt")
    
    # We use a small number of epochs (10) for demonstration/speed. 
    # For a real production model, this should be 50-100+ depending on dataset size.
    print("Starting training...")
    results = model.train(
        data=str(yaml_path),
        epochs=10,
        imgsz=640,
        batch=16, # Adjust if out of memory
        name="custom_yolov8n",
        device=0 # Using GPU 0 for training
    )
    
    # Copy best weights to a known location
    train_dir = base_dir / "runs" / "detect"
    
    # Ultralytics puts it in the most recent train* folder
    # We can just look for the most recently created 'custom_yolov8n*' folder
    # or rely on model.trainer.save_dir if accessible
    save_dir = Path(model.trainer.save_dir) if hasattr(model, 'trainer') else None
    
    if save_dir:
        best_pt = save_dir / "weights" / "best.pt"
        if best_pt.exists():
            models_dir = base_dir / "models"
            models_dir.mkdir(exist_ok=True)
            target_pt = models_dir / "best_custom_yolov8n.pt"
            shutil.copy(str(best_pt), str(target_pt))
            print(f"Training complete! Best weights saved to: {target_pt}")
            print(f"Update your config/detection_config.json to use this new model.")
        else:
            print("Could not find best.pt in the runs directory.")
            
if __name__ == "__main__":
    train_model()
