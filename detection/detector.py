import json
from pathlib import Path
from ultralytics import YOLO

class NavigationDetector:
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to the project config
            config_path = Path(__file__).parent.parent / "config" / "detection_config.json"
            
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.target_classes = set(self.config.get("target_classes", []))
        
        # Load YOLOv8 models from config, fallback to a single pretrained nano
        model_paths = self.config.get("model_paths", [])
        if not model_paths:
            # Backwards compatibility if they still use 'model_path'
            single_path = self.config.get("model_path", "yolov8n.pt")
            model_paths = [single_path]
            
        self.models = []
        for path_str in model_paths:
            if path_str != "yolov8n.pt":
                path_str = str(Path(__file__).parent.parent / path_str)
            self.models.append(YOLO(path_str))
        
        # Build a mapping from YOLO's class IDs to class names to filter effectively
        # Each model might have different names mappings, so we handle it per-model
        # during inference.

    def detect(self, frame):
        """
        Runs inference on a single frame using all loaded models and returns filtered detections.
        """
        detections = []
        
        for model in self.models:
            # Run inference (stream=False for single frame, verbose=False to keep console clean)
            results = model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    confidence = box.conf[0].item()
                    class_name = model.names[cls_id]
                    
                    # Filter by config
                    if class_name in self.target_classes:
                        # Get bounding box coordinates [x1, y1, x2, y2]
                        xyxy = box.xyxy[0].tolist()
                        xyxyn = box.xyxyn[0].tolist() # Normalized coordinates [0, 1]
                        detections.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": xyxy,
                            "bbox_norm": xyxyn
                        })
                        
        return detections
