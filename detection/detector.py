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
        # Load YOLOv8 nano model (will auto-download on first run)
        self.model = YOLO("yolov8n.pt")
        
        # Build a mapping from YOLO's class IDs to class names to filter effectively
        self.names = self.model.names
        self.target_class_ids = [
            class_id for class_id, class_name in self.names.items() 
            if class_name in self.target_classes
        ]

    def detect(self, frame):
        """
        Runs inference on a single frame and returns filtered detections.
        """
        # Run inference (stream=False for single frame)
        # verbose=False to keep console clean during loops
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                class_name = self.names[cls_id]
                
                # Filter by config
                if class_name in self.target_classes:
                    # Get bounding box coordinates [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].tolist()
                    detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": xyxy
                    })
                    
        return detections
