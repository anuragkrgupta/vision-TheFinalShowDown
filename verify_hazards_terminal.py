import sys
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    model_path = Path("runs/detect/runs/detect/hazards_model/weights/best.pt")
    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found at {model_path}")
        return

    print("=" * 80)
    print("HAZARDS MODEL TERMINAL INFERENCE VERIFICATION")
    print("=" * 80)
    print(f"Loading checkpoint: {model_path}")

    model = YOLO(str(model_path))
    original_names = dict(model.names)
    print(f"Embedded model.names in checkpoint: {original_names}")

    # Correct in-memory class mapping to match ground truth annotations
    CORRECTED_NAMES = {0: "door", 1: "stairs", 2: "window"}
    model.model.names = CORRECTED_NAMES
    print(f"Active in-memory model.names:       {model.names}")
    print("=" * 80)

    # Categories to test
    test_suite = [
        (
            "DOOR IMAGES",
            [
                Path("dataset/hazards_dataset/images/val/004a39badfc14cb4.jpg"),
                Path("dataset/hazards_dataset/images/val/00a096ef77b91be2.jpg"),
                Path("dataset/hazards_dataset/images/val/00b08e610dd83c90.jpg"),
            ]
        ),
        (
            "STAIRS IMAGES (Test set & Validation set)",
            [
                Path("model_test_images/stairs/images (8).jpg"),
                Path("model_test_images/stairs/images (9).jpg"),
                Path("model_test_images/stairs/images (10).jpg"),
                Path("model_test_images/stairs/images (11).jpg"),
                Path("model_test_images/stairs/images (12).jpg"),
                Path("model_test_images/stairs/images (13).jpg"),
                Path("model_test_images/stairs/images (14).jpg"),
                Path("dataset/hazards_dataset/images/val/107171693_c3a8f003cf.jpg"),
                Path("dataset/hazards_dataset/images/val/109877264_50266fea4a.jpg"),
            ]
        ),
        (
            "WINDOW IMAGES",
            [
                Path("dataset/hazards_dataset/images/val/0000e8b36676338b.jpg"),
                Path("dataset/hazards_dataset/images/val/00033bc5b18e08ad.jpg"),
                Path("dataset/hazards_dataset/images/val/00041da5b52f4197.jpg"),
            ]
        ),
        (
            "MULTI-CLASS IMAGES",
            [
                Path("dataset/hazards_dataset/images/val/000a05dd07294ce4.jpg"),  # Door & Windows
                Path("dataset/hazards_dataset/images/val/356661655_4c9de71b71.jpg"), # Stairs & Door
                Path("dataset/hazards_dataset/images/val/423401646_bb0132385d.jpg"), # Stairs & Windows
            ]
        ),
        (
            "BACKGROUND / NON-HAZARD SCENE",
            [
                Path("model_test_images/traffic/images (7).jpg"),
            ]
        ),
    ]

    for category_name, images in test_suite:
        print(f"\n### {category_name}\n")
        for img_path in images:
            if not img_path.exists():
                print(f"Image: {img_path.name}")
                print(f"Error: File not found at {img_path}\n")
                continue

            results = model.predict(
                source=str(img_path),
                imgsz=640,
                conf=0.25,
                device="cpu",
                verbose=False
            )
            res = results[0]

            print(f"Image: {img_path.name}\n")
            print("Detected objects:")
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    cls_name = model.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy[0].tolist()]
                    print(f"Class: {cls_name}")
                    print(f"Confidence: {conf:.2f}")
                    print(f"Bounding box: {x1}, {y1}, {x2}, {y2}")
                    print()
            else:
                print("None\n")

if __name__ == "__main__":
    main()

