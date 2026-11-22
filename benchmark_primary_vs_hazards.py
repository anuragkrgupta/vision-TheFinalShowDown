import os
import sys
import time
from pathlib import Path
import torch
from ultralytics import YOLO

def format_bbox(box):
    return [round(float(v), 2) for v in box]

def main():
    print("=" * 80)
    print("BENCHMARK COMPARISON: PRIMARY MODEL vs HAZARDS MODEL")
    print("=" * 80)

    primary_path = Path("models/best_custom_yolov8n.pt")
    hazards_path = Path("runs/detect/runs/detect/hazards_model/weights/best.pt")

    if not primary_path.exists():
        print(f"ERROR: Primary model not found at {primary_path}")
        return
    if not hazards_path.exists():
        print(f"ERROR: Hazards model not found at {hazards_path}")
        return

    # Select device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Inference Device: {device}")
    if device == "cuda:0":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # 1. Load Primary Model
    print(f"\nLoading Primary Model: {primary_path}")
    primary_model = YOLO(str(primary_path))
    print(f"  Primary Model Names: {primary_model.names}")

    # 2. Load Hazards Model
    print(f"\nLoading Hazards Model: {hazards_path}")
    hazards_model = YOLO(str(hazards_path))
    print(f"  Raw Embedded Names:  {hazards_model.names}")
    # Apply in-memory class correction
    hazards_model.model.names = {0: "door", 1: "stairs", 2: "window"}
    print(f"  Corrected Names:     {hazards_model.names}")

    # Warmup
    dummy_img = "model_test_images/traffic/images (7).jpg"
    _ = primary_model.predict(dummy_img, imgsz=640, conf=0.25, device=device, verbose=False)
    _ = hazards_model.predict(dummy_img, imgsz=640, conf=0.25, device=device, verbose=False)

    # Test Image Groups
    test_suite = [
        (
            "DOORS",
            [
                (Path("dataset/hazards_dataset/images/val/004a39badfc14cb4.jpg"), Path("dataset/hazards_dataset/labels/val/004a39badfc14cb4.txt"), "Exterior/Interior residential doors"),
                (Path("dataset/hazards_dataset/images/val/00a096ef77b91be2.jpg"), Path("dataset/hazards_dataset/labels/val/00a096ef77b91be2.txt"), "Large double doors"),
            ]
        ),
        (
            "STAIRS",
            [
                (Path("model_test_images/stairs/images (8).jpg"), None, "Outdoor park staircase with handrail"),
                (Path("model_test_images/stairs/images (14).jpg"), None, "Underground metro staircase with ascending commuter"),
                (Path("dataset/hazards_dataset/images/val/109877264_50266fea4a.jpg"), Path("dataset/hazards_dataset/labels/val/109877264_50266fea4a.txt"), "Concrete stair treads"),
            ]
        ),
        (
            "WINDOWS",
            [
                (Path("dataset/hazards_dataset/images/val/00041da5b52f4197.jpg"), Path("dataset/hazards_dataset/labels/val/00041da5b52f4197.txt"), "Building facade with multiple window rows"),
                (Path("dataset/hazards_dataset/images/val/0000e8b36676338b.jpg"), Path("dataset/hazards_dataset/labels/val/0000e8b36676338b.txt"), "Large glass window / door pane"),
            ]
        ),
        (
            "INDOOR OBSTACLES",
            [
                (Path("dataset/Indoor Objects Detection/test/images/1003.png"), Path("dataset/Indoor Objects Detection/test/labels/1003.txt"), "Indoor living room interior"),
                (Path("dataset/Indoor Objects Detection/test/images/1014.png"), Path("dataset/Indoor Objects Detection/test/labels/1014.txt"), "Indoor furniture / chairs / doorway"),
                (Path("dataset/Indoor Objects Detection/test/images/1020.png"), Path("dataset/Indoor Objects Detection/test/labels/1020.txt"), "Indoor office / table interior"),
            ]
        ),
        (
            "MIXED SCENES",
            [
                (Path("dataset/hazards_dataset/images/val/000a05dd07294ce4.jpg"), Path("dataset/hazards_dataset/labels/val/000a05dd07294ce4.txt"), "Greenhouse entrance (door + windows)"),
                (Path("dataset/hazards_dataset/images/val/423401646_bb0132385d.jpg"), Path("dataset/hazards_dataset/labels/val/423401646_bb0132385d.txt"), "Stairs + windows + door facade"),
                (Path("model_test_images/crowd/toronto-on-canada-september-7-600w-2719199917.webp"), None, "Urban street: crowd + transit + background buildings"),
            ]
        ),
        (
            "BACKGROUND / TRAFFIC SCENES",
            [
                (Path("model_test_images/traffic/images (7).jpg"), None, "Sunlit roadway with scooters and cars"),
                (Path("model_test_images/traffic/ALL_7480.jpg"), None, "Urban congested junction: rickshaws, motorcycles, commuters"),
            ]
        ),
    ]

    total_primary_time = 0.0
    total_hazards_time = 0.0
    total_images = 0

    print("\n" + "=" * 80)
    print("DETAILED HEAD-TO-HEAD INFERENCE RESULTS")
    print("=" * 80)

    for group_name, images in test_suite:
        print(f"\n================================================================================")
        print(f"CATEGORY: {group_name}")
        print(f"================================================================================")

        for img_path, label_path, description in images:
            if not img_path.exists():
                print(f"SKIPPING: File not found {img_path}")
                continue

            total_images += 1
            print(f"\n--------------------------------------------------------------------------------")
            print(f"Image:       {img_path.name}")
            print(f"Description: {description}")
            print(f"Path:        {img_path}")

            # Ground truth if available
            if label_path and label_path.exists():
                gt_lines = label_path.read_text().splitlines()
                gt_desc = []
                for line in gt_lines:
                    if line.strip():
                        parts = line.split()
                        cid = int(parts[0])
                        # Check whether it's hazards dataset or indoor dataset
                        if "hazards_dataset" in str(label_path):
                            cname = {0: "door", 1: "stairs", 2: "window"}.get(cid, str(cid))
                        else:
                            indoor_names = ["door", "cabinetDoor", "refrigeratorDoor", "window", "chair", "table", "cabinet", "couch", "openedDoor", "pole"]
                            cname = indoor_names[cid] if cid < len(indoor_names) else str(cid)
                        gt_desc.append(cname)
                print(f"Ground Truth: {len(gt_desc)} objects -> {', '.join(gt_desc[:10])}{' ...' if len(gt_desc) > 10 else ''}")
            else:
                print("Ground Truth: Unannotated test media / visual benchmark")

            # --- Primary Model Inference ---
            t0 = time.perf_counter()
            p_res = primary_model.predict(
                source=str(img_path),
                imgsz=640,
                conf=0.25,
                device=device,
                verbose=False
            )[0]
            p_time = (time.perf_counter() - t0) * 1000.0
            total_primary_time += p_time

            # --- Hazards Model Inference ---
            t0 = time.perf_counter()
            h_res = hazards_model.predict(
                source=str(img_path),
                imgsz=640,
                conf=0.25,
                device=device,
                verbose=False
            )[0]
            h_time = (time.perf_counter() - t0) * 1000.0
            total_hazards_time += h_time

            # Display Primary Results
            print(f"\n[1] PRIMARY MODEL (best_custom_yolov8n.pt) | Latency: {p_time:.2f} ms")
            if p_res.boxes is not None and len(p_res.boxes) > 0:
                for b in p_res.boxes:
                    cid = int(b.cls.item())
                    conf = float(b.conf.item())
                    cname = primary_model.names.get(cid, str(cid))
                    xyxy = format_bbox(b.xyxy[0].tolist())
                    print(f"    - Class: {cname:<15} | Conf: {conf:.2f} | Box: {xyxy}")
            else:
                print("    - None (No target navigation objects detected)")

            # Display Hazards Results
            print(f"\n[2] HAZARDS MODEL (hazards_model/best.pt)   | Latency: {h_time:.2f} ms")
            if h_res.boxes is not None and len(h_res.boxes) > 0:
                for b in h_res.boxes:
                    cid = int(b.cls.item())
                    conf = float(b.conf.item())
                    cname = hazards_model.names.get(cid, str(cid))
                    xyxy = format_bbox(b.xyxy[0].tolist())
                    print(f"    - Class: {cname:<15} | Conf: {conf:.2f} | Box: {xyxy}")
            else:
                print("    - None (No doors, stairs, or windows detected)")

    print("\n" + "=" * 80)
    print("SPEED & LATENCY SUMMARY")
    print("=" * 80)
    avg_primary = total_primary_time / max(1, total_images)
    avg_hazards = total_hazards_time / max(1, total_images)
    print(f"Images Evaluated:          {total_images}")
    print(f"Primary Model Avg Latency: {avg_primary:.2f} ms/image ({1000.0/avg_primary:.1f} FPS)")
    print(f"Hazards Model Avg Latency: {avg_hazards:.2f} ms/image ({1000.0/avg_hazards:.1f} FPS)")

if __name__ == "__main__":
    main()

