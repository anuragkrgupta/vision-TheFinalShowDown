import json
from pathlib import Path
import numpy as np
import yaml
from ultralytics import YOLO

def main():
    print("=" * 80)
    print("PART A: HAZARDS MODEL VALIDATION METRICS VERIFICATION")
    print("=" * 80)

    # 1. Inspect data.yaml
    yaml_path = Path("dataset/hazards_dataset/data.yaml").resolve()
    print(f"Dataset YAML Path: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    print("Dataset YAML Content:")
    print(json.dumps(yaml_data, indent=2))

    # Resolve val images directory
    val_rel = yaml_data.get("val", "images/val")
    val_dir = (yaml_path.parent / val_rel).resolve()
    print(f"Exact Validation Images Directory: {val_dir}")
    print(f"Validation Directory Exists: {val_dir.exists()}")

    # 2. Check labels directory
    val_labels_dir = (yaml_path.parent / "labels" / "val").resolve()
    print(f"Exact Validation Labels Directory: {val_labels_dir}")
    print(f"Validation Labels Directory Exists: {val_labels_dir.exists()}")

    # Count ground truth instances directly from label files
    gt_counts = {0: 0, 1: 0, 2: 0}
    label_files = list(val_labels_dir.glob("*.txt"))
    for lf in label_files:
        for line in lf.read_text().splitlines():
            line = line.strip()
            if line:
                cid = int(line.split()[0])
                if cid in gt_counts:
                    gt_counts[cid] += 1
                else:
                    gt_counts[cid] = 1

    print(f"\nGround Truth Label Files Count: {len(label_files)}")
    print(f"Ground Truth Instance Counts from disk: {gt_counts}")
    print(f"  Class 0 (door):   {gt_counts.get(0, 0)} instances")
    print(f"  Class 1 (stairs): {gt_counts.get(1, 0)} instances")
    print(f"  Class 2 (window): {gt_counts.get(2, 0)} instances")
    print(f"  Total GT Instances: {sum(gt_counts.values())}")

    # 3. Load Model and verify class mapping
    checkpoint_path = Path("runs/detect/runs/detect/hazards_model/weights/best.pt").resolve()
    print(f"\nLoading Model Checkpoint: {checkpoint_path}")
    model = YOLO(str(checkpoint_path))
    print(f"Model embedded names (raw):   {model.names}")

    # Apply in-memory mapping correction
    CORRECTED_NAMES = {0: "door", 1: "stairs", 2: "window"}
    model.model.names = CORRECTED_NAMES
    print(f"Model active names (corrected): {model.names}")

    # 4. Run model.val() on GPU
    print("\nRunning Ultralytics model.val()...")
    metrics = model.val(
        data=str(yaml_path),
        imgsz=640,
        conf=0.25,
        device=0,
        split="val",
        plots=False,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("DETAILED METRIC STRUCTURE FROM ULTRALYTICS DetMetrics")
    print("=" * 80)
    print(f"Validation Images Evaluated: {metrics.box.nc} classes")
    print(f"Mean Precision (all):        {metrics.box.mp:.4f}")
    print(f"Mean Recall (all):           {metrics.box.mr:.4f}")
    print(f"Mean mAP@50 (all):           {metrics.box.map50:.4f}")
    print(f"Mean mAP@50-95 (all):        {metrics.box.map:.4f}")

    # Class-wise breakdowns
    # Note: metrics.box.all_ap has shape (nc, 10) for IoU 0.50 to 0.95
    # ap[:, 0] is mAP@50; ap.mean(axis=1) is mAP@50-95
    ap50_per_class = metrics.box.all_ap[:, 0]
    ap50_95_per_class = metrics.box.all_ap.mean(axis=1)

    print("\nClass-wise Breakdown:")
    for idx, cname in model.names.items():
        p = metrics.box.p[idx]
        r = metrics.box.r[idx]
        ap50 = ap50_per_class[idx]
        ap50_95 = ap50_95_per_class[idx]
        print(f"  Class {idx} ({cname:<7}): Precision={p:.4f}, Recall={r:.4f}, mAP@50={ap50:.4f}, mAP@50-95={ap50_95:.4f}")

if __name__ == "__main__":
    main()

