import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent
ROOT_IMAGES = PROJECT_ROOT / "model_test_images"
OUTPUT_DIR = PROJECT_ROOT / "model_test_results"
SUMMARY_FILE = PROJECT_ROOT / "model_test_summary.txt"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TEST_CONFIG = {
    "crowd": {
        "images": ROOT_IMAGES / "crowd",
        "models": {
            "best_custom_yolov8n": PROJECT_ROOT / "models" / "best_custom_yolov8n.pt",
            "custom_person_model": PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "custom_person_model" / "weights" / "best.pt",
        },
    },
    "traffic": {
        "images": ROOT_IMAGES / "traffic",
        "models": {
            "best_custom_yolov8n": PROJECT_ROOT / "models" / "best_custom_yolov8n.pt",
            "traffic_signs_model": PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "traffic_signs_model" / "weights" / "best.pt",
        },
    },
    "stairs": {
        "images": ROOT_IMAGES / "stairs",
        "models": {
            "hazards_model": PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "hazards_model" / "weights" / "best.pt",
        },
    },
}

def get_device():
    if torch.cuda.is_available():
        try:
            import torchvision.ops
            boxes = torch.tensor([[0., 0., 10., 10.]], device="cuda")
            scores = torch.tensor([0.9], device="cuda")
            torchvision.ops.nms(boxes, scores, 0.5)
            return 0, f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception as e:
            return "cpu", f"CPU (CUDA detected but torchvision::nms lacks CUDA kernel, falling back to CPU)"
    return "cpu", "CPU"

def main():
    device, device_name = get_device()
    print(f"Starting model evaluation...")
    print(f"Device: {device_name} (device={device})")
    print(f"Root test images: {ROOT_IMAGES}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("VISION MODEL EVALUATION SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Device: {device_name}")
    summary_lines.append(f"Confidence Threshold: 0.25 | Image Size: 640")
    summary_lines.append("=" * 80 + "\n")

    overall_errors = []

    for category, cfg in TEST_CONFIG.items():
        image_dir = cfg["images"]
        models_dict = cfg["models"]

        print("=" * 70)
        print(f"CATEGORY: {category.upper()}")
        print(f"Directory: {image_dir}")
        print("=" * 70)

        if not image_dir.exists() or not image_dir.is_dir():
            err_msg = f"Directory not found: {image_dir}"
            print(f"ERROR: {err_msg}")
            overall_errors.append(f"[{category}] {err_msg}")
            summary_lines.append(f"Category: {category} - ERROR: {err_msg}\n")
            continue

        images = sorted(
            [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        )
        print(f"Found {len(images)} images in {category}")

        if not images:
            warn_msg = f"No supported images found in {image_dir}"
            print(f"WARNING: {warn_msg}")
            overall_errors.append(f"[{category}] {warn_msg}")
            summary_lines.append(f"Category: {category} - WARNING: {warn_msg}\n")
            continue

        for model_name, weights_path in models_dict.items():
            print(f"\n--- Model: {model_name} ---")
            print(f"Weights path: {weights_path}")

            if not weights_path.exists():
                err_msg = f"Model weights not found: {weights_path}"
                print(f"ERROR: {err_msg}")
                overall_errors.append(f"[{category}/{model_name}] {err_msg}")
                summary_lines.append(f"Category: {category} | Model: {model_name}")
                summary_lines.append(f"  ERROR: {err_msg}\n")
                continue

            save_dir = OUTPUT_DIR / category / model_name
            save_dir.mkdir(parents=True, exist_ok=True)

            try:
                model = YOLO(str(weights_path))
                # If loading hazards_model with uncorrected checkpoint mapping, apply correct in-memory mapping
                if model_name == "hazards_model" and model.names == {0: "door", 1: "window", 2: "stairs"}:
                    model.model.names = {0: "door", 1: "stairs", 2: "window"}
            except Exception as e:
                err_msg = f"Failed to load model {weights_path}: {e}"
                print(f"ERROR: {err_msg}")
                overall_errors.append(f"[{category}/{model_name}] {err_msg}")
                summary_lines.append(f"Category: {category} | Model: {model_name}")
                summary_lines.append(f"  ERROR: {err_msg}\n")
                continue

            processed_count = 0
            no_detection_images = []
            detection_records = []

            for img_path in images:
                try:
                    results = model.predict(
                        source=str(img_path),
                        imgsz=640,
                        conf=0.25,
                        device=device,
                        save=False,
                        verbose=False,
                    )
                    res = results[0]
                    processed_count += 1

                    # Extract detections
                    detections = []
                    if res.boxes is not None and len(res.boxes) > 0:
                        for box in res.boxes:
                            cls_id = int(box.cls.item())
                            conf = float(box.conf.item())
                            cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, "names") and model.names else str(cls_id)
                            detections.append((cls_name, conf))

                    # Save annotated image preserving source image name & model
                    output_filename = f"{img_path.stem}_{model_name}{img_path.suffix}"
                    output_filepath = save_dir / output_filename
                    res.save(filename=str(output_filepath))

                    if detections:
                        det_str = ", ".join([f"{c} {s:.2f}" for c, s in detections])
                        print(f"  {img_path.name}: {det_str}")
                        detection_records.append((img_path.name, det_str))
                    else:
                        print(f"  {img_path.name}: NO DETECTIONS")
                        no_detection_images.append(img_path.name)
                        detection_records.append((img_path.name, "NO DETECTIONS"))

                except Exception as e:
                    err_msg = f"Error processing {img_path.name}: {e}"
                    print(f"  ERROR: {err_msg}")
                    overall_errors.append(f"[{category}/{model_name}/{img_path.name}] {err_msg}")
                    detection_records.append((img_path.name, f"ERROR: {e}"))

            # Append to summary
            summary_lines.append(f"Category: {category} | Model: {model_name}")
            summary_lines.append(f"  Weights: {weights_path.relative_to(PROJECT_ROOT) if weights_path.is_relative_to(PROJECT_ROOT) else weights_path}")
            summary_lines.append(f"  Annotated Images Output: {save_dir.relative_to(PROJECT_ROOT) if save_dir.is_relative_to(PROJECT_ROOT) else save_dir}")
            summary_lines.append(f"  Images Processed: {processed_count}/{len(images)}")
            summary_lines.append(f"  Images With Detections: {processed_count - len(no_detection_images)}")
            summary_lines.append(f"  Images With No Detections: {len(no_detection_images)}")
            if no_detection_images:
                summary_lines.append(f"    - " + ", ".join(no_detection_images))
            summary_lines.append("  Detections:")
            for img_name, det_str in detection_records:
                summary_lines.append(f"    * {img_name}: {det_str}")
            summary_lines.append("")

    if overall_errors:
        summary_lines.append("ERRORS ENCOUNTERED:")
        for err in overall_errors:
            summary_lines.append(f"  - {err}")
    else:
        summary_lines.append("No errors encountered during evaluation.")
    summary_lines.append("\n" + "=" * 80)

    # Write summary file
    summary_text = "\n".join(summary_lines)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nEvaluation complete!")
    print(f"Annotated results saved to: {OUTPUT_DIR}")
    print(f"Summary saved to: {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
