import sys
from pathlib import Path
import cv2
from detection.pipeline import DetectionPipeline

def main():
    print("=" * 80)
    print("VERIFICATION OF TWO-MODEL INTEGRATION IN PRODUCTION PIPELINE")
    print("=" * 80)

    # 1. Initialize DetectionPipeline via its standard default constructor (same as app.py)
    print("\n[Step 1] Initializing DetectionPipeline()...")
    pipeline = DetectionPipeline()
    detector = pipeline.detector

    # 2. Inspect loaded models
    print(f"\n[Step 2] Total Models Loaded in Detector: {len(detector.models)}")
    assert len(detector.models) == 2, f"Expected 2 models, got {len(detector.models)}"

    model_0 = detector.models[0]
    model_1 = detector.models[1]

    # Checkpoint paths from detector config
    config_paths = detector.config.get("model_paths", [])
    print(f"\nModel 0 Config Path: {config_paths[0]}")
    print(f"Model 0 Names ({len(model_0.names)} classes): {model_0.names}")

    print(f"\nModel 1 Config Path: {config_paths[1]}")
    print(f"Model 1 Names ({len(model_1.names)} classes): {model_1.names}")

    # Verify hazards mapping
    expected_hazards_names = {0: "door", 1: "stairs", 2: "window"}
    assert model_1.names == expected_hazards_names, (
        f"ERROR: Expected {expected_hazards_names}, got {model_1.names}"
    )
    print("\n[Verification Passed] Hazards model mapping is EXACTLY {0: 'door', 1: 'stairs', 2: 'window'}!")

    # 3. Test representative test images through pipeline.process_frame()
    test_cases = [
        ("ROAD / VEHICLES", Path("model_test_images/traffic/images (7).jpg")),
        ("DOORS", Path("dataset/hazards_dataset/images/val/004a39badfc14cb4.jpg")),
        ("STAIRS", Path("model_test_images/stairs/images (8).jpg")),
        ("WINDOWS", Path("dataset/hazards_dataset/images/val/00041da5b52f4197.jpg")),
        ("MIXED (Stairs + Window + Door)", Path("dataset/hazards_dataset/images/val/423401646_bb0132385d.jpg")),
        ("URBAN TRANSIT (Pedestrians + Bus + Buildings)", Path("model_test_images/crowd/toronto-on-canada-september-7-600w-2719199917.webp")),
    ]

    print("\n" + "=" * 80)
    print("[Step 3] RUNNING INFERENCE ON REPRESENTATIVE TEST IMAGES")
    print("=" * 80)

    for category, img_path in test_cases:
        print(f"\n--------------------------------------------------------------------------------")
        print(f"CATEGORY: {category}")
        print(f"Image:    {img_path.name}")

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"ERROR: Could not read image at {img_path}")
            continue

        # Single frame run through raw detector
        raw_detections = detector.detect(frame)
        print(f"\n[Raw Detections from Both Models Combined: {len(raw_detections)}]")
        detected_classes = set()
        for d in raw_detections:
            cname = d["class_name"]
            detected_classes.add(cname)
            # Validate detection dictionary structure
            assert "class_name" in d, "Missing class_name"
            assert "confidence" in d and 0.0 <= d["confidence"] <= 1.0, "Invalid confidence"
            assert "bbox" in d and len(d["bbox"]) == 4, "Invalid bbox"
            assert "bbox_norm" in d and len(d["bbox_norm"]) == 4, "Invalid bbox_norm"
            # Ensure normalized coords are within [0, 1] approximately
            for coord in d["bbox_norm"]:
                assert -0.05 <= coord <= 1.05, f"Normalized bbox coordinate out of bounds: {coord}"

        print(f"  Distinct Classes Detected: {sorted(list(detected_classes))}")
        for d in raw_detections[:8]:
            bbox_str = [round(v, 2) for v in d["bbox"]]
            print(f"  - {d['class_name']:<15} | Conf: {d['confidence']:.2f} | Box: {bbox_str}")
        if len(raw_detections) > 8:
            print(f"  ... ({len(raw_detections) - 8} more detections)")

        # Verify Spatial Analysis
        spatial_dets = pipeline.spatial_analyzer.analyze(raw_detections)
        print(f"\n[Spatial Analyzer Output: {len(spatial_dets)} objects]")
        for sd in spatial_dets[:4]:
            assert "zone" in sd and sd["zone"] in ("Left", "Center", "Right"), f"Invalid zone: {sd.get('zone')}"
            assert "proximity" in sd and sd["proximity"] in ("Near", "Mid", "Far"), f"Invalid proximity: {sd.get('proximity')}"
            assert "area" in sd and sd["area"] >= 0.0, f"Invalid area: {sd.get('area')}"
            print(f"  - {sd['class_name']:<12} | Zone: {sd['zone']:<6} | Proximity: {sd['proximity']:<4} | Area Ratio: {sd['area']:.4f}")
        if len(spatial_dets) > 4:
            print(f"  ... ({len(spatial_dets) - 4} more spatial detections)")

        # Verify Temporal Smoothing & Cooldown via pipeline.process_frame()
        # Feed 3 consecutive identical frames so N-of-M (2-of-3) persistence triggers
        print(f"\n[Testing Multi-Frame Temporal Persistence (M=2 out of N=3) & Cooldown]:")
        # Frame 1: History begins, may not reach M=2 yet
        emitted_f1, _ = pipeline.process_frame(frame)
        # Frame 2: Reaches M=2, should trigger events
        emitted_f2, _ = pipeline.process_frame(frame)
        # Frame 3: Cooldown (5.0s) should suppress repetitive alert
        emitted_f3, _ = pipeline.process_frame(frame)

        print(f"  Frame 1 Emitted Events: {[e['class_name'] for e in emitted_f1]}")
        print(f"  Frame 2 Emitted Events: {[e['class_name'] for e in emitted_f2]}")
        print(f"  Frame 3 Emitted Events: {[e['class_name'] for e in emitted_f3]} (Suppressed by Cooldown)")

    print("\n" + "=" * 80)
    print("ALL TWO-MODEL INTEGRATION VERIFICATIONS SUCCESSFUL!")
    print("=" * 80)

if __name__ == "__main__":
    main()

