import pytest
import json
from pathlib import Path
from detection.pipeline import DetectionPipeline

BASE_DIR = Path(__file__).parent / "fixtures"
PHOTOS_DIR = BASE_DIR / "photos"
EXPECTED_RESULTS_FILE = BASE_DIR / "expected_results.json"

@pytest.fixture(scope="module")
def pipeline():
    return DetectionPipeline()

@pytest.fixture(scope="module")
def expected_results():
    if not EXPECTED_RESULTS_FILE.exists():
        pytest.skip("expected_results.json not found. Run generate_baseline.py first.")
    with open(EXPECTED_RESULTS_FILE, "r") as f:
        return json.load(f)

def test_static_photos(pipeline, expected_results):
    for filename, expected_data in expected_results.items():
        img_path = PHOTOS_DIR / filename
        if not img_path.exists():
            continue
            
        detections = pipeline.process_image(img_path)
        classes_detected = set(d["class_name"] for d in detections)
        expected_classes = set(expected_data.get("expected_classes", []))
        
        # We assert that our filtered pipeline output contains the same classes
        # that the baseline recorded (which was also filtered by the same config).
        # We check subset/superset to ensure no regressions in expected classes.
        missing_classes = expected_classes - classes_detected
        assert not missing_classes, f"Regression on {filename}: Missing expected classes {missing_classes}. Detected: {classes_detected}"
