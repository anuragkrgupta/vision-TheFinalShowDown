# Assistive Vision Software

An edge-optimized computer vision pipeline designed for assistive navigation. This software runs real-time object detection (via a fine-tuned YOLOv8 model), applies spatial analysis to determine object zones and distances, and utilizes temporal smoothing and cooldowns to generate clean, actionable events.

## Features

* **Custom Object Detection:** Fine-tuned YOLOv8n model optimized specifically for navigation-critical classes (people, cars, bicycles, traffic lights, etc.).
* **Spatial Zone Classification:** Automatically segments the visual field into `Left`, `Center`, and `Right` zones.
* **Proximity Proxy:** Uses normalized bounding box area as a fast heuristic to classify objects into `Near`, `Mid`, and `Far` distance bands.
* **Temporal Smoothing:** Employs an N-of-M persistence algorithm to eliminate flickering and false positives.
* **Event Cooldowns:** Suppresses repetitive announcements for the same object in the same zone, preventing user overwhelm while remaining responsive to moving obstacles.

## Architecture

1. **Camera/Input Stream:** Captures frames from a local webcam or offline video.
2. **Inference (`detector.py`):** Runs YOLOv8 inference and filters for target classes.
3. **Spatial Analyzer (`spatial_analyzer.py`):** Calculates centers and bounding box areas to assign zones and proximity.
4. **Temporal Smoother (`pipeline.py`):** Tracks unique `(class, zone, proximity)` keys across frames to ensure stability.
5. **Cooldown Manager (`cooldown.py`):** Filters the active smoothed events to ensure a quiet, non-spammy user experience.

## Setup & Installation

**1. Create and Activate Virtual Environment:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**2. Install Dependencies:**
```powershell
pip install ultralytics opencv-python numpy psutil
```

**3. Install PyTorch (GPU Support):**
To leverage NVIDIA GPUs for significantly faster inference:
```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Configuration

All system parameters are easily configurable via `config/detection_config.json`:
- Modify `target_classes` to change what objects the system looks for.
- Adjust `zone_boundaries` to shift the definitions of Left/Center/Right.
- Adjust `cooldown_seconds` to control how often repeat events are emitted.

## Usage & Startup

**Run Live Software (Webcam)**
To start the full pipeline using your computer's webcam:
```powershell
# Make sure your virtual environment is activated
.\venv\Scripts\python.exe main.py
```
*Note: A window will appear showing the live feed with bounding boxes, spatial zones, and the actual events the system emits will be printed to your console.*

**Run Offline Visual Tests**
To verify the system's behavior against the suite of sample videos (ideal for testing cooldowns and zones without physically moving around):
```powershell
$env:PYTHONPATH="C:\Users\kumar\Desktop\VISION"
.\venv\Scripts\python.exe tests\run_videos_visual.py
```

**Run Automated Unit Tests**
To run the automated test suite (verifying spatial math, cooldown logic, etc.):
```powershell
$env:PYTHONPATH="C:\Users\kumar\Desktop\VISION"
.\venv\Scripts\pytest.exe tests\
```

## Roadmap / Pending Phases

- [ ] **Text-to-Speech (TTS):** Integrate `pyttsx3` to announce the emitted events audibly.
- [ ] **Hardware Distance Fusion:** Replace the bounding box area proxy with actual distance sensor data (e.g. ToF I2C sensors).
- [ ] **OCR Backend:** Implement a FastAPI WebSocket to capture and read textual signs on-demand via Tesseract.
