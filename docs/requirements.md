# Assistive Vision Software — Requirements & Dependencies

## 1. Software Functional Requirements

| ID | Requirement |
|----|-------------|
| SR-1 | Capture video frames from a camera source at a configurable sampled rate (2–5 fps). |
| SR-2 | Run object detection inference on captured frames using a YOLOv8n model, restricted to a configurable class subset. |
| SR-3 | Read distance data from a connected distance sensor and fuse it with detected bounding boxes. |
| SR-4 | Classify each detection by distance band (Near/Mid/Ignore) and horizontal zone (Left/Center/Right). |
| SR-5 | Apply temporal smoothing (N-of-M frame persistence) before emitting a detection event. |
| SR-6 | Apply per-object cooldown to suppress repeat announcements. |
| SR-7 | Maintain a priority queue that orders/interrupts voice events based on zone + distance-band urgency. |
| SR-8 | Convert queued text to speech and play it through an audio output device. |
| SR-9 | Accept a user-triggered "read sign" event, capture a frame, and send it to the OCR backend over WebSocket. |
| SR-10 | Expose a FastAPI WebSocket endpoint that accepts an image frame and returns extracted text as JSON. |
| SR-11 | Run OCR extraction on the received frame server-side and return `{ text, confidence }`. |
| SR-12 | Continue operating the detection/voice loop with no network connection; OCR requests fail gracefully when offline. |
| SR-13 | Log detection events, distances, and OCR results locally for debugging and later dataset collection. |
| SR-14 | Load configuration (class list, distance thresholds, zone boundaries, cooldown windows) from a config file rather than hardcoded values. |

## 2. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | Detection-to-voice latency ≤ ~1 second for Near-band, Center-zone events. |
| NFR-2 | Capture, inference, and voice output run as separate threads/processes; none blocks the others. |
| NFR-3 | System starts the detection loop within a few seconds of power-on (no long init delay). |
| NFR-4 | Sensor or model init failure degrades gracefully (skip that feature) rather than crashing the process. |
| NFR-5 | Codebase is modular: detection, distance fusion, zone/priority logic, voice queue, and OCR client are independently swappable/testable units. |

## 3. Python Package Dependencies

### Detection & vision
| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8n model loading and inference |
| `opencv-python` (or `opencv-python-headless` on headless Pi setups) | Camera capture, frame preprocessing |
| `numpy` | Array/frame manipulation |
| `torch` / `torchvision` | Backend inference engine for Ultralytics — **on Raspberry Pi/ARM, use the official ARM wheel builds, not generic pip install, or export the model to ONNX/NCNN/Hailo format for hardware-accelerated inference** |

### Distance sensing
| Package | Purpose |
|---|---|
| Sensor-specific driver library (e.g. `VL53L0X` / `smbus2` for I2C ToF sensors) | Reading distance sensor data |

### Backend & communication
| Package | Purpose |
|---|---|
| `fastapi` | Backend web framework for the OCR endpoint |
| `uvicorn` | ASGI server to run FastAPI |
| `websockets` | WebSocket client/server communication |
| `python-multipart` | Handling image uploads if using HTTP fallback alongside WebSocket |
| `pydantic` | Request/response schema validation (ships with FastAPI) |

### OCR
| Package | Purpose |
|---|---|
| `pytesseract` | Python wrapper for Tesseract OCR |
| Tesseract OCR engine (system package, not pip) | Actual OCR binary — installed via OS package manager |
| `Pillow` | Image handling for OCR preprocessing |

### Text-to-speech
| Package | Purpose |
|---|---|
| `pyttsx3` | Offline, fully free TTS engine (cross-platform, no internet dependency, no API costs) |

### Orchestration / utilities
| Package | Purpose |
|---|---|
| `python-dotenv` or `PyYAML` | Loading configuration files |
| `queue` (standard library) | Priority/voice event queue |
| `threading` / `multiprocessing` (standard library) | Parallel capture/inference/voice pipeline |
| `logging` (standard library) | Event and error logging |

## 4. System-Level (OS) Dependencies

| Dependency | Notes |
|---|---|
| Python 3.10+ | Base runtime |
| `tesseract-ocr` (apt package on Debian/Raspberry Pi OS) | Required by `pytesseract` — install via `sudo apt install tesseract-ocr` |
| I2C/SPI enabled on the Pi (`raspi-config`) | Required if using an I2C/SPI distance sensor |
| Camera interface enabled (`raspi-config` or `libcamera` stack) | Required for Pi Camera Module access |
| Hailo runtime/SDK (if using Hailo-8L accelerator) | Separate install from Hailo's own package repo, not pip |
| Audio output configured (ALSA/PulseAudio) | Required for `pyttsx3` playback on Linux |

## 5. Compatibility Notes

- **ARM architecture (Raspberry Pi):** Standard `pip install torch` may not have a prebuilt wheel for ARM — check for official ARM builds or plan to export the trained model to ONNX/NCNN format for lighter, faster inference without needing full PyTorch on-device.
- **Hailo-8L acceleration:** If using the Hailo AI kit, the model needs to be compiled/converted through Hailo's toolchain (separate from the Ultralytics training/export flow) — budget time for this conversion step.
- **`opencv-python` vs `opencv-python-headless`:** Use the headless variant on a device with no display server to avoid unnecessary GUI dependency bloat.
- **Tesseract accuracy:** Default Tesseract works reasonably on clean, well-lit printed signage; performance drops on stylized fonts or poor lighting — worth noting as a known POC-stage limitation, not a bug to chase down early.

## 6. Free & Open-Source Compliance Notes

Everything in Sections 3–4 is free/open-source, with two things worth knowing:

- **`ultralytics` license (AGPL-3.0):** Free to use, including for a POC and personal/academic work. AGPL only becomes relevant if you distribute a modified version as a hosted service without releasing source — not a concern at this stage, just something to be aware of if this ever goes commercial. No cost either way.
- **Tesseract vs cloud OCR APIs:** Stick with Tesseract (fully free, offline, no API key/billing) rather than Google Vision API or AWS Textract — those have free tiers but incur cost beyond a request quota. Tesseract is the right default here, not just a budget fallback.
- **Annotation tooling for future fine-tuning:** Use **CVAT** (fully free, self-hosted, open-source) rather than Roboflow's hosted paid tiers, if you get to the fine-tuning phase later.
- **Datasets:** Public datasets like Mapillary Vistas are free for research/non-commercial use — worth confirming license terms if this ever moves toward commercial deployment.

Nothing else in this document requires payment, an API key with billing, or a subscription.

## 7. Suggested `requirements.txt` (POC baseline)

```
ultralytics
opencv-python-headless
numpy
fastapi
uvicorn[standard]
websockets
pytesseract
Pillow
pyttsx3
PyYAML
python-dotenv
```
