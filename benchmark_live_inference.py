
import time
import cv2
from detection.pipeline import DetectionPipeline

TOTAL_FRAMES = 100
CAMERA_INDEX = 0

print("Loading VISION detection pipeline...")
pipeline = DetectionPipeline()

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print(f"Camera started. Processing {TOTAL_FRAMES} frames...")
print("Please keep the camera pointed at your test environment.")

frame_count = 0
total_inference_time = 0.0
start_time = time.perf_counter()

try:
    while frame_count < TOTAL_FRAMES:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame unavailable.")
            break

        inference_start = time.perf_counter()

        events, detections = pipeline.process_frame(frame)

        inference_time = time.perf_counter() - inference_start

        total_inference_time += inference_time
        frame_count += 1

        print(
            f"Frame {frame_count}/{TOTAL_FRAMES} | "
            f"Inference: {inference_time * 1000:.1f} ms | "
            f"Detections: {len(detections)} | "
            f"Events: {len(events)}"
        )

finally:
    elapsed = time.perf_counter() - start_time
    cap.release()

    print("\n========== VISION BENCHMARK RESULTS ==========")
    print("Frames processed:", frame_count)
    print("Total duration:", round(elapsed, 2), "seconds")

    if frame_count > 0:
        average_inference = total_inference_time / frame_count

        print(
            "Average inference latency:",
            round(average_inference * 1000, 2),
            "ms"
        )

        print(
            "Average inference FPS:",
            round(1 / average_inference, 2)
            if average_inference > 0 else 0
        )

        print(
            "Overall processing FPS:",
            round(frame_count / elapsed, 2)
            if elapsed > 0 else 0
        )

    print("==============================================")

