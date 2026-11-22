import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import psutil
import torch

from detection.pipeline import DetectionPipeline

# Initialize the production pipeline
pipeline = DetectionPipeline()

class MockVideoFrame:
    """Emulates av.VideoFrame to mirror streamlit-webrtc without external av dependency."""
    def __init__(self, ndarray):
        self._arr = ndarray

    def to_ndarray(self, format="bgr24"):
        return self._arr.copy()

    @staticmethod
    def from_ndarray(arr, format="bgr24"):
        return arr

# Define VideoProcessor identically to app.py lines 20-68
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        emitted_events, spatial_detections = pipeline.process_frame(img)

        for d in spatial_detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d['class_name']} {d['confidence']:.2f} ({d['zone']}, {d['proximity']})"
            cv2.putText(img, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if emitted_events:
            event_classes = [e['class_name'] for e in emitted_events]
            event_text = f"Emitted: {', '.join(event_classes)}"
            cv2.putText(img, event_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return MockVideoFrame.from_ndarray(img, format="bgr24")

def create_av_frame(cv2_bgr_img):
    return MockVideoFrame(cv2_bgr_img)

def main():
    print("=" * 80)
    print("VISION — LIVE CAMERA INTEGRATION & CONTINUOUS SOAK TEST")
    print("=" * 80)

    detector = pipeline.detector
    print(f"Loaded Models Count: {len(detector.models)}")
    for i, m in enumerate(detector.models):
        print(f"  Model [{i}]: {detector.config['model_paths'][i]}")
        print(f"    Class Names: {m.names}")

    processor = VideoProcessor()
    print("Initialized VideoProcessor successfully.")

    assets = {
        "road": cv2.imread("model_test_images/traffic/images (7).jpg"),
        "doors": cv2.imread("dataset/hazards_dataset/images/val/004a39badfc14cb4.jpg"),
        "stairs": cv2.imread("model_test_images/stairs/images (8).jpg"),
        "stairs_metro": cv2.imread("model_test_images/stairs/images (14).jpg"),
        "windows": cv2.imread("dataset/hazards_dataset/images/val/00041da5b52f4197.jpg"),
        "mixed": cv2.imread("dataset/hazards_dataset/images/val/423401646_bb0132385d.jpg"),
    }

    for name, img in assets.items():
        assert img is not None, f"Failed to load asset for {name}"
    print(f"Loaded {len(assets)} test scene assets into memory.")

    # --------------------------------------------------------------------------
    # SECTION 1: DEDICATED SCENARIO & ZONE/PROXIMITY VALIDATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 1: FUNCTIONAL VERIFICATION OF ZONES, PROXIMITY & SMOOTHING")
    print("=" * 80)

    # Scenarios 1-5
    for name, img in assets.items():
        av_in = create_av_frame(img)
        t0 = time.perf_counter()
        av_out = processor.recv(av_in)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        out_bgr = av_out
        assert out_bgr is not None and out_bgr.shape == img.shape

        _, spatial = pipeline.process_frame(cv2.flip(img, 1))
        classes = list(set([d["class_name"] for d in spatial]))
        zones = list(set([d["zone"] for d in spatial]))
        proximities = list(set([d["proximity"] for d in spatial]))

        print(f"  Scene: {name.upper():<12} | Latency: {lat_ms:6.2f} ms | Classes: {classes} | Zones: {zones} | Prox: {proximities}")

    # Scenario 6: Spatial Zone Shifting (Left -> Center -> Right)
    print("\nTesting Spatial Zone Shifting (Left -> Center -> Right)...")
    stairs_crop = assets["stairs"]
    h, w = stairs_crop.shape[:2]

    for target_zone, shift_factor in [("Left", -0.3), ("Center", 0.0), ("Right", 0.3)]:
        M = np.float32([[1, 0, shift_factor * w], [0, 1, 0]])
        shifted_img = cv2.warpAffine(stairs_crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        av_in = create_av_frame(shifted_img)
        _ = processor.recv(av_in)
        _, spatial = pipeline.process_frame(cv2.flip(shifted_img, 1))
        detected_zones = [d["zone"] for d in spatial if d["class_name"] == "stairs"]
        print(f"    Shift {target_zone:<6} (dx={shift_factor*w:+.0f}px) -> Detected Stairs Zones: {detected_zones}")

    # Scenario 7: Proximity Scaling (Far -> Mid -> Near)
    print("\nTesting Proximity Scaling (Zoom / Distance Variation)...")
    for target_prox, scale in [("Far (<10% area)", 0.3), ("Mid (10-30% area)", 0.7), ("Near (>30% area)", 1.4)]:
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        scaled = cv2.resize(stairs_crop, (scaled_w, scaled_h))
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        min_h = min(h, scaled_h)
        min_w = min(w, scaled_w)
        canvas[:min_h, :min_w] = scaled[:min_h, :min_w]
        av_in = create_av_frame(canvas)
        _ = processor.recv(av_in)
        _, spatial = pipeline.process_frame(cv2.flip(canvas, 1))
        detected_prox = [d["proximity"] for d in spatial]
        print(f"    Scale {scale:.1f}x -> Target {target_prox:<16} | Detected Proximities: {detected_prox}")

    # Scenario 8: Temporal Smoothing & Cooldown
    print("\nTesting Temporal Persistence (2-of-3) & Event Cooldown (5.0s)...")
    smooth_door = assets["doors"]
    e1, _ = pipeline.process_frame(smooth_door)
    e2, _ = pipeline.process_frame(smooth_door)
    e3, _ = pipeline.process_frame(smooth_door)
    time.sleep(5.1)
    e4, _ = pipeline.process_frame(smooth_door)

    print(f"    Frame 1: {[d['class_name'] for d in e1]} (Persistence filtering: Expected None)")
    print(f"    Frame 2: {[d['class_name'] for d in e2]} (Persistence reached: Expected Alert)")
    print(f"    Frame 3: {[d['class_name'] for d in e3]} (Cooldown active: Expected None)")
    print(f"    Frame 4 (t+5.1s): {[d['class_name'] for d in e4]} (Cooldown expired: Expected Alert)")

    # --------------------------------------------------------------------------
    # SECTION 2: 10-MINUTE CONTINUOUS CAMERA STREAM SOAK TEST (600 SECONDS)
    # --------------------------------------------------------------------------
    duration_seconds = 600 # 10 full minutes
    target_fps = 5.0
    target_interval = 1.0 / target_fps

    print("\n" + "=" * 80)
    print(f"SECTION 2: 10-MINUTE CONTINUOUS CAMERA STREAM SOAK TEST ({duration_seconds}s @ {target_fps} FPS)")
    print("=" * 80)

    process = psutil.Process()
    start_rss_mb = process.memory_info().rss / (1024 * 1024)
    start_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    print(f"Initial Process Memory: {start_rss_mb:.2f} MB RSS")
    print(f"Initial GPU VRAM:       {start_vram_mb:.2f} MB")
    print(f"Target Frame Rate:      {target_fps} FPS (Pacing: {target_interval*1000:.1f} ms)")

    scene_keys = list(assets.keys())
    total_frames = 0
    frame_latencies_ms = []

    log_csv_path = Path("live_soak_metrics.csv")
    with open(log_csv_path, "w", encoding="utf-8") as f:
        f.write("Elapsed_Seconds,Frames_Processed,Current_FPS,Mean_Latency_ms,Max_Latency_ms,CPU_Percent,RSS_Memory_MB,GPU_VRAM_MB\n")

    start_time = time.time()
    last_log_time = start_time
    frames_since_last_log = 0
    latencies_since_last_log = []

    print("\nStreaming continuous frames into VideoProcessor.recv()...\n")
    print("Time (s) | Total Frames | Window FPS | Avg Latency | Max Latency | CPU % | RAM (MB) | VRAM (MB)")
    print("-" * 88)

    while True:
        loop_start = time.perf_counter()
        elapsed = time.time() - start_time
        if elapsed >= duration_seconds:
            break

        scene_idx = (total_frames // 10) % len(scene_keys)
        current_scene_img = assets[scene_keys[scene_idx]]

        jitter_x = int(np.random.randint(-2, 3))
        jitter_y = int(np.random.randint(-2, 3))
        M = np.float32([[1, 0, jitter_x], [0, 1, jitter_y]])
        jittered_frame = cv2.warpAffine(current_scene_img, M, (current_scene_img.shape[1], current_scene_img.shape[0]))

        av_frame = create_av_frame(jittered_frame)

        t_recv_start = time.perf_counter()
        _ = processor.recv(av_frame)
        t_recv_end = time.perf_counter()

        frame_lat_ms = (t_recv_end - t_recv_start) * 1000.0
        frame_latencies_ms.append(frame_lat_ms)
        latencies_since_last_log.append(frame_lat_ms)
        total_frames += 1
        frames_since_last_log += 1

        now = time.time()
        if (now - last_log_time) >= 30.0:
            window_sec = now - last_log_time
            cur_fps = frames_since_last_log / window_sec
            mean_lat = np.mean(latencies_since_last_log)
            max_lat = np.max(latencies_since_last_log)
            cpu_pct = process.cpu_percent()
            rss_mb = process.memory_info().rss / (1024 * 1024)
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

            print(f"{elapsed:7.1f}s | {total_frames:12d} | {cur_fps:10.1f} | {mean_lat:9.2f}ms | {max_lat:9.2f}ms | {cpu_pct:4.1f}% | {rss_mb:7.1f} | {vram_mb:8.1f}")

            with open(log_csv_path, "a", encoding="utf-8") as f:
                f.write(f"{elapsed:.1f},{total_frames},{cur_fps:.2f},{mean_lat:.2f},{max_lat:.2f},{cpu_pct:.1f},{rss_mb:.1f},{vram_mb:.1f}\n")

            last_log_time = now
            frames_since_last_log = 0
            latencies_since_last_log = []

        elapsed_loop = time.perf_counter() - loop_start
        sleep_needed = target_interval - elapsed_loop
        if sleep_needed > 0:
            time.sleep(sleep_needed)

    total_elapsed = time.time() - start_time
    final_rss_mb = process.memory_info().rss / (1024 * 1024)
    final_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    print("-" * 88)
    print("\n" + "=" * 80)
    print("10-MINUTE CONTINUOUS SOAK TEST COMPLETE")
    print("=" * 80)
    print(f"Total Duration:             {total_elapsed:.1f} seconds ({total_elapsed/60.0:.1f} minutes)")
    print(f"Total Frames Processed:     {total_frames}")
    print(f"Overall Application FPS:    {total_frames / total_elapsed:.2f} FPS")
    print(f"Frame Latency Statistics:")
    print(f"  - Mean:                   {np.mean(frame_latencies_ms):.2f} ms")
    print(f"  - Median:                 {np.median(frame_latencies_ms):.2f} ms")
    print(f"  - P95:                    {np.percentile(frame_latencies_ms, 95):.2f} ms")
    print(f"  - Min:                    {np.min(frame_latencies_ms):.2f} ms")
    print(f"  - Max:                    {np.max(frame_latencies_ms):.2f} ms")
    print(f"Resource Stability:")
    print(f"  - Initial RAM:            {start_rss_mb:.2f} MB")
    print(f"  - Final RAM:              {final_rss_mb:.2f} MB (Change: {final_rss_mb - start_rss_mb:+.2f} MB)")
    print(f"  - Initial GPU VRAM:       {start_vram_mb:.2f} MB")
    print(f"  - Final GPU VRAM:         {final_vram_mb:.2f} MB (Change: {final_vram_mb - start_vram_mb:+.2f} MB)")
    print(f"  - Log Saved To:           {log_csv_path.resolve()}")

if __name__ == "__main__":
    main()
