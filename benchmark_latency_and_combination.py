import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def compute_stats(latencies_ms):
    arr = np.array(latencies_ms)
    mean = np.mean(arr)
    median = np.median(arr)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    std = np.std(arr)
    fps = 1000.0 / mean if mean > 0 else 0
    return {
        "mean": mean,
        "median": median,
        "p95": p95,
        "p99": p99,
        "std": std,
        "fps": fps
    }

def main():
    print("=" * 80)
    print("PARTS B & C: ADVANCED LATENCY, COMBINED INFERENCE & GPU BENCHMARK")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != "cuda:0":
        print("ERROR: CUDA GPU is required for accurate GPU benchmark.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"Inference Device: {device} ({gpu_name})")
    print("Inference Settings: imgsz=640, conf=0.25, iterations=50 per test image")

    # Load models
    primary_path = Path("models/best_custom_yolov8n.pt")
    hazards_path = Path("runs/detect/runs/detect/hazards_model/weights/best.pt")

    primary_model = YOLO(str(primary_path))
    hazards_model = YOLO(str(hazards_path))
    # In-memory mapping correction for hazards model
    hazards_model.model.names = {0: "door", 1: "stairs", 2: "window"}

    # Test images across diverse domains
    test_image_paths = [
        Path("dataset/hazards_dataset/images/val/004a39badfc14cb4.jpg"),    # Doors
        Path("model_test_images/stairs/images (8).jpg"),                   # Stairs
        Path("dataset/hazards_dataset/images/val/00041da5b52f4197.jpg"),    # Windows
        Path("dataset/Indoor Objects Detection/test/images/1003.png"),       # Indoor
        Path("dataset/hazards_dataset/images/val/423401646_bb0132385d.jpg"),# Mixed (Stairs+Windows+Door)
        Path("model_test_images/traffic/images (7).jpg"),                  # Traffic / Background
    ]

    # Pre-load all test images into memory as numpy arrays (emulating camera frames in app.py)
    loaded_images = []
    for p in test_image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: Failed to load {p}")
            continue
        loaded_images.append((p.name, img))

    print(f"\nPre-loaded {len(loaded_images)} test images into RAM (zero disk I/O during timing).")

    # Warm-up GPU with 15 iterations on both models
    print("\nWarming up GPU with 15 cycles on both models...")
    warmup_img = loaded_images[0][1]
    for _ in range(15):
        _ = primary_model.predict(warmup_img, imgsz=640, conf=0.25, device=device, verbose=False)
        _ = hazards_model.predict(warmup_img, imgsz=640, conf=0.25, device=device, verbose=False)
    torch.cuda.synchronize()
    print("Warm-up complete.")

    # --------------------------------------------------------------------------
    # PART B: Separate Model Benchmark (50 iterations per image)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART B: STANDALONE LATENCY PROFILES (50 runs per image, CUDA synchronized)")
    print("=" * 80)

    num_iterations = 50

    all_primary_pure_latencies = []
    all_hazards_pure_latencies = []

    for img_name, img_arr in loaded_images:
        print(f"\nEvaluating: {img_name} ({img_arr.shape[1]}x{img_arr.shape[0]})")

        # 1. Primary Model 50 runs
        primary_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = primary_model.predict(img_arr, imgsz=640, conf=0.25, device=device, verbose=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            primary_times.append((t1 - t0) * 1000.0)

        # 2. Hazards Model 50 runs
        hazards_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = hazards_model.predict(img_arr, imgsz=640, conf=0.25, device=device, verbose=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            hazards_times.append((t1 - t0) * 1000.0)

        p_stats = compute_stats(primary_times)
        h_stats = compute_stats(hazards_times)

        all_primary_pure_latencies.extend(primary_times)
        all_hazards_pure_latencies.extend(hazards_times)

        print(f"  Primary Model: Mean={p_stats['mean']:.2f}ms | Median={p_stats['median']:.2f}ms | P95={p_stats['p95']:.2f}ms | FPS={p_stats['fps']:.1f}")
        print(f"  Hazards Model: Mean={h_stats['mean']:.2f}ms | Median={h_stats['median']:.2f}ms | P95={h_stats['p95']:.2f}ms | FPS={h_stats['fps']:.1f}")

    overall_p_stats = compute_stats(all_primary_pure_latencies)
    overall_h_stats = compute_stats(all_hazards_pure_latencies)

    print("\n" + "-" * 80)
    print("OVERALL STANDALONE LATENCY METRICS (Aggregated across all 300 test runs):")
    print("-" * 80)
    print(f"Primary Model ({len(all_primary_pure_latencies)} runs):")
    print(f"  Mean Latency:    {overall_p_stats['mean']:.2f} ms")
    print(f"  Median Latency:  {overall_p_stats['median']:.2f} ms")
    print(f"  P95 Latency:     {overall_p_stats['p95']:.2f} ms")
    print(f"  P99 Latency:     {overall_p_stats['p99']:.2f} ms")
    print(f"  Std Deviation:   {overall_p_stats['std']:.2f} ms")
    print(f"  Approximate FPS: {overall_p_stats['fps']:.1f} FPS")

    print(f"\nHazards Model ({len(all_hazards_pure_latencies)} runs):")
    print(f"  Mean Latency:    {overall_h_stats['mean']:.2f} ms")
    print(f"  Median Latency:  {overall_h_stats['median']:.2f} ms")
    print(f"  P95 Latency:     {overall_h_stats['p95']:.2f} ms")
    print(f"  P99 Latency:     {overall_h_stats['p99']:.2f} ms")
    print(f"  Std Deviation:   {overall_h_stats['std']:.2f} ms")
    print(f"  Approximate FPS: {overall_h_stats['fps']:.1f} FPS")

    # --------------------------------------------------------------------------
    # PART C: Measured Sequential Two-Model Inference & GPU Memory Usage
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART C: DIRECT SEQUENTIAL TWO-MODEL COMBINED INFERENCE & VRAM METRICS")
    print("=" * 80)

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    base_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"Base Allocated GPU Memory (models loaded in VRAM): {base_memory_mb:.2f} MB")

    combined_latencies = []
    primary_isolated_latencies = []
    hazards_isolated_latencies = []

    for img_name, img_arr in loaded_images:
        print(f"\nSequential Two-Model Testing on: {img_name}")
        img_combined_times = []

        for _ in range(num_iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Execute Model 1 (Primary)
            _ = primary_model.predict(img_arr, imgsz=640, conf=0.25, device=device, verbose=False)
            torch.cuda.synchronize()
            t_mid = time.perf_counter()

            # Execute Model 2 (Hazards)
            _ = hazards_model.predict(img_arr, imgsz=640, conf=0.25, device=device, verbose=False)
            torch.cuda.synchronize()
            t_end = time.perf_counter()

            primary_isolated_latencies.append((t_mid - t0) * 1000.0)
            hazards_isolated_latencies.append((t_end - t_mid) * 1000.0)
            img_combined_times.append((t_end - t0) * 1000.0)

        combined_latencies.extend(img_combined_times)
        c_stats = compute_stats(img_combined_times)
        print(f"  Combined: Mean={c_stats['mean']:.2f}ms | Median={c_stats['median']:.2f}ms | P95={c_stats['p95']:.2f}ms | FPS={c_stats['fps']:.1f}")

    # VRAM stats
    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)

    final_combined_stats = compute_stats(combined_latencies)
    final_p_in_seq = compute_stats(primary_isolated_latencies)
    final_h_in_seq = compute_stats(hazards_isolated_latencies)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY: DIRECT SEQUENTIAL TWO-MODEL PERFORMANCE")
    print("=" * 80)
    print(f"Total Combined Runs:        {len(combined_latencies)}")
    print(f"Primary-Only Mean Latency:  {final_p_in_seq['mean']:.2f} ms")
    print(f"Hazards-Only Mean Latency:  {final_h_in_seq['mean']:.2f} ms")
    print(f"Direct Combined Latency:")
    print(f"  - Mean:                   {final_combined_stats['mean']:.2f} ms")
    print(f"  - Median:                 {final_combined_stats['median']:.2f} ms")
    print(f"  - P95:                    {final_combined_stats['p95']:.2f} ms")
    print(f"  - P99:                    {final_combined_stats['p99']:.2f} ms")
    print(f"  - Combined Throughput:    {final_combined_stats['fps']:.1f} FPS")
    print(f"\nGPU VRAM Usage:")
    print(f"  - Current Allocated VRAM: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
    print(f"  - Peak Allocated VRAM:    {peak_allocated_mb:.2f} MB")
    print(f"  - Reserved PyTorch VRAM:  {reserved_mb:.2f} MB")
    print(f"  - Total GPU Capacity:     {torch.cuda.get_device_properties(0).total_memory / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()

