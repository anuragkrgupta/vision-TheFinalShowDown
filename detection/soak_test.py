import argparse
from pathlib import Path
from detection.pipeline import DetectionPipeline

def main():
    parser = argparse.ArgumentParser(description="Run live detection or 30-min soak test")
    parser.add_argument("--soak", action="store_true", help="Run a 30 minute soak test without visualization")
    parser.add_argument("--log", type=str, default="soak_metrics.csv", help="CSV log file path")
    args = parser.parse_args()

    pipeline = DetectionPipeline()
    config = pipeline.detector.config
    
    duration = config.get("soak_test_duration_minutes", 30) if args.soak else None
    visualize = not args.soak
    
    print(f"Starting live detection loop...")
    if args.soak:
        print(f"Mode: SOAK TEST ({duration} minutes)")
        print(f"Logging metrics to: {args.log}")
    else:
        print("Mode: DEV LIVE (Press 'q' to quit)")
        
    pipeline.process_live_stream(duration_minutes=duration, visualize=visualize, log_file=args.log)
    
    print("Execution complete.")

if __name__ == "__main__":
    main()
