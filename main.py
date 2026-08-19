import os
from detection.pipeline import DetectionPipeline

def main():
    print("Starting Assistive Vision Software...")
    print("Initializing detection pipeline...")
    pipeline = DetectionPipeline()
    
    print("\nStarting live webcam stream. Press 'q' in the visualizer window to exit.")
    
    # process_live_stream is patched into DetectionPipeline
    # If running headless, set visualize=False
    pipeline.process_live_stream(visualize=True)
    
    print("Software stopped gracefully.")

if __name__ == "__main__":
    # Ensure PYTHONPATH includes the current directory if run directly
    os.environ['PYTHONPATH'] = os.getcwd()
    main()
