from pathlib import Path
from detection.pipeline import DetectionPipeline

def main():
    base_dir = Path(__file__).parent / "fixtures"
    videos_dir = base_dir / "videos"
    
    pipeline = DetectionPipeline()
    
    video_files = list(videos_dir.glob("*.mp4"))
    if not video_files:
        print("No videos found in tests/fixtures/videos.")
        return
        
    print(f"Found {len(video_files)} videos. Starting visual playback...")
    print("Press 'q' during playback to skip to the next video.\n")
    
    for video_path in video_files:
        print(f"Playing: {video_path.name}")
        pipeline.process_video_offline(video_path, visualize=True)
        print(f"Finished: {video_path.name}\n")
        
    print("All videos completed.")

if __name__ == "__main__":
    main()
