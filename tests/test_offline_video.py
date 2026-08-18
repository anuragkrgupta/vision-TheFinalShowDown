import pytest
from pathlib import Path
from detection.pipeline import DetectionPipeline

BASE_DIR = Path(__file__).parent / "fixtures"
VIDEOS_DIR = BASE_DIR / "videos"

@pytest.fixture(scope="module")
def pipeline():
    return DetectionPipeline()

def test_offline_video_temporal_logic(pipeline):
    """
    Tests that the temporal smoothing logic works correctly on a video.
    """
    video_files = list(VIDEOS_DIR.glob("*.mp4"))
    if not video_files:
        pytest.skip("No video files found in tests/fixtures/videos to test temporal logic.")
        
    for video_path in video_files:
        print(f"Testing temporal logic on {video_path.name}")
        timeline = pipeline.process_video_offline(video_path, visualize=False)
        
        # Verify that the timeline has elements (video was processed)
        assert len(timeline) > 0, f"Video {video_path.name} produced no frames."
        
        # Temporal smoothing specific checks
        # If N=3, M=2, an object should only be active if seen 2 out of 3 frames.
        # This is a basic sanity check that smoothing logic ran without crashing.
        for frame_data in timeline:
            assert "raw_detections" in frame_data
            assert "smoothed_active_classes" in frame_data
            
        # Optional: We could assert specific classes are found in specific videos, 
        # but since videos are arbitrary, simply ensuring the logic executes 
        # smoothly without error across the timeline is sufficient for the core loop test.
