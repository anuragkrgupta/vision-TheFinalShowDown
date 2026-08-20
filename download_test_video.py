import os
import sys
import subprocess

def install_ytdlp():
    try:
        import yt_dlp
    except ImportError:
        print("Installing yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
        import yt_dlp

def download_video(url, output_name="test_video.mp4"):
    import yt_dlp
    
    print(f"Downloading video from {url}...")
    ydl_opts = {
        # 'b' downloads the best pre-merged file available (mp4 or webm), avoiding ffmpeg!
        'format': 'b',
        'outtmpl': output_name
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    print(f"\nDownload complete! Saved as {output_name}")
    
    # Generate a helper script to run the pipeline on this video
    test_script_content = f"""import os
from detection.pipeline import DetectionPipeline

print("Testing Assistive Vision on '{output_name}'...")
pipeline = DetectionPipeline()
pipeline.process_video_offline(video_path="{output_name}", visualize=True)
"""
    with open("run_video_test.py", "w") as f:
        f.write(test_script_content)
        
    print(f"I also generated 'run_video_test.py'. Run `python run_video_test.py` to test the multi-model pipeline on the video!")

if __name__ == "__main__":
    install_ytdlp()
    
    # A default video search query to find driving footage with traffic and potholes
    default_url = "ytsearch1:driving dashcam india traffic pothole"  
    
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    else:
        print("No URL provided. Using default busy street driving video...")
        print("Usage: python download_test_video.py <YouTube_URL>")
        video_url = default_url
        
    download_video(video_url)
