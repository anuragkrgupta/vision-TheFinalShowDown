import os
import requests
from pathlib import Path

# URLs of sample images (public domain / open source)
PHOTO_URLS = [
    ("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", "bus_street.jpg"),
    ("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg", "people_zidane.jpg"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Crosswalk_in_pedestrian_perspective.jpg/800px-Crosswalk_in_pedestrian_perspective.jpg", "crosswalk.jpg"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Bicycles_in_Amsterdam.jpg/800px-Bicycles_in_Amsterdam.jpg", "bicycles.jpg")
]

# URLs of sample videos (from intel-iot-devkit sample videos)
VIDEO_URLS = [
    ("https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4", "person_bicycle_car.mp4"),
    ("https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4", "pedestrians.mp4")
]

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Success!")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    base_dir = Path(__file__).parent / "fixtures"
    photos_dir = base_dir / "photos"
    videos_dir = base_dir / "videos"
    
    photos_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    for url, filename in PHOTO_URLS:
        dest = photos_dir / filename
        if not dest.exists():
            download_file(url, dest)
        else:
            print(f"{filename} already exists, skipping.")
            
    for url, filename in VIDEO_URLS:
        dest = videos_dir / filename
        if not dest.exists():
            download_file(url, dest)
        else:
            print(f"{filename} already exists, skipping.")

    print("Media download complete.")

if __name__ == "__main__":
    main()
