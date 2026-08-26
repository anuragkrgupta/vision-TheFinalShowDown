import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import uvicorn
from detection.pipeline import DetectionPipeline
import sys
import os

app = FastAPI()

# Mount frontend directory for static HTML
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

print("Initializing Detection Pipeline...")
pipeline = DetectionPipeline()
print("Pipeline Initialized!")

@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to WebSocket!")
    try:
        while True:
            data = await websocket.receive_text()
            
            # The browser sends a data URL: 'data:image/jpeg;base64,...'
            if "," in data:
                encoded_data = data.split(",")[1]
            else:
                encoded_data = data
                
            img_bytes = base64.b64decode(encoded_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Mirror the frame horizontally so left/right match the web display
                frame = cv2.flip(frame, 1)
                
                emitted_events, spatial_detections = pipeline.process_frame(frame)
                
                # Format response with standard python float (json serializable)
                # Ensure numpy types are converted
                clean_detections = []
                for d in spatial_detections:
                    clean_detections.append({
                        "class_name": str(d["class_name"]),
                        "confidence": float(d["confidence"]),
                        "zone": str(d["zone"]),
                        "proximity": str(d["proximity"]),
                        "bbox": [float(x) for x in d["bbox"]]
                    })

                await websocket.send_json({
                    "detections": clean_detections
                })
            else:
                await websocket.send_json({"detections": []})
                
    except Exception as e:
        print(f"WebSocket closed or error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server... open your browser to: http://127.0.0.1:{port}/frontend/index.html")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
