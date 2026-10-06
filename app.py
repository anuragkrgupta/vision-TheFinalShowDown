import streamlit as st
import cv2
import av
from detection.pipeline import DetectionPipeline
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Vision Assistant", layout="wide")

st.title("Vision Assistant - Real-Time Detection")
st.write("This application runs a live computer-vision detection pipeline directly in your browser using WebRTC.")

# Cache the pipeline so it is initialized only once and not on every UI interaction.
# This prevents the heavy ML models from being reloaded.
@st.cache_resource
def load_pipeline():
    return DetectionPipeline()

pipeline = load_pipeline()

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert incoming WebRTC frame to NumPy/OpenCV BGR format
        img = frame.to_ndarray(format="bgr24")

        # Flip horizontally to match the natural webcam mirror effect
        img = cv2.flip(img, 1)

        # Run the existing DetectionPipeline inference and spatial logic
        emitted_events, spatial_detections = pipeline.process_frame(img)

        # Draw bounding boxes and text overlays using OpenCV
        for d in spatial_detections:
            # Parse bounding box coordinates
            x1, y1, x2, y2 = map(int, d["bbox"])
            
            # Draw green bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Construct label with class name, confidence, zone, and proximity
            label = f"{d['class_name']} {d['confidence']:.2f} ({d['zone']}, {d['proximity']})"
            
            # Draw label above the bounding box
            cv2.putText(
                img, 
                label, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )

        # Provide a visual indicator for recently emitted audio events
        if emitted_events:
            event_classes = [e['class_name'] for e in emitted_events]
            event_text = f"Emitted: {', '.join(event_classes)}"
            cv2.putText(
                img, 
                event_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )

        # Return the processed frame back to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown("### Live Webcam Feed")

# Build ICE server configuration
ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

# Add TURN server if credentials are provided in Streamlit Secrets
if hasattr(st, "secrets") and "TURN_SERVER" in st.secrets:
    ice_servers.append({
        "urls": [st.secrets["TURN_SERVER"]],
        "username": st.secrets["TURN_USERNAME"],
        "credential": st.secrets["TURN_CREDENTIAL"],
    })

# Streamlit-WebRTC component to handle the video stream
webrtc_streamer(
    key="vision-assistant",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": ice_servers
    },
    media_stream_constraints={"video": True, "audio": False},
)
