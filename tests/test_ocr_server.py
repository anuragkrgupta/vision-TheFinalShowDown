import pytest
from fastapi.testclient import TestClient
from backend.ocr_server import app
import cv2
import numpy as np
import base64

def test_ocr_websocket():
    client = TestClient(app)
    
    # Create a dummy image with some text
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(img, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    ret, buffer = cv2.imencode('.jpg', img)
    assert ret
    
    with client.websocket_connect("/ws/ocr") as websocket:
        websocket.send_bytes(buffer.tobytes())
        data = websocket.receive_json()
        
        # It should either return the OCR data, or an error if tesseract is missing
        if "error" in data:
            assert "Tesseract" in data["error"] or "installed" in data["error"]
        else:
            assert "text" in data
            assert "confidence" in data
            assert isinstance(data["confidence"], (int, float))
