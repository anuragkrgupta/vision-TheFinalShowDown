import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json

app = FastAPI()

# Specify tesseract path explicitly since it's commonly required on Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.websocket("/ws/ocr")
async def ocr_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("OCR Client connected.")
    try:
        while True:
            # Receive binary frame (JPEG)
            data = await websocket.receive_bytes()
            
            # Decode image
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                await websocket.send_json({"error": "Invalid image data"})
                continue

            try:
                # Convert to grayscale for better OCR
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Get detailed data including confidence
                ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
                
                # Filter out low confidence and empty text
                valid_words = []
                confidences = []
                
                for i in range(len(ocr_data['text'])):
                    text = ocr_data['text'][i].strip()
                    conf = int(ocr_data['conf'][i])
                    
                    if text and conf > 40: # Threshold of 40% confidence
                        valid_words.append(text)
                        confidences.append(conf)
                
                if valid_words:
                    full_text = " ".join(valid_words)
                    avg_conf = sum(confidences) / len(confidences)
                    await websocket.send_json({
                        "text": full_text,
                        "confidence": avg_conf
                    })
                else:
                    await websocket.send_json({
                        "text": "",
                        "confidence": 0
                    })
            except pytesseract.TesseractNotFoundError:
                await websocket.send_json({
                    "error": "Tesseract is not installed on the system. Please install from https://github.com/UB-Mannheim/tesseract/wiki and restart."
                })
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("OCR Client disconnected.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
