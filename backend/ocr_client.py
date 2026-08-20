import asyncio
import websockets
import threading
import json
import cv2
import time
from audio.voice_engine import VoiceEvent

class OCRClient:
    def __init__(self, audio_engine=None):
        self.uri = "ws://127.0.0.1:8000/ws/ocr"
        self.audio_engine = audio_engine
        self.loop = asyncio.new_event_loop()
        
        # Start the asyncio loop in a separate thread
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    async def _send_frame_async(self, frame):
        try:
            async with websockets.connect(self.uri) as websocket:
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    return
                
                # Send binary data
                await websocket.send(buffer.tobytes())
                
                # Wait for response
                response_str = await websocket.recv()
                response = json.loads(response_str)
                
                if "error" in response:
                    print(f"OCR Error: {response['error']}")
                    if self.audio_engine:
                        self.audio_engine.enqueue(VoiceEvent(
                            priority=3,
                            timestamp=time.time(),
                            text="OCR error.",
                            class_name="system",
                            zone="Center"
                        ))
                else:
                    text = response.get("text", "")
                    if text and self.audio_engine:
                        self.audio_engine.enqueue(VoiceEvent(
                            priority=2,
                            timestamp=time.time(),
                            text=f"Sign reads: {text}",
                            class_name="sign",
                            zone="Center"
                        ))
                        print(f"[OCR Success] {text}")
                    elif self.audio_engine:
                        self.audio_engine.enqueue(VoiceEvent(
                            priority=3,
                            timestamp=time.time(),
                            text="No clear text detected.",
                            class_name="system",
                            zone="Center"
                        ))
        except (websockets.exceptions.WebSocketException, ConnectionRefusedError, OSError) as e:
            print(f"OCR Backend unreachable: {e}")
            if self.audio_engine:
                self.audio_engine.enqueue(VoiceEvent(
                    priority=3,
                    timestamp=time.time(),
                    text="OCR disconnected.",
                    class_name="system",
                    zone="Center"
                ))
        except Exception as e:
            print(f"Unexpected OCR Error: {e}")
            
    def send_frame(self, frame):
        """Non-blocking call to send a frame for OCR processing."""
        # Fire and forget in the asyncio loop
        asyncio.run_coroutine_threadsafe(self._send_frame_async(frame), self.loop)
