import cv2

def test_camera():
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
        
    ret, frame = cap.read()
    if ret:
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        shape = frame.shape
        print("Success: Camera grabbed a frame successfully!")
        print(f"Timestamp: {timestamp}")
        print(f"Frame Shape: {shape}")
        cv2.imwrite("test_frame.jpg", frame)
        print("Saved test frame to 'test_frame.jpg'")
    else:
        print("Error: Could not read frame from camera.")
        
    cap.release()
    return ret

if __name__ == "__main__":
    test_camera()
