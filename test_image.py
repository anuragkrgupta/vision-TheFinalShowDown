import cv2
from detection.pipeline import DetectionPipeline

pipeline = DetectionPipeline()
img_path = r'C:\Users\kumar\Desktop\VISION\dataset\WIN_20260827_04_13_10_Pro.jpg'
img = cv2.imread(img_path)

if img is None:
    print('Failed to load image:', img_path)
else:
    emitted, spatial = pipeline.process_frame(img)
    print('\n================ RESULTS ================')
    if not spatial:
        print('No objects detected.')
    for d in spatial:
        print(f"Detected: {d['class_name']}")
        print(f"  Confidence: {d['confidence']:.2f}")
        print(f"  Zone: {d['zone']}")
        print(f"  Proximity: {d['proximity']}")
        
    # Also save the image with bounding boxes to see the visual output
    for d in spatial:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    output_path = 'test_output.jpg'
    cv2.imwrite(output_path, img)
    print(f'Saved visualized output to {output_path}')
    print('=========================================')
