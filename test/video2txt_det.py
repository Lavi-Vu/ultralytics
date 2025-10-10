import cv2
from ultralytics import YOLO

# Label mapping
label_map = {
    0: "BSD",
    1: "BSV",
    2: "BUS",
    3: "CAR",
    4: "HUMAN",
    5: "MOTOR",
    6: "TRUCK"
}

# Load YOLO model (replace 'yolov8n.pt' with your model if needed)
model = YOLO('/home/lavi/Documents/yolov11m_1280_vh_01_10_25/weights/best.pt')

# Open video file
video_path = '/home/lavi/Videos/vph.mp4'  # Change to your video path
cap = cv2.VideoCapture(video_path)

# Output txt file
output_txt = '/home/lavi/Videos/detections.txt'
f = open(output_txt, 'w')

frame_id = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label_idx = int(box.cls[0])
            label = label_map.get(label_idx, str(label_idx))
            # Convert to <x> <y> <width> <height>
            x = x1
            y = y1
            width = x2 - x1
            height = y2 - y1
            f.write(f"{frame_id} {label} {x:.1f} {y:.1f} {width:.1f} {height:.1f}\n")

    frame_id += 1

cap.release()
f.close()
print(f"Detections written to {output_txt}")