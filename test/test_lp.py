import cv2
from ultralytics import YOLO

# Load the main vehicle detector + tracker model
vehicle_model = YOLO("/home/lavi/Documents/ultralytic/best_vh_512_2006.pt")

# Load the secondary model (e.g., license plate, fine-grained detection)
license_plate_model = YOLO("/home/lavi/sshfs/best_lp_128.pt")

# Open RTSP stream
rtsp_url = "rtsp://192.169.1.251/stream1"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Step 1: Vehicle detection + tracking
    results = vehicle_model.track(source=frame, persist=True, conf=0.25, imgsz=512)

    annotated_frame = frame.copy()

    if results and results[0].boxes is not None:
        boxes = results[0].boxes

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            # Draw vehicle box in GREEN
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Vehicle ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Crop vehicle
            cropped_vehicle = frame[y1:y2, x1:x2]
            if cropped_vehicle.size == 0:
                continue

            # Step 2: License plate detection on vehicle crop
            secondary_results = license_plate_model.predict(source=cropped_vehicle, conf=0.3, imgsz=128, verbose=False)

            if secondary_results and secondary_results[0].boxes is not None:
                lp_boxes = secondary_results[0].boxes

                for lp_box in lp_boxes:
                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box.xyxy[0])

                    # Translate license plate box to original frame coordinates
                    abs_x1 = x1 + lp_x1
                    abs_y1 = y1 + lp_y1
                    abs_x2 = x1 + lp_x2
                    abs_y2 = y1 + lp_y2

                    # Draw license plate box in RED
                    cv2.rectangle(annotated_frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, 'LP', (abs_x1, abs_y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show final frame with both sets of boxes
    annotated_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("Vehicle + License Plate Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()