import cv2
import numpy as np
import time

from ultralytics import YOLO

# Load the YOLO model
det_model = YOLO("/home/lavi/Documents/fasternet_relu6/weights/best.pt")
human_attr_model = YOLO("runs/multi_label_classify/train20/weights/best.pt")
# Open the video file
# video_path = "rtsp://admin:Mkv@1234@14.224.163.142:551/stream0"
video_path = "rtsp://admin:Mkv@1234@14.224.186.46:551/stream0"
# video_path = "rtsp://admin:Mkv@1234@14.224.186.46:552/stream0"
# video_path = "test_video.mp4"
record = False
cap = cv2.VideoCapture(video_path)
# Output video settings
out_dir = "runs/human_attr"
out_name = "inference_1.mp4"
out_path = f"{out_dir}/{out_name}"
cv2.os.makedirs(out_dir, exist_ok=True)
writer = None

# FPS measurement
prev_time = 0.0
fps = 0.0
avg_fps = 0.0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run detection model on the frame (person class only)
        results = det_model(frame, classes=[1], iou=0.4)

        # Prepare a copy of the frame to draw on
        annotated_frame = frame.copy()

        if len(results) > 0:
            det = results[0]
            boxes = getattr(det, 'boxes', None)

            if boxes is not None and len(boxes) > 0:
                # iterate through detected boxes
                for box in boxes:
                    try:
                        # get xyxy and confidence
                        xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
                        conf = float(box.conf.cpu().numpy()) if hasattr(box.conf, 'cpu') else float(box.conf)
                    except Exception:
                        # fallback if fields differ
                        xyxy = box.xyxy if isinstance(box.xyxy, (list, tuple)) else box.xyxy.cpu().numpy().astype(int).flatten()
                        conf = float(getattr(box, 'conf', 0.0))

                    # skip low-confidence detections
                    if conf < 0.3:
                        continue

                    x1, y1, x2, y2 = xyxy
                    # clamp coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

                    # crop the person region
                    crop = frame[y1:y2, x1:x2]
                    if crop is None or crop.size == 0:
                        continue

                    # run attribute model on the cropped person
                    labels = []
                    names = getattr(human_attr_model, 'names', None) or {}
                    attr_results = human_attr_model(crop, imgsz=[128,64])
                    probs = None
                    # direct probs on the returned object
                    
                    probs = attr_results[0].probs.data.cpu().numpy()
                    
                    if probs is not None:
                        names = getattr(human_attr_model, 'names', None) or {}
                        for i, p in enumerate(probs):
                            if p > 0.1:
                                if isinstance(names, (list, tuple)) and i < len(names):
                                    name = names[i]
                                elif isinstance(names, dict):
                                    name = names.get(i, str(i))
                                else:
                                    name = str(i)
                                labels.append(f"{name}:{p:.2f}")
                    
                    # draw detection box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # draw labels above the box
                    if len(labels) > 0:
                        font_scale = 0.8          # increased font size
                        thickness = 2             # thicker text
                        line_height = 25          # increased line height
                        
                        tx, ty = x1 + 5, y1 + 30  # padding inside bbox

                        for lbl in labels:
                            # Calculate text size
                            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                            # Draw background for the text
                            cv2.rectangle(
                                annotated_frame,
                                (tx - 3, ty - th - 3),
                                (tx + tw + 3, ty + 3),
                                (0, 255, 0),
                                -1
                            )

                            # Draw the text label
                            cv2.putText(
                                annotated_frame,
                                lbl,
                                (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                (0, 0, 0),
                                thickness,
                                cv2.LINE_AA
                            )

                            ty += line_height  # Move to the next line

        # compute FPS
        now = time.time()
        if prev_time > 0:
            dt = now - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            # smooth FPS for a steadier readout
            if avg_fps == 0.0:
                avg_fps = fps
            else:
                avg_fps = avg_fps * 0.9 + fps * 0.1
        prev_time = now

        # overlay FPS on frame
        fps_text = f"FPS: {avg_fps:.2f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.namedWindow("YOLO Inference", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLO Inference", annotated_frame)
        if record :
        # Initialize VideoWriter once we have a frame size (do it lazily)
            if writer is None:
                # try to get fps, fall back to 25 if unavailable
                fps = cap.get(cv2.CAP_PROP_FPS)
                try:
                    fps = float(fps) if fps and fps > 0 else 25.0
                except Exception:
                    fps = 25.0
                height, width = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                if not writer.isOpened():
                    print(f"Warning: failed to open VideoWriter for {out_path}")

            # Write annotated frame to output if writer is available
            if writer is not None and writer.isOpened():
                # Ensure frame is BGR uint8
                frame_to_write = annotated_frame
                if frame_to_write.dtype != "uint8":
                    frame_to_write = (frame_to_write * 255).astype("uint8")
                writer.write(frame_to_write)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()