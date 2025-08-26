import cv2
from ultralytics import YOLO

# 🔧 Load your models
model = YOLO("/media/lavi/Data/ultralytics/ultralytics/cfg/models/mbn_yolo_nano.yaml")

model.info()  # Print model information