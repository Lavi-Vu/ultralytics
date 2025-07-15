import cv2
from ultralytics import YOLO

# 🔧 Load your models
model = YOLO("/home/lavi/Documents/ultralytics/ultralytics/cfg/models/mbn_yolo.yaml")          # your license plate detector

model.info()  # Print model information