import cv2
from ultralytics import YOLO

# 🔧 Load your models
# model = YOLO("/media/lavi/Data/ultralytics/ultralytics/cfg/models/MobileNetV3/MobileNetV3.yaml")
model = YOLO('/media/lavi/Data/ultralytics/ultralytics/cfg/models/yolov8_BiFPN.yaml')

model.info()  # Print model information