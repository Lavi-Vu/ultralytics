import cv2
from ultralytics import YOLO

# 🔧 Load your models
# model = YOLO("/media/lavi/Data/ultralytics/ultralytics/cfg/models/MobileNetV3/MobileNetV3.yaml")
# model = YOLO('/media/lavi/Data/ultralytics/ultralytics/cfg/models/v8/yolov8-convnext.yaml')
model = YOLO('/media/lavi/Data/ultralytics/ultralytics/cfg/models/v8/yolov8-fasternet.yaml')

model.export(format="onnx", opset=11, imgsz=512) # Export the model to ONNX format
model.info()  # Print model information