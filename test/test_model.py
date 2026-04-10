import cv2
from ultralytics import YOLO
from copy import deepcopy
from ultralytics.nn.tasks import parse_model
# 🔧 Load your models
model = YOLO("/media/lavi/Data/ultralytics/ultralytics/cfg/models/v8/yolo_MobileVitxxs.yaml")
# model= YOLO('/media/lavi/Data/ultralytics/ultralytics/cfg/models/11/yolo11n-cls.yaml')
# model = YOLO('/media/lavi/Data/ultralytics/ultralytics/cfg/models/v8/yolov8-convnext.yaml')
# model = YOLO('/media/lavi/Data/ultralytics/ultralytics/cfg/models/v8/yolov8-MobileNeXt.yaml')
# model= YOLO('/media/lavi/Data/ultralytics/ultralytics/cfg/models/v8/yolov8-fasternet.yaml')
# model= YOLO('/media/lavi/Data/ultralytics/ultralytics/cfg/models/v8/yolov8-EfficentNet.yaml')

# model.info(verbose=True, detailed=False)  # Print model information
# _, _ = parse_model(deepcopy(model.yaml), ch=3, verbose=True)
model.info()
model.export(format="onnx", opset=11) # Export the model to ONNX format
# print(model)