import cv2
from ultralytics import YOLO
from copy import deepcopy
from ultralytics.nn.tasks import parse_model
# 🔧 Load your models
model= YOLO('/home/lavi/Documents/ultralytics/ultralytics/cfg/models/edge.yaml')

# model.info(verbose=True, detailed=False)  # Print model information
# _, _ = parse_model(deepcopy(model.yaml), ch=3, verbose=True)
model.info()
model.export(format="onnx") # Export the model to ONNX format
# print(model)