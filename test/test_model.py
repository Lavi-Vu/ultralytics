import cv2
from ultralytics import YOLO
from copy import deepcopy
from ultralytics.nn.tasks import parse_model
import argparse
# # 🔧 Load your models
# model= YOLO('/home/lavi/Documents/ultralytics/ultralytics/cfg/models/v8/my_custom.yaml')

# # model.info(verbose=True, detailed=False)  # Print model information
# # _, _ = parse_model(deepcopy(model.yaml), ch=3, verbose=True)
# model.info()
# model.export(format="onnx") # Export the model to ONNX format
# # print(model)

# Load model form args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to the model file')
args = parser.parse_args()
model = YOLO(args.model)
model.info()
model.export(format="onnx") # Export the model to ONNX format
