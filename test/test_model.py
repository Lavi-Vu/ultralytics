import cv2
from ultralytics import YOLO
from copy import deepcopy
from ultralytics.nn.tasks import parse_model
import argparse

# Load model form args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to the model file')
parser.add_argument("--export", action="store_true", help="Export the model to ONNX format")
parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
args = parser.parse_args()
model = YOLO(args.model)
model.info()
if args.export:
    model.export(format="onnx", opset=args.opset) # Export the model to ONNX format
