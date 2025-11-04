# import cv2
# from ultralytics import YOLO

# # 🔧 Load your models
# # model = YOLO("/media/lavi/Data/ultralytics/ultralytics/cfg/models/MobileNetV3/MobileNetV3.yaml")
# # model = YOLO('/media/lavi/Data/ultralytics/ultralytics/cfg/models/v8/yolov8-convnext.yaml')
# # model = YOLO('/media/lavi/Data/ultralytics/ultralytics/cfg/models/v8/yolov8-multi-label-cls.yaml')
# model=YOLO('/media/lavi/Data/ultralytics/runs/multi_label_classify/train113/weights/best.pt')

# model.export(format="onnx", opset=11, imgsz=(224,112)) # Export the model to ONNX format
# model.info()  # Print model information



from ultralytics import YOLO

# Load a model

model = YOLO('/media/lavi/Data/ultralytics/ultralytics/cfg/models/v8/yolov8-multi-label-cls.yaml')

# Train the model
results = model.train(data="/media/lavi/Data/ultralytics/ultralytics/cfg/datasets/pa100k.yaml", 
                      epochs=20,
                      rect=True, 
                      imgsz=(224,112), 
                      batch=64, amp=False, 
                      erasing=0.0, 
                      auto_augment=False)