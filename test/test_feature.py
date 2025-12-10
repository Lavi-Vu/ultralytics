from ultralytics import YOLO
import torch
import cv2
import os

# model = YOLO("runs/multi_label_classify/train16/weights/best.pt")
# model = YOLO("/media/lavi/Data/ultralytics/ultralytics/cfg/models/v8/yolov8-fasternet.yaml")
model = YOLO("/media/lavi/Data/ultralytics/ultralytics/cfg/models/v8/yolov8-multi-label-cls.yaml")
# img_path = "/home/lavi/Downloads/a5d55df7-02b6-427d-ad48-be0703a4fe58.jpeg"
img_path = "test.jpg"
# img_path = "/home/lavi/Pictures/data_human_attr/data_gen/CCTV_person_resize/male/glasses_backpack_pants_upperLongSleeve_upred_lowerLong_downpink_1.png"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.tensor(img_rgb).permute(2,0,1).float().unsqueeze(0) / 255.0

os.makedirs("features", exist_ok=True)

feature_maps = []

def hook_fn(module, input, output):
    # Một số layer trả về tuple → lấy output chính
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        return
    feature_maps.append(output.cpu().detach())

hooks = []
for layer in model.model.model:
    hooks.append(layer.register_forward_hook(hook_fn))

with torch.no_grad():
    _ = model.model(img_tensor)

for h in hooks:
    h.remove()

print("Số feature map lấy được:", len(feature_maps))



import numpy as np
import cv2
import torch

# for i, fmap in enumerate(feature_maps):

#     # Bỏ qua feature không phải tensor
#     if not torch.is_tensor(fmap):
#         print(f"Skip layer {i} (not tensor)")
#         continue

#     fmap = fmap.squeeze(0)  # remove batch dim

#     # fmap shape phải là (C,H,W)
#     if fmap.ndim != 3:
#         print(f"Skip layer {i} (invalid ndim: {fmap.ndim}, shape: {fmap.shape})")
#         continue

#     C, H, W = fmap.shape

#     # lấy kênh đầu tiên
#     fm = fmap[0].cpu().numpy()

#     # normalize về 0–255
#     fm_min, fm_max = fm.min(), fm.max()
#     if fm_max - fm_min < 1e-6:
#         print(f"Skip layer {i} (flat feature map)")
#         continue

#     fm = (fm - fm_min) / (fm_max - fm_min)
#     fm = (fm * 255).astype(np.uint8)

#     # lưu ảnh
#     save_path = f"features/layer_{i}.png"
#     cv2.imwrite(save_path, fm)
#     print("Saved", save_path)

scale = 16  # scale factor, có thể chỉnh lớn hơn nếu muốn

for i, fmap in enumerate(feature_maps):
    if not torch.is_tensor(fmap):
        continue

    fmap = fmap.squeeze(0)  # remove batch dim
    if fmap.ndim != 3:
        continue

    C, H, W = fmap.shape
    fm = fmap[0].cpu().numpy()  # lấy channel đầu tiên

    fm_min, fm_max = fm.min(), fm.max()
    if fm_max - fm_min < 1e-6:
        continue

    fm = (fm - fm_min) / (fm_max - fm_min) * 255
    fm = fm.astype(np.uint8)

    # resize ảnh lên lớn hơn
    fm_large = cv2.resize(fm, (W*scale, H*scale), interpolation=cv2.INTER_NEAREST)

    save_path = f"features/layer_{i}_large.png"
    cv2.imwrite(save_path, fm_large)
    print("Saved", save_path)
