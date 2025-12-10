# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import torch
from PIL import Image
import cv2
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.data.augment import classify_transforms
import numpy as np

class MultiLabelClassificationPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a multi label classification model.

    Notes:
        - Torchvision multi label classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.multi_label_classify import MultiLabelClassificationPredictor

        args = dict(model="yolo11n-cls.pt", source=ASSETS)
        predictor = MultiLabelClassificationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes MultiLabelClassificationPredictor setting the task to 'multi_label_classify'."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "multi_label_classify"
        self._legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"

    def setup_source(self, source):
        """Set up source and inference mode and classify transforms."""
        super().setup_source(source)
        updated = (
            self.model.model.transforms.transforms[0].size != max(self.imgsz)
            if hasattr(self.model.model, "transforms") and hasattr(self.model.model.transforms.transforms[0], "size")
            else False
        )
        self.transforms = (
            classify_transforms(self.imgsz) if updated or not self.model.pt else self.model.model.transforms
        )
    def save_tensor_image(self, tensor_img, path="output.jpg"):
        """
        Save a preprocessed model-input tensor back to an image file FOR DEBUG ONLY.
        tensor_img: shape [C, H, W] or [1, C, H, W]
        """

        # Remove batch dimension if needed
        if tensor_img.dim() == 4:
            tensor_img = tensor_img[0]

        # Convert to float32 CPU tensor
        img = tensor_img.detach().float().cpu()

        # Undo normalization if your transforms normalized the images
        # Example for YOLO (0..1 range) – adjust if needed
        img = torch.clamp(img, 0, 1)

        # Convert CHW → HWC and to numpy
        img_np = img.permute(1, 2, 0).numpy()

        # Convert RGB → BGR for cv2
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        cv2.imwrite(path, img_np)
        print(f"Saved: {path}")
        
    def preprocess(self, img):
        """Convert input images to model-compatible tensor format with appropriate normalization."""
        if not isinstance(img, torch.Tensor):
            img = torch.stack(
                [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
            )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        # self.save_tensor_image(img, "saved_preprocessed.jpg")
        return img.half() if self.model.fp16 else img.float()  # Convert uint8 to fp16/32
    
    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
    