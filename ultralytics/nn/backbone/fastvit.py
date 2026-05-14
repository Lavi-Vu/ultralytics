import torch
import torch.nn as nn
import timm


class FastViTBackbone(nn.Module):

    def __init__(
        self,
        model_name="fastvit_t8",
        stage_idx=0,
        pretrained=True
    ):
        super().__init__()

        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        self.stage = model

        self.stage_idx = stage_idx

        self.channels = model.feature_info.channels()[stage_idx]

    def forward(self, x):

        features = self.stage(x)

        return features[self.stage_idx]