import torch
import torch.nn as nn
from mamba_ssm import Mamba    

class MambaTracker(nn.Module):
    def __init__(self, d_model=128, d_state=8, d_conv=2, expand=1):
        super().__init__()
        self.backbone = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.conv1 = nn.Conv2d(3, d_model, kernel_size=3, padding=1)
        self.det_head = nn.Conv2d(d_model, 5, kernel_size=1)  # [x, y, w, h, conf]
        self.id_head = nn.Conv2d(d_model, 64, kernel_size=1)   # Reduced ID embedding
        self.pool = nn.AdaptiveAvgPool2d((16, 16))  # Downsample
        print(f"Initialized MambaTracker with d_model={d_model}, d_state={d_state}, d_conv={d_conv}, expand={expand}")

    def forward(self, x):
        print(f"Input shape to MambaTracker: {x.shape}")
        x = self.conv1(x)
        print(f"After conv1: {x.shape}")
        x = self.pool(x)
        print(f"After pool: {x.shape}")
        B, C, H, W = x.shape
        seq = x.view(B, C, H * W).permute(2, 0, 1)
        print(f"Sequence shape before Mamba: {seq.shape}")
        features = self.backbone(seq)
        print(f"After Mamba backbone: {features.shape}")
        features = features.permute(1, 2, 0).view(B, C, H, W)
        print(f"Reshaped features: {features.shape}")
        bboxes = self.det_head(features)
        print(f"Bounding boxes shape: {bboxes.shape}")
        ids = self.id_head(features)
        print(f"IDs shape: {ids.shape}")
        return bboxes, ids