"""A compact single-stage (YOLO-style) object detector for OxidizedVision examples.

A real, working anchor-free detection architecture: a small strided-conv
backbone downsamples the input by 32x, then a detection head predicts, per
grid cell, `num_anchors` boxes each with (cx, cy, w, h, objectness,
class_logits...). This mirrors the grid-based prediction structure YOLO
models use, at a scale that's fast to export/convert/benchmark as an example
— for a production detector, swap in a deeper backbone (e.g. CSPDarknet) and
multi-scale heads (FPN/PAN), and add anchor decoding + NMS as a
post-processing step (not part of the exported graph, same as this example).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv + BatchNorm + SiLU, the standard YOLO-family building block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Backbone(nn.Module):
    """Strided-conv backbone, downsampling 32x (5 stride-2 stages)."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.stem = ConvBlock(in_channels, 16, stride=2)  # /2
        self.stage1 = ConvBlock(16, 32, stride=2)  # /4
        self.stage2 = ConvBlock(32, 64, stride=2)  # /8
        self.stage3 = ConvBlock(64, 128, stride=2)  # /16
        self.stage4 = ConvBlock(128, 256, stride=2)  # /32

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class DetectionHead(nn.Module):
    """Per-cell grid predictions: `num_anchors` boxes x (4 box + 1 obj + num_classes)."""

    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        out_channels = num_anchors * (5 + num_classes)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class TinyYOLO(nn.Module):
    """Compact single-scale, anchor-based detector.

    Output shape: [B, num_anchors * (5 + num_classes), H/32, W/32].
    Each of the `num_anchors` predictions per cell is
    (tx, ty, tw, th, objectness_logit, class_logits...) in YOLO's
    grid-relative parameterization — decode with the standard YOLO
    sigmoid(tx)/sigmoid(ty) + cell offset, exp(tw)/exp(th) * anchor formula
    at inference time (not included here, same as this repo's UNet example
    leaves argmax/softmax decoding to the caller).
    """

    def __init__(self, num_classes: int = 80, num_anchors: int = 3):
        super().__init__()
        self.backbone = Backbone()
        self.head = DetectionHead(256, num_classes, num_anchors)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


if __name__ == "__main__":
    model = TinyYOLO(num_classes=80, num_anchors=3)
    dummy = torch.randn(1, 3, 320, 320)
    output = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {output.shape}")  # [1, 3*(5+80), 10, 10] = [1, 255, 10, 10]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
