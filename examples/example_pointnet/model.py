"""A PointNet-style point cloud classifier for OxidizedVision examples.

3D vision workloads are a real deployment target for this toolkit too —
point clouds from LiDAR/depth sensors, not just 2D images. PointNet
(Qi et al., 2017) is the canonical "vision on point sets" architecture: it
applies a shared per-point MLP (implemented as 1D convolutions) to lift each
point into a high-dimensional feature, then uses a symmetric (order-
invariant) global max-pool to aggregate an unordered point cloud into a
single feature vector for classification.

This is the vanilla PointNet backbone (shared-MLP + global max-pool +
classifier head), without the T-Net input/feature alignment networks —
those use a matrix-multiply-based spatial transform that is harder to
export cleanly across TorchScript/ONNX/tract/ONNX Runtime and isn't needed
to demonstrate the same conversion/benchmark/serve pipeline as the other
examples in this repo.
"""

import torch
import torch.nn as nn


class SharedMLP(nn.Module):
    """A per-point Conv1d + BatchNorm + ReLU — the PointNet "shared MLP" building block.

    Applying a 1x1 Conv1d across the point dimension is equivalent to
    applying the same MLP independently to every point, which is what makes
    PointNet permutation-invariant to point order.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PointNetClassifier(nn.Module):
    """Point cloud classifier: shared MLPs -> global max-pool -> MLP head.

    Input: [B, 3, N] — a batch of point clouds, each N points with (x, y, z)
    coordinates. Output: [B, num_classes] class logits.
    """

    def __init__(self, num_classes: int = 10, num_points: int = 1024):
        super().__init__()
        self.num_points = num_points

        self.mlp1 = nn.Sequential(
            SharedMLP(3, 64),
            SharedMLP(64, 64),
        )
        self.mlp2 = nn.Sequential(
            SharedMLP(64, 64),
            SharedMLP(64, 128),
            SharedMLP(128, 1024),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: [B, 3, N]
        x = self.mlp1(x)  # [B, 64, N]
        x = self.mlp2(x)  # [B, 1024, N]
        x = torch.max(x, dim=2).values  # symmetric global feature: [B, 1024]
        return self.classifier(x)


if __name__ == "__main__":
    model = PointNetClassifier(num_classes=10, num_points=1024)
    model.eval()
    dummy = torch.randn(1, 3, 1024)
    output = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
