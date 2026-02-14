"""A real U-Net implementation for OxidizedVision examples.

This is a proper encoder-decoder architecture with skip connections,
suitable for image segmentation and image-to-image tasks.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two conv layers with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling: MaxPool then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling: Upsample then DoubleConv with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if necessary (handles non-divisible-by-2 dimensions)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                      diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net: Encoder-Decoder with skip connections.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB).
        out_channels: Number of output channels (default: 1 for segmentation mask).
        base_features: Number of features in the first layer (doubles in each level).
        bilinear: Use bilinear upsampling (True) or transposed convolutions (False).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_features: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        f = base_features
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16 // factor)

        # Decoder
        self.up1 = Up(f * 16, f * 8 // factor, bilinear)
        self.up2 = Up(f * 8, f * 4 // factor, bilinear)
        self.up3 = Up(f * 4, f * 2 // factor, bilinear)
        self.up4 = Up(f * 2, f, bilinear)

        # Output
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    dummy = torch.randn(1, 3, 256, 256)
    output = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
