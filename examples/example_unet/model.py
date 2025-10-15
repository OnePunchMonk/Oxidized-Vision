import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
