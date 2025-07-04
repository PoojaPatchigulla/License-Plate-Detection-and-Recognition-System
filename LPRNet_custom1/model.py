import torch.nn as nn
import torch.nn.functional as F

class LPRNet(nn.Module):
    def __init__(self, num_classes):
        super(LPRNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # -> [64, H, W]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> [64, H/2, W/2]
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> [128, H/2, W/2]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> [128, H/4, W/4]
            nn.Dropout(0.1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # -> [256, H/4, W/4]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, None))  # -> [256, 1, W]
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1),  # -> [num_classes, 1, W]
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.backbone(x)         # [B, 256, 1, W]
        x = self.head(x)             # [B, num_classes, 1, W]
        x = x.squeeze(2)             # [B, num_classes, W]
        x = x.permute(2, 0, 1)       # [W, B, num_classes] for CTC loss

        return x
