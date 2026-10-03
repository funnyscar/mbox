#CLAUDE

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        # 1x1 projection when shape changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Stem: 224 -> 56
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        # 4 stages x 2 blocks
        self.layer1 = self._stage(64, 64, stride=1)    # 56x56
        self.layer2 = self._stage(64, 128, stride=2)   # 28x28
        self.layer3 = self._stage(128, 256, stride=2)  # 14x14
        self.layer4 = self._stage(256, 512, stride=2)  # 7x7
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _stage(self, in_c, out_c, stride):
        return nn.Sequential(
            BasicBlock(in_c, out_c, stride),
            BasicBlock(out_c, out_c, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.pool(x), 1)
        return self.fc(x)


if __name__ == "__main__":
    model = ResNet18()
    x = torch.randn(2, 3, 224, 224)
    print("Output:", model(x).shape)
    print("Params:", sum(p.numel() for p in model.parameters()))