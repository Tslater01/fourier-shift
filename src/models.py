import torch
import torchvision
from torch import nn
from torch.nn import functional as F

CONVNEXT_RES = 128  # keeps the /4 stem + 3 downsamples geometry (4x4 final map); 224 is pure upsampling cost from 32x32


def cifar_resnet18(num_classes=2, in_ch=3):
    m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(in_ch, 64, 3, stride=1, padding=1, bias=False)  # 32x32 stem: no 7x7/2, no maxpool
    m.maxpool = nn.Identity()
    return m


class ConvNeXtFT(nn.Module):
    def __init__(self, num_classes=2, in_ch=3):
        super().__init__()
        self.net = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1")
        self.net.classifier[2] = nn.Linear(768, num_classes)
        if in_ch != 3:
            w = self.net.features[0][0].weight.data.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
            self.net.features[0][0] = nn.Conv2d(in_ch, 96, 4, 4)
            self.net.features[0][0].weight.data = w

    def forward(self, x):
        return self.net(F.interpolate(x, CONVNEXT_RES, mode="bilinear", align_corners=False))


class TwoStream(nn.Module):
    def __init__(self, num_classes=2, in_ch=3):
        super().__init__()
        self.rgb, self.fft = cifar_resnet18(), cifar_resnet18(in_ch=in_ch)
        self.rgb.fc, self.fft.fc = nn.Identity(), nn.Identity()
        self.head = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, a, b):
        return self.head(torch.cat([self.rgb(a), self.fft(b)], 1))


MODELS = {"resnet18": cifar_resnet18, "convnext": ConvNeXtFT, "twostream": TwoStream}
