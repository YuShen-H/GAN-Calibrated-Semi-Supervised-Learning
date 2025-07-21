
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

def weights_init_normal(m: nn.Module) -> None:
    """Initializes `Conv` weights to N(0,0.02) & Batch/InstanceNorm gamma to N(1,0.02)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1 or classname.find("InstanceNorm") != -1:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class UNetDown(nn.Module):
    def __init__(self, in_size: int, out_size: int, normalize: bool = True, dropout: float | None = None):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout: float | None = None):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    """A 4-down/4-up U-Net for predicting a 4-dim bounding box correction Δ."""
    def __init__(self, delta_scale: float = 0.25):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.up1 = UNetUp(512, 256, dropout=0.5)
        self.up2 = UNetUp(512, 128, dropout=0.5)
        self.up3 = UNetUp(256, 64)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_delta = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 4),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3)
        pooled = self.avg_pool(u4)
        delta_raw = self.fc_delta(pooled)
        return delta_raw * self.delta_scale

class GeneratorSimpleRegressor(nn.Module):
    def __init__(self, delta_scale: float = 0.25):
        super().__init__()
        self.delta_scale = delta_scale
        self.regressor = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 4),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.regressor(x) * self.delta_scale

class Discriminator(nn.Module):
    """A PatchGAN discriminator for (pred_patch, other_patch) pairs."""
    def __init__(self, spectral_norm: bool = False):
        super().__init__()
        def conv_block(in_ch: int, out_ch: int, norm: bool = True):
            conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
            if spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers = [conv]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *conv_block(6, 64, norm=False),
            *conv_block(64, 128),
            *conv_block(128, 256),
            *conv_block(256, 512),
            nn.Conv2d(512, 1, 4, stride=1, padding=1, bias=False)
        )

    def forward(self, pred_patch: torch.Tensor, other_patch: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pred_patch, other_patch], dim=1)
        return self.model(x)
