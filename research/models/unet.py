"""Compact U-Net encoder–decoder with skip connections (classic U-Net).

Each encoder/decoder block uses convolution → **BatchNorm2d** → ReLU (see ``_DoubleConv``).
Decoder blocks concatenate upsampled features with matching encoder skips (channel dim).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Encoder–decoder with skip connections (upsample ⊕ encoder feature → decoder conv)."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = _DoubleConv(in_channels, c)
        self.enc2 = _DoubleConv(c, c * 2)
        self.enc3 = _DoubleConv(c * 2, c * 4)
        self.enc4 = _DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = _DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = _DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = _DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = _DoubleConv(c * 2, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.head(d1)
