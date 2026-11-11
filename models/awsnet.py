"""Canonical AWSnet (Attention U-Net with Auto-Weighted Deep Supervision) Architecture.

Reference Paper:
Wang, Kai-Ni, et al. "AWSnet: an auto-weighted supervision attention network for
myocardial scar and edema segmentation in multi-sequence cardiac magnetic resonance images."
Medical Image Analysis 77 (2022): 102362.

Benchmark Protocol: myops380_train304_test76_last_v1
Model ID: awsnet
Checkpoint: <save_dir>/awsnet_seed<seed>.pt
"""
from __future__ import annotations

from typing import Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Dual 3x3 Conv + BatchNorm + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):
    """Upsampling followed by 3x3 Conv + BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AttentionBlock(nn.Module):
    """Additive Attention Gate Block (Oktay et al., 2018)."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AWSnet(nn.Module):
    """AWSnet Segmentation Model: 2D Attention U-Net with 4 Deep Supervision Stages.

    Args:
        in_channels: Number of input channels (default 3 for bSSFP, LGE, T2w).
        num_classes: Number of target segmentation classes (default 4 for MyoPS380).
        base_filters: Number of initial feature channels (default 32).
        deep_supervision: Whether to output multi-scale deep supervision heads in train mode.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        base_filters: int = 32,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.model_id = "awsnet"

        n1 = base_filters
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # [32, 64, 128, 256, 512]

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(in_channels, filters[0])
        self.Conv2 = ConvBlock(filters[0], filters[1])
        self.Conv3 = ConvBlock(filters[1], filters[2])
        self.Conv4 = ConvBlock(filters[2], filters[3])
        self.Conv5 = ConvBlock(filters[3], filters[4])

        self.Up5 = UpConv(filters[4], filters[3])
        self.Att5 = AttentionBlock(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = ConvBlock(filters[4], filters[3])

        self.Up4 = UpConv(filters[3], filters[2])
        self.Att4 = AttentionBlock(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = ConvBlock(filters[3], filters[2])

        self.Up3 = UpConv(filters[2], filters[1])
        self.Att3 = AttentionBlock(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = ConvBlock(filters[2], filters[1])

        self.Up2 = UpConv(filters[1], filters[0])
        self.Att2 = AttentionBlock(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = ConvBlock(filters[1], filters[0])

        self.final3 = nn.Conv2d(filters[3], num_classes, kernel_size=1, stride=1, padding=0)
        self.final2 = nn.Conv2d(filters[2], num_classes, kernel_size=1, stride=1, padding=0)
        self.final1 = nn.Conv2d(filters[1], num_classes, kernel_size=1, stride=1, padding=0)
        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, *images: torch.Tensor, return_all: Optional[bool] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass supporting single concatenated tensor or tuple of modal tensors.

        Args:
            *images: Either 1 tensor (B, 3, H, W) or 3 tensors (B, 1, H, W) for (c0, lge, t2).
            return_all: If True, returns all 4 deep supervision stages + primary output.
                        If False, returns only the primary output logits.
                        If None, returns all stages when self.deep_supervision and self.training, else primary.

        Returns:
            If return_all or (self.deep_supervision and self.training and return_all is not False):
                Tuple: (output4, output3, output2, output1, output)
            Otherwise:
                Primary output logits tensor of shape (B, num_classes, H, W).
        """
        if len(images) == 1:
            input_img = images[0]
        elif len(images) == 3:
            input_img = torch.cat(images, dim=1)
        elif len(images) == 2:
            input_img = torch.cat(images, dim=1)
        else:
            raise ValueError(f"Expected 1, 2, or 3 image tensors, got {len(images)}")

        e1 = self.Conv1(input_img)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        should_return_all = (return_all is True) or (
            return_all is None and self.deep_supervision and self.training
        )

        if should_return_all:
            output4 = self.final3(self.up8(d5))
            output3 = self.final2(self.up4(d4))
            output2 = self.final1(self.up(d3))
            output1 = self.final(d2)
            output = output1
            return output4, output3, output2, output1, output
        else:
            output = self.final(d2)
            return output


# Alias for backward compatibility
AttU_Net = AWSnet
