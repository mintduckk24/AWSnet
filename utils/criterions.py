"""Native Loss Functions for AWSnet MyoPS380 Adaptation.

Faithfully implements Section 5 of Promt_train_test_no_val.txt:
Preserves AWSnet's native objective: LossPRSnet_dicece combining multi-scale
deep supervision cross-entropy and soft multi-class Dice loss.

Formula:
  L_stage_i = w_ce * CE(logits_i, target) + w_dc * MultiDiceLoss(logits_i, target)
  L_total = sum_{i=0}^3 (w_stage_i * L_stage_i)

Default weights in AWSnet codebase:
  Stage weights: [0.125, 0.25, 0.5, 1.0] (for 1/8, 1/4, 1/2, full-res heads)
  Loss weights: w_ce = 0.2, w_dc = 1.0
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Binary soft Dice loss for a single channel."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, input_prob: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
        N = target_prob.size(0)
        input_flat = input_prob.contiguous().view(N, -1)
        target_flat = target_prob.contiguous().view(N, -1)
        intersection = (input_flat * target_flat).sum(1)
        dice = (2.0 * intersection + self.smooth) / (input_flat.sum(1) + target_flat.sum(1) + self.smooth)
        loss = 1.0 - dice.mean()
        return loss


class MultiDiceLoss(nn.Module):
    """Multi-class Dice loss computed per category and aggregated.

    Matches AWSnet ASAN_training/loss_deepsupvision.py MultiDiceLoss,
    vectorized and device-agnostic for CPU/CUDA compatibility.
    """

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        input_logits: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    ) -> torch.Tensor:
        """Computes multi-class Dice loss.

        Args:
            input_logits: Tensor of shape (B, C, H, W).
            target: Long tensor of shape (B, H, W) or (B, 1, H, W).
            weights: Optional per-class weights of length C.
        """
        probs = F.softmax(input_logits, dim=1)
        if target.dim() == 4:
            target = target.squeeze(1)
        elif target.dim() == 2:
            target = target.unsqueeze(0)

        # One-hot target: (B, C, H, W)
        target_oh = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        C = input_logits.shape[1]
        if weights is None:
            class_weights = torch.ones(C, device=input_logits.device, dtype=input_logits.dtype)
        elif not isinstance(weights, torch.Tensor):
            class_weights = torch.tensor(weights, device=input_logits.device, dtype=input_logits.dtype)
        else:
            class_weights = weights.to(device=input_logits.device, dtype=input_logits.dtype)

        total_loss = torch.zeros(1, device=input_logits.device, dtype=input_logits.dtype)
        for i in range(C):
            loss_c = self.dice(probs[:, i], target_oh[:, i])
            total_loss = total_loss + class_weights[i] * loss_c

        return total_loss.squeeze()


class LossPRSnet_dicece(nn.Module):
    """Multi-scale Deep Supervision Loss from AWSnet.

    Default weights: [w_stage0, w_stage1, w_stage2, w_stage3, w_ce, w_dice]
    Default values: [0.125, 0.25, 0.5, 1.0, 0.2, 1.0]
    """

    def __init__(
        self,
        num_classes: int = 4,
        weights: Sequence[float] = (0.125, 0.25, 0.5, 1.0, 0.2, 1.0),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weights = list(weights)
        self.criterion_dice = MultiDiceLoss(num_classes=num_classes)
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs4: torch.Tensor,
        outputs3: torch.Tensor,
        outputs2: torch.Tensor,
        outputs1: torch.Tensor,
        target: torch.Tensor,
        *extra_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes weighted deep supervision loss.

        Args:
            outputs4: 1/8 resolution logits (B, C, H, W).
            outputs3: 1/4 resolution logits (B, C, H, W).
            outputs2: 1/2 resolution logits (B, C, H, W).
            outputs1: Full resolution logits (B, C, H, W).
            target: Ground truth label tensor (B, H, W) or (B, 1, H, W).
            *extra_targets: Optional additional target tensors for backward compatibility.
        """
        if target.dim() == 4:
            target = target.squeeze(1)
        target = target.long()

        dc4 = self.criterion_dice(outputs4, target)
        dc3 = self.criterion_dice(outputs3, target)
        dc2 = self.criterion_dice(outputs2, target)
        dc1 = self.criterion_dice(outputs1, target)

        ce4 = self.criterion_ce(outputs4, target)
        ce3 = self.criterion_ce(outputs3, target)
        ce2 = self.criterion_ce(outputs2, target)
        ce1 = self.criterion_ce(outputs1, target)

        w_ce = self.weights[4]
        w_dc = self.weights[5]

        loss4 = w_ce * ce4 + w_dc * dc4
        loss3 = w_ce * ce3 + w_dc * dc3
        loss2 = w_ce * ce2 + w_dc * dc2
        loss1 = w_ce * ce1 + w_dc * dc1

        total_loss = (
            self.weights[0] * loss4
            + self.weights[1] * loss3
            + self.weights[2] * loss2
            + self.weights[3] * loss1
        )

        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_stage4": float(loss4.detach().item()),
            "loss_stage3": float(loss3.detach().item()),
            "loss_stage2": float(loss2.detach().item()),
            "loss_stage1": float(loss1.detach().item()),
            "ce1": float(ce1.detach().item()),
            "dc1": float(dc1.detach().item()),
        }

        return total_loss, metrics


class AWSCompositeLoss(nn.Module):
    """High-level loss adapter for AWSnet supporting flexible forward outputs."""

    def __init__(
        self,
        num_classes: int = 4,
        weights: Sequence[float] = (0.125, 0.25, 0.5, 1.0, 0.2, 1.0),
    ):
        super().__init__()
        self.loss_fn = LossPRSnet_dicece(num_classes=num_classes, weights=weights)

    def forward(
        self,
        model_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(model_output, (tuple, list)) and len(model_output) >= 4:
            out4, out3, out2, out1 = model_output[0], model_output[1], model_output[2], model_output[3]
            return self.loss_fn(out4, out3, out2, out1, target)
        else:
            # Single output fallback
            if isinstance(model_output, (tuple, list)):
                primary = model_output[-1]
            else:
                primary = model_output
            ce = F.cross_entropy(primary, target.squeeze(1).long() if target.dim() == 4 else target.long())
            dice = self.loss_fn.criterion_dice(primary, target)
            total = self.loss_fn.weights[4] * ce + self.loss_fn.weights[5] * dice
            return total, {"loss_total": float(total.detach().item()), "ce": float(ce.detach().item()), "dice": float(dice.detach().item())}
