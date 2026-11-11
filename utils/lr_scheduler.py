"""Learning rate scheduling matching SCAR and I-MMSeg's polynomial decay.

Protocol: im_mseg_adamw_poly_v1
LR formula after u successful updates:
base_lr * max(0.0, 1.0 - max(0, u - 1) / total_updates) ** 0.9
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


OPTIMIZATION_PROTOCOL = {
    "id": "im_mseg_adamw_poly_v1",
    "optimizer": "AdamW",
    "weight_decay_scope": "all_parameters",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "scheduler": "polynomial_after_update_zero_based_iteration",
    "power": 0.9,
}


def build_optimizer(model: torch.nn.Module, base_lr: float = 0.0003333, weight_decay: float = 0.0001) -> torch.optim.AdamW:
    """Builds standard AdamW optimizer applying weight decay to all model parameters."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(optimizer: Optimizer, total_updates: int, power: float = 0.9) -> LambdaLR:
    """Builds polynomial learning rate scheduler with I-MMSeg zero-based iteration offset."""
    if not isinstance(total_updates, int) or total_updates < 1:
        raise ValueError(f"total_updates must be a positive integer, got {total_updates}")

    return LambdaLR(
        optimizer,
        lambda step: max(0.0, 1.0 - max(0, step - 1) / total_updates) ** power,
    )
