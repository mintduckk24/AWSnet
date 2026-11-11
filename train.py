"""Production Training Engine for AWSnet on MyoPS380 Benchmark.

Strictly adheres to:
- Benchmark Protocol: myops380_train304_test76_last_v1
- Model Identifier: awsnet
- Checkpoint: <save_dir>/awsnet_seed<seed>.pt
- Requirements:
  1. 304 train patients (1,972 slices) from train.txt, zero validation (no val split, no val loader)
  2. Data augmentation: SCAR RandomGenerator on (bSSFP, LGE, T2w) aligned slices in [0.0, 1.0]
  3. Fixed 300 epochs budget (max_epochs = 300), no early stopping, no best.pt
  4. Target effective batch size 16 with sample-weighted accumulation and drop_last=False
  5. AdamW optimizer (weight_decay=0.0001, lr=0.0003333), step-level Polynomial LR Decay (power=0.9)
  6. Native AWSnet objective: LossPRSnet_dicece across multi-scale deep supervision heads
  7. Deterministic execution (seed=1234), atomic checkpoint saving to awsnet_seed<seed>.pt
  8. Strict test set isolation (no test evaluation during training)
  9. Precision policy matching SCAR: amp=auto (CUDA prefers BF16, else FP16 with GradScaler; CPU no AMP)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.myops380_dataset import MyoPS380TrainDataset
from models.awsnet import AWSnet
from utils.criterions import AWSCompositeLoss, LossPRSnet_dicece
from utils.lr_scheduler import build_optimizer, build_scheduler, OPTIMIZATION_PROTOCOL
from utils.paths import resolve_data_root, resolve_split_dir, resolve_save_dir


def resolve_amp(mode: str, device: torch.device) -> Optional[torch.dtype]:
    """Resolves mixed precision data type matching SCAR protocol."""
    mode_lower = str(mode).lower()
    if mode_lower == "none" or (mode_lower == "auto" and device.type == "cpu"):
        return None
    if device.type != "cuda":
        raise ValueError("Mixed precision is enabled only on CUDA; use --amp none on CPU.")
    if mode_lower not in {"auto", "none", "fp16", "bf16"}:
        raise ValueError(f"Unknown AMP mode: {mode}")
    if mode_lower == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode_lower == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16.")
        return torch.bfloat16
    return torch.float16


def set_seed(seed: int = 1234, deterministic: bool = True) -> None:
    """Sets random seeds across Python, NumPy, PyTorch, and CUDA for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    """Seeds DataLoader worker processes using PyTorch initial seed modulo 2**32."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def compute_manifest_hash(path: Path) -> str:
    """Computes SHA256 of manifest file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def count_parameters(model: nn.Module) -> int:
    """Calculates total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    criterion: AWSCompositeLoss,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    accum_steps: int = 1,
    clip_grad: float = 0.0,
    epoch: int = 0,
    global_step: int = 0,
    successful_updates: int = 0,
    log_interval: int = 20,
    max_batches: Optional[int] = None,
) -> Tuple[Dict[str, float], int, int, int]:
    """Executes one training epoch with sample-weighted gradient accumulation and AMP.

    Conforms to Promt_train_test_no_val.txt Section 4:
    - Sample-weighted accumulation: backward loss * cur_batch_size
    - At boundary: unscale (if scaler), divide gradients by actual group sample count
    - Advance scheduler and global_step ONLY on successful optimizer updates
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    epoch_loss = 0.0
    epoch_stage1 = 0.0
    epoch_stage2 = 0.0
    epoch_stage3 = 0.0
    epoch_stage4 = 0.0
    total_samples = 0
    group_samples = 0
    skipped_steps = 0

    use_amp = (amp_dtype is not None and device.type == "cuda")
    t0 = time.perf_counter()
    num_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if "image_3ch" in batch:
            image = batch["image_3ch"]
        elif "image" in batch and "image1" in batch and "image2" in batch and batch["image"].size(1) == 1:
            image = torch.cat([batch["image"], batch["image1"], batch["image2"]], dim=1)
        else:
            image = batch["image"]
        image = image.to(device, non_blocking=loader.pin_memory if hasattr(loader, "pin_memory") else False)  # (B, 3, 128, 128)
        target = batch["label"].to(device, non_blocking=loader.pin_memory if hasattr(loader, "pin_memory") else False)  # (B, 128, 128)
        cur_batch_size = image.size(0)

        # Forward pass in training mode with optional AMP
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(image)
            loss, loss_metrics = criterion(out, target)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss {loss.item()} at epoch {epoch}, batch {batch_idx}")

        # Sample-weighted accumulation: backward loss * cur_batch_size
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss * cur_batch_size).backward()
        else:
            (loss * cur_batch_size).backward()

        group_samples += cur_batch_size
        total_samples += cur_batch_size
        epoch_loss += float(loss.detach().item()) * cur_batch_size

        if "loss_stage1" in loss_metrics:
            epoch_stage1 += loss_metrics["loss_stage1"] * cur_batch_size
            epoch_stage2 += loss_metrics["loss_stage2"] * cur_batch_size
            epoch_stage3 += loss_metrics["loss_stage3"] * cur_batch_size
            epoch_stage4 += loss_metrics["loss_stage4"] * cur_batch_size

        is_accum_boundary = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == num_batches)

        if is_accum_boundary:
            # Unscale before dividing by actual group sample count
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)

            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(group_samples)

            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad if (clip_grad and clip_grad > 0) else float("inf")
            )
            finite_gradient = bool(torch.isfinite(norm))
            if not finite_gradient and not (scaler is not None and scaler.is_enabled()):
                raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch}, batch {batch_idx}.")

            if scaler is not None and scaler.is_enabled():
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
            else:
                optimizer.step()
                skipped = False

            if skipped:
                skipped_steps += 1
            else:
                scheduler.step()
                global_step += 1
                successful_updates += 1

            optimizer.zero_grad(set_to_none=True)
            group_samples = 0

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            current_lr = scheduler.get_last_lr()[0]
            step_loss = float(loss.detach().item())
            print(
                f"  Epoch [{epoch+1:03d}] Batch [{batch_idx+1:03d}/{num_batches:03d}] "
                f"Loss: {step_loss:.4f} | LR: {current_lr:.6f}"
            )

    elapsed = time.perf_counter() - t0
    avg_loss = epoch_loss / max(1, total_samples)

    metrics = {
        "loss": avg_loss,
        "stage1_loss": epoch_stage1 / max(1, total_samples),
        "stage2_loss": epoch_stage2 / max(1, total_samples),
        "stage3_loss": epoch_stage3 / max(1, total_samples),
        "stage4_loss": epoch_stage4 / max(1, total_samples),
        "epoch_time_seconds": elapsed,
        "samples_per_second": total_samples / max(1e-5, elapsed),
        "amp_skipped_steps": skipped_steps,
    }

    return metrics, global_step, successful_updates, skipped_steps


def train(
    data_dir: str | Path | None = None,
    split_dir: str | Path | None = None,
    save_dir: str | Path | None = None,
    batch_size: int = 16,
    accum_steps: int = 1,
    max_epochs: int = 300,
    base_lr: float = 0.0003333,
    weight_decay: float = 0.0001,
    base_filters: int = 32,
    seed: int = 1234,
    resume: str | None = None,
    device_name: str | None = None,
    amp: str = "auto",
    num_workers: int = 4,
    clip_grad: float = 0.0,
    log_interval: int = 20,
    max_batches: int | None = None,
) -> Path:
    """Runs standard 300-epoch training strictly conforming to Promt_train_test_no_val.txt."""
    resolved_data = resolve_data_root(data_dir)
    resolved_split = resolve_split_dir(split_dir)
    resolved_save = resolve_save_dir(save_dir)

    train_manifest = resolved_split / "train.txt"
    if not train_manifest.is_file():
        raise FileNotFoundError(f"Missing training manifest: {train_manifest}")

    manifest_hash = compute_manifest_hash(train_manifest)

    # 1. Device and precision resolution
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp(amp, device)
    print(f"[AWSnet] Execution device: {device} | AMP mode: {amp} (dtype: {amp_dtype})")

    # 2. Seed & Determinism
    set_seed(seed=seed, deterministic=True)
    g = torch.Generator()
    g.manual_seed(seed)

    # 3. Training Dataset & DataLoader (Strictly NO validation loader)
    train_dataset = MyoPS380TrainDataset(
        data_dir=resolved_data,
        split_dir=resolved_split,
        split="train",
        img_size=(128, 128),
    )
    print(f"[AWSnet] Loaded train dataset: {len(train_dataset)} slices (304 patients, zero validation)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,  # Keep all 1,972 slices
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    batches_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(batches_per_epoch / accum_steps)
    total_updates = max_epochs * updates_per_epoch
    effective_bs = batch_size * accum_steps

    print(
        f"[AWSnet] Training Protocol & Budget:\n"
        f"  - protocol_id: myops380_train304_test76_last_v1\n"
        f"  - model_id: awsnet\n"
        f"  - max_epochs: {max_epochs}\n"
        f"  - batch_size: {batch_size}, accum_steps: {accum_steps} (Effective BS: {effective_bs})\n"
        f"  - batches_per_epoch: {batches_per_epoch}, total_updates: {total_updates}\n"
        f"  - base_lr: {base_lr}, weight_decay: {weight_decay}\n"
        f"  - manifest_hash (train.txt): {manifest_hash[:16]}..."
    )

    # 4. Model initialization
    model = AWSnet(
        in_channels=3,
        num_classes=4,
        base_filters=base_filters,
        deep_supervision=True,
    ).to(device)

    total_params = count_parameters(model)
    print(f"[AWSnet] Instantiated AWSnet (Trainable parameters: {total_params:,})")

    # 5. Optimization & Scheduling (AdamW + Polynomial Decay)
    optimizer = build_optimizer(model, base_lr=base_lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, total_updates=total_updates, power=0.9)
    criterion = AWSCompositeLoss(num_classes=4)
    use_scaler = (amp_dtype == torch.float16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_epoch = 0
    global_step = 0
    successful_updates = 0
    cumulative_skipped = 0
    history = []

    # 6. Resume handling
    ckpt_filename = f"awsnet_seed{seed}.pt"
    ckpt_path = resolved_save / ckpt_filename

    resume_target = Path(resume) if (resume and resume != "auto") else ckpt_path
    if (resume and resume != "auto") or (resume == "auto" and resume_target.is_file()):
        if not resume_target.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_target}")
        print(f"[AWSnet] Resuming from checkpoint: {resume_target}")
        checkpoint = torch.load(resume_target, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler and "scaler_state" in checkpoint and checkpoint["scaler_state"] and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = checkpoint.get("completed_epochs", checkpoint.get("epoch", 0))
        global_step = checkpoint.get("global_step", start_epoch * updates_per_epoch)
        successful_updates = checkpoint.get("successful_updates", global_step)
        cumulative_skipped = checkpoint.get("amp_skipped_updates", 0)
        history = checkpoint.get("history", [])

        if "rng_state_torch" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state_torch"])
        if "rng_state_numpy" in checkpoint:
            np.random.set_state(checkpoint["rng_state_numpy"])
        if "rng_state_python" in checkpoint:
            random.setstate(checkpoint["rng_state_python"])
        if torch.cuda.is_available() and "rng_state_cuda" in checkpoint and checkpoint["rng_state_cuda"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng_state_cuda"])
        if "loader_rng" in checkpoint and checkpoint["loader_rng"] is not None:
            g.set_state(checkpoint["loader_rng"])

        print(
            f"[AWSnet] Resumed at epoch {start_epoch}, global_step {global_step} "
            f"(successful {successful_updates}, skipped {cumulative_skipped})"
        )

    if start_epoch >= max_epochs:
        print(f"[AWSnet] Checkpoint already completed full budget ({start_epoch}/{max_epochs} epochs).")
        return ckpt_path

    # 7. Training loop (Strictly NO validation)
    print(f"\n[AWSnet] Starting training for {max_epochs - start_epoch} epochs...")
    total_training_start = time.perf_counter()

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.perf_counter()
        metrics, global_step, successful_updates, skipped_steps = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            amp_dtype=amp_dtype,
            scaler=scaler,
            accum_steps=accum_steps,
            clip_grad=clip_grad,
            epoch=epoch,
            global_step=global_step,
            successful_updates=successful_updates,
            log_interval=log_interval,
            max_batches=max_batches,
        )
        cumulative_skipped += skipped_steps
        epoch_duration = time.perf_counter() - epoch_start
        metrics["epoch"] = epoch + 1
        metrics["epoch_wall_time"] = epoch_duration
        history.append(metrics)

        print(
            f"[Epoch {epoch+1:03d}/{max_epochs:03d}] "
            f"Loss: {metrics['loss']:.4f} | "
            f"Speed: {metrics['samples_per_second']:.1f} samples/s | "
            f"Duration: {epoch_duration:.2f}s | "
            f"Updates: {successful_updates}/{total_updates} (Skipped: {cumulative_skipped})"
        )

        # Atomic checkpoint save at end of each completed epoch
        completed_epochs = epoch + 1
        is_final_epoch = (completed_epochs == max_epochs)
        state = {
            "model_id": "awsnet",
            "model_architecture": "AWSnet (Attention U-Net with Auto-Weighted Deep Supervision)",
            "benchmark_protocol": "myops380_train304_test76_last_v1",
            "epoch": completed_epochs,
            "completed_epochs": completed_epochs,
            "planned_epochs": max_epochs,
            "training_complete": is_final_epoch,
            "checkpoint_selection": "last_completed_epoch",
            "checkpoint_selection_metric": None,
            "global_step": global_step,
            "total_updates": total_updates,
            "successful_updates": successful_updates,
            "amp_skipped_updates": cumulative_skipped,
            "amp_mode": amp,
            "amp_dtype": str(amp_dtype),
            "seed": seed,
            "batch_size": batch_size,
            "accum_steps": accum_steps,
            "base_lr": base_lr,
            "weight_decay": weight_decay,
            "base_filters": base_filters,
            "trainable_parameters": total_params,
            "manifest_hash": manifest_hash,
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if (scaler and scaler.is_enabled()) else None,
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_python": random.getstate(),
            "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader_rng": g.get_state(),
            "history": history,
            "label_order": "canonical",
            "canonical_classes": {0: "background", 1: "normal_myocardium", 2: "edema", 3: "scar"},
        }

        # Atomic write
        temp_ckpt = resolved_save / f"{ckpt_filename}.tmp"
        torch.save(state, temp_ckpt)
        os.replace(temp_ckpt, ckpt_path)

    total_training_duration = time.perf_counter() - total_training_start
    print(f"\n[AWSnet] Training complete in {total_training_duration/60:.2f} minutes.")
    print(f"[AWSnet] Checkpoint saved: {ckpt_path.resolve()}")

    # Save summary metadata
    summary_path = resolved_save / f"training_summary_seed{seed}.json"
    summary_data = {
        "model_id": "awsnet",
        "benchmark_protocol": "myops380_train304_test76_last_v1",
        "seed": seed,
        "max_epochs": max_epochs,
        "completed_epochs": max_epochs,
        "total_updates": total_updates,
        "successful_updates": successful_updates,
        "amp_skipped_updates": cumulative_skipped,
        "amp_mode": amp,
        "amp_dtype": str(amp_dtype),
        "trainable_parameters": total_params,
        "total_training_seconds": total_training_duration,
        "final_loss": history[-1]["loss"] if history else None,
        "checkpoint_file": str(ckpt_path.resolve()),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(description="AWSnet MyoPS380 Training Engine")
    parser.add_argument("--data_dir", "--data-root", "--data_root", dest="data_dir", type=str, default=None, help="Root directory of preprocessed dataset")
    parser.add_argument("--split_dir", "--list-dir", "--list_dir", dest="split_dir", type=str, default=None, help="Directory containing split txt files")
    parser.add_argument("--save_dir", "--run-root", "--run_root", "--output-dir", "--output_dir", dest="save_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=16, help="Physical batch size")
    parser.add_argument("--accum_steps", "--accum-steps", dest="accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_epochs", "--epochs", "--max-epochs", dest="max_epochs", type=int, default=300, help="Maximum epochs (default: 300)")
    parser.add_argument("--base_lr", "--lr", "--base-lr", dest="base_lr", type=float, default=0.0003333, help="Base learning rate")
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--base_filters", type=int, default=32, help="Initial conv filters (default: 32)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        type=str,
        default=None,
        help="Path to checkpoint to resume from, or pass without value to resume from default checkpoint"
    )
    parser.add_argument("--device", type=str, default=None, help="Execution device (e.g. cuda:0 or cpu)")
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=4, help="DataLoader workers count (default: 4)")
    parser.add_argument("--log_interval", "--log-every", dest="log_interval", type=int, default=20, help="Batch logging interval")
    parser.add_argument("--amp", type=str, default="auto", choices=["auto", "none", "fp16", "bf16"], help="Mixed precision mode (default: auto)")
    parser.add_argument("--clip_grad", "--clip-grad", dest="clip_grad", type=float, default=0.0, help="Gradient clipping norm (0.0 to disable)")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path (for compatibility with SCAR launch scripts)")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches per epoch (for quick testing)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        max_epochs=args.max_epochs,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        base_filters=args.base_filters,
        seed=args.seed,
        resume=args.resume,
        device_name=args.device,
        amp=args.amp,
        num_workers=args.num_workers,
        clip_grad=args.clip_grad,
        log_interval=args.log_interval,
        max_batches=args.max_batches,
    )
