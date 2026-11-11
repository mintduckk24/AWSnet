"""Production Inference Pipeline for AWSnet on MyoPS380 Test Volumes.

Strictly conforms to Section 7 of Promt_train_test_no_val.txt:
- Loads completed checkpoint: awsnet_seed<seed>.pt
- Evaluates exactly 76 test cases from test_vol.txt
- Model in eval() mode, single-pass forward + argmax slice-by-slice
- Assembles predictions into 3D volume (128, 128, D) with canonical labels {0, 1, 2, 3}
- Zero morphological postprocessing (no connected components filtering, no LCC)
- Exports test_predictions/<CaseID>_pred.npz with key 'prediction' as uint8 array
- Respects precision policy matching SCAR: amp=auto (CUDA prefers BF16, else FP16; CPU no AMP)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from dataset.myops380_dataset import MyoPS380TestVolumeDataset
from models.awsnet import AWSnet
from train import resolve_amp
from utils.paths import resolve_data_root, resolve_split_dir, resolve_save_dir


def single_pass_volume_inference(
    model: nn.Module,
    c0_vol: np.ndarray,
    lge_vol: np.ndarray,
    t2_vol: np.ndarray,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
) -> np.ndarray:
    """Runs slice-by-slice single-pass forward + argmax on 3D volume (128, 128, D).

    Strictly avoids morphological post-processing.
    Returns uint8 array (128, 128, D) with canonical classes in {0, 1, 2, 3}.
    """
    model.eval()
    H, W, D = c0_vol.shape
    pred_3d = np.zeros((H, W, D), dtype=np.uint8)
    use_amp = (amp_dtype is not None and device.type == "cuda")

    with torch.no_grad():
        for d in range(D):
            slice_c0 = torch.from_numpy(c0_vol[:, :, d]).unsqueeze(0).unsqueeze(0).float()
            slice_lge = torch.from_numpy(lge_vol[:, :, d]).unsqueeze(0).unsqueeze(0).float()
            slice_t2 = torch.from_numpy(t2_vol[:, :, d]).unsqueeze(0).unsqueeze(0).float()

            slice_3ch = torch.cat([slice_c0, slice_lge, slice_t2], dim=1).to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                model_out = model(slice_3ch)
                if isinstance(model_out, (tuple, list)):
                    logits = model_out[-1]
                else:
                    logits = model_out

            slice_pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_3d[:, :, d] = slice_pred

    return pred_3d


def run_prediction(
    checkpoint_path: str | Path,
    data_dir: str | Path | None = None,
    split_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    device_name: str | None = None,
    amp: str = "auto",
    allow_incomplete: bool = False,
) -> Path:
    """Generates canonical test predictions for all 76 cases in test_vol.txt."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    resolved_data = resolve_data_root(data_dir)
    resolved_split = resolve_split_dir(split_dir)

    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = ckpt_path.parent.parent / "test_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp(amp, device)
    print(f"[AWSnet Predict] Execution device: {device} | AMP mode: {amp} (dtype: {amp_dtype})")

    # 1. Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not allow_incomplete and not ckpt.get("training_complete", False):
        completed = ckpt.get("completed_epochs", ckpt.get("epoch", 0))
        raise ValueError(
            f"Checkpoint is incomplete ({completed} completed epochs). "
            "Evaluation strictly requires completed 300 epochs under protocol myops380_train304_test76_last_v1. "
            "Use --allow_incomplete to override for dry-run testing."
        )

    base_filters = ckpt.get("base_filters", 32)
    model = AWSnet(
        in_channels=3,
        num_classes=4,
        base_filters=base_filters,
        deep_supervision=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[AWSnet Predict] Loaded weights from {ckpt_path.name}")

    # 2. Test dataset
    test_manifest = resolved_split / "test_vol.txt"
    if not test_manifest.is_file():
        raise FileNotFoundError(f"Missing test manifest: {test_manifest}")

    test_dataset = MyoPS380TestVolumeDataset(
        data_dir=resolved_data,
        split_dir=resolved_split,
        split_file="test_vol.txt",
    )
    num_cases = len(test_dataset)
    print(f"[AWSnet Predict] Running single-pass inference on {num_cases} test cases...")

    t0 = time.perf_counter()
    for idx in range(num_cases):
        item = test_dataset[idx]
        case_id = item["case_id"]

        pred_3d = single_pass_volume_inference(
            model=model,
            c0_vol=item["c0"],
            lge_vol=item["lge"],
            t2_vol=item["t2"],
            device=device,
            amp_dtype=amp_dtype,
        )

        out_file = out_dir / f"{case_id}_pred.npz"
        np.savez_compressed(
            out_file,
            prediction=pred_3d,
            case_id=case_id,
            exported_label_order="canonical",
            canonical_classes={0: "bg", 1: "myo", 2: "edema", 3: "scar"},
        )

        if (idx + 1) % 10 == 0 or (idx + 1) == num_cases:
            print(f"  Processed [{idx+1:02d}/{num_cases:02d}] {case_id} -> {out_file.name}")

    elapsed = time.perf_counter() - t0
    print(f"[AWSnet Predict] Inference complete for {num_cases} cases in {elapsed:.2f}s ({elapsed/num_cases:.2f}s/vol).")
    print(f"[AWSnet Predict] Predictions saved to: {out_dir.resolve()}")
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(description="AWSnet MyoPS380 Inference Engine")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to awsnet_seed<seed>.pt")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory of preprocessed dataset")
    parser.add_argument("--split_dir", type=str, default=None, help="Directory containing split txt files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for predictions")
    parser.add_argument("--device", type=str, default=None, help="Execution device (e.g. cuda:0 or cpu)")
    parser.add_argument("--amp", type=str, default="auto", choices=["auto", "none", "fp16", "bf16"], help="Mixed precision mode (default: auto)")
    parser.add_argument("--allow_incomplete", action="store_true", help="Allow inference on checkpoints < 300 epochs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        amp=args.amp,
        allow_incomplete=args.allow_incomplete,
    )
