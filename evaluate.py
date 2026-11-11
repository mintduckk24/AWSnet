"""Production Inference & Shared Evaluation for AWSnet on MyoPS380 Benchmark.

Strictly adheres to:
- Benchmark Protocol: myops380_train304_test76_last_v1
- Metric Protocol: myops380_voxel_v3
- Shared Evaluator: Reuses SCAR surface_distance.py (dataset_rows, summarize_dataset_rows)
- Requirements:
  - Loads atomic awsnet_seed<seed>.pt at epoch 300 (rejects incomplete checkpoints unless overridden)
  - Single-pass forward + argmax on all 76 test patients from test_vol.txt
  - Zero morphological postprocessing (no LCC, no small connected components filtering)
  - Exports test_predictions/{CaseID}_pred.npz (uint8, 128x128xD, key 'prediction')
  - Produces metrics.json and per_case.csv with complete protocol metadata
  - Respects precision policy matching SCAR: amp=auto (CUDA prefers BF16, else FP16; CPU no AMP)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from dataset.myops380_dataset import MyoPS380TestVolumeDataset, extract_patient_id
from models.awsnet import AWSnet
from predict import single_pass_volume_inference
from train import resolve_amp
from utils.paths import resolve_data_root, resolve_split_dir, resolve_save_dir


def load_surface_distance_module(scar_root: str | Path | None = None):
    """Loads SCAR's surface_distance.py or bundled local copy."""
    candidates = []
    if scar_root is not None:
        candidates.append(Path(scar_root) / "training" / "metrics" / "surface_distance.py")
    candidates.extend([
        Path(__file__).resolve().parent / "utils" / "surface_distance.py",
        Path(r"D:\nckh_model\SCAR\training\metrics\surface_distance.py"),
        Path(r"../SCAR/training/metrics/surface_distance.py"),
    ])

    for p in candidates:
        if p.is_file():
            spec = importlib.util.spec_from_file_location("scar_surface_distance", str(p.resolve()))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod

    raise FileNotFoundError(
        f"Cannot locate surface_distance.py in candidates: {[str(c) for c in candidates]}"
    )


def compute_file_sha256(path: str | Path) -> str:
    """Computes SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate(
    checkpoint: str | Path | None = None,
    predictions_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    split_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    split: str = "test_vol",
    batch_size: int = 8,
    scar_root: str | Path | None = None,
    device_name: str | None = None,
    amp: str = "auto",
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Runs prediction and shared evaluation for AWSnet on the 76 MyoPS380 test patients."""
    resolved_data = resolve_data_root(data_dir)
    resolved_split = resolve_split_dir(split_dir)
    surface_dist_mod = load_surface_distance_module(scar_root)

    manifest_name = f"{split}.txt" if not split.endswith(".txt") else split
    test_manifest = resolved_split / manifest_name
    if not test_manifest.is_file():
        raise FileNotFoundError(f"Missing test manifest: {test_manifest}")

    test_dataset = MyoPS380TestVolumeDataset(
        data_dir=resolved_data,
        split_dir=resolved_split,
        split_file=manifest_name,
    )
    expected_cases = len(test_dataset)
    if expected_cases != 76 and "test" in split:
        print(f"[WARNING] Expected 76 test cases, but {manifest_name} has {expected_cases}")

    # Determine execution device and precision
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp(amp, device)
    print(f"[AWSnet Eval] Execution device: {device} | AMP mode: {amp} (dtype: {amp_dtype})")

    # Step 1: Predictions resolution or generation
    model = None
    ckpt_meta: dict[str, Any] = {}
    checkpoint_hash = None
    pred_dir = None

    if predictions_dir is not None:
        pred_dir = Path(predictions_dir)
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"Provided predictions directory does not exist: {pred_dir}")
        print(f"[AWSnet Eval] Using existing predictions from: {pred_dir}")
    elif checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")

        checkpoint_hash = compute_file_sha256(ckpt_path)
        ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)

        if not allow_incomplete and not ckpt_data.get("training_complete", False):
            completed = ckpt_data.get("completed_epochs", ckpt_data.get("epoch", 0))
            raise ValueError(
                f"Checkpoint {ckpt_path.name} is incomplete ({completed} completed epochs). "
                "Official shared evaluation strictly requires completed 300 epochs under protocol "
                "myops380_train304_test76_last_v1. Use --allow_incomplete to override for dry-run testing."
            )

        ckpt_meta = {
            "model_id": ckpt_data.get("model_id", "awsnet"),
            "model_architecture": ckpt_data.get("model_architecture", "AWSnet"),
            "completed_epochs": ckpt_data.get("completed_epochs", ckpt_data.get("epoch", 0)),
            "planned_epochs": ckpt_data.get("planned_epochs", 300),
            "training_complete": ckpt_data.get("training_complete", False),
            "seed": ckpt_data.get("seed", 1234),
            "total_updates": ckpt_data.get("total_updates", None),
            "successful_updates": ckpt_data.get("successful_updates", None),
            "amp_skipped_updates": ckpt_data.get("amp_skipped_updates", 0),
            "amp_mode": ckpt_data.get("amp_mode", amp),
            "amp_dtype": ckpt_data.get("amp_dtype", str(amp_dtype)),
            "checkpoint_sha256": checkpoint_hash,
        }

        base_filters = ckpt_data.get("base_filters", 32)
        model = AWSnet(
            in_channels=3,
            num_classes=4,
            base_filters=base_filters,
            deep_supervision=False,
        ).to(device)
        model.load_state_dict(ckpt_data["model_state"])
        model.eval()

        if output_dir is not None:
            pred_dir = Path(output_dir) / "test_predictions"
        else:
            pred_dir = ckpt_path.parent.parent / "test_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("Either --checkpoint or --predictions_dir must be provided.")

    if output_dir is not None:
        eval_out_dir = Path(output_dir)
    elif checkpoint is not None:
        eval_out_dir = Path(checkpoint).parent.parent / "evaluation"
    else:
        eval_out_dir = pred_dir.parent / "evaluation"
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Run inference if model is loaded
    all_case_ids = test_dataset.cases
    print(f"[AWSnet Eval] Commencing volume-level evaluation for {len(all_case_ids)} cases...")

    all_rows = []
    t_start = time.perf_counter()

    for idx, case_id in enumerate(all_case_ids):
        item = test_dataset[idx]
        target_vol = item["label"]  # Ground truth canonical 0..3 (128, 128, D)

        pred_file = pred_dir / f"{case_id}_pred.npz"

        if pred_file.is_file():
            with np.load(pred_file, allow_pickle=False) as npz:
                pred_vol = npz["prediction"]
        elif model is not None:
            pred_vol = single_pass_volume_inference(
                model=model,
                c0_vol=item["c0"],
                lge_vol=item["lge"],
                t2_vol=item["t2"],
                device=device,
                amp_dtype=amp_dtype,
            )
            np.savez_compressed(
                pred_file,
                prediction=pred_vol,
                case_id=case_id,
                exported_label_order="canonical",
                canonical_classes={0: "bg", 1: "myo", 2: "edema", 3: "scar"},
            )
        else:
            raise FileNotFoundError(f"Prediction file missing: {pred_file}")

        # Validate prediction volume properties
        if pred_vol.shape != target_vol.shape:
            raise ValueError(f"Shape mismatch for {case_id}: pred {pred_vol.shape} vs GT {target_vol.shape}")
        if pred_vol.dtype != np.uint8:
            pred_vol = pred_vol.astype(np.uint8)

        # Execute shared evaluator dataset_rows for myops380
        case_rows = surface_dist_mod.dataset_rows(
            prediction=pred_vol,
            target=target_vol,
            case=case_id,
            compute_distance=True,
            dataset_id="myops380",
        )
        all_rows.extend(case_rows)

        if (idx + 1) % 15 == 0 or (idx + 1) == len(all_case_ids):
            print(f"  Evaluated [{idx+1:02d}/{len(all_case_ids):02d}] {case_id}")

    elapsed_eval = time.perf_counter() - t_start
    print(f"[AWSnet Eval] Finished case evaluations in {elapsed_eval:.2f}s.")

    # Step 3: Summarize metrics using SCAR summarize_dataset_rows
    summary = surface_dist_mod.summarize_dataset_rows(all_rows, dataset_id="myops380")

    # Step 4: Write per_case.csv
    csv_path = eval_out_dir / "per_case.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
    else:
        fieldnames = [
            "case",
            "region",
            "dice",
            "iou",
            "hd95_voxel",
            "asd_voxel",
            "status",
            "precision",
            "recall",
            "prediction_voxels",
            "target_voxels",
        ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"[AWSnet Eval] Exported per_case.csv: {csv_path}")

    # Step 5: Write metrics.json and summary.json with complete metadata
    # Flatten summary metrics directly to root level matching SCAR reference
    results_payload = dict(summary)
    results_payload.update({
        "model_id": "awsnet",
        "benchmark_protocol": "myops380_train304_test76_last_v1",
        "metric_protocol": "myops380_voxel_v3",
        "dataset_profile": "myops380",
        "test_cases_count": len(all_case_ids),
        "split_manifest_sha256": compute_file_sha256(test_manifest),
        "loss_objective": "AWSnet_LossPRSnet_dicece_0.2CE_1.0Dice_4StageDeepSupervision",
        "exported_label_order": "canonical",
        "canonical_classes": {0: "background", 1: "normal_myocardium", 2: "edema", 3: "scar"},
        "regions": {
            "normal_myocardium": [1],
            "edema": [2],
            "scar": [3],
            "edema_inclusive": [2, 3],
            "myocardial_ring": [1, 2, 3],
        },
        "amp_mode": amp,
        "amp_dtype": str(amp_dtype),
        "checkpoint_metadata": ckpt_meta,
        "summary": summary,  # Keep nested for backward compatibility
        "evaluation_time_seconds": elapsed_eval,
    })

    json_path = eval_out_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[AWSnet Eval] Exported metrics.json: {json_path}")

    summary_json_path = eval_out_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[AWSnet Eval] Exported summary.json: {summary_json_path}")

    # Step 6: Print clean report table
    print("\n" + "=" * 80)
    print("                    AWSnet MyoPS380 Official Evaluation Report                    ")
    print("=" * 80)
    print(f"Protocol: myops380_voxel_v3 | Cases: {len(all_case_ids)} | Unit: voxel")
    print("-" * 80)
    header = f"{'Region':<22} | {'Dice Mean':<10} | {'IoU Mean':<10} | {'HD95 (vox)':<12} | {'ASD (vox)':<10} | {'N (valid)':<8}"
    print(header)
    print("-" * 80)

    for region_name, metrics in summary.items():
        if not isinstance(metrics, dict):
            continue
        dice_m = metrics.get("mean_dice", metrics.get("dice_mean"))
        iou_m = metrics.get("mean_iou", metrics.get("iou_mean"))
        hd95_m = metrics.get("mean_hd95_voxel", metrics.get("hd95_voxel_mean"))
        asd_m = metrics.get("mean_asd_voxel", metrics.get("asd_voxel_mean"))
        n_def = metrics.get("dice_defined_cases", metrics.get("dice_count", 0))
        n_tot = metrics.get("cases", metrics.get("total_cases", 0))

        dice_str = f"{dice_m:.4f}" if dice_m is not None else "N/A"
        iou_str = f"{iou_m:.4f}" if iou_m is not None else "N/A"
        hd95_str = f"{hd95_m:.4f}" if hd95_m is not None else "N/A"
        asd_str = f"{asd_m:.4f}" if asd_m is not None else "N/A"
        n_valid = f"{n_def}/{n_tot}"
        print(f"{region_name:<22} | {dice_str:<10} | {iou_str:<10} | {hd95_str:<12} | {asd_str:<10} | {n_valid:<8}")

    print("-" * 80)
    if "avg_pathology_dice" in summary and summary["avg_pathology_dice"] is not None:
        print(f"avg_pathology_dice (Scar + Edema-exclusive): {summary['avg_pathology_dice']:.4f}")
    if "avg_pathology_inclusive_dice" in summary and summary["avg_pathology_inclusive_dice"] is not None:
        print(f"avg_pathology_inclusive_dice (Scar + Edema-inclusive): {summary['avg_pathology_inclusive_dice']:.4f}")
    print("=" * 80 + "\n")

    return results_payload


def parse_args():
    parser = argparse.ArgumentParser(description="AWSnet MyoPS380 Shared Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to awsnet_seed<seed>.pt")
    parser.add_argument("--predictions_dir", type=str, default=None, help="Directory containing pre-generated npz")
    parser.add_argument("--data_dir", "--data-root", "--data_root", dest="data_dir", type=str, default=None, help="Root directory of preprocessed dataset")
    parser.add_argument("--split_dir", "--list-dir", "--list_dir", dest="split_dir", type=str, default=None, help="Directory containing split txt files")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default=None, help="Output directory for evaluations")
    parser.add_argument("--split", type=str, default="test_vol", choices=["test_vol", "val_vol"], help="Split manifest to evaluate (default: test_vol)")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--scar_root", type=str, default=None, help="Path to reference SCAR repository")
    parser.add_argument("--device", type=str, default=None, help="Execution device (e.g. cuda:0 or cpu)")
    parser.add_argument("--amp", type=str, default="auto", choices=["auto", "none", "fp16", "bf16"], help="Mixed precision mode (default: auto)")
    parser.add_argument("--allow_incomplete", action="store_true", help="Allow evaluation on incomplete checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    return evaluate(
        checkpoint=args.checkpoint,
        predictions_dir=args.predictions_dir,
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        scar_root=args.scar_root,
        device_name=args.device,
        amp=args.amp,
        allow_incomplete=args.allow_incomplete,
    )


if __name__ == "__main__":
    main()
