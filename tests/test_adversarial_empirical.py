"""Adversarial stress-test harness for AWSnet MyoPS380 Adaptation.

Empirically verifies:
1. Stochastic reproducibility & DataLoader generator state.
2. Gradient flow across all 4 deep supervision stages & gradient clipping.
3. Strict normalization boundary checks [-1e-5, 1.0 + 1e-5].
4. Evaluation engine metric calculation & empty mask policy.
5. Output format verification: root-level keys in metrics.json and summary.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure AWSnet repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dataset.data_augmentation import RandomGenerator, ResizeGenerator
from dataset.myops380_dataset import (
    MyoPS380TestVolumeDataset,
    MyoPS380TrainDataset,
    canonicalize_label,
    ingest_image_array,
)
from evaluate import evaluate
from models.awsnet import AWSnet
from train import seed_worker, set_seed, train_epoch
from utils.criterions import AWSCompositeLoss, LossPRSnet_dicece
from utils.lr_scheduler import build_optimizer, build_scheduler
from utils.surface_distance import (
    BENCHMARK_PROTOCOL,
    binary_metrics,
    dataset_rows,
    summarize_dataset_rows,
    summarize_rows,
)


class DummyTrainDataset(Dataset):
    """Synthetic dataset for testing reproducibility."""

    def __init__(self, num_samples: int = 16, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.data = []
        for i in range(num_samples):
            img = np.full((128, 128), fill_value=(i + 1) / (num_samples + 1), dtype=np.float32)
            lbl = np.zeros((128, 128), dtype=np.int64)
            lbl[30:50, 30:50] = (i % 3) + 1
            self.data.append((img, lbl))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, lbl = self.data[idx]
        sample = {
            "image": img.copy(),
            "image1": img.copy(),
            "image2": img.copy(),
            "label": lbl.copy(),
            "case_id": f"sample_{idx:03d}",
        }
        if self.transform:
            sample = self.transform(sample)

        t_c0 = torch.from_numpy(np.ascontiguousarray(sample["image"], dtype=np.float32)).unsqueeze(0)
        t_lge = torch.from_numpy(np.ascontiguousarray(sample["image1"], dtype=np.float32)).unsqueeze(0)
        t_t2 = torch.from_numpy(np.ascontiguousarray(sample["image2"], dtype=np.float32)).unsqueeze(0)
        t_lbl = torch.from_numpy(np.ascontiguousarray(sample["label"], dtype=np.int64))
        t_3ch = torch.cat([t_c0, t_lge, t_t2], dim=0)

        return {
            "image": t_c0,
            "image1": t_lge,
            "image2": t_t2,
            "image_3ch": t_3ch,
            "label": t_lbl,
            "sample_idx": idx,
        }


class TestAdversarialEmpirical(unittest.TestCase):
    """Empirical verification suite for Challenger 1."""

    # =========================================================================
    # Mission Item 1: Stochastic Reproducibility & DataLoader Generator State
    # =========================================================================
    def test_dataloader_generator_resumption_reproducibility(self):
        """Verifies that saving and restoring loader_rng genuinely produces identical batch sequence."""
        seed = 42
        set_seed(seed, deterministic=True)
        g = torch.Generator()
        g.manual_seed(seed)

        # Dataset with SCAR RandomGenerator augmentation
        ds = DummyTrainDataset(num_samples=20, transform=RandomGenerator(output_size=(128, 128)))
        loader = DataLoader(
            ds,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # Run Epoch 0
        epoch0_indices = []
        for batch in loader:
            epoch0_indices.extend(batch["sample_idx"].tolist())

        # Save RNG states at boundary between Epoch 0 and Epoch 1 (exactly as in train.py)
        saved_loader_rng = g.get_state().clone()
        saved_torch_rng = torch.get_rng_state().clone()
        saved_np_rng = np.random.get_state()
        saved_py_rng = random.getstate()

        # Run Epoch 1 uninterrupted (Sequence A)
        seq_a_indices = []
        seq_a_data_sums = []
        for batch in loader:
            seq_a_indices.extend(batch["sample_idx"].tolist())
            seq_a_data_sums.append(float(batch["image_3ch"].sum().item()))

        # Now simulate a fresh session resumption:
        # Create a new generator and restore the exact saved states
        resumed_g = torch.Generator()
        resumed_g.set_state(saved_loader_rng)
        torch.set_rng_state(saved_torch_rng)
        np.random.set_state(saved_np_rng)
        random.setstate(saved_py_rng)

        resumed_loader = DataLoader(
            ds,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=resumed_g,
        )

        # Run Epoch 1 resumed (Sequence B)
        seq_b_indices = []
        seq_b_data_sums = []
        for batch in resumed_loader:
            seq_b_indices.extend(batch["sample_idx"].tolist())
            seq_b_data_sums.append(float(batch["image_3ch"].sum().item()))

        # Empirical assertion: sample order and batch sequence MUST be identical
        self.assertEqual(
            seq_a_indices,
            seq_b_indices,
            "Resumed DataLoader produced a DIFFERENT sample index sequence than uninterrupted run!",
        )

        # Check augmented pixel data values: MUST match with high precision
        for i, (val_a, val_b) in enumerate(zip(seq_a_data_sums, seq_b_data_sums)):
            self.assertAlmostEqual(
                val_a,
                val_b,
                places=4,
                msg=f"Batch {i} augmented data tensor differed between resumed and uninterrupted run!",
            )

    # =========================================================================
    # Mission Item 2: Gradient Flow & Deep Supervision
    # =========================================================================
    def test_gradient_flow_all_four_stages_and_clip_grad(self):
        """Verifies non-zero, finite gradients on all 4 deep supervision stages and clip_grad_norm_."""
        device = torch.device("cpu")
        model = AWSnet(
            in_channels=3,
            num_classes=4,
            base_filters=16,
            deep_supervision=True,
        ).to(device)
        model.train()

        batch_size = 2
        x = torch.randn(batch_size, 3, 128, 128, device=device, requires_grad=False)
        target = torch.randint(0, 4, (batch_size, 128, 128), device=device)

        # Forward pass
        outputs = model(x)
        self.assertIsInstance(outputs, tuple, "Model forward in train mode did not return a tuple.")
        self.assertEqual(len(outputs), 5, f"Expected 5 output tensors (4 stages + primary), got {len(outputs)}")

        out4, out3, out2, out1, out_primary = outputs

        # 1. Verify individual gradient flow for each stage
        # Stage 4: 1/8 resolution head (final3)
        # Stage 3: 1/4 resolution head (final2)
        # Stage 2: 1/2 resolution head (final1)
        # Stage 1: Full resolution head (final)
        heads = [
            ("Stage 4 (1/8 res)", out4, model.final3),
            ("Stage 3 (1/4 res)", out3, model.final2),
            ("Stage 2 (1/2 res)", out2, model.final1),
            ("Stage 1 (full res)", out1, model.final),
        ]

        criterion = LossPRSnet_dicece(num_classes=4)

        for name, stage_out, head_module in heads:
            model.zero_grad()
            stage_loss = F.cross_entropy(stage_out, target)
            stage_loss.backward(retain_graph=True)

            head_grad = head_module.weight.grad
            self.assertIsNotNone(head_grad, f"{name} head weight grad is None.")
            self.assertTrue(torch.isfinite(head_grad).all(), f"{name} head weight grad has non-finite values.")
            self.assertGreater(torch.count_nonzero(head_grad).item(), 0, f"{name} head weight grad is entirely zero.")

        # 2. Joint loss backward through full LossPRSnet_dicece objective
        model.zero_grad()
        total_loss, metrics = criterion(out4, out3, out2, out1, target)
        self.assertTrue(torch.isfinite(total_loss), "Total deep supervision loss is non-finite.")
        total_loss.backward()

        # Verify gradient flow to encoder, decoders, attention gates, and all 4 heads
        components_to_check = [
            ("Encoder stage 1", model.Conv1.conv[0].weight),
            ("Encoder stage 5 (bottleneck)", model.Conv5.conv[0].weight),
            ("Attention gate 5", model.Att5.W_g[0].weight),
            ("Attention gate 2", model.Att2.psi[0].weight),
            ("Decoder stage 5", model.Up_conv5.conv[0].weight),
            ("Decoder stage 2", model.Up_conv2.conv[0].weight),
            ("Deep supervision head 4", model.final3.weight),
            ("Deep supervision head 3", model.final2.weight),
            ("Deep supervision head 2", model.final1.weight),
            ("Deep supervision head 1", model.final.weight),
        ]

        for comp_name, param in components_to_check:
            self.assertIsNotNone(param.grad, f"{comp_name} gradient is None.")
            self.assertTrue(torch.isfinite(param.grad).all(), f"{comp_name} gradient is non-finite.")
            self.assertGreater(
                torch.count_nonzero(param.grad).item(),
                0,
                f"{comp_name} gradient is all zeros; deep supervision flow interrupted!",
            )

        # 3. Test torch.nn.utils.clip_grad_norm_ functionality
        # Amplify gradients artificially to test clipping
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(1000.0)

        pre_clip_norm = torch.sqrt(sum(p.grad.norm(2) ** 2 for p in model.parameters() if p.grad is not None)).item()
        self.assertGreater(pre_clip_norm, 10.0, "Pre-clip norm is not sufficiently large for clipping test.")

        clip_max = 1.0
        reported_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max)
        self.assertTrue(torch.isfinite(reported_norm), "clip_grad_norm_ returned non-finite norm.")
        self.assertAlmostEqual(reported_norm.item(), pre_clip_norm, places=2)

        post_clip_norm = torch.sqrt(sum(p.grad.norm(2) ** 2 for p in model.parameters() if p.grad is not None)).item()
        self.assertLessEqual(
            post_clip_norm,
            clip_max + 1e-4,
            f"clip_grad_norm_ failed: post-clip norm {post_clip_norm} exceeds max_norm {clip_max}.",
        )

    # =========================================================================
    # Mission Item 3: Strict Normalization Boundary Check
    # =========================================================================
    def test_strict_normalization_boundary_checks(self):
        """Verifies strict boundary validation: [-1e-5, 1.0 + 1e-5] passes, out-of-bounds raises ValueError."""
        # 1. In-bound valid arrays within [0.0, 1.0] -> zero errors
        valid_cases = [
            np.array([0.0, 0.5, 1.0], dtype=np.float32),
            np.zeros((128, 128), dtype=np.float32),
            np.ones((128, 128), dtype=np.float32),
            np.full((64, 64), fill_value=0.42, dtype=np.float32),
            # Boundary exact tolerances: -1e-5 and 1.0 + 1e-5
            np.array([-1e-5, 0.5, 1.0 + 1e-5], dtype=np.float32),
            np.array([-0.000009, 1.000009], dtype=np.float32),
        ]
        for arr in valid_cases:
            res = ingest_image_array(arr, name="valid_test")
            self.assertIsInstance(res, np.ndarray)
            self.assertEqual(res.dtype, np.float32)

        # 2. Out-of-bounds below -1e-5 -> strictly raises ValueError
        lower_violations = [
            np.array([-1e-4, 0.5, 1.0], dtype=np.float32),
            np.array([-2e-5, 0.5, 1.0], dtype=np.float32),
            np.array([-0.001, 0.0, 1.0], dtype=np.float32),
            np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            np.array([-255.0, 0.0, 1.0], dtype=np.float32),
        ]
        for arr in lower_violations:
            with self.assertRaises(ValueError, msg=f"Failed to raise ValueError for lower violation min={arr.min()}"):
                ingest_image_array(arr, name="lower_violation")

        # 3. Out-of-bounds above 1.0 + 1e-5 -> strictly raises ValueError
        upper_violations = [
            np.array([0.0, 0.5, 1.001], dtype=np.float32),
            np.array([0.0, 0.5, 1.0001], dtype=np.float32),
            np.array([0.0, 0.5, 1.00005], dtype=np.float32),
            np.array([0.0, 0.5, 1.01], dtype=np.float32),
            np.array([0.0, 0.5, 2.0], dtype=np.float32),
            np.array([0.0, 127.5, 255.0], dtype=np.float32),  # Raw 0..255 without preprocessing
        ]
        for arr in upper_violations:
            with self.assertRaises(ValueError, msg=f"Failed to raise ValueError for upper violation max={arr.max()}"):
                ingest_image_array(arr, name="upper_violation")

        # 4. Non-finite values -> strictly raises ValueError
        non_finite_cases = [
            np.array([0.0, np.nan, 1.0], dtype=np.float32),
            np.array([0.0, np.inf, 0.5], dtype=np.float32),
            np.array([-np.inf, 0.5, 1.0], dtype=np.float32),
        ]
        for arr in non_finite_cases:
            with self.assertRaises(ValueError, msg="Failed to raise ValueError on non-finite array"):
                ingest_image_array(arr, name="non_finite")

    # =========================================================================
    # Mission Item 4: Evaluation Engine Metric Calculation & Empty Policy
    # =========================================================================
    def test_evaluation_engine_metric_calculation_and_empty_policy(self):
        """Feeds synthetic 3D volume masks testing all 4 empty policy conditions."""
        shape = (128, 128, 6)

        # Condition (a): Both pred and target are empty for a class (verify None returned)
        pred_a = np.zeros(shape, dtype=np.uint8)
        target_a = np.zeros(shape, dtype=np.uint8)
        # Put non-empty label only in class 1 (myo) to keep volumes non-empty
        pred_a[10:30, 10:30, :] = 1
        target_a[10:30, 10:30, :] = 1
        # Class 2 (edema) and Class 3 (scar) are BOTH completely empty in both pred and target
        metrics_a = binary_metrics(pred_a == 3, target_a == 3, compute_distance=True)
        self.assertIsNone(metrics_a["dice"], "Condition (a): Expected None for dice when both empty.")
        self.assertIsNone(metrics_a["iou"], "Condition (a): Expected None for iou when both empty.")
        self.assertIsNone(metrics_a["hd95"], "Condition (a): Expected None for hd95 when both empty.")
        self.assertIsNone(metrics_a["asd"], "Condition (a): Expected None for asd when both empty.")
        self.assertEqual(metrics_a["status"], "both_empty")

        rows_a = dataset_rows(pred_a, target_a, case="case_both_empty", compute_distance=True, dataset_id="myops380")
        scar_row_a = next(r for r in rows_a if r["region"] == "scar")
        self.assertIsNone(scar_row_a["dice"])
        self.assertIsNone(scar_row_a["hd95_voxel"])
        self.assertEqual(scar_row_a["status"], "both_empty")

        # Condition (b): Pred has pixels but target is empty (verify Dice = 0.0, HD95 = None)
        pred_b = np.zeros(shape, dtype=np.uint8)
        target_b = np.zeros(shape, dtype=np.uint8)
        pred_b[40:60, 40:60, :] = 3  # Pred has scar pixels
        # Target has no scar pixels (empty)
        target_b[10:30, 10:30, :] = 1  # Foreground elsewhere so volume valid

        metrics_b = binary_metrics(pred_b == 3, target_b == 3, compute_distance=True)
        self.assertEqual(metrics_b["dice"], 0.0, "Condition (b): Expected Dice=0.0 when target is empty.")
        self.assertEqual(metrics_b["iou"], 0.0, "Condition (b): Expected IoU=0.0 when target is empty.")
        self.assertIsNone(metrics_b["hd95"], "Condition (b): Expected HD95=None when target is empty.")
        self.assertIsNone(metrics_b["asd"], "Condition (b): Expected ASD=None when target is empty.")
        self.assertEqual(metrics_b["status"], "target_empty")

        rows_b = dataset_rows(pred_b, target_b, case="case_target_empty", compute_distance=True, dataset_id="myops380")
        scar_row_b = next(r for r in rows_b if r["region"] == "scar")
        self.assertEqual(scar_row_b["dice"], 0.0)
        self.assertIsNone(scar_row_b["hd95_voxel"])
        self.assertEqual(scar_row_b["status"], "target_empty")

        # Condition (c): Target has pixels but pred is empty (verify Dice = 0.0, HD95 = None)
        pred_c = np.zeros(shape, dtype=np.uint8)
        target_c = np.zeros(shape, dtype=np.uint8)
        pred_c[10:30, 10:30, :] = 1  # Pred has myo
        target_c[40:60, 40:60, :] = 3  # Target has scar
        # Pred has no scar pixels (empty)

        metrics_c = binary_metrics(pred_c == 3, target_c == 3, compute_distance=True)
        self.assertEqual(metrics_c["dice"], 0.0, "Condition (c): Expected Dice=0.0 when pred is empty.")
        self.assertEqual(metrics_c["iou"], 0.0, "Condition (c): Expected IoU=0.0 when pred is empty.")
        self.assertIsNone(metrics_c["hd95"], "Condition (c): Expected HD95=None when pred is empty.")
        self.assertIsNone(metrics_c["asd"], "Condition (c): Expected ASD=None when pred is empty.")
        self.assertEqual(metrics_c["status"], "prediction_empty")

        rows_c = dataset_rows(pred_c, target_c, case="case_pred_empty", compute_distance=True, dataset_id="myops380")
        scar_row_c = next(r for r in rows_c if r["region"] == "scar")
        self.assertEqual(scar_row_c["dice"], 0.0)
        self.assertIsNone(scar_row_c["hd95_voxel"])
        self.assertEqual(scar_row_c["status"], "prediction_empty")

        # Condition (d): Both have non-empty overlapping pixels (verify finite Dice, HD95, ASD)
        pred_d = np.zeros(shape, dtype=np.uint8)
        target_d = np.zeros(shape, dtype=np.uint8)
        # Partially overlapping scar masks
        pred_d[30:50, 30:50, :] = 3
        target_d[35:55, 35:55, :] = 3

        metrics_d = binary_metrics(pred_d == 3, target_d == 3, compute_distance=True)
        self.assertGreater(metrics_d["dice"], 0.0, "Condition (d): Expected Dice > 0.0.")
        self.assertLessEqual(metrics_d["dice"], 1.0, "Condition (d): Expected Dice <= 1.0.")
        self.assertIsNotNone(metrics_d["hd95"], "Condition (d): Expected finite HD95.")
        self.assertIsNotNone(metrics_d["asd"], "Condition (d): Expected finite ASD.")
        self.assertTrue(np.isfinite(metrics_d["hd95"]), "Condition (d): HD95 is not finite.")
        self.assertTrue(np.isfinite(metrics_d["asd"]), "Condition (d): ASD is not finite.")
        self.assertEqual(metrics_d["status"], "ok")

        rows_d = dataset_rows(pred_d, target_d, case="case_overlapping", compute_distance=True, dataset_id="myops380")
        scar_row_d = next(r for r in rows_d if r["region"] == "scar")
        self.assertAlmostEqual(scar_row_d["dice"], metrics_d["dice"], places=5)
        self.assertAlmostEqual(scar_row_d["hd95_voxel"], metrics_d["hd95"], places=5)
        self.assertAlmostEqual(scar_row_d["asd_voxel"], metrics_d["asd"], places=5)
        self.assertEqual(scar_row_d["status"], "ok")

    # =========================================================================
    # Mission Item 5: Output Format Verification
    # =========================================================================
    def test_output_format_metrics_and_summary_json(self):
        """Verifies BOTH metrics.json and summary.json are written with flat root-level keys."""
        temp_dir = Path(tempfile.mkdtemp(prefix="awsnet_test_output_format_"))
        try:
            data_dir = temp_dir / "Dataset"
            split_dir = temp_dir / "splits"
            pred_dir = temp_dir / "test_predictions"
            eval_dir = temp_dir / "evaluation"

            for p in (data_dir, split_dir, pred_dir, eval_dir):
                p.mkdir(parents=True, exist_ok=True)

            for mod in ("bSSFP", "LGE", "T2w"):
                (data_dir / mod / "test_vol_h5").mkdir(parents=True, exist_ok=True)

            case_ids = ["test_case_001", "test_case_002"]
            # Write both train.txt and test_vol.txt so resolve_split_dir recognizes split_dir
            (split_dir / "train.txt").write_text("dummy_slice\n", encoding="utf-8")
            (split_dir / "test_vol.txt").write_text("\n".join(case_ids) + "\n", encoding="utf-8")

            # Create mock test volumes and predictions
            shape = (128, 128, 4)
            for cid in case_ids:
                vol = np.random.uniform(0.0, 1.0, shape).astype(np.float32)
                # Ground truth with myocardium, edema, and scar
                gt = np.zeros(shape, dtype=np.int64)
                gt[20:40, 20:40, :] = 1  # normal myocardium
                gt[40:50, 40:50, :] = 2  # edema
                gt[50:60, 50:60, :] = 3  # scar

                for mod in ("bSSFP", "LGE", "T2w"):
                    with h5py.File(data_dir / mod / "test_vol_h5" / f"{cid}.npy.h5", "w") as f:
                        f.create_dataset("image", data=vol)
                        f.create_dataset("label", data=gt)

                # Matching prediction with slight offset
                pred = np.zeros(shape, dtype=np.uint8)
                pred[20:42, 20:42, :] = 1
                pred[40:52, 40:52, :] = 2
                pred[50:62, 50:62, :] = 3

                np.savez_compressed(pred_dir / f"{cid}_pred.npz", prediction=pred)

            # Run evaluation using existing predictions
            eval_results = evaluate(
                predictions_dir=pred_dir,
                data_dir=data_dir,
                split_dir=split_dir,
                output_dir=eval_dir,
                split="test_vol",
                device_name="cpu",
                amp="none",
            )

            metrics_path = eval_dir / "metrics.json"
            summary_path = eval_dir / "summary.json"
            csv_path = eval_dir / "per_case.csv"

            # 1. Verify existence of all 3 evaluation artifacts
            self.assertTrue(metrics_path.is_file(), "metrics.json was NOT generated!")
            self.assertTrue(summary_path.is_file(), "summary.json was NOT generated!")
            self.assertTrue(csv_path.is_file(), "per_case.csv was NOT generated!")

            # 2. Inspect metrics.json
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            # 3. Inspect summary.json
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)

            # 4. Verify flat root-level keys required by benchmark protocol
            required_root_keys = [
                "normal_myocardium",
                "edema",
                "scar",
                "avg_pathology_dice",
            ]

            for key in required_root_keys:
                self.assertIn(
                    key,
                    metrics_data,
                    f"metrics.json missing required flat root-level key: {key!r}",
                )
                self.assertIn(
                    key,
                    summary_data,
                    f"summary.json missing required flat root-level key: {key!r}",
                )

            # Verify contents of regional summary dictionaries
            for reg in ("normal_myocardium", "edema", "scar"):
                reg_info = metrics_data[reg]
                self.assertIsInstance(reg_info, dict)
                self.assertIn("mean_dice", reg_info)
                self.assertIn("mean_hd95_voxel", reg_info)
                self.assertIn("mean_asd_voxel", reg_info)
                self.assertTrue(np.isfinite(reg_info["mean_dice"]))

            # Verify avg_pathology_dice is finite float
            self.assertIsNotNone(metrics_data["avg_pathology_dice"])
            self.assertTrue(np.isfinite(metrics_data["avg_pathology_dice"]))
            self.assertAlmostEqual(
                metrics_data["avg_pathology_dice"],
                (metrics_data["scar"]["mean_dice"] + metrics_data["edema"]["mean_dice"]) / 2.0,
                places=5,
            )

            # Verify metrics.json and summary.json root-level consistency
            for key in required_root_keys:
                self.assertEqual(metrics_data[key], summary_data[key])

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
