"""Edge case unit tests for AWSnet MyoPS380 pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import json
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.myops380_dataset import (
    canonicalize_label,
    ingest_image_array,
    read_split_names,
    extract_patient_id,
    MyoPS380TrainDataset,
    MyoPS380TestVolumeDataset,
)
from dataset.data_augmentation import RandomGenerator, ResizeGenerator
from train import resolve_amp, parse_args as parse_train_args
from evaluate import parse_args as parse_eval_args
from utils.lr_scheduler import build_optimizer, build_scheduler
from utils.surface_distance import dataset_rows, summarize_dataset_rows
import test as test_wrapper


class TestEdgeCases(unittest.TestCase):
    def test_canonicalize_label_mapping(self):
        """Tests that legacy 2 (scar) becomes canonical 3, and legacy 3 (edema) becomes 2."""
        legacy = np.array([0, 1, 2, 3], dtype=np.int64)
        canonical = canonicalize_label(legacy, source_order="legacy")
        np.testing.assert_array_equal(canonical, np.array([0, 1, 3, 2], dtype=np.int64))

        # Identity on canonical
        id_canonical = canonicalize_label(canonical, source_order="canonical")
        np.testing.assert_array_equal(id_canonical, canonical)

    def test_ingest_image_array_strict_validation(self):
        """Tests that [0, 1] data is preserved while unnormalized data raises ValueError."""
        arr_01 = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        res_01 = ingest_image_array(arr_01)
        np.testing.assert_allclose(res_01, arr_01)

        # Boundary tolerance: within 1e-5 is accepted
        arr_boundary = np.array([-0.000005, 1.000005], dtype=np.float32)
        res_boundary = ingest_image_array(arr_boundary)
        np.testing.assert_allclose(res_boundary, arr_boundary)

        # Unnormalized data in [0, 255] strictly raises ValueError (no silent division)
        arr_255 = np.array([0.0, 127.5, 255.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            ingest_image_array(arr_255)

        # Negative out of bounds strictly raises ValueError
        with self.assertRaises(ValueError):
            ingest_image_array(np.array([-0.1, 0.5, 1.0]))

        # Positive out of bounds strictly raises ValueError
        with self.assertRaises(ValueError):
            ingest_image_array(np.array([0.0, 0.5, 1.5]))

        # Non-finite raises ValueError
        with self.assertRaises(ValueError):
            ingest_image_array(np.array([0.0, np.nan, 1.0]))

    def test_polynomial_lr_schedule(self):
        """Tests polynomial scheduler zero-based iteration offset."""
        model = torch.nn.Linear(10, 2)
        optimizer = build_optimizer(model, base_lr=0.0003333)
        total_updates = 100
        scheduler = build_scheduler(optimizer, total_updates=total_updates, power=0.9)

        # First update (step 0 before advance or step 1): LR remains base_lr
        initial_lr = scheduler.get_last_lr()[0]
        self.assertAlmostEqual(initial_lr, 0.0003333, places=7)

        # Advance 50 steps
        for _ in range(50):
            optimizer.step()
            scheduler.step()

        mid_lr = scheduler.get_last_lr()[0]
        expected_mid = 0.0003333 * (max(0.0, 1.0 - (50 - 1) / total_updates) ** 0.9)
        self.assertAlmostEqual(mid_lr, expected_mid, places=7)

    def test_resolve_amp_modes(self):
        """Tests precision resolution for CPU and CUDA."""
        cpu_device = torch.device("cpu")
        self.assertIsNone(resolve_amp("auto", cpu_device))
        self.assertIsNone(resolve_amp("none", cpu_device))
        with self.assertRaises(ValueError):
            resolve_amp("fp16", cpu_device)

    def test_sample_weighted_accumulation_gradient_division(self):
        """Verifies mathematical correctness of sample-weighted gradient accumulation."""
        # Simple linear model: y = w * x
        model = nn.Linear(1, 1, bias=False)
        nn.init.constant_(model.weight, 1.0)

        # 2 micro-batches with different batch sizes (batch 1: size 2, batch 2: size 1)
        x1 = torch.tensor([[2.0], [4.0]])
        y1 = torch.tensor([[0.0], [0.0]])
        # loss1 per sample = mean((w*x)^2) = (4 + 16)/2 = 10.0
        # loss1 * batch_size = 20.0
        # d(loss1*2)/dw = d(w^2 * (4+16))/dw = 2*w*20 = 40.0

        x2 = torch.tensor([[3.0]])
        y2 = torch.tensor([[0.0]])
        # loss2 per sample = (w*3)^2 = 9.0
        # loss2 * batch_size = 9.0
        # d(loss2*1)/dw = d(w^2 * 9)/dw = 2*w*9 = 18.0

        model.zero_grad()
        group_samples = 0

        # Step 1
        l1 = F.mse_loss(model(x1), y1)
        (l1 * 2).backward()
        group_samples += 2

        # Step 2
        l2 = F.mse_loss(model(x2), y2)
        (l2 * 1).backward()
        group_samples += 1

        # Total unscaled gradient = 40.0 + 18.0 = 58.0
        self.assertEqual(group_samples, 3)

        # Correct sample-weighted accumulation: divide by group_samples (3)
        model.weight.grad.div_(group_samples)
        expected_grad = (40.0 + 18.0) / 3.0
        self.assertAlmostEqual(model.weight.grad.item(), expected_grad, places=5)

        # In contrast, hardcoded division by accum_steps=2 gives 58.0 / 2 = 29.0 (WRONG!)
        self.assertNotAlmostEqual(model.weight.grad.item(), 58.0 / 2.0, places=3)

    def test_manifest_integrity(self):
        """Tests that train.txt has 1972 slices, test_vol.txt has 76 cases, and zero overlap."""
        train_path = repo_root / "dataset" / "splits" / "myops380_no_val" / "train.txt"
        test_path = repo_root / "dataset" / "splits" / "myops380_no_val" / "test_vol.txt"

        if not train_path.is_file() or not test_path.is_file():
            self.skipTest("Manifest files not present in expected local path.")

        train_slices = read_split_names(train_path)
        test_cases = read_split_names(test_path)

        self.assertEqual(len(train_slices), 1972)
        self.assertEqual(len(test_cases), 76)

        train_patients = {extract_patient_id(s) for s in train_slices}
        self.assertEqual(len(train_patients), 304)

        overlap = train_patients.intersection(set(test_cases))
        self.assertEqual(len(overlap), 0, f"Data leakage detected! Overlapping patients: {overlap}")

    def test_surface_distance_empty_masks(self):
        """Tests empty-mask handling under myops380_voxel_v3 protocol."""
        pred = np.zeros((128, 128, 5), dtype=np.uint8)
        gt = np.zeros((128, 128, 5), dtype=np.uint8)

        # Both empty: dice should be None / undefined, status 'both_empty'
        rows_both_empty = dataset_rows(pred, gt, case="case_empty", compute_distance=True, dataset_id="myops380")
        for r in rows_both_empty:
            self.assertEqual(r["status"], "both_empty")
            self.assertIsNone(r["dice"])
            self.assertIsNone(r["hd95_voxel"])

        # One empty (GT has foreground, pred empty): dice=0, distances undefined, status 'prediction_empty'
        gt[60:70, 60:70, :] = 3  # Scar foreground
        rows_one_empty = dataset_rows(pred, gt, case="case_one_empty", compute_distance=True, dataset_id="myops380")
        scar_row = next(r for r in rows_one_empty if r["region"] == "scar")
        self.assertEqual(scar_row["status"], "prediction_empty")
        self.assertEqual(scar_row["dice"], 0.0)
        self.assertIsNone(scar_row["hd95_voxel"])

    def test_evaluate_summary_keys_alignment(self):
        """Tests that summarize_dataset_rows produces mean_dice and mean_hd95_voxel correctly."""
        pred = np.zeros((128, 128, 2), dtype=np.uint8)
        gt = np.zeros((128, 128, 2), dtype=np.uint8)
        pred[50:60, 50:60, :] = 3
        gt[50:60, 50:60, :] = 3
        rows = dataset_rows(pred, gt, case="case_perfect", compute_distance=True, dataset_id="myops380")
        summary = summarize_dataset_rows(rows, dataset_id="myops380")
        scar_summary = summary["scar"]
        self.assertIn("mean_dice", scar_summary)
        self.assertAlmostEqual(scar_summary["mean_dice"], 1.0)

    def test_train_dataset_keys_contract(self):
        """Tests that MyoPS380TrainDataset returns both SCAR canonical keys and AWSnet convenience keys."""
        temp_dir = Path(tempfile.mkdtemp(prefix="awsnet_test_train_keys_"))
        try:
            data_dir = temp_dir / "Dataset"
            split_dir = temp_dir / "splits"
            split_dir.mkdir(parents=True)
            for mod in ("bSSFP", "LGE", "T2w"):
                (data_dir / mod / "train_npz").mkdir(parents=True)

            slice_name = "case0002_slice000"
            (split_dir / "train.txt").write_text(f"{slice_name}\n", encoding="utf-8")

            img = np.zeros((128, 128), dtype=np.float32)
            lbl = np.zeros((128, 128), dtype=np.int64)
            for mod in ("bSSFP", "LGE", "T2w"):
                np.savez_compressed(data_dir / mod / "train_npz" / f"{slice_name}.npz", image=img, label=lbl)

            ds = MyoPS380TrainDataset(data_dir=data_dir, split_dir=split_dir, split="train", transform=ResizeGenerator((128, 128)))
            item = ds[0]

            # Canonical SCAR keys
            self.assertIn("image", item)
            self.assertIn("image1", item)
            self.assertIn("image2", item)
            self.assertIn("label", item)
            self.assertIn("case_name", item)
            self.assertIn("patient_id", item)
            self.assertEqual(item["case_name"], slice_name)
            self.assertEqual(item["patient_id"], "case0002")
            self.assertEqual(item["image"].shape, (1, 128, 128))
            self.assertEqual(item["image1"].shape, (1, 128, 128))
            self.assertEqual(item["image2"].shape, (1, 128, 128))
            self.assertEqual(item["label"].shape, (128, 128))

            # Convenience AWSnet keys
            self.assertIn("image_3ch", item)
            self.assertIn("c0", item)
            self.assertIn("lge", item)
            self.assertIn("t2", item)
            self.assertIn("case_id", item)
            self.assertEqual(item["image_3ch"].shape, (3, 128, 128))
            self.assertEqual(item["case_id"], slice_name)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_test_volume_dataset_keys_contract(self):
        """Tests that MyoPS380TestVolumeDataset returns SCAR canonical keys and geometry metadata."""
        import h5py
        temp_dir = Path(tempfile.mkdtemp(prefix="awsnet_test_vol_keys_"))
        try:
            data_dir = temp_dir / "Dataset"
            split_dir = temp_dir / "splits"
            split_dir.mkdir(parents=True)
            for mod in ("bSSFP", "LGE", "T2w"):
                (data_dir / mod / "test_vol_h5").mkdir(parents=True)

            case_id = "case0001"
            (split_dir / "test_vol.txt").write_text(f"{case_id}\n", encoding="utf-8")

            vol = np.zeros((128, 128, 4), dtype=np.float32)
            lbl = np.zeros((128, 128, 4), dtype=np.int64)
            for mod in ("bSSFP", "LGE", "T2w"):
                with h5py.File(data_dir / mod / "test_vol_h5" / f"{case_id}.npy.h5", "w") as f:
                    f.create_dataset("image", data=vol)
                    f.create_dataset("label", data=lbl)

            ds = MyoPS380TestVolumeDataset(data_dir=data_dir, split_dir=split_dir, split_file="test_vol.txt")
            item = ds[0]

            # Canonical SCAR keys
            self.assertIn("image", item)
            self.assertIn("image1", item)
            self.assertIn("image2", item)
            self.assertIn("label", item)
            self.assertIn("case_name", item)
            self.assertIn("patient_id", item)
            self.assertIn("spacing", item)
            self.assertIn("affine", item)
            self.assertIn("spacing_unit", item)
            self.assertIn("has_geometry", item)
            self.assertEqual(item["case_name"], case_id)
            self.assertEqual(item["patient_id"], case_id)
            self.assertFalse(item["has_geometry"])

            # Convenience keys
            self.assertIn("c0", item)
            self.assertIn("lge", item)
            self.assertIn("t2", item)
            self.assertIn("case_id", item)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_train_cli_parser_aliases(self):
        """Tests that train.py parse_args properly resolves all kebab-case aliases."""
        test_args = [
            "train.py",
            "--data-root", "/dummy/data",
            "--list-dir", "/dummy/splits",
            "--run-root", "/dummy/runs",
            "--batch-size", "8",
            "--accum-steps", "2",
            "--max-epochs", "250",
            "--base-lr", "0.0005",
            "--weight-decay", "0.0002",
            "--num-workers", "4",
            "--clip-grad", "1.0",
            "--config", "some_config.yaml",
            "--log-every", "10",
        ]
        import unittest.mock
        with unittest.mock.patch.object(sys, "argv", test_args):
            parsed = parse_train_args()
            self.assertEqual(parsed.data_dir, "/dummy/data")
            self.assertEqual(parsed.split_dir, "/dummy/splits")
            self.assertEqual(parsed.save_dir, "/dummy/runs")
            self.assertEqual(parsed.batch_size, 8)
            self.assertEqual(parsed.accum_steps, 2)
            self.assertEqual(parsed.max_epochs, 250)
            self.assertEqual(parsed.base_lr, 0.0005)
            self.assertEqual(parsed.weight_decay, 0.0002)
            self.assertEqual(parsed.num_workers, 4)
            self.assertEqual(parsed.clip_grad, 1.0)
            self.assertEqual(parsed.config, "some_config.yaml")
            self.assertEqual(parsed.log_interval, 10)

    def test_eval_cli_parser_aliases(self):
        """Tests that evaluate.py parse_args resolves aliases and default options."""
        test_args = [
            "evaluate.py",
            "--checkpoint", "/dummy/ckpt.pt",
            "--data-root", "/dummy/data",
            "--list-dir", "/dummy/splits",
            "--output-dir", "/dummy/eval",
            "--split", "test_vol",
            "--batch-size", "16",
        ]
        import unittest.mock
        with unittest.mock.patch.object(sys, "argv", test_args):
            parsed = parse_eval_args()
            self.assertEqual(parsed.checkpoint, "/dummy/ckpt.pt")
            self.assertEqual(parsed.data_dir, "/dummy/data")
            self.assertEqual(parsed.split_dir, "/dummy/splits")
            self.assertEqual(parsed.output_dir, "/dummy/eval")
            self.assertEqual(parsed.split, "test_vol")
            self.assertEqual(parsed.batch_size, 16)

    def test_test_wrapper_entrypoint(self):
        """Tests that test.py correctly exposes parse_args and forwards to evaluate."""
        self.assertTrue(hasattr(test_wrapper, "main"))
        self.assertTrue(hasattr(test_wrapper, "parse_args"))
        self.assertTrue(hasattr(test_wrapper, "evaluate"))

    def test_loader_rng_serialization(self):
        """Tests that generator state is accurately serialized and deserialized."""
        g = torch.Generator()
        g.manual_seed(42)
        state_saved = g.get_state()

        # Draw a random number
        r1 = torch.rand(1, generator=g).item()

        # Restore state
        g_restored = torch.Generator()
        g_restored.set_state(state_saved)
        r2 = torch.rand(1, generator=g_restored).item()

        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
