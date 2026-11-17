"""Integration and regression test suite for MyoPS++ and AWSnet benchmark environment."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.myops380_dataset import (
    MyoPS380TrainDataset,
    MyoPS380TestVolumeDataset,
    canonicalize_raw_myopspp,
)
from models.awsnet import AWSnet
from utils.criterions import LossPRSnet_dicece, AWSCompositeLoss
from utils.surface_distance import protocol_for_dataset, dataset_rows
from benchmark_efficiency import run_benchmark
from train import parse_args as parse_train_args
from evaluate import parse_args as parse_eval_args, load_surface_distance_module
from predict import parse_args as parse_predict_args
from utils.paths import resolve_data_root, resolve_split_dir


class TestMyoPSppLabelAndConsistency(unittest.TestCase):
    """Test label mappings, model architecture invariance, and loss symmetry."""

    def test_canonicalize_raw_myopspp_labels(self):
        """Verify raw MyoPS++ NIfTI labels map correctly to canonical indices."""
        # 0: bg, 500: LV cavity, 600: RV cavity -> 0 (background)
        # 200: normal myocardium -> 1
        # 1220: edema -> 2
        # 2221: scar -> 3
        raw = np.array([0, 500, 600, 200, 1220, 2221], dtype=np.int32)
        expected = np.array([0, 0, 0, 1, 2, 3], dtype=np.int64)
        mapped = canonicalize_raw_myopspp(raw)
        np.testing.assert_array_equal(mapped, expected)

    def test_model_architecture_and_loss_invariance(self):
        """Verify model output shape and loss calculation with 4 canonical classes."""
        model = AWSnet(in_channels=3, num_classes=4, base_filters=16, deep_supervision=True)
        x = torch.randn(2, 3, 128, 128, dtype=torch.float32)

        # In train mode with deep supervision
        model.train()
        outputs = model(x)
        self.assertEqual(len(outputs), 5)
        for out in outputs:
            self.assertEqual(out.shape, (2, 4, 128, 128))

        # In eval mode
        model.eval()
        with torch.no_grad():
            out_eval = model(x)
        self.assertEqual(out_eval.shape, (2, 4, 128, 128))

        # Loss calculation using AWSCompositeLoss
        criterion = AWSCompositeLoss(num_classes=4)
        target = torch.randint(0, 4, (2, 128, 128), dtype=torch.long)
        loss, metrics = criterion(outputs, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        # Single output evaluation loss
        loss_single, _ = criterion(out_eval, target)
        self.assertTrue(torch.isfinite(loss_single))


class TestMyoPSppDatasetLoading(unittest.TestCase):
    """Test real MyoPS++ dataset loading from D:/nckh_model/Dataset/MyoPSpp_UNet2D_ROI128_1mm."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = PROJECT_ROOT / "Dataset" / "MyoPSpp_UNet2D_ROI128_1mm"
        cls.split_dir = REPO_ROOT / "dataset" / "splits" / "myopspp_unet2d_roi128_1mm"

    def test_train_dataset_real_data(self):
        """Verify MyoPS++ train split loads 296 slices with correct shapes and labels."""
        if not self.data_dir.exists():
            self.skipTest(f"Data dir not found: {self.data_dir}")

        dataset = MyoPS380TrainDataset(
            data_dir=str(self.data_dir),
            split_dir=str(self.split_dir),
            split="train",
            dataset_id="myopspp_unet2d_roi128_1mm",
            label_order="canonical",
        )
        self.assertEqual(len(dataset), 296)

        sample = dataset[0]
        self.assertIn("image", sample)
        self.assertIn("image1", sample)
        self.assertIn("image2", sample)
        self.assertIn("label", sample)
        self.assertEqual(sample["image"].shape, (1, 128, 128))
        self.assertEqual(sample["image1"].shape, (1, 128, 128))
        self.assertEqual(sample["image2"].shape, (1, 128, 128))
        self.assertEqual(sample["label"].shape, (128, 128))

        # Check labels belong to {0, 1, 2, 3}
        unique_labels = np.unique(sample["label"])
        for lbl in unique_labels:
            self.assertIn(lbl, [0, 1, 2, 3])

    def test_test_dataset_real_data(self):
        """Verify MyoPS++ test volume loads 16 cases with mm spacing and geometry."""
        if not self.data_dir.exists():
            self.skipTest(f"Data dir not found: {self.data_dir}")

        test_dataset = MyoPS380TestVolumeDataset(
            data_dir=str(self.data_dir),
            split_dir=str(self.split_dir),
            split="test_vol",
            dataset_id="myopspp_unet2d_roi128_1mm",
            label_order="canonical",
        )
        self.assertEqual(len(test_dataset), 16)

        vol_sample = test_dataset[0]
        self.assertIn("image", vol_sample)
        self.assertIn("label", vol_sample)
        self.assertIn("spacing", vol_sample)
        self.assertIn("spacing_unit", vol_sample)

        # Crucial check: spacing is mm
        self.assertEqual(vol_sample["spacing_unit"], "mm")
        self.assertTrue(vol_sample.get("has_geometry", False))
        self.assertEqual(len(vol_sample["spacing"]), 3)


class TestSurfaceDistanceMetricsUnit(unittest.TestCase):
    """Test HD95 metric units for MyoPS++ (mm) vs MyoPS380 (voxel)."""

    def test_protocol_for_dataset(self):
        """Verify protocol metadata specifies correct distance units."""
        myopspp_proto = protocol_for_dataset("myopspp_unet2d_roi128_1mm")
        self.assertEqual(myopspp_proto.get("distance_unit"), "mm")

        myops380_proto = protocol_for_dataset("myops380")
        self.assertEqual(myops380_proto.get("distance_unit"), "voxel")

    def test_hd95_computation_in_mm(self):
        """Verify dataset_rows outputs hd95 in mm when physical spacing is given."""
        H, W, D = 128, 128, 5
        pred = np.zeros((H, W, D), dtype=np.int32)
        gt = np.zeros((H, W, D), dtype=np.int32)

        # Put a small square in pred and gt with offset
        pred[50:60, 50:60, :] = 1
        gt[52:62, 50:60, :] = 1

        rows = dataset_rows(
            prediction=pred,
            target=gt,
            case="case0001",
            dataset_id="myopspp_unet2d_roi128_1mm",
            spacing=(1.5, 1.5, 10.0),
        )
        self.assertGreater(len(rows), 0)
        row = rows[0]
        self.assertEqual(row["hd95_unit"], "mm")
        self.assertIn("hd95_mm", row)
        self.assertIsNotNone(row["hd95_mm"])


class TestEnvironmentScriptsSyntax(unittest.TestCase):
    """Test script structure, shebang, set -e, and CLI flag compatibility."""

    def test_script_existence_and_bash_syntax(self):
        """Verify both new environment scripts exist and pass syntax validation."""
        f1 = PROJECT_ROOT / "Environment_A100_4seeds_AWSnet.txt"
        f2 = PROJECT_ROOT / "Environment_Benchmark_Seed1234_AWSnet.txt"

        self.assertTrue(f1.exists(), f"Missing {f1}")
        self.assertTrue(f2.exists(), f"Missing {f2}")

        for f_path in (f1, f2):
            content = f_path.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("#!/bin/bash"), f"Missing bash shebang in {f_path.name}")
            self.assertIn("set -e", content, f"Missing set -e in {f_path.name}")
            self.assertEqual(content.count("("), content.count(")"), f"Unbalanced () in {f_path.name}")
            self.assertEqual(content.count("["), content.count("]"), f"Unbalanced [] in {f_path.name}")

    def test_script_4seeds_flags_parse(self):
        """Verify CLI flags from Environment_A100_4seeds_AWSnet.txt parse in train.py and evaluate.py."""
        train_args = [
            "train.py",
            "--data_dir", "/dummy/data",
            "--split_dir", "/dummy/split",
            "--save_dir", "/dummy/save",
            "--dataset", "myopspp_unet2d_roi128_1mm",
            "--label_order", "canonical",
            "--seed", "6868",
            "--batch_size", "16",
            "--accum_steps", "1",
            "--max_epochs", "300",
            "--base_lr", "0.0003333",
            "--weight_decay", "0.0001",
            "--num_workers", "4",
            "--amp", "auto",
            "--resume",
        ]
        import unittest.mock
        with unittest.mock.patch.object(sys, "argv", train_args):
            parsed = parse_train_args()
            self.assertEqual(parsed.seed, 6868)
            self.assertEqual(parsed.dataset, "myopspp_unet2d_roi128_1mm")

        eval_args = [
            "evaluate.py",
            "--checkpoint", "/dummy/ckpt.pt",
            "--data_dir", "/dummy/data",
            "--split_dir", "/dummy/split",
            "--dataset", "myopspp_unet2d_roi128_1mm",
            "--output_dir", "/dummy/eval",
            "--batch_size", "8",
            "--amp", "auto",
        ]
        with unittest.mock.patch.object(sys, "argv", eval_args):
            parsed_eval = parse_eval_args()
            self.assertEqual(parsed_eval.dataset, "myopspp_unet2d_roi128_1mm")

    def test_script_seed1234_flags_parse(self):
        """Verify CLI flags from Environment_Benchmark_Seed1234_AWSnet.txt parse."""
        train_args = [
            "train.py",
            "--data_dir", "/dummy/data",
            "--split_dir", "/dummy/split",
            "--save_dir", "/dummy/save",
            "--dataset", "myopspp_unet2d_roi128_1mm",
            "--label_order", "canonical",
            "--seed", "1234",
            "--max_epochs", "300",
            "--batch_size", "16",
            "--accum_steps", "1",
            "--base_lr", "0.0003333",
            "--weight_decay", "0.0001",
            "--amp", "none",
            "--num_workers", "0",
            "--pin_memory",
            "--resume",
        ]
        import unittest.mock
        with unittest.mock.patch.object(sys, "argv", train_args):
            parsed = parse_train_args()
            self.assertEqual(parsed.seed, 1234)
            self.assertEqual(parsed.amp, "none")
            self.assertEqual(parsed.num_workers, 0)
            self.assertTrue(parsed.pin_memory)


class TestBenchmarkEfficiencyExecution(unittest.TestCase):
    """Test running benchmark_efficiency.py in dry-run mode."""

    def test_benchmark_efficiency_cpu(self):
        """Verify benchmark_efficiency runs and produces valid 7-metric report."""
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="test_bench_eff_"))
        out_file = temp_dir / "efficiency_metrics.json"

        try:
            report = run_benchmark(
                output_path=out_file,
                device_name="cpu",
                seed=1234,
                base_filters=16,
            )
            self.assertTrue(out_file.exists())
            self.assertEqual(report["model_id"], "awsnet")
            self.assertIn("complexity", report)
            self.assertIn("inference", report)
            self.assertIn("training", report)
            self.assertGreater(report["complexity"]["total_params_M"], 0)
            self.assertGreater(report["complexity"]["flops_G"], 0)
            self.assertGreater(report["inference"]["latency_ms_per_slice"], 0)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_benchmark_efficiency_myopspp_metadata(self):
        """Verify benchmark_efficiency training block reflects MyoPS++ when selected."""
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="test_bench_eff_pp_"))
        out_file = temp_dir / "efficiency_metrics.json"

        try:
            report = run_benchmark(
                output_path=out_file,
                device_name="cpu",
                seed=1234,
                base_filters=16,
                dataset="myopspp_unet2d_roi128_1mm",
            )
            self.assertTrue(out_file.exists())
            self.assertEqual(report["protocol"], "myopspp_roi128_train64_test16_no_val_v1")
            self.assertEqual(report["training"]["train_slices"], 296)
            self.assertEqual(report["training"]["train_patients"], 64)
            self.assertEqual(report["training"]["test_patients"], 16)
            self.assertEqual(report["training"]["steps_per_epoch"], 19)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestPathResolutionAndCLI(unittest.TestCase):
    """Test path resolution heuristics and CLI argument handling."""

    def test_path_resolution_respects_dataset_flag(self):
        """Verify resolve_data_root and resolve_split_dir prioritize dataset argument."""
        # MyoPS380 resolution
        root_380 = resolve_data_root(None, dataset="myops380")
        split_380 = resolve_split_dir(None, dataset="myops380")
        self.assertIn("MyoPS380", str(root_380))
        self.assertIn("myops380_no_val", str(split_380))

        # MyoPS++ resolution
        root_pp = resolve_data_root(None, dataset="myopspp_unet2d_roi128_1mm")
        split_pp = resolve_split_dir(None, dataset="myopspp_unet2d_roi128_1mm")
        self.assertIn("MyoPSpp", str(root_pp))
        self.assertIn("myopspp_unet2d_roi128_1mm", str(split_pp))

        # Infer split from data_root when dataset is auto
        split_auto_pp = resolve_split_dir(None, dataset="auto", data_root=root_pp)
        self.assertIn("myopspp_unet2d_roi128_1mm", str(split_auto_pp))

    def test_predict_cli_supports_dataset(self):
        """Verify predict.py CLI parses --dataset argument."""
        import unittest.mock
        predict_args = [
            "predict.py",
            "--checkpoint", "/dummy/ckpt.pt",
            "--dataset", "myopspp_unet2d_roi128_1mm",
            "--output_dir", "/dummy/pred",
        ]
        with unittest.mock.patch.object(sys, "argv", predict_args):
            parsed = parse_predict_args()
            self.assertEqual(parsed.dataset, "myopspp_unet2d_roi128_1mm")

    def test_surface_distance_module_loads_myopspp_protocol(self):
        """Verify load_surface_distance_module provides valid profile for myopspp_unet2d_roi128_1mm."""
        mod = load_surface_distance_module()
        proto = mod.protocol_for_dataset("myopspp_unet2d_roi128_1mm")
        self.assertEqual(proto["distance_unit"], "mm")
        self.assertEqual(proto["id"], "myopspp_unet2d_roi128_1mm_roi_mm_v2")


if __name__ == "__main__":
    unittest.main()
