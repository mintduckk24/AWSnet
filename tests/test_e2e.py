"""End-to-end integration test for AWSnet MyoPS380 pipeline."""
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

import h5py
import numpy as np
import torch

from train import train
from predict import run_prediction
from evaluate import evaluate


class TestAWSnetE2E(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="awsnet_test_e2e_"))

        # Setup mock dataset structure
        self.data_dir = self.temp_dir / "Dataset"
        self.split_dir = self.temp_dir / "splits"
        self.save_dir = self.temp_dir / "checkpoints"
        self.pred_dir = self.temp_dir / "test_predictions"
        self.eval_dir = self.temp_dir / "evaluation"

        for p in (self.split_dir, self.save_dir, self.pred_dir, self.eval_dir):
            p.mkdir(parents=True, exist_ok=True)

        for mod in ("bSSFP", "LGE", "T2w"):
            (self.data_dir / mod / "train_npz").mkdir(parents=True, exist_ok=True)
            (self.data_dir / mod / "test_vol_h5").mkdir(parents=True, exist_ok=True)

        # Create mock train slices: 4 slices from 2 patients
        self.train_slices = ["case0002_slice000", "case0002_slice001", "case0003_slice000", "case0003_slice001"]
        train_txt = self.split_dir / "train.txt"
        train_txt.write_text("\n".join(self.train_slices) + "\n", encoding="utf-8")

        for slice_name in self.train_slices:
            img = np.random.uniform(0.0, 1.0, (128, 128)).astype(np.float32)
            lbl = np.random.choice([0, 1, 2, 3], size=(128, 128)).astype(np.int64)
            for mod in ("bSSFP", "LGE", "T2w"):
                p = self.data_dir / mod / "train_npz" / f"{slice_name}.npz"
                np.savez_compressed(p, image=img, label=lbl)

        # Create mock test volumes: 2 patients, depth 4
        self.test_cases = ["case0001", "case0006"]
        test_txt = self.split_dir / "test_vol.txt"
        test_txt.write_text("\n".join(self.test_cases) + "\n", encoding="utf-8")

        for case_id in self.test_cases:
            vol = np.random.uniform(0.0, 1.0, (128, 128, 4)).astype(np.float32)
            lbl_vol = np.random.choice([0, 1, 2, 3], size=(128, 128, 4)).astype(np.int64)
            for mod in ("bSSFP", "LGE", "T2w"):
                p = self.data_dir / mod / "test_vol_h5" / f"{case_id}.npy.h5"
                with h5py.File(p, "w") as f:
                    f.create_dataset("image", data=vol)
                    f.create_dataset("label", data=lbl_vol)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_pipeline_train_predict_eval(self):
        """Tests complete pipeline: train 2 epochs (accum_steps=2) -> resume 1 epoch -> predict -> evaluate."""
        # Step 1: Initial training for 2 epochs with gradient accumulation
        ckpt_path = train(
            data_dir=self.data_dir,
            split_dir=self.split_dir,
            save_dir=self.save_dir,
            batch_size=2,
            accum_steps=2,
            max_epochs=2,
            base_filters=8,  # Tiny model for ultra-fast unit testing
            seed=1234,
            device_name="cpu",
            amp="none",
            num_workers=0,
            log_interval=1,
        )

        self.assertTrue(ckpt_path.is_file())
        self.assertEqual(ckpt_path.name, "awsnet_seed1234.pt")

        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt_data["completed_epochs"], 2)
        self.assertEqual(ckpt_data["model_id"], "awsnet")
        self.assertIn("loader_rng", ckpt_data)
        self.assertIn("successful_updates", ckpt_data)
        self.assertIn("amp_skipped_updates", ckpt_data)
        self.assertEqual(ckpt_data["amp_mode"], "none")

        # Step 2: Resume training to epoch 3
        resumed_ckpt = train(
            data_dir=self.data_dir,
            split_dir=self.split_dir,
            save_dir=self.save_dir,
            batch_size=2,
            accum_steps=2,
            max_epochs=3,
            base_filters=8,
            seed=1234,
            resume=str(ckpt_path),
            device_name="cpu",
            amp="none",
            num_workers=0,
            log_interval=1,
        )

        ckpt_data3 = torch.load(resumed_ckpt, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt_data3["completed_epochs"], 3)
        self.assertEqual(len(ckpt_data3["history"]), 3)
        self.assertGreater(ckpt_data3["successful_updates"], ckpt_data["successful_updates"])

        # Step 3: Run prediction
        pred_out = run_prediction(
            checkpoint_path=resumed_ckpt,
            data_dir=self.data_dir,
            split_dir=self.split_dir,
            output_dir=self.pred_dir,
            device_name="cpu",
            amp="none",
            allow_incomplete=True,
        )

        for case_id in self.test_cases:
            pred_file = pred_out / f"{case_id}_pred.npz"
            self.assertTrue(pred_file.is_file())
            with np.load(pred_file) as npz:
                self.assertIn("prediction", npz)
                pred_vol = npz["prediction"]
                self.assertEqual(pred_vol.shape, (128, 128, 4))
                self.assertEqual(pred_vol.dtype, np.uint8)
                self.assertTrue(np.all(pred_vol >= 0) and np.all(pred_vol <= 3))

        # Step 4: Run shared evaluation
        eval_results = evaluate(
            checkpoint=resumed_ckpt,
            predictions_dir=self.pred_dir,
            data_dir=self.data_dir,
            split_dir=self.split_dir,
            output_dir=self.eval_dir,
            device_name="cpu",
            amp="none",
            allow_incomplete=True,
        )

        self.assertIn("summary", eval_results)
        self.assertEqual(eval_results["model_id"], "awsnet")
        self.assertEqual(eval_results["metric_protocol"], "myops380_voxel_v3")
        self.assertIn("regions", eval_results)

        metrics_file = self.eval_dir / "metrics.json"
        summary_file = self.eval_dir / "summary.json"
        csv_file = self.eval_dir / "per_case.csv"
        self.assertTrue(metrics_file.is_file())
        self.assertTrue(summary_file.is_file())
        self.assertTrue(csv_file.is_file())

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_json = json.load(f)
        self.assertEqual(metrics_json["model_id"], "awsnet")
        self.assertIn("summary", metrics_json)
        self.assertIn("scar", metrics_json["summary"])
        # SCAR root-level metrics compatibility
        self.assertIn("scar", metrics_json)
        self.assertIn("normal_myocardium", metrics_json)
        self.assertIn("edema", metrics_json)

        with open(summary_file, "r", encoding="utf-8") as f:
            summary_json = json.load(f)
        self.assertEqual(summary_json["model_id"], "awsnet")
        self.assertIn("scar", summary_json)


if __name__ == "__main__":
    unittest.main()
