"""Adversarial stress-testing suite for Challenger 2:
Environment scripts & CLI parser compatibility.
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import parse_args as parse_train_args
from evaluate import parse_args as parse_eval_args
import test as test_entrypoint


class TestCLIParserExhaustion(unittest.TestCase):
    """Test 1: Exhaustive CLI argument & flag parsing."""

    def test_train_flags_from_env_script(self):
        """Verify exact flags invoked in Environment_A100_5seeds_AWSnet.txt parse without error."""
        # Exact flags from lines 149-160
        argv = [
            "train.py",
            "--data_dir", "/dummy/dataset",
            "--split_dir", "/dummy/splits",
            "--save_dir", "/dummy/runs",
            "--seed", "1234",
            "--batch_size", "16",
            "--accum_steps", "1",
            "--max_epochs", "300",
            "--base_lr", "0.0003333",
            "--weight_decay", "0.0001",
            "--num_workers", "4",
            "--amp", "auto",
        ]
        import unittest.mock
        with unittest.mock.patch.object(sys, "argv", argv):
            args = parse_train_args()
            self.assertEqual(args.data_dir, "/dummy/dataset")
            self.assertEqual(args.split_dir, "/dummy/splits")
            self.assertEqual(args.save_dir, "/dummy/runs")
            self.assertEqual(args.seed, 1234)
            self.assertEqual(args.batch_size, 16)
            self.assertEqual(args.accum_steps, 1)
            self.assertEqual(args.max_epochs, 300)
            self.assertAlmostEqual(args.base_lr, 0.0003333)
            self.assertAlmostEqual(args.weight_decay, 0.0001)
            self.assertEqual(args.num_workers, 4)
            self.assertEqual(args.amp, "auto")

    def test_eval_flags_from_env_script(self):
        """Verify exact flags invoked in Environment_Eval_5seeds_AWSnet.txt parse without error."""
        # Exact flags from lines 43-49
        argv = [
            "test.py",
            "--checkpoint", "/dummy/ckpt.pt",
            "--data-root", "/dummy/dataset",
            "--list-dir", "/dummy/splits",
            "--split", "test_vol",
            "--batch-size", "8",
            "--amp", "auto",
        ]
        import unittest.mock
        with unittest.mock.patch.object(sys, "argv", argv):
            args = parse_eval_args()
            self.assertEqual(args.checkpoint, "/dummy/ckpt.pt")
            self.assertEqual(args.data_dir, "/dummy/dataset")
            self.assertEqual(args.split_dir, "/dummy/splits")
            self.assertEqual(args.split, "test_vol")
            self.assertEqual(args.batch_size, 8)
            self.assertEqual(args.amp, "auto")

    def test_train_kebab_and_snake_combinations(self):
        """Test kebab-case and snake_case variations for train.py."""
        variations = [
            # kebab-case variants
            [
                "train.py",
                "--data-root", "/path/data",
                "--list-dir", "/path/splits",
                "--run-root", "/path/runs",
                "--batch-size", "8",
                "--accum-steps", "2",
                "--max-epochs", "150",
                "--base-lr", "0.0001",
                "--weight-decay", "0.00005",
                "--num-workers", "2",
                "--clip-grad", "1.0",
                "--log-every", "5",
                "--amp", "fp16",
            ],
            # alternate aliases
            [
                "train.py",
                "--data_root", "/path/data2",
                "--list_dir", "/path/splits2",
                "--output-dir", "/path/runs2",
                "--epochs", "200",
                "--lr", "0.0002",
                "--amp", "none",
            ],
            [
                "train.py",
                "--output_dir", "/path/runs3",
                "--config", "config.yaml",
                "--base_filters", "16",
                "--device", "cpu",
                "--amp", "bf16",
                "--max_batches", "10",
            ],
        ]
        import unittest.mock
        for argv in variations:
            with unittest.mock.patch.object(sys, "argv", argv):
                try:
                    args = parse_train_args()
                    self.assertIsNotNone(args)
                except SystemExit as e:
                    self.fail(f"parse_train_args failed with SystemExit for args: {argv}")

    def test_eval_kebab_and_snake_combinations(self):
        """Test kebab-case and snake_case variations for evaluate.py / test.py."""
        variations = [
            [
                "test.py",
                "--checkpoint", "/path/ckpt.pt",
                "--data_dir", "/path/data",
                "--split_dir", "/path/splits",
                "--output_dir", "/path/out",
                "--split", "test_vol",
                "--batch_size", "4",
                "--device", "cpu",
                "--amp", "none",
                "--allow_incomplete",
            ],
            [
                "test.py",
                "--predictions_dir", "/path/preds",
                "--data_root", "/path/data2",
                "--list_dir", "/path/splits2",
                "--output-dir", "/path/out2",
                "--split", "val_vol",
                "--batch-size", "16",
                "--scar_root", "/path/scar",
                "--amp", "fp16",
            ],
        ]
        import unittest.mock
        for argv in variations:
            with unittest.mock.patch.object(sys, "argv", argv):
                try:
                    args = parse_eval_args()
                    self.assertIsNotNone(args)
                except SystemExit as e:
                    self.fail(f"parse_eval_args failed with SystemExit for args: {argv}")

    def test_potential_unsupported_flags_audit(self):
        """Audit whether commonly expected flags like --save-dir, --split-dir, --predictions-dir, --scar-root, --allow-incomplete work or fail."""
        unsupported_eval = []
        for test_flag in [
            ["--save-dir", "/path"],
            ["--save_dir", "/path"],
            ["--run-root", "/path"],
            ["--run_root", "/path"],
            ["--predictions-dir", "/path"],
            ["--split-dir", "/path"],
            ["--scar-root", "/path"],
            ["--allow-incomplete"],
        ]:
            argv = ["test.py"] + test_flag
            import unittest.mock
            with unittest.mock.patch.object(sys, "argv", argv):
                try:
                    parse_eval_args()
                except SystemExit:
                    unsupported_eval.append(test_flag[0])

        unsupported_train = []
        for test_flag in [
            ["--save-dir", "/path"],
            ["--split-dir", "/path"],
        ]:
            argv = ["train.py"] + test_flag
            import unittest.mock
            with unittest.mock.patch.object(sys, "argv", argv):
                try:
                    parse_train_args()
                except SystemExit:
                    unsupported_train.append(test_flag[0])

        print(f"\n[CLI Audit] Unsupported eval flags: {unsupported_eval}")
        print(f"[CLI Audit] Unsupported train flags: {unsupported_train}")
        # Note: we record these for documentation, but verify that the environment scripts ONLY use supported flags!


class TestScriptSyntaxAndLinting(unittest.TestCase):
    """Test 2: Bash syntax, linting, quote matching, and embedded Python AST parsing."""

    def setUp(self):
        self.train_env_path = PROJECT_ROOT / "Environment_A100_5seeds_AWSnet.txt"
        self.eval_env_path = PROJECT_ROOT / "Environment_Eval_5seeds_AWSnet.txt"
        self.assertTrue(self.train_env_path.exists(), f"Missing {self.train_env_path}")
        self.assertTrue(self.eval_env_path.exists(), f"Missing {self.eval_env_path}")
        self.train_content = self.train_env_path.read_text(encoding="utf-8")
        self.eval_content = self.eval_env_path.read_text(encoding="utf-8")

    def test_bash_shebang_and_strict_mode(self):
        """Verify scripts start with proper bash shebang and set -e."""
        self.assertTrue(self.train_content.startswith("#!/bin/bash"))
        self.assertTrue(self.eval_content.startswith("#!/bin/bash"))
        self.assertIn("set -e", self.train_content)
        self.assertIn("set -e", self.eval_content)

    def test_brackets_and_subshell_matching(self):
        """Verify that brackets, braces, and parentheses balance in both scripts."""
        for name, content in [("Train Env", self.train_content), ("Eval Env", self.eval_content)]:
            # Check balanced brackets
            counts = {
                "(": content.count("("),
                ")": content.count(")"),
                "[": content.count("["),
                "]": content.count("]"),
                "{": content.count("{"),
                "}": content.count("}"),
            }
            self.assertEqual(counts["("], counts[")"], f"Unbalanced () in {name}")
            self.assertEqual(counts["["], counts["]"], f"Unbalanced [] in {name}")
            self.assertEqual(counts["{"], counts["}"], f"Unbalanced {{}} in {name}")

    def test_tmux_commands_integrity(self):
        """Verify tmux commands in training environment script."""
        # Should have kill-session, new-session, and new-window for 5 seeds
        self.assertIn("tmux kill-session -t train5_awsnet", self.train_content)
        self.assertIn("tmux new-session -d -s train5_awsnet", self.train_content)
        self.assertIn("tmux new-window -t train5_awsnet", self.train_content)
        # Verify 5 seeds
        self.assertIn("SEEDS=(1234 1098 1024 2026 6868)", self.train_content)
        self.assertIn("for i in {1..4}; do", self.train_content)

    def test_inline_python_scripts_ast_parse(self):
        """Extract and parse all inline python -c scripts with ast.parse."""
        # 1. Drive mount script in Train Env
        mount_match = re.search(r'python3\s+-c\s+"([^"]+)"', self.train_content)
        self.assertIsNotNone(mount_match, "Could not extract drive mount python snippet")
        mount_code = mount_match.group(1)
        try:
            tree = ast.parse(mount_code)
            self.assertIsNotNone(tree)
        except SyntaxError as e:
            self.fail(f"Drive mount inline script failed ast.parse: {e}")

        # 2. Aggregator script in Eval Env
        eval_match = re.search(r'python3\s+-c\s+"([\s\S]+?)"\s*$', self.eval_content)
        self.assertIsNotNone(eval_match, "Could not extract eval aggregator python snippet")
        raw_eval_code = eval_match.group(1)

        # In bash execution, bash expands variables and unescapes \" inside double quotes
        # Test simulated bash unquoting:
        bash_expanded = raw_eval_code.replace('\\"', '"')
        bash_expanded = bash_expanded.replace("$RUN_ROOT", "/content/drive/MyDrive/AWSnet_Runs")
        bash_expanded = bash_expanded.replace("$MODEL_NAME", "awsnet")

        try:
            tree = ast.parse(bash_expanded)
            self.assertIsNotNone(tree)
        except SyntaxError as e:
            self.fail(f"Eval aggregator inline script (bash-expanded) failed ast.parse: {e}")

        # Also test raw script AST parsing across Python syntax versions
        try:
            tree_raw = ast.parse(raw_eval_code)
            self.assertIsNotNone(tree_raw)
        except SyntaxError as e:
            print(f"[AST Warning] Raw code ast.parse note: {e}")


class TestMetricParsingAggregatorResilience(unittest.TestCase):
    """Test 3: Metric parsing aggregator line-by-line simulation under stress conditions."""

    def run_aggregator(self, run_root: Path, model_name: str = "awsnet") -> tuple[int, str, list[dict]]:
        """Executes the exact aggregator logic from Environment_Eval_5seeds_AWSnet.txt."""
        import json, glob, os, csv
        import numpy as np

        seeds = [1234, 1098, 1024, 2026, 6868]

        seed_summaries = {}
        for s in seeds:
            matches = glob.glob(os.path.join(str(run_root), f'*{model_name}*seed{s}*/**/metrics.json'), recursive=True)
            if not matches:
                matches = glob.glob(os.path.join(str(run_root), f'*{model_name}*seed{s}*/**/summary.json'), recursive=True)
            if not matches:
                all_jsons = glob.glob(os.path.join(str(run_root), '**/metrics.json'), recursive=True) + glob.glob(os.path.join(str(run_root), '**/summary.json'), recursive=True)
                for sm in all_jsons:
                    try:
                        data = json.load(open(sm))
                        ckpt_meta = data.get('checkpoint_metadata', {})
                        if str(ckpt_meta.get('seed')) == str(s) or str(data.get('seed')) == str(s):
                            matches = [sm]
                            break
                    except Exception:
                        pass
            if matches:
                raw_data = json.load(open(matches[-1]))
                if 'summary' in raw_data and isinstance(raw_data['summary'], dict) and 'scar' in raw_data['summary']:
                    merged = dict(raw_data['summary'])
                    merged.update(raw_data)
                    seed_summaries[s] = merged
                else:
                    seed_summaries[s] = raw_data

        metrics_def = [
            ('Cơ tim lành (Myocardium) Dice', 'normal_myocardium', 'mean_dice', True),
            ('Phù nề (Edema) Dice', 'edema', 'mean_dice', True),
            ('Sẹo nhồi máu (Scar) Dice', 'scar', 'mean_dice', True),
            ('Phù nề gộp {2,3} Dice', 'edema_inclusive', 'mean_dice', True),
            ('Toàn bộ vòng cơ tim Dice', 'myocardial_ring', 'mean_dice', True),
            ('HD95 Sẹo nhồi máu (voxel)', 'scar', 'mean_hd95_voxel', False),
            ('HD95 Phù nề (voxel)', 'edema', 'mean_hd95_voxel', False),
            ('ASSD Sẹo nhồi máu (voxel)', 'scar', 'mean_asd_voxel', False),
        ]

        export_rows = []
        output_lines = []

        if not seed_summaries:
            output_lines.append('⚠️ Không tìm thấy file metrics.json hoặc summary.json nào.')
            return len(seed_summaries), "\n".join(output_lines), export_rows

        for title, reg, met, is_pct in metrics_def:
            row_vals = []
            row_strs = []
            for s, d in seed_summaries.items():
                val = d.get(reg, {}).get(met)
                if val is not None:
                    calc_val = val * 100 if is_pct else val
                    row_vals.append(calc_val)
                    row_strs.append(f'{calc_val:6.2f}')
                else:
                    row_strs.append('   N/A')

            if row_vals:
                mean = np.mean(row_vals)
                std = np.std(row_vals)
                unit = '%' if is_pct else ' vox'
                stat_str = f'{mean:6.2f} ± {std:4.2f}{unit}'
            else:
                stat_str = 'N/A'
                mean, std = 0, 0

            output_lines.append(f'{title:<32} | ' + ' | '.join(row_strs) + f' | {stat_str}')
            export_rows.append({'Metric': title, 'Mean': round(mean, 2), 'Std': round(std, 2)})

        for label, key in [('⭐ Avg Pathology Dice', 'avg_pathology_dice'), ('⭐ Avg Pathology Inclusive Dice', 'avg_pathology_inclusive_dice')]:
            row_vals = []
            row_strs = []
            for s, d in seed_summaries.items():
                val = d.get(key)
                if val is not None:
                    calc_val = val * 100
                    row_vals.append(calc_val)
                    row_strs.append(f'{calc_val:6.2f}')
                else:
                    row_strs.append('   N/A')
            if row_vals:
                mean = np.mean(row_vals)
                std = np.std(row_vals)
                stat_str = f'{mean:6.2f} ± {std:4.2f}%'
            else:
                stat_str = 'N/A'
                mean, std = 0, 0
            output_lines.append(f'{label:<32} | ' + ' | '.join(row_strs) + f' | {stat_str}')
            export_rows.append({'Metric': label, 'Mean': round(mean, 2), 'Std': round(std, 2)})

        out_csv = os.path.join(str(run_root), f'Bao_Cao_5Seeds_AWSnet.csv')
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Metric', 'Mean', 'Std'])
            writer.writeheader()
            writer.writerows(export_rows)

        return len(seed_summaries), "\n".join(output_lines), export_rows

    def test_aggregator_full_5seeds_metrics_json(self):
        """Simulate all 5 seeds with standard metrics.json files."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_agg_full_"))
        try:
            seeds = [1234, 1098, 1024, 2026, 6868]
            for s in seeds:
                s_dir = temp_dir / f"eval_awsnet_seed{s}"
                s_dir.mkdir(parents=True)
                payload = {
                    "checkpoint_metadata": {"seed": s, "completed_epochs": 300},
                    "summary": {
                        "normal_myocardium": {"mean_dice": 0.85, "mean_iou": 0.74, "mean_hd95_voxel": 1.2, "mean_asd_voxel": 0.4},
                        "edema": {"mean_dice": 0.65, "mean_iou": 0.49, "mean_hd95_voxel": 2.1, "mean_asd_voxel": 0.8},
                        "scar": {"mean_dice": 0.60, "mean_iou": 0.45, "mean_hd95_voxel": 3.0, "mean_asd_voxel": 1.1},
                        "edema_inclusive": {"mean_dice": 0.72, "mean_iou": 0.58, "mean_hd95_voxel": 1.8, "mean_asd_voxel": 0.6},
                        "myocardial_ring": {"mean_dice": 0.88, "mean_iou": 0.78, "mean_hd95_voxel": 1.1, "mean_asd_voxel": 0.3},
                        "avg_pathology_dice": 0.625,
                        "avg_pathology_inclusive_dice": 0.660,
                    }
                }
                with open(s_dir / "metrics.json", "w") as f:
                    json.dump(payload, f)

            count, out_str, rows = self.run_aggregator(temp_dir)
            self.assertEqual(count, 5)
            self.assertEqual(len(rows), 10)  # 8 region metrics + 2 summary metrics
            csv_path = temp_dir / "Bao_Cao_5Seeds_AWSnet.csv"
            self.assertTrue(csv_path.exists())
            with open(csv_path, encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                self.assertEqual(len(reader), 10)
                self.assertEqual(reader[0]["Metric"], "Cơ tim lành (Myocardium) Dice")
                self.assertEqual(float(reader[0]["Mean"]), 85.0)
                self.assertEqual(float(reader[0]["Std"]), 0.0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_aggregator_summary_json_fallback(self):
        """Simulate seeds producing summary.json instead of metrics.json."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_agg_summary_"))
        try:
            seeds = [1234, 1098, 1024, 2026, 6868]
            for s in seeds:
                s_dir = temp_dir / f"run_awsnet_seed{s}" / "eval_output"
                s_dir.mkdir(parents=True)
                payload = {
                    "seed": s,
                    "summary": {
                        "normal_myocardium": {"mean_dice": 0.80, "mean_hd95_voxel": 1.5, "mean_asd_voxel": 0.5},
                        "edema": {"mean_dice": 0.60, "mean_hd95_voxel": 2.5, "mean_asd_voxel": 0.9},
                        "scar": {"mean_dice": 0.55, "mean_hd95_voxel": 3.2, "mean_asd_voxel": 1.2},
                        "edema_inclusive": {"mean_dice": 0.70, "mean_hd95_voxel": 2.0, "mean_asd_voxel": 0.7},
                        "myocardial_ring": {"mean_dice": 0.85, "mean_hd95_voxel": 1.3, "mean_asd_voxel": 0.4},
                        "avg_pathology_dice": 0.575,
                        "avg_pathology_inclusive_dice": 0.625,
                    }
                }
                with open(s_dir / "summary.json", "w") as f:
                    json.dump(payload, f)

            count, out_str, rows = self.run_aggregator(temp_dir)
            self.assertEqual(count, 5)
            self.assertEqual(len(rows), 10)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_aggregator_partial_seeds_resilience(self):
        """Stress-test when only 2 seeds completed, or directory names lack seed prefix."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_agg_partial_"))
        try:
            # Only seeds 1234 and 2026 exist
            for s in [1234, 2026]:
                s_dir = temp_dir / f"arbitrary_folder_{s}"
                s_dir.mkdir(parents=True)
                payload = {
                    "checkpoint_metadata": {"seed": s},
                    "summary": {
                        "scar": {"mean_dice": 0.50, "mean_hd95_voxel": 3.0, "mean_asd_voxel": 1.0},
                        "edema": {"mean_dice": 0.60, "mean_hd95_voxel": 2.0, "mean_asd_voxel": 0.8},
                    }
                }
                with open(s_dir / "metrics.json", "w") as f:
                    json.dump(payload, f)

            count, out_str, rows = self.run_aggregator(temp_dir)
            self.assertEqual(count, 2)
            # Verify that partial seeds produce valid Mean and Std without crashing
            scar_row = next(r for r in rows if "Scar" in r["Metric"])
            self.assertEqual(scar_row["Mean"], 50.0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_aggregator_none_and_missing_metrics_resilience(self):
        """Stress-test edge cases where some seeds report None or missing for empty regions (e.g. no scar)."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_agg_none_"))
        try:
            seeds = [1234, 1098, 1024, 2026, 6868]
            for i, s in enumerate(seeds):
                s_dir = temp_dir / f"eval_awsnet_seed{s}"
                s_dir.mkdir(parents=True)
                # Seed 1098 and 2026 have None for scar HD95 and ASD (e.g., ground truth has no scar in some cases)
                scar_hd95 = None if i in (1, 3) else 2.5
                scar_asd = None if i in (1, 3) else 0.8
                payload = {
                    "summary": {
                        "normal_myocardium": {"mean_dice": 0.82, "mean_hd95_voxel": 1.4, "mean_asd_voxel": 0.4},
                        "edema": {"mean_dice": 0.62, "mean_hd95_voxel": 2.2, "mean_asd_voxel": 0.7},
                        "scar": {"mean_dice": 0.58, "mean_hd95_voxel": scar_hd95, "mean_asd_voxel": scar_asd},
                        "edema_inclusive": {"mean_dice": 0.71, "mean_hd95_voxel": 1.9, "mean_asd_voxel": 0.6},
                        "myocardial_ring": {"mean_dice": 0.86, "mean_hd95_voxel": 1.2, "mean_asd_voxel": 0.3},
                        "avg_pathology_dice": 0.60,
                        "avg_pathology_inclusive_dice": 0.645,
                    }
                }
                with open(s_dir / "metrics.json", "w") as f:
                    json.dump(payload, f)

            count, out_str, rows = self.run_aggregator(temp_dir)
            self.assertEqual(count, 5)
            # HD95 scar row had only 3 valid numbers (seeds 0, 2, 4)
            scar_hd95_row = next(r for r in rows if "HD95 Sẹo nhồi máu" in r["Metric"])
            self.assertEqual(scar_hd95_row["Mean"], 2.5)
            self.assertEqual(scar_hd95_row["Std"], 0.0)

            # All None case: imagine a completely missing metric
            empty_dir = Path(tempfile.mkdtemp(prefix="test_agg_all_none_"))
            for s in seeds:
                s_d = empty_dir / f"eval_awsnet_seed{s}"
                s_d.mkdir(parents=True)
                with open(s_d / "metrics.json", "w") as f:
                    json.dump({"summary": {"scar": {"mean_dice": None}}}, f)
            c2, o2, r2 = self.run_aggregator(empty_dir)
            self.assertEqual(c2, 5)
            scar_r = next(r for r in r2 if "Sẹo nhồi máu (Scar) Dice" in r["Metric"])
            self.assertEqual(scar_r["Mean"], 0)
            self.assertEqual(scar_r["Std"], 0)
            shutil.rmtree(empty_dir, ignore_errors=True)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_aggregator_zero_seeds_empty_dir(self):
        """Stress-test when no json files exist at all."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_agg_empty_"))
        try:
            count, out_str, rows = self.run_aggregator(temp_dir)
            self.assertEqual(count, 0)
            self.assertIn("Không tìm thấy file metrics.json hoặc summary.json nào", out_str)
            self.assertEqual(len(rows), 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSubprocessDryRun(unittest.TestCase):
    """Test 4: Verify real subprocess dry-runs of train.py, evaluate.py, test.py."""

    def test_help_flags_across_all_entrypoints(self):
        """Verify --help exits 0 on train.py, evaluate.py, test.py."""
        for script in ["train.py", "evaluate.py", "test.py"]:
            cmd = [sys.executable, str(REPO_ROOT / script), "--help"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"{script} --help failed: {res.stderr}")

    def test_synthetic_train_and_test_subprocesses(self):
        """End-to-end dry-run of train.py and test.py with mock dataset via subprocess."""
        import h5py
        temp_dir = Path(tempfile.mkdtemp(prefix="subproc_e2e_"))
        try:
            data_dir = temp_dir / "Dataset"
            split_dir = temp_dir / "splits"
            save_dir = temp_dir / "runs"
            eval_dir = temp_dir / "eval"
            for p in (data_dir, split_dir, save_dir, eval_dir):
                p.mkdir(parents=True, exist_ok=True)

            for mod in ("bSSFP", "LGE", "T2w"):
                (data_dir / mod / "train_npz").mkdir(parents=True, exist_ok=True)
                (data_dir / mod / "test_vol_h5").mkdir(parents=True, exist_ok=True)

            # Create mock train slice
            slice_name = "case0001_slice000"
            (split_dir / "train.txt").write_text(slice_name + "\n", encoding="utf-8")
            img = np.random.uniform(0.0, 1.0, (128, 128)).astype(np.float32)
            lbl = np.random.choice([0, 1, 2, 3], size=(128, 128)).astype(np.int64)
            for mod in ("bSSFP", "LGE", "T2w"):
                np.savez_compressed(data_dir / mod / "train_npz" / f"{slice_name}.npz", image=img, label=lbl)

            # Create mock test volume
            case_id = "case0001"
            (split_dir / "test_vol.txt").write_text(case_id + "\n", encoding="utf-8")
            for mod in ("bSSFP", "LGE", "T2w"):
                with h5py.File(data_dir / mod / "test_vol_h5" / f"{case_id}.h5", "w") as f:
                    f.create_dataset("image", data=np.random.uniform(0.0, 1.0, (128, 128, 2)).astype(np.float32))
                    f.create_dataset("label", data=np.random.choice([0, 1, 2, 3], size=(128, 128, 2)).astype(np.int64))

            # 1. Subprocess run of train.py using exact flags from Environment_A100_5seeds_AWSnet.txt
            train_cmd = [
                sys.executable, str(REPO_ROOT / "train.py"),
                "--data_dir", str(data_dir),
                "--split_dir", str(split_dir),
                "--save_dir", str(save_dir),
                "--seed", "1234",
                "--batch_size", "1",
                "--accum_steps", "1",
                "--max_epochs", "1",
                "--base_lr", "0.0003333",
                "--weight_decay", "0.0001",
                "--num_workers", "0",
                "--amp", "none",
                "--device", "cpu",
                "--max_batches", "1",
                "--base_filters", "8",
            ]
            res_train = subprocess.run(train_cmd, capture_output=True, text=True)
            self.assertEqual(res_train.returncode, 0, f"train.py subprocess failed:\n{res_train.stderr}")
            ckpt_path = save_dir / "awsnet_seed1234.pt"
            self.assertTrue(ckpt_path.exists(), "Checkpoint awsnet_seed1234.pt not created")

            # 2. Subprocess run of test.py using exact flags from Environment_Eval_5seeds_AWSnet.txt
            test_cmd = [
                sys.executable, str(REPO_ROOT / "test.py"),
                "--checkpoint", str(ckpt_path),
                "--data-root", str(data_dir),
                "--list-dir", str(split_dir),
                "--split", "test_vol",
                "--batch-size", "2",
                "--amp", "none",
                "--device", "cpu",
                "--output_dir", str(eval_dir),
                "--allow_incomplete",
            ]
            res_test = subprocess.run(test_cmd, capture_output=True, text=True)
            self.assertEqual(res_test.returncode, 0, f"test.py subprocess failed:\n{res_test.stderr}")
            self.assertTrue((eval_dir / "metrics.json").exists(), "metrics.json was not created")
            self.assertTrue((eval_dir / "summary.json").exists(), "summary.json was not created")
            self.assertTrue((eval_dir / "per_case.csv").exists(), "per_case.csv was not created")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
