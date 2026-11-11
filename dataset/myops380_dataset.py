"""Dataset and DataLoader implementations for AWSnet on the MyoPS380 benchmark.

Conforms strictly to benchmark protocol myops380_train304_test76_last_v1:
- 304 train patients (1,972 slices) from train.txt
- 76 test patients from test_vol.txt
- Zero validation split (val.txt strictly prohibited, no validation loader)
- Direct ingestion of [0.0, 1.0] preprocessed cache (no double division by 255.0)
- Canonical label mapping: legacy {0:bg, 1:myo, 2:scar, 3:edema} -> canonical {0:bg, 1:myo, 2:edema, 3:scar}
- Returns dict with keys for individual modalities ('c0', 'lge', 't2') and stacked tensor ('image')
"""
from __future__ import annotations

import os
from pathlib import Path
import random
import re
from typing import Any, Callable, Generator, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.data_augmentation import RandomGenerator, ResizeGenerator


def read_split_names(split_path: str | Path) -> list[str]:
    """Reads slice or volume case IDs from a manifest file.

    Args:
        split_path: Path to the manifest file (.txt).

    Returns:
        List of trimmed sample names without empty lines or file extensions.

    Raises:
        FileNotFoundError: If manifest file does not exist.
        ValueError: If manifest is empty, has duplicates, or contains extensions.
    """
    path = Path(split_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")

    content = path.read_text(encoding="utf-8-sig")
    names = [line.strip() for line in content.splitlines() if line.strip()]

    if not names:
        raise ValueError(f"Split manifest is empty: {path}")

    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate entries found in manifest: {path}")

    for name in names:
        if Path(name).suffix != "":
            raise ValueError(
                f"Manifest entries must not contain file extensions: {name!r} in {path}"
            )

    return names


def extract_patient_id(sample_name: str) -> str:
    """Extracts base patient ID from a sample name or file path."""
    name = str(sample_name).strip()
    for ext in (".npy.h5", ".npz", ".h5", ".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return re.sub(r"_slice\d+$", "", name)


def canonicalize_label(label: np.ndarray, source_order: str = "legacy") -> np.ndarray:
    """Converts labels to canonical order: {0:bg, 1:myo, 2:edema, 3:scar}."""
    arr = np.asarray(label, dtype=np.int64)
    if source_order == "legacy":
        lut = np.array([0, 1, 3, 2], dtype=np.int64)
        return lut[arr]
    elif source_order == "canonical":
        return arr
    else:
        raise ValueError(f"Unsupported source label order: {source_order}")


def ingest_image_array(image: np.ndarray, name: str = "sample") -> np.ndarray:
    """Ingests and strictly validates preprocessed image array into [0.0, 1.0] float32.

    Strictly validates that values lie in [-1e-5, 1.0 + 1e-5] matching SCAR reference.
    Raises ValueError on non-finite or out-of-range values. Does not perform silent scaling.
    """
    img = np.asarray(image, dtype=np.float32)
    if not np.isfinite(img).all():
        raise ValueError(f"Non-finite values (NaN or Inf) encountered in image array for {name}.")
    if img.min() < -1e-5 or img.max() > 1.0 + 1e-5:
        raise ValueError(
            f"Images must be normalized to [0,1] in preprocessing; out-of-range intensity in {name} "
            f"(min={img.min():.4f}, max={img.max():.4f})."
        )
    return img


class MyoPS380TrainDataset(Dataset):
    """2D Slice Training Dataset for AWSnet on MyoPS380 under no_val protocol.

    Loads aligned bSSFP, LGE, and T2w slices from preprocessed NPZ files, applies
    SCAR RandomGenerator augmentation, and outputs standardized batch items.
    """

    def __init__(
        self,
        data_dir: str | Path = r"D:\nckh_model\Dataset\MyoPS380\Processed_data",
        split_dir: str | Path = r"D:\nckh_model\AWSnet\dataset\splits\myops380_no_val",
        split: str = "train",
        img_size: int | tuple[int, int] = (128, 128),
        transform: Callable | None = None,
        validate_files: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if "data_root" in kwargs and kwargs["data_root"] is not None:
            data_dir = kwargs["data_root"]
        if "split_path" in kwargs and kwargs["split_path"] is not None:
            resolved_split_path = Path(kwargs["split_path"])
        else:
            resolved_split_path = Path(split_dir) / f"{split}.txt" if Path(split_dir).is_dir() else Path(split_dir)

        split_lower = str(split).lower()
        if "val" in split_lower:
            raise ValueError(
                "Validation split is forbidden under benchmark protocol 'myops380_train304_test76_last_v1'. "
                "All 304 non-test patients are included in train.txt. Validation loaders and loops must not be created."
            )

        p_data = Path(data_dir)
        if (p_data / "Processed_data" / "bSSFP" / "train_npz").is_dir():
            p_data = p_data / "Processed_data"
        elif p_data.name == "Raw_data" and (p_data.parent / "Processed_data" / "bSSFP" / "train_npz").is_dir():
            p_data = p_data.parent / "Processed_data"
        elif not (p_data / "bSSFP" / "train_npz").is_dir():
            for cand in [p_data / "Processed_data", p_data.parent / "Processed_data"]:
                if (cand / "bSSFP" / "train_npz").is_dir():
                    p_data = cand
                    break

        self.data_root = p_data
        self.split_path = resolved_split_path
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)

        # 1. Read manifest
        self.sample_list = read_split_names(self.split_path)

        # 2. Modality directories
        self.modality_dirs = {
            "C0": self.data_root / "bSSFP" / "train_npz",
            "LGE": self.data_root / "LGE" / "train_npz",
            "T2": self.data_root / "T2w" / "train_npz",
        }

        # 3. Optional integrity check
        if validate_files:
            for name in self.sample_list:
                for mod, mod_dir in self.modality_dirs.items():
                    p = mod_dir / f"{name}.npz"
                    if not p.is_file():
                        raise FileNotFoundError(f"Missing {mod} file for {name}: {p}")

        # 4. Transform setup (SCAR RandomGenerator by default)
        if transform is None:
            self.transform = RandomGenerator(output_size=self.img_size)
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        slice_name = self.sample_list[idx]

        # 1. Read aligned modalities from NPZ
        p_c0 = self.modality_dirs["C0"] / f"{slice_name}.npz"
        p_lge = self.modality_dirs["LGE"] / f"{slice_name}.npz"
        p_t2 = self.modality_dirs["T2"] / f"{slice_name}.npz"

        with np.load(p_c0, allow_pickle=False) as d_c0, \
             np.load(p_lge, allow_pickle=False) as d_lge, \
             np.load(p_t2, allow_pickle=False) as d_t2:
            img_c0 = ingest_image_array(d_c0["image"], name=f"{slice_name}_c0")
            img_lge = ingest_image_array(d_lge["image"], name=f"{slice_name}_lge")
            img_t2 = ingest_image_array(d_t2["image"], name=f"{slice_name}_t2")
            label = np.asarray(d_c0["label"])

        # 2. Canonicalize label: legacy {0:bg, 1:myo, 2:scar, 3:edema} -> canonical {0:bg, 1:myo, 2:edema, 3:scar}
        label = np.round(label).astype(np.int64)
        label = canonicalize_label(label, source_order="legacy")
        if label.min() < 0 or label.max() > 3:
            raise ValueError(f"Slice {slice_name} has invalid label classes: {np.unique(label)}")

        # 3. Apply augmentation
        sample = {
            "image": img_c0,
            "image1": img_lge,
            "image2": img_t2,
            "label": label,
            "case_id": slice_name,
        }
        augmented = self.transform(sample)

        t_c0 = augmented["image"]
        t_lge = augmented["image1"]
        t_t2 = augmented["image2"]
        t_label = augmented["label"]

        if not isinstance(t_c0, torch.Tensor):
            t_c0 = torch.from_numpy(np.ascontiguousarray(t_c0, dtype=np.float32))
        if t_c0.dim() == 2:
            t_c0 = t_c0.unsqueeze(0)

        if not isinstance(t_lge, torch.Tensor):
            t_lge = torch.from_numpy(np.ascontiguousarray(t_lge, dtype=np.float32))
        if t_lge.dim() == 2:
            t_lge = t_lge.unsqueeze(0)

        if not isinstance(t_t2, torch.Tensor):
            t_t2 = torch.from_numpy(np.ascontiguousarray(t_t2, dtype=np.float32))
        if t_t2.dim() == 2:
            t_t2 = t_t2.unsqueeze(0)

        if not isinstance(t_label, torch.Tensor):
            t_label = torch.from_numpy(np.ascontiguousarray(t_label, dtype=np.int64))
        else:
            t_label = t_label.long()

        # Concatenated 3-channel tensor (3, H, W)
        t_stacked = torch.cat([t_c0, t_lge, t_t2], dim=0)

        patient_id = extract_patient_id(slice_name)
        return {
            # SCAR canonical keys
            "image": t_c0,
            "image1": t_lge,
            "image2": t_t2,
            "label": t_label,
            "case_name": slice_name,
            "patient_id": patient_id,
            # AWSnet model convenience keys
            "c0": t_c0,
            "lge": t_lge,
            "t2": t_t2,
            "image_3ch": t_stacked,
            "case_id": slice_name,
        }


def _resolve_vol_path(mod_dir: Path, case_id: str) -> Path:
    """Resolve test volume path supporting both .npy.h5 and .h5 extensions."""
    p = mod_dir / f"{case_id}.npy.h5"
    if p.exists():
        return p
    p_alt = mod_dir / f"{case_id}.h5"
    if p_alt.exists():
        return p_alt
    return p


class MyoPS380TestVolumeDataset(Dataset):
    """3D Volume Test Dataset for AWSnet on MyoPS380 test set (76 cases)."""

    def __init__(
        self,
        data_dir: str | Path = r"D:\nckh_model\Dataset\MyoPS380\Processed_data",
        split_dir: str | Path = r"D:\nckh_model\AWSnet\dataset\splits\myops380_no_val",
        split_file: str = "test_vol.txt",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if "data_root" in kwargs and kwargs["data_root"] is not None:
            data_dir = kwargs["data_root"]

        p_data = Path(data_dir)
        if (p_data / "Processed_data" / "bSSFP" / "test_vol_h5").is_dir():
            p_data = p_data / "Processed_data"
        elif p_data.name == "Raw_data" and (p_data.parent / "Processed_data" / "bSSFP" / "test_vol_h5").is_dir():
            p_data = p_data.parent / "Processed_data"
        elif not (p_data / "bSSFP" / "test_vol_h5").is_dir():
            for cand in [p_data / "Processed_data", p_data.parent / "Processed_data"]:
                if (cand / "bSSFP" / "test_vol_h5").is_dir():
                    p_data = cand
                    break

        self.data_root = p_data
        self.split_path = Path(split_dir) / split_file if Path(split_dir).is_dir() else Path(split_dir)
        self.cases = read_split_names(self.split_path)

        self.modality_dirs = {
            "C0": self.data_root / "bSSFP" / "test_vol_h5",
            "LGE": self.data_root / "LGE" / "test_vol_h5",
            "T2": self.data_root / "T2w" / "test_vol_h5",
        }

    @staticmethod
    def _resolve_vol_path(mod_dir: Path, case_id: str) -> Path:
        """Resolve test volume path supporting both .npy.h5 and .h5 extensions."""
        return _resolve_vol_path(mod_dir, case_id)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case_id = self.cases[idx]

        p_c0 = self._resolve_vol_path(self.modality_dirs["C0"], case_id)
        p_lge = self._resolve_vol_path(self.modality_dirs["LGE"], case_id)
        p_t2 = self._resolve_vol_path(self.modality_dirs["T2"], case_id)

        with h5py.File(p_c0, "r") as f_c0, \
             h5py.File(p_lge, "r") as f_lge, \
             h5py.File(p_t2, "r") as f_t2:
            c0_vol = ingest_image_array(f_c0["image"][:], name=f"{case_id}_c0")
            lge_vol = ingest_image_array(f_lge["image"][:], name=f"{case_id}_lge")
            t2_vol = ingest_image_array(f_t2["image"][:], name=f"{case_id}_t2")
            raw_label = np.asarray(f_c0["label"][:], dtype=np.int64)

        label_vol = canonicalize_label(raw_label, source_order="legacy")

        return {
            # SCAR canonical keys
            "image": c0_vol,
            "image1": lge_vol,
            "image2": t2_vol,
            "label": label_vol,
            "case_name": case_id,
            "patient_id": case_id,
            "spacing": np.array([np.nan, np.nan, np.nan], dtype=np.float64),
            "affine": np.full((4, 4), np.nan, dtype=np.float64),
            "spacing_unit": "unknown",
            "has_geometry": False,
            # AWSnet model convenience keys
            "c0": c0_vol,
            "lge": lge_vol,
            "t2": t2_vol,
            "case_id": case_id,
        }
