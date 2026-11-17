"""Dataset and DataLoader implementations for AWSnet on MyoPS380 and MyoPS++ benchmarks.

Strictly supports:
1. MyoPS380 Benchmark:
   - Protocol: myops380_train304_test76_last_v1
   - 304 train patients (1,972 slices) from train.txt, zero validation
   - 76 test patients from test_vol.txt (unit grid, HD95 in voxels)
   - Source labels in legacy order {0:bg, 1:myo, 2:scar, 3:edema} canonicalized to
     {0:bg, 1:myo, 2:edema, 3:scar}

2. MyoPS++ Benchmark (Profile myopspp_unet2d_roi128_1mm):
   - 64 train patients (296 slices) from train.txt, zero validation
   - 16 test patients from test_vol.txt (physical grid, HD95 in mm)
   - Source labels:
     - Raw NIfTI: {0:bg, 200:myo, 500:LV, 600:RV, 1220:edema, 2221:scar} mapped to
       canonical {0:bg, 1:myo, 2:edema, 3:scar}
     - Cached NPZ/H5: already in canonical order {0:bg, 1:myo, 2:edema, 3:scar}
   - Preserves physical voxel spacing in mm (e.g. [1.0, 1.0, slice_thickness])

Consistent Across Both Benchmarks:
- Input channels: 3 (C0, LGE, T2w) in [0.0, 1.0] float32
- Target spatial shape: 128x128
- Canonical 4 classes: 0: background, 1: normal myocardium, 2: edema, 3: scar
- Data augmentation: SCAR RandomGenerator (flips, rotations, scaling, noise)
- Model architecture and loss function: 100% unchanged
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.data_augmentation import RandomGenerator, ResizeGenerator

try:
    import nibabel as nib
except ImportError:
    nib = None


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


def canonicalize_raw_myopspp(label: np.ndarray) -> np.ndarray:
    """Maps raw MyoPS++ labels (0, 200, 500, 600, 1220, 2221) to canonical 0..3."""
    raw = np.asarray(label, dtype=np.int64)
    out = np.zeros_like(raw, dtype=np.int64)
    out[raw == 200] = 1   # Normal myocardium
    out[raw == 1220] = 2  # Edema
    out[raw == 2221] = 3  # Scar
    # 0, 500 (LV), 600 (RV) remain 0 (background)
    return out


def canonicalize_label(label: np.ndarray, source_order: str = "legacy") -> np.ndarray:
    """Converts labels to canonical order: {0:bg, 1:myo, 2:edema, 3:scar}."""
    arr = np.asarray(label, dtype=np.int64)
    if source_order == "legacy":
        lut = np.array([0, 1, 3, 2], dtype=np.int64)
        return lut[arr]
    elif source_order == "canonical":
        return arr
    elif source_order == "raw_myopspp":
        return canonicalize_raw_myopspp(arr)
    elif source_order == "auto":
        if arr.max() > 3:
            return canonicalize_raw_myopspp(arr)
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


def _find_cases_directory(path: Path) -> Optional[Path]:
    """Finds directory containing cases/CenterB or cases/CenterC for MyoPS++."""
    candidates = [
        path,
        path / "cases",
        path / "MyoPSpp_UNet2D_ROI128_1mm" / "cases",
        path.parent / "cases",
        path.parent / "MyoPSpp_UNet2D_ROI128_1mm" / "cases",
    ]
    for c in candidates:
        if c.is_dir() and ((c / "CenterB").is_dir() or (c / "CenterC").is_dir()):
            return c
        if (c / "cases").is_dir() and (((c / "cases") / "CenterB").is_dir() or ((c / "cases") / "CenterC").is_dir()):
            return c / "cases"
    return None


class MyoPS380TrainDataset(Dataset):
    """2D Slice Training Dataset for AWSnet supporting both MyoPS380 and MyoPS++.

    Loads aligned bSSFP, LGE, and T2w slices from preprocessed NPZ files or direct
    MyoPS++ ROI NIfTI cases, applies SCAR RandomGenerator augmentation, and outputs
    standardized canonical batch items.
    """

    def __init__(
        self,
        data_dir: str | Path = r"D:\nckh_model\Dataset\MyoPS380\Processed_data",
        split_dir: str | Path | None = None,
        split: str = "train",
        img_size: int | tuple[int, int] = (128, 128),
        transform: Callable | None = None,
        validate_files: bool = False,
        label_order: str = "auto",
        dataset: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if "data_root" in kwargs and kwargs["data_root"] is not None:
            data_dir = kwargs["data_root"]

        split_lower = str(split).lower()
        if "val" in split_lower:
            raise ValueError(
                "Validation split is forbidden under no_val benchmark protocol. "
                "Validation loaders and loops must not be created."
            )

        p_data = Path(data_dir)
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)

        # Detect MyoPS++ NIfTI export format
        cases_dir = _find_cases_directory(p_data)
        self.is_nifti_export = (cases_dir is not None)
        self.nifti_cases_dir = cases_dir

        # Fallback to default split dir if not supplied
        if split_dir is None or not str(split_dir).strip():
            if self.is_nifti_export or "myopspp" in str(p_data).lower() or "myopspp" in str(dataset).lower():
                # MyoPS++ split
                cand_splits = [
                    Path(r"D:\nckh_model\AWSnet\dataset\splits\myopspp_unet2d_roi128_1mm"),
                    Path(__file__).resolve().parent / "splits" / "myopspp_unet2d_roi128_1mm",
                    Path(r"/content/AWSnet/dataset/splits/myopspp_unet2d_roi128_1mm"),
                ]
                for cs in cand_splits:
                    if cs.is_dir() and (cs / f"{split}.txt").is_file():
                        split_dir = cs
                        break
                if split_dir is None:
                    split_dir = Path(__file__).resolve().parent / "splits" / "myopspp_unet2d_roi128_1mm"
            else:
                # MyoPS380 split
                split_dir = Path(__file__).resolve().parent / "splits" / "myops380_no_val"

        if "split_path" in kwargs and kwargs["split_path"] is not None:
            resolved_split_path = Path(kwargs["split_path"])
        else:
            resolved_split_path = Path(split_dir) / f"{split}.txt" if Path(split_dir).is_dir() else Path(split_dir)

        self.split_path = resolved_split_path
        self.sample_list = read_split_names(self.split_path)

        # Memory cache for loaded 3D volumes when reading NIfTI files
        self._vol_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        if not self.is_nifti_export:
            # Handle NPZ cache structure (MyoPS380 or converted MyoPS++ cache)
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
            self.modality_dirs = {
                "C0": self.data_root / "bSSFP" / "train_npz",
                "LGE": self.data_root / "LGE" / "train_npz",
                "T2": self.data_root / "T2w" / "train_npz",
            }

            # Determine label ordering for NPZ cache
            meta_json = self.data_root / "dataset_metadata.json"
            if not meta_json.is_file():
                meta_json = self.data_root.parent / "dataset_metadata.json"

            if label_order != "auto":
                self.source_order = label_order
            elif meta_json.is_file():
                try:
                    info = json.loads(meta_json.read_text(encoding="utf-8"))
                    stored = info.get("label_order", "legacy")
                    self.source_order = "canonical" if "canonical" in stored else "legacy"
                except Exception:
                    self.source_order = "legacy"
            elif "myopspp" in str(self.data_root).lower() or "myopspp" in str(dataset).lower():
                self.source_order = "canonical"
            else:
                self.source_order = "legacy"

            if validate_files:
                for name in self.sample_list:
                    for mod, mod_dir in self.modality_dirs.items():
                        p = mod_dir / f"{name}.npz"
                        if not p.is_file():
                            raise FileNotFoundError(f"Missing {mod} file for {name}: {p}")
        else:
            self.data_root = cases_dir.parent
            self.source_order = "raw_myopspp"

        # Transform setup (SCAR RandomGenerator by default)
        if transform is None:
            self.transform = RandomGenerator(output_size=self.img_size)
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_list)

    def _load_nifti_case(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads and caches 3D MRI modalities and label for a patient."""
        if patient_id in self._vol_cache:
            return self._vol_cache[patient_id]

        if nib is None:
            raise ImportError("nibabel is required to read NIfTI cases directly. Run: pip install nibabel")

        center = "CenterB" if patient_id.startswith("Case2") else "CenterC"
        case_dir = self.nifti_cases_dir / center / patient_id
        if not case_dir.is_dir():
            if (self.nifti_cases_dir / patient_id).is_dir():
                case_dir = self.nifti_cases_dir / patient_id
            else:
                for cand in self.nifti_cases_dir.glob(f"*/{patient_id}"):
                    if cand.is_dir():
                        case_dir = cand
                        break

        p_c0 = case_dir / f"{patient_id}_C0.nii.gz"
        p_lge = case_dir / f"{patient_id}_LGE.nii.gz"
        p_t2 = case_dir / f"{patient_id}_T2.nii.gz"
        p_gd = case_dir / f"{patient_id}_gd.nii.gz"

        for p in (p_c0, p_lge, p_t2, p_gd):
            if not p.is_file():
                raise FileNotFoundError(f"Missing MyoPS++ modality file: {p}")

        c0_vol = ingest_image_array(nib.load(str(p_c0)).get_fdata(dtype=np.float32), name=f"{patient_id}_c0")
        lge_vol = ingest_image_array(nib.load(str(p_lge)).get_fdata(dtype=np.float32), name=f"{patient_id}_lge")
        t2_vol = ingest_image_array(nib.load(str(p_t2)).get_fdata(dtype=np.float32), name=f"{patient_id}_t2")
        raw_label = np.asarray(nib.load(str(p_gd)).dataobj)
        label_vol = canonicalize_raw_myopspp(raw_label)

        vol_tuple = (c0_vol, lge_vol, t2_vol, label_vol)
        self._vol_cache[patient_id] = vol_tuple
        return vol_tuple

    def __getitem__(self, idx: int) -> dict[str, Any]:
        slice_name = self.sample_list[idx]

        if self.is_nifti_export:
            # Parse slice_name: CaseXXXX_sliceYYY
            match = re.match(r"^(Case\d+)_slice(\d+)$", slice_name)
            if not match:
                raise ValueError(f"Invalid MyoPS++ slice format: {slice_name}; expected CaseXXXX_sliceYYY")
            patient_id, slice_idx_str = match.group(1), match.group(2)
            slice_idx = int(slice_idx_str)

            c0_vol, lge_vol, t2_vol, label_vol = self._load_nifti_case(patient_id)
            img_c0 = c0_vol[:, :, slice_idx]
            img_lge = lge_vol[:, :, slice_idx]
            img_t2 = t2_vol[:, :, slice_idx]
            label = label_vol[:, :, slice_idx]
        else:
            # Read from NPZ files
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

            label = np.round(label).astype(np.int64)
            label = canonicalize_label(label, source_order=self.source_order)

        if label.min() < 0 or label.max() > 3:
            raise ValueError(f"Slice {slice_name} has invalid label classes: {np.unique(label)}")

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

        t_stacked = torch.cat([t_c0, t_lge, t_t2], dim=0)
        patient_id = extract_patient_id(slice_name)

        return {
            "image": t_c0,
            "image1": t_lge,
            "image2": t_t2,
            "label": t_label,
            "case_name": slice_name,
            "patient_id": patient_id,
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
    """3D Volume Test Dataset for AWSnet on MyoPS380 (76 cases) and MyoPS++ (16 cases)."""

    def __init__(
        self,
        data_dir: str | Path = r"D:\nckh_model\Dataset\MyoPS380\Processed_data",
        split_dir: str | Path | None = None,
        split_file: str = "test_vol.txt",
        dataset: str = "auto",
        label_order: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.label_order = label_order
        if "data_root" in kwargs and kwargs["data_root"] is not None:
            data_dir = kwargs["data_root"]

        p_data = Path(data_dir)

        # Detect MyoPS++ NIfTI export format
        cases_dir = _find_cases_directory(p_data)
        self.is_nifti_export = (cases_dir is not None)
        self.nifti_cases_dir = cases_dir

        if split_dir is None or not str(split_dir).strip():
            if self.is_nifti_export or "myopspp" in str(p_data).lower() or "myopspp" in str(dataset).lower():
                cand_splits = [
                    Path(r"D:\nckh_model\AWSnet\dataset\splits\myopspp_unet2d_roi128_1mm"),
                    Path(__file__).resolve().parent / "splits" / "myopspp_unet2d_roi128_1mm",
                    Path(r"/content/AWSnet/dataset/splits/myopspp_unet2d_roi128_1mm"),
                ]
                for cs in cand_splits:
                    if cs.is_dir() and (cs / split_file).is_file():
                        split_dir = cs
                        break
                if split_dir is None:
                    split_dir = Path(__file__).resolve().parent / "splits" / "myopspp_unet2d_roi128_1mm"
            else:
                split_dir = Path(__file__).resolve().parent / "splits" / "myops380_no_val"

        self.split_path = Path(split_dir) / split_file if Path(split_dir).is_dir() else Path(split_dir)
        self.cases = read_split_names(self.split_path)

        if not self.is_nifti_export:
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
            self.modality_dirs = {
                "C0": self.data_root / "bSSFP" / "test_vol_h5",
                "LGE": self.data_root / "LGE" / "test_vol_h5",
                "T2": self.data_root / "T2w" / "test_vol_h5",
            }
        else:
            self.data_root = cases_dir.parent

    @staticmethod
    def _resolve_vol_path(mod_dir: Path, case_id: str) -> Path:
        return _resolve_vol_path(mod_dir, case_id)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case_id = self.cases[idx]

        if self.is_nifti_export:
            if nib is None:
                raise ImportError("nibabel is required to load MyoPS++ NIfTI volumes directly.")
            center = "CenterB" if case_id.startswith("Case2") else "CenterC"
            case_dir = self.nifti_cases_dir / center / case_id
            if not case_dir.is_dir():
                if (self.nifti_cases_dir / case_id).is_dir():
                    case_dir = self.nifti_cases_dir / case_id
                else:
                    for cand in self.nifti_cases_dir.glob(f"*/{case_id}"):
                        if cand.is_dir():
                            case_dir = cand
                            break

            p_c0 = case_dir / f"{case_id}_C0.nii.gz"
            p_lge = case_dir / f"{case_id}_LGE.nii.gz"
            p_t2 = case_dir / f"{case_id}_T2.nii.gz"
            p_gd = case_dir / f"{case_id}_gd.nii.gz"

            c0_vol = ingest_image_array(nib.load(str(p_c0)).get_fdata(dtype=np.float32), name=f"{case_id}_c0")
            lge_vol = ingest_image_array(nib.load(str(p_lge)).get_fdata(dtype=np.float32), name=f"{case_id}_lge")
            t2_vol = ingest_image_array(nib.load(str(p_t2)).get_fdata(dtype=np.float32), name=f"{case_id}_t2")
            raw_label = np.asarray(nib.load(str(p_gd)).dataobj)
            label_vol = canonicalize_raw_myopspp(raw_label)

            # Extract physical spacing in mm
            meta_json = case_dir / "metadata.json"
            spacing = np.array([1.0, 1.0, 10.0], dtype=np.float64)
            affine = np.eye(4, dtype=np.float64)
            if meta_json.is_file():
                try:
                    meta = json.loads(meta_json.read_text(encoding="utf-8"))
                    if "output_spacing_mm" in meta:
                        spacing = np.asarray(meta["output_spacing_mm"], dtype=np.float64)
                    elif "native_spacing_mm" in meta:
                        spacing = np.asarray(meta["native_spacing_mm"], dtype=np.float64)
                    if "output_affine" in meta:
                        affine = np.asarray(meta["output_affine"], dtype=np.float64)
                    elif "native_affine" in meta:
                        affine = np.asarray(meta["native_affine"], dtype=np.float64)
                except Exception:
                    pass
            else:
                nii_c0 = nib.load(str(p_c0))
                affine = np.asarray(nii_c0.affine, dtype=np.float64)
                spacing = np.linalg.norm(affine[:3, :3], axis=0)

            return {
                "image": c0_vol,
                "image1": lge_vol,
                "image2": t2_vol,
                "label": label_vol,
                "case_name": case_id,
                "patient_id": case_id,
                "spacing": spacing,
                "affine": affine,
                "spacing_unit": "mm",
                "has_geometry": True,
                "c0": c0_vol,
                "lge": lge_vol,
                "t2": t2_vol,
                "case_id": case_id,
            }
        else:
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

                # Check attributes for metadata
                attrs = dict(f_c0.attrs)
                stored_order = attrs.get("label_order", "legacy")
                if self.label_order != "auto":
                    order_to_use = self.label_order
                elif "canonical" in str(stored_order):
                    order_to_use = "canonical"
                else:
                    order_to_use = "legacy"
                label_vol = canonicalize_label(raw_label, source_order=order_to_use)

                has_geom = False
                spacing = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                affine = np.full((4, 4), np.nan, dtype=np.float64)
                spacing_unit = "unknown"

                if "spacing" in f_c0:
                    spacing = np.asarray(f_c0["spacing"][:], dtype=np.float64)
                    has_geom = True
                    spacing_unit = str(attrs.get("spacing_unit", "mm"))
                elif "spacing" in attrs:
                    spacing = np.asarray(attrs["spacing"], dtype=np.float64)
                    has_geom = True
                    spacing_unit = str(attrs.get("spacing_unit", "mm"))

                if "affine" in f_c0:
                    affine = np.asarray(f_c0["affine"][:], dtype=np.float64)
                elif "affine" in attrs:
                    affine = np.asarray(attrs["affine"], dtype=np.float64)

            return {
                "image": c0_vol,
                "image1": lge_vol,
                "image2": t2_vol,
                "label": label_vol,
                "case_name": case_id,
                "patient_id": case_id,
                "spacing": spacing,
                "affine": affine,
                "spacing_unit": spacing_unit,
                "has_geometry": has_geom,
                "c0": c0_vol,
                "lge": lge_vol,
                "t2": t2_vol,
                "case_id": case_id,
            }


# Backwards compatible aliases
MyoPSTrainDataset = MyoPS380TrainDataset
MyoPSTestVolumeDataset = MyoPS380TestVolumeDataset
MyoPSDataset = MyoPS380TrainDataset
