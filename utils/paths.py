"""Path resolution utilities for AWSnet on local and server environments."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_split_dir(candidate: str | Path | None = None) -> Path:
    """Resolves split directory containing train.txt and test_vol.txt."""
    if candidate and str(candidate).strip():
        p = Path(candidate)
        if p.is_dir() and (p / "train.txt").is_file():
            return p

    fallbacks = [
        Path(r"D:\nckh_model"),
        Path(r"D:\nckh_model\AWSnet\dataset\splits\myops380_no_val"),
        Path(r"dataset/splits/myops380_no_val"),
        Path(r"/content/AWSnet/dataset/splits/myops380_no_val"),
        Path(r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val"),
        Path(r"../SCAR/preprocessing/splits/myops380_no_val"),
        Path(r"D:\nckh_model\splits\myops380_no_val"),
        Path(r"/content/SCAR/preprocessing/splits/myops380_no_val"),
    ]
    for fb in fallbacks:
        if fb.is_dir() and (fb / "train.txt").is_file():
            return fb

    return Path(r"D:\nckh_model\AWSnet\dataset\splits\myops380_no_val")


def resolve_data_root(candidate: str | Path | None = None) -> Path:
    """Resolves dataset root containing bSSFP, LGE, T2w directories."""
    if candidate and str(candidate).strip():
        p = Path(candidate)
        if p.is_dir():
            if (p / "bSSFP" / "train_npz").is_dir() or (p / "bSSFP" / "test_vol_h5").is_dir():
                return p
            if (p / "Processed_data" / "bSSFP" / "train_npz").is_dir() or (p / "Processed_data" / "bSSFP" / "test_vol_h5").is_dir():
                return p / "Processed_data"
            if p.name == "Raw_data" and ((p.parent / "Processed_data" / "bSSFP" / "train_npz").is_dir() or (p.parent / "Processed_data" / "bSSFP" / "test_vol_h5").is_dir()):
                return p.parent / "Processed_data"
            if (p / "bSSFP").is_dir():
                return p

    search_dirs = [
        Path(r"D:\nckh_model\Dataset\MyoPS380\Processed_data"),
        Path(r"D:\nckh_model\Dataset\MyoPS380"),
        Path(r"/content/AWSnet/MyoPS380_dataset/Processed_data"),
        Path(r"/content/drive/MyDrive/MyoPS380_dataset/Processed_data"),
        Path(r"/content/drive/MyDrive/Processed_data"),
        Path(r"/content/Dataset/MyoPS380/Processed_data"),
        Path(r"/content/MyoPS380_dataset/Processed_data"),
        Path(r"/content/MyoPS380_dataset"),
        Path(r"../Dataset/MyoPS380/Processed_data"),
    ]
    for d in search_dirs:
        if d.is_dir():
            if (d / "bSSFP" / "train_npz").is_dir() or (d / "bSSFP" / "test_vol_h5").is_dir():
                return d
            if (d / "Processed_data" / "bSSFP" / "train_npz").is_dir() or (d / "Processed_data" / "bSSFP" / "test_vol_h5").is_dir():
                return d / "Processed_data"
            if (d / "bSSFP").is_dir():
                return d

    return Path(r"D:\nckh_model\Dataset\MyoPS380\Processed_data")


def resolve_save_dir(candidate: str | Path | None = None) -> Path:
    """Resolves checkpoint output directory."""
    if candidate and str(candidate).strip():
        p = Path(candidate)
    else:
        p = Path(r"D:\nckh_model\AWSnet\checkpoints")
    p.mkdir(parents=True, exist_ok=True)
    return p
