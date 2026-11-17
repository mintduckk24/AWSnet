"""Path resolution utilities for AWSnet on local and server environments."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_split_dir(
    candidate: str | Path | None = None,
    dataset: str = "auto",
    data_root: str | Path | None = None,
) -> Path:
    """Resolves split directory containing train.txt and test_vol.txt."""
    if candidate and str(candidate).strip():
        p = Path(candidate)
        if p.is_dir() and ((p / "train.txt").is_file() or (p / "splits.json").is_file()):
            return p
        if (p / "lists").is_dir() and (p / "lists" / "train.txt").is_file():
            return p / "lists"

    is_myopspp = (
        "myopspp" in str(candidate).lower()
        or "myopspp" in str(dataset).lower()
        or (data_root is not None and "myopspp" in str(data_root).lower())
    )
    is_myops380 = "myops380" in str(dataset).lower()

    if is_myopspp and not is_myops380:
        fallbacks = [
            Path(r"D:\nckh_model\AWSnet\dataset\splits\myopspp_unet2d_roi128_1mm"),
            Path(__file__).resolve().parent.parent / "dataset" / "splits" / "myopspp_unet2d_roi128_1mm",
            Path(r"/content/AWSnet/dataset/splits/myopspp_unet2d_roi128_1mm"),
            Path(r"/content/SCAR/data/myopspp_unet2d_roi128_1mm/cache/lists"),
            Path(r"D:\nckh_model\SCAR\data\myopspp_unet2d_roi128_1mm\cache\lists"),
        ]
        for fb in fallbacks:
            if fb.is_dir() and (fb / "train.txt").is_file():
                return fb

    fallbacks_380 = [
        Path(r"D:\nckh_model\AWSnet\dataset\splits\myops380_no_val"),
        Path(__file__).resolve().parent.parent / "dataset" / "splits" / "myops380_no_val",
        Path(r"/content/AWSnet/dataset/splits/myops380_no_val"),
        Path(r"D:\nckh_model\SCAR\preprocessing\splits\myops380_no_val"),
        Path(r"../SCAR/preprocessing/splits/myops380_no_val"),
        Path(r"D:\nckh_model\splits\myops380_no_val"),
        Path(r"/content/SCAR/preprocessing/splits/myops380_no_val"),
    ]
    for fb in fallbacks_380:
        if fb.is_dir() and (fb / "train.txt").is_file():
            return fb

    # Fallback to myopspp if myops380 not found
    fb_pp = Path(r"D:\nckh_model\AWSnet\dataset\splits\myopspp_unet2d_roi128_1mm")
    if fb_pp.is_dir() and (fb_pp / "train.txt").is_file():
        return fb_pp

    return Path(r"D:\nckh_model\AWSnet\dataset\splits\myops380_no_val")


def resolve_data_root(candidate: str | Path | None = None, dataset: str = "auto") -> Path:
    """Resolves dataset root containing bSSFP, LGE, T2w directories or MyoPS++ cases."""
    if candidate and str(candidate).strip():
        p = Path(candidate)
        if p.is_dir():
            # Check MyoPS++ NIfTI export
            if (p / "cases").is_dir() or (p / "CenterB").is_dir():
                return p
            if (p / "MyoPSpp_UNet2D_ROI128_1mm" / "cases").is_dir():
                return p / "MyoPSpp_UNet2D_ROI128_1mm"
            if (p / "bSSFP" / "train_npz").is_dir() or (p / "bSSFP" / "test_vol_h5").is_dir():
                return p
            if (p / "Processed_data" / "bSSFP" / "train_npz").is_dir() or (p / "Processed_data" / "bSSFP" / "test_vol_h5").is_dir():
                return p / "Processed_data"
            if p.name == "Raw_data" and ((p.parent / "Processed_data" / "bSSFP" / "train_npz").is_dir() or (p.parent / "Processed_data" / "bSSFP" / "test_vol_h5").is_dir()):
                return p.parent / "Processed_data"
            if (p / "bSSFP").is_dir():
                return p
            # Even if it's the root containing dataset_manifest.json
            if (p / "dataset_manifest.json").is_file():
                return p

    is_explicit_myops380 = "myops380" in str(dataset).lower()
    is_explicit_myopspp = "myopspp" in str(dataset).lower()

    myopspp_dirs = [
        Path(r"D:\nckh_model\Dataset\MyoPSpp_UNet2D_ROI128_1mm\MyoPSpp_UNet2D_ROI128_1mm"),
        Path(r"D:\nckh_model\Dataset\MyoPSpp_UNet2D_ROI128_1mm"),
        Path(r"/content/SCAR/data/myopspp_unet2d_roi128_1mm/cache"),
        Path(r"/content/SCAR/data/MyoPSpp_UNet2D_ROI128_1mm"),
        Path(r"/content/MyoPSpp_UNet2D_ROI128_1mm"),
        Path(r"/content/drive/MyDrive/MyoPSpp_UNet2D_ROI128_1mm/MyoPSpp_UNet2D_ROI128_1mm"),
        Path(r"/content/drive/MyDrive/MyoPSpp_UNet2D_ROI128_1mm"),
    ]

    myops380_dirs = [
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

    if is_explicit_myops380:
        search_dirs = myops380_dirs + myopspp_dirs
    elif is_explicit_myopspp:
        search_dirs = myopspp_dirs + myops380_dirs
    else:
        # Default priority: MyoPS++ then MyoPS380
        search_dirs = myopspp_dirs + myops380_dirs

    for d in search_dirs:
        if d.is_dir():
            if (d / "cases").is_dir() or (d / "CenterB").is_dir() or (d / "dataset_manifest.json").is_file():
                return d
            if (d / "bSSFP" / "train_npz").is_dir() or (d / "bSSFP" / "test_vol_h5").is_dir():
                return d
            if (d / "Processed_data" / "bSSFP" / "train_npz").is_dir() or (d / "Processed_data" / "bSSFP" / "test_vol_h5").is_dir():
                return d / "Processed_data"
            if (d / "bSSFP").is_dir():
                return d

    if is_explicit_myopspp:
        return Path(r"D:\nckh_model\Dataset\MyoPSpp_UNet2D_ROI128_1mm\MyoPSpp_UNet2D_ROI128_1mm")
    return Path(r"D:\nckh_model\Dataset\MyoPS380\Processed_data")


def resolve_save_dir(candidate: str | Path | None = None) -> Path:
    """Resolves checkpoint output directory."""
    if candidate and str(candidate).strip():
        p = Path(candidate)
    else:
        p = Path(r"D:\nckh_model\AWSnet\checkpoints")
    p.mkdir(parents=True, exist_ok=True)
    return p
