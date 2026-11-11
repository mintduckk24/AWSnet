"""Exact SCAR RandomGenerator and ResizeGenerator for MyoPS380.

Faithfully implements Section 3 of Promt_train_test_no_val.txt:
- 1) if/elif/elif sequence with synchronized rot/flip (50%), rotate -20..19 (25%), gaussian noise (12.5%)
- 2) np.roll shift [-5, 5] (50%)
- 3) shared brightness uniform(0.8, 1.2) clip(0, 255) (50%)
- 4) shared gamma uniform(0.8, 1.2) power(x/255.0, gamma)*255.0 (50%)
- 5) top/left crop (50%)
- 6) zoom resize: order=3 for images, order=0 for mask
- 7) returns float32 (1, H, W) images and int64 (H, W) label
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
import torch
from torch.nn import functional as F


def random_rot_flip(image, image1, image2, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    image1 = np.rot90(image1, k)
    image2 = np.rot90(image2, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    image1 = np.flip(image1, axis=axis).copy()
    image2 = np.flip(image2, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, image1, image2, label


def random_rotate(image, image1, image2, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    image1 = ndimage.rotate(image1, angle, order=0, reshape=False)
    image2 = ndimage.rotate(image2, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, image1, image2, label


def add_gaussian_noise(image, image1, image2, label, mean=0, std=0.2):
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    image1 = image1 + noise
    image2 = image2 + noise
    image = np.clip(image, 0, 1)
    image1 = np.clip(image1, 0, 1)
    image2 = np.clip(image2, 0, 1)
    return image, image1, image2, label


class ResizeGenerator:
    """Deterministic, alignment-preserving image resize and tensor conversion."""

    def __init__(self, output_size: tuple[int, int] | int = (128, 128)):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if len(output_size) != 2 or any(int(v) <= 0 for v in output_size):
            raise ValueError("output_size must contain two positive dimensions.")
        self.output_size = tuple(int(v) for v in output_size)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        result = dict(sample)
        shape = sample["label"].shape
        if len(shape) != 2:
            raise ValueError("Slice transforms require 2D samples; volume resize belongs in inference.")
        for key in ("image", "image1", "image2", "label", "C0", "LGE", "T2"):
            if key not in sample:
                continue
            value = np.asarray(sample[key])
            if value.shape != shape:
                raise ValueError("Modalities and mask must have identical spatial shapes.")
            dtype = np.int64 if key == "label" else np.float32
            tensor = torch.from_numpy(np.array(value, dtype=dtype, copy=True, order="C"))
            if shape != self.output_size:
                if key == "label":
                    tensor = F.interpolate(tensor[None, None].float(), size=self.output_size, mode="nearest")[0, 0].long()
                else:
                    tensor = F.interpolate(tensor[None, None], size=self.output_size, mode="bilinear", align_corners=False)[0, 0]
            result[key] = tensor if key == "label" else tensor.unsqueeze(0)
        return result


class RandomGenerator:
    """Reproduce I-MMSeg-Fork / SCAR datasets/myops_dataset.py, including RNG order."""

    def __init__(self, output_size: tuple[int, int] | int = (128, 128)):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(int(v) for v in output_size)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = sample.get("image", sample.get("C0"))
        image1 = sample.get("image1", sample.get("LGE"))
        image2 = sample.get("image2", sample.get("T2"))
        label = sample["label"]

        if random.random() > 0.5:
            image, image1, image2, label = random_rot_flip(image, image1, image2, label)
        elif random.random() > 0.5:
            image, image1, image2, label = random_rotate(image, image1, image2, label)
        elif random.random() > 0.5:
            image, image1, image2, label = add_gaussian_noise(image, image1, image2, label)

        if random.random() > 0.5:
            shift_x = random.randint(-5, 5)
            shift_y = random.randint(-5, 5)
            image = np.roll(image, (shift_x, shift_y), axis=(0, 1))
            image1 = np.roll(image1, (shift_x, shift_y), axis=(0, 1))
            image2 = np.roll(image2, (shift_x, shift_y), axis=(0, 1))
            label = np.roll(label, (shift_x, shift_y), axis=(0, 1))

        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255)
            image1 = np.clip(image1 * factor, 0, 255)
            image2 = np.clip(image2 * factor, 0, 255)

        if random.random() > 0.5:
            gamma = random.uniform(0.8, 1.2)
            image = np.power(image / 255.0, gamma) * 255.0
            image1 = np.power(image1 / 255.0, gamma) * 255.0
            image2 = np.power(image2 / 255.0, gamma) * 255.0

        if random.random() > 0.5:
            top = random.randint(0, image.shape[0] // 5)
            left = random.randint(0, image.shape[1] // 5)
            image = image[top:top + self.output_size[0], left:left + self.output_size[1]]
            image1 = image1[top:top + self.output_size[0], left:left + self.output_size[1]]
            image2 = image2[top:top + self.output_size[0], left:left + self.output_size[1]]
            label = label[top:top + self.output_size[0], left:left + self.output_size[1]]

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image1 = zoom(image1, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image2 = zoom(image2, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(np.ascontiguousarray(image.astype(np.float32))).unsqueeze(0)
        image1 = torch.from_numpy(np.ascontiguousarray(image1.astype(np.float32))).unsqueeze(0)
        image2 = torch.from_numpy(np.ascontiguousarray(image2.astype(np.float32))).unsqueeze(0)
        label = torch.from_numpy(np.ascontiguousarray(label.astype(np.float32))).long()

        result = dict(sample)
        result.update(
            image=image,
            image1=image1,
            image2=image2,
            C0=image,
            LGE=image1,
            T2=image2,
            label=label,
        )
        return result
