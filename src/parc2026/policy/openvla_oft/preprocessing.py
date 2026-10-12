"""Shared camera preprocessing at the LIBERO/OpenVLA-OFT policy boundary."""

from __future__ import annotations

import numpy as np
from PIL import Image


def prepare_libero_camera(
    image: np.ndarray,
    *,
    sensor_size: int = 128,
    rotate_images_180: bool = True,
) -> np.ndarray:
    """Match upstream LIBERO camera orientation before model preprocessing."""

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB uint8 image, got {array.shape}")
    if rotate_images_180:
        array = np.ascontiguousarray(array[::-1, ::-1])
    if array.shape[:2] != (sensor_size, sensor_size):
        array = np.asarray(
            Image.fromarray(array).resize(
                (sensor_size, sensor_size), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
    return np.ascontiguousarray(array)
