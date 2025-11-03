#!/usr/bin/env python3
"""Update meta.mat with Unity camera intrinsics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

# ---------- 依需求調整 ----------
META_PATH = Path("meta.mat")          # meta.mat 的路徑
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOCAL_LENGTH_MM = 2.052764
SENSOR_SIZE_MM = (3.896, 2.453)       # (sensor width, sensor height)
HORIZONTAL_FOV_DEG = 87.0             # 只做紀錄用，可省略
NEAR_CLIP = 0.3
FAR_CLIP = 12.0
FRAME_ID = "camera"
# --------------------------------

def compute_intrinsics() -> dict[str, float | np.ndarray]:
    sensor_w, sensor_h = SENSOR_SIZE_MM

    fx = FOCAL_LENGTH_MM * IMAGE_WIDTH / sensor_w
    fy = FOCAL_LENGTH_MM * IMAGE_HEIGHT / sensor_h
    cx = IMAGE_WIDTH / 2.0
    cy = IMAGE_HEIGHT / 2.0

    k_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    return {
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": k_matrix,
        "near": NEAR_CLIP,
        "far": FAR_CLIP,
        "sensorSize": np.array(SENSOR_SIZE_MM, dtype=float),
        "focalLength": float(FOCAL_LENGTH_MM),
        "horizontalFov": float(HORIZONTAL_FOV_DEG),
        "frame_id": FRAME_ID,
        "intrinsic_matrix": k_matrix,
    }

def update_meta(meta: dict, intrinsics: dict) -> dict:
    for key, value in intrinsics.items():
        meta[key] = value
    return meta

def main() -> None:
    meta_path = META_PATH.resolve()
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.mat not found at {meta_path}")

    print(f"Loading meta from {meta_path}")
    meta = loadmat(meta_path)
    meta = {k: v for k, v in meta.items() if not k.startswith("__")}

    intrinsics = compute_intrinsics()
    meta = update_meta(meta, intrinsics)

    savemat(meta_path, meta, do_compression=False)
    print("meta.mat updated successfully.")

if __name__ == "__main__":
    main()
