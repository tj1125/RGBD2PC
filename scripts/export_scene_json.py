#!/usr/bin/env python3
"""Export a scene JSON (object + scene info) and visualize the point cloud."""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

DEFAULT_GRASP_POSE = [
    [1.0, 0.0, 0.0, 0.0],   # gripper X axis stays horizontal (world X)
    [0.0, 0.0, 1.0, 0.0],   # gripper Y axis points up (world Z)
    [0.0, -1.0, 0.0, 0.0],  # gripper Z axis (approach) points forward (world Y)
    [0.0, 0.0, 0.0, 0.0],
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_dataset = project_root / "dataset" / "Room_1_1"
    parser = argparse.ArgumentParser(
        description="Export a scene JSON similar to result_scene_json/1745767724_664606.json."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default=default_dataset,
        help=f"Dataset directory (default: {default_dataset})",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Optional path to meta.mat (defaults to dataset/meta.mat).",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional object mask (defaults to dataset/segment_mask.png if present).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to result_scene_json/<timestamp_random>.json.",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="針對生成深度圖啟用 FOV 對應的 Y 軸縮放（預設關閉）。",
    )
    return parser.parse_args()


def load_intrinsics(meta_path: Path):
    meta = sio.loadmat(str(meta_path))
    try:
        fx = float(meta["fx"].squeeze())
        fy = float(meta["fy"].squeeze())
        cx = float(meta["cx"].squeeze())
        cy = float(meta["cy"].squeeze())
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"meta.mat 缺少欄位：{exc}") from exc
    return fx, fy, cx, cy, meta


def resolve_dataset_path(raw_path: Path) -> Path:
    raw_path = raw_path.expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    project_root = Path(__file__).resolve().parents[1]
    return (project_root / raw_path).resolve()


def load_mask(dataset_dir: Path, mask_override: Optional[Path], shape: Tuple[int, int]) -> np.ndarray:
    mask_path = mask_override
    if mask_path is None:
        candidate = dataset_dir / "segment_mask.png"
        mask_path = candidate if candidate.exists() else None
    if mask_path is None:
        return np.ones(shape, dtype=bool)
    mask_path = mask_path.expanduser()
    if not mask_path.is_absolute():
        mask_path = dataset_dir / mask_path
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"無法讀取 mask：{mask_path}")
    return mask > 0


def _extract_scalar(meta_dict: dict, key: str) -> Optional[float]:
    value = meta_dict.get(key)
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(value.squeeze())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _estimate_fake_scale(meta_dict: dict) -> Tuple[float, float, str]:
    fov = _extract_scalar(meta_dict, "verticalFov")
    fov_key = "verticalFov"
    if fov is None:
        fov = _extract_scalar(meta_dict, "horizontalFov")
        fov_key = "horizontalFov"
    if fov is None or fov <= 0:
        raise ValueError("meta.mat 缺少可用的 FOV 欄位。")
    exponent = 0.274653 * fov - 3.951243
    scale = math.exp(exponent) * 0.5
    scale = float(np.clip(scale, 0.01, 1.0))
    return scale, fov, fov_key


def compute_point_cloud(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray]:
    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_m
    valid = (Z > 0) & np.isfinite(Z)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points = np.stack([X, Z, Y], axis=-1)
    return points, valid


def pixel_to_point(x: int, y: int, z: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    return np.array([X, z, Y], dtype=np.float64)


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_dataset_path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"找不到 dataset：{dataset_dir}")

    color_path = dataset_dir / "color.png"
    depth_path = dataset_dir / "depth.png"
    meta_path = args.meta if args.meta is not None else dataset_dir / "meta.mat"

    for path in (color_path, depth_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(f"找不到必要檔案：{path}")

    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise ValueError(f"無法讀取 color 圖像：{color_path}")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise ValueError(f"無法讀取 depth 圖像：{depth_path}")
    depth = depth_raw.astype(np.float32)
    if depth.max() > 10:
        depth *= 0.001  # mm → m

    obj_mask = load_mask(dataset_dir, args.mask, depth.shape)
    fx, fy, cx, cy, meta_dict = load_intrinsics(meta_path)
    if args.fake:
        depth_valid_mask = obj_mask & np.isfinite(depth) & (depth > 0)
        if np.any(depth_valid_mask):
            ys, xs = np.where(depth_valid_mask)
            depths_valid = depth[depth_valid_mask]

            top_idx = np.argmin(ys)
            bottom_idx = np.argmax(ys)
            left_idx = np.argmin(xs)
            right_idx = np.argmax(xs)
            top_x, top_y = int(xs[top_idx]), int(ys[top_idx])
            bottom_x, bottom_y = int(xs[bottom_idx]), int(ys[bottom_idx])
            left_x, left_y = int(xs[left_idx]), int(ys[left_idx])
            right_x, right_y = int(xs[right_idx]), int(ys[right_idx])
            max_idx = np.argmax(depths_valid)
            max_y = ys[max_idx]
            max_x = xs[max_idx]

            top_pt = pixel_to_point(top_x, top_y, float(depth[top_y, top_x]), fx, fy, cx, cy)
            bottom_pt = pixel_to_point(bottom_x, bottom_y, float(depth[bottom_y, bottom_x]), fx, fy, cx, cy)
            left_pt = pixel_to_point(left_x, left_y, float(depth[left_y, left_x]), fx, fy, cx, cy)
            right_pt = pixel_to_point(right_x, right_y, float(depth[right_y, right_x]), fx, fy, cx, cy)
            plane_point = pixel_to_point(max_x, max_y, float(depth[max_y, max_x]), fx, fy, cx, cy)

            v1 = bottom_pt - top_pt
            v2 = right_pt - left_pt
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                normal = np.array([0.0, 1.0, 0.0])
            else:
                normal /= norm

            n_dot_p0 = float(np.dot(normal, plane_point))

            H, W = depth.shape
            u = np.arange(W)
            v = np.arange(H)
            uu, vv = np.meshgrid(u, v)
            directions = np.stack(
                [
                    (uu - cx) / fx,
                    np.ones_like(uu, dtype=np.float32),
                    (vv - cy) / fy,
                ],
                axis=-1,
            )
            denom = directions @ normal
            plane_depth = np.full_like(depth, np.nan, dtype=np.float32)
            valid_denom = np.abs(denom) > 1e-6
            plane_depth[valid_denom] = n_dot_p0 / denom[valid_denom]
            fallback_depth = depths_valid.max()
            plane_depth[~valid_denom] = fallback_depth

            depth[~obj_mask] = plane_depth[~obj_mask]

            raw_valid = depth_raw[depth_valid_mask]
            if raw_valid.size > 0 and np.median(depths_valid) > 0:
                scale = float(np.median(raw_valid)) / float(np.median(depths_valid))
            else:
                scale = 1000.0
            depth_raw_plane = plane_depth * scale
            if np.issubdtype(depth_raw.dtype, np.integer):
                depth_raw_plane = np.clip(
                    depth_raw_plane, 0, np.iinfo(depth_raw.dtype).max
                )
            depth_raw_fill = depth_raw_plane.astype(depth_raw.dtype, copy=False)
            depth_raw[~obj_mask] = depth_raw_fill[~obj_mask]
    points_all, valid_depth = compute_point_cloud(depth, fx, fy, cx, cy)

    depth_scale = 1.0
    if args.fake:
        try:
            depth_scale, used_fov, fov_key = _estimate_fake_scale(meta_dict)
            print(
                f"ℹ️ fake 模式：使用 {fov_key}={used_fov:.3f}°，"
                f"Y 軸（深度）縮放 {depth_scale:.3f}"
            )
        except ValueError as exc:
            print(f"⚠️ fake 模式無法套用：{exc}")
    points_all = points_all.copy()
    points_all[:, :, 1] *= depth_scale

    object_mask = obj_mask & valid_depth
    if not np.any(object_mask):
        raise ValueError("object mask 與 depth 沒有重疊，無法輸出點雲。")

    colors_flat = color_rgb.reshape(-1, 3)
    full_pc_flat = points_all.reshape(-1, 3)
    object_flat_mask = object_mask.reshape(-1)
    object_points = full_pc_flat[object_flat_mask]
    object_colors = colors_flat[object_flat_mask].astype(int)

    rng = np.random.default_rng()
    obj_limit = min(10000, object_points.shape[0])
    full_limit = min(10000, full_pc_flat.shape[0])
    if object_points.shape[0] > obj_limit:
        idx = rng.choice(object_points.shape[0], obj_limit, replace=False)
        object_points = object_points[idx]
        object_colors = object_colors[idx]
    if full_pc_flat.shape[0] > full_limit:
        idx = rng.choice(full_pc_flat.shape[0], full_limit, replace=False)
        full_pc_flat = full_pc_flat[idx]
        colors_flat = colors_flat[idx]
        object_flat_mask = object_flat_mask[idx]

    full_mask_flat = object_flat_mask.astype(int)
    full_pc_augmented = np.concatenate(
        [
            full_pc_flat,
            colors_flat.astype(np.float32),
            full_mask_flat[:, None].astype(np.float32),
        ],
        axis=1,
    )

    scene_info = {
        "img_depth": depth_raw.astype(np.int32).tolist(),
        "img_color": color_rgb.tolist(),
        "full_pc": [full_pc_augmented.tolist()],
        "full_pc_fields": ["x", "y", "z", "r", "g", "b", "mask"],
        "obj_mask": object_mask.tolist(),
    }
    object_info = {
        "pc": object_points.tolist(),
        "pc_color": object_colors.tolist(),
    }
    grasp_poses = [DEFAULT_GRASP_POSE]
    grasp_conf = [0.0 for _ in grasp_poses]
    grasp_info = {"grasp_poses": grasp_poses, "grasp_conf": grasp_conf}

    project_root = Path(__file__).resolve().parents[1]
    result_dir = project_root / "result_scene_json"
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        output_path = resolve_dataset_path(args.output)
    else:
        timestamp = int(time.time())
        random_suffix = np.random.randint(100000, 999999)
        output_path = result_dir / f"{timestamp}_{random_suffix}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "object_info": object_info,
                "grasp_info": grasp_info,
                "scene_info": scene_info,
            },
            f,
        )
    print(f"✅ 已輸出 JSON：{output_path}")
    print(f"   Object points：{len(object_points)}，Full cloud：{full_pc_flat.shape[0]}")

    # Matplotlib visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, len(object_points) // 5000)
    sample_pts = object_points[::step]
    sample_colors = (object_colors[::step] / 255.0).clip(0, 1)
    ax.scatter(
        sample_pts[:, 0],
        sample_pts[:, 1],
        sample_pts[:, 2],
        c=sample_colors,
        s=2,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Depth Z (m)")
    ax.set_zlabel("Y (m)")
    ax.set_title("Object Point Cloud Preview")

    plt.show()


if __name__ == "__main__":
    main()
