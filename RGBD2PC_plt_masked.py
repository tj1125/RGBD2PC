import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RGB-D data into a masked point cloud and export GraspGen input."
    )
    default_dataset = Path(__file__).resolve().parent / "dataset" / "example_data_apple"
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default=default_dataset,
        help=f"Path to the dataset directory (default: {default_dataset})",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional path to a workspace mask image. "
        "Non-zero pixels are kept. Defaults to dataset/workspace_mask.png if present.",
    )
    parser.add_argument(
        "-m",
        "--max-distance",
        type=float,
        default=None,
        help="Maximum depth (meters) to keep in the point cloud; discard farther points.",
    )
    parser.add_argument(
        "--focus-percentile",
        type=float,
        default=1.0,
        help="Keep only the nearest X fraction of valid depths (0 < X ≤ 1).",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=None,
        help="Optional number of points to sample from the near region for visualization/export.",
    )
    return parser.parse_args()


def resolve_image_path(directory: Path, stem: str) -> Path:
    extensions = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    for ext in extensions:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    options = "、".join(f"{stem}{ext}" for ext in extensions[:3])
    raise FileNotFoundError(
        f"❌ 找不到必要檔案，請確認資料夾包含 {options}：{directory}"
    )


def load_workspace_mask(
    dataset_dir: Path, mask_override: Optional[Path]
) -> Tuple[Optional[np.ndarray], Optional[Path]]:
    mask_path = mask_override if mask_override is not None else dataset_dir / "workspace_mask.png"
    if mask_path is None or not mask_path.exists():
        return None, None

    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        raise ValueError(f"❌ 無法讀取 workspace mask：{mask_path}")
    if mask_image.ndim == 3:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    mask_bool = mask_image > 0
    mask_bool = np.flipud(mask_bool)  # mirror to stay aligned with flipped RGB
    return mask_bool, mask_path


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset

    if not dataset_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 dataset：{dataset_dir}")

    mask_override = args.mask
    if mask_override is not None:
        mask_override = mask_override.expanduser()
        if not mask_override.is_absolute():
            mask_override = dataset_dir / mask_override

    color_path = resolve_image_path(dataset_dir, "color")
    depth_path = dataset_dir / "depth.png"
    meta_path = dataset_dir / "meta.mat"

    missing = []
    if not depth_path.exists():
        missing.append("depth.png")
    if not meta_path.exists():
        missing.append("meta.mat")
    if missing:
        joined = "、".join(missing)
        raise FileNotFoundError(
            f"❌ 找不到 dataset 檔案，請確認資料夾包含 {joined}：{dataset_dir}"
        )

    color = cv2.cvtColor(cv2.imread(str(color_path)), cv2.COLOR_BGR2RGB)
    color = np.flipud(color)
    print("ℹ️ 已對 RGB 圖像進行 Y 軸鏡像（垂直翻轉）。")

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    print(depth.min(), depth.max())
    if depth.max() > 10:
        depth /= 1000.0
        print("ℹ️ 偵測到深度單位為 mm，已自動轉換為公尺。")
    else:
        print("ℹ️ 偵測到深度單位為公尺。")

    mask_from_file, mask_path = load_workspace_mask(dataset_dir, mask_override)
    if mask_from_file is None:
        print("ℹ️ 找不到 workspace mask，將使用全部有效深度像素。")
    else:
        print(f"ℹ️ 已讀取 workspace mask：{mask_path}（已套用垂直鏡像）")

    meta = sio.loadmat(str(meta_path))
    fx = float(meta["fx"].squeeze())
    fy = float(meta["fy"].squeeze())
    cx = float(meta["cx"].squeeze())
    cy = float(meta["cy"].squeeze())

    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    mask = (Z > 0) & np.isfinite(Z)
    if args.max_distance is not None:
        mask &= Z <= args.max_distance
        print(
            f"ℹ️ 已套用最大距離 {args.max_distance:.2f} m，保留 {mask.sum()} / {mask.size} 個有效像素。"
        )
    if mask_from_file is not None:
        if mask_from_file.shape != depth.shape:
            raise ValueError(
                f"❌ workspace mask 尺寸不符：{mask_from_file.shape} vs depth {depth.shape}"
            )
        before = mask.sum()
        mask &= mask_from_file
        after = mask.sum()
        print(f"ℹ️ workspace mask 篩選：{after} / {before} 像素保留。")

    if not (0 < args.focus_percentile <= 1.0):
        raise ValueError("--focus-percentile 必須介於 0 與 1 之間。")

    valid_indices = np.flatnonzero(mask)
    if valid_indices.size == 0:
        raise ValueError("❌ 沒有符合條件的深度像素可用於生成點雲。")

    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    colors_flat = color.reshape(-1, 3)

    depth_valid = Z_flat[valid_indices]
    order = np.argsort(depth_valid)
    sorted_indices = valid_indices[order]

    focus_count = int(np.ceil(sorted_indices.size * args.focus_percentile))
    if focus_count <= 0:
        raise ValueError("❌ focus-percentile 太小，沒有點可保留。")
    if focus_count < sorted_indices.size:
        sorted_indices = sorted_indices[:focus_count]
        print(
            f"ℹ️ 依照 focus-percentile={args.focus_percentile:.2f} 保留最近的 {focus_count} / {valid_indices.size} 點。"
        )

    if args.target_points is not None and args.target_points < sorted_indices.size:
        sampled_indices = sorted_indices[: args.target_points]
        print(
            f"ℹ️ 針對近距離採樣 {args.target_points} 點（原本 {sorted_indices.size} 點）。"
        )
    else:
        sampled_indices = sorted_indices

    X_sel = X_flat[sampled_indices]
    Y_sel = Y_flat[sampled_indices]
    Z_sel = Z_flat[sampled_indices]
    colors_sampled = colors_flat[sampled_indices]
    depth_sampled = Z_sel

    depth_norm = (depth_sampled - depth_sampled.min()) / (
        depth_sampled.max() - depth_sampled.min() + 1e-8
    )
    depth_brightness = 0.3 + 0.7 * (1.0 - depth_norm)
    colors_for_plot = np.clip(
        (colors_sampled.astype(np.float32) / 255.0) * depth_brightness[:, None],
        0.0,
        1.0,
    )
    print(
        f"ℹ️ 最終點雲深度範圍：{depth_sampled.min():.3f} m → {depth_sampled.max():.3f} m，"
        f"點數 {depth_sampled.size}。"
    )
    points = np.stack([X_sel, -Z_sel, Y_sel], axis=1)
    print("ℹ️ 座標系統：X 向右、Y 向後（深度鏡像）、Z 向下（鏡像）。")
    print("ℹ️ 圖形顏色 = 原始 RGB × 深度亮度（亮 = 近）。")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_slice = slice(None, None, 10)
    ax.scatter(
        points[plot_slice, 0],
        points[plot_slice, 1],
        points[plot_slice, 2],
        c=colors_for_plot[plot_slice],
        s=1,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y (mirrored)")
    ax.set_zlabel("Z (mirrored)")
    ax.set_title("RGB-D → 3D Point Cloud (Masked)")
    ax.set_box_aspect([1, 1, 1])
    plt.show()

    default_pose = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    payload = {
        "pc": points.tolist(),
        "pc_color": colors_sampled.astype(np.uint8).tolist(),
        "grasp_poses": [default_pose],
        "grasp_conf": [0.0],
    }

    result_dir = Path(__file__).resolve().parent / "result_json"
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_dir.name
    object_name = dataset_name.split("_")[-1] if "_" in dataset_name else dataset_name
    output_path = result_dir / f"{object_name}_graspgen.json"

    with open(output_path, "w") as f:
        json.dump(payload, f)
    print(f"✅ Saved {len(points)} points to {output_path}")


if __name__ == "__main__":
    main()
