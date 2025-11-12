# import argparse
# import json
# from pathlib import Path

# import cv2
# import numpy as np
# import scipy.io as sio
# import matplotlib.pyplot as plt


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Convert RGB-D data inside a dataset folder into a point cloud and export GraspGen input."
#     )
#     default_dataset = Path(__file__).resolve().parent / "example_data_apple"
#     parser.add_argument(
#         "-d",
#         "--dataset",
#         type=Path,
#         default=default_dataset,
#         help=f"Path to the dataset directory (default: {default_dataset})",
#     )
#     parser.add_argument(
#         "-m",
#         "--max-distance",
#         type=float,
#         default=None,
#         help="Maximum depth (meters) to keep in the point cloud; discard farther points.",
#     )
#     parser.add_argument(
#         "--focus-percentile",
#         type=float,
#         default=1.0,
#         help="Keep only the nearest X fraction of valid depths (0 < X ≤ 1).",
#     )
#     parser.add_argument(
#         "--target-points",
#         type=int,
#         default=None,
#         help="Optional number of points to sample from the near region for visualization/export.",
#     )
#     return parser.parse_args()


# def resolve_image_path(directory: Path, stem: str) -> Path:
#     extensions = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
#     for ext in extensions:
#         candidate = directory / f"{stem}{ext}"
#         if candidate.exists():
#             return candidate
#     options = "、".join(f"{stem}{ext}" for ext in extensions[:3])
#     raise FileNotFoundError(
#         f"❌ 找不到必要檔案，請確認資料夾包含 {options}：{directory}"
#     )
# def main() -> None:
#     args = parse_args()
#     dataset_dir = args.dataset

#     color_path = resolve_image_path(dataset_dir, "color")
#     depth_path = dataset_dir / "depth.png"
#     meta_path = dataset_dir / "meta.mat"

#     missing = []
#     if not depth_path.exists():
#         missing.append("depth.png")
#     if not meta_path.exists():
#         missing.append("meta.mat")
#     if missing:
#         joined = "、".join(missing)
#         raise FileNotFoundError(
#             f"❌ 找不到 dataset 檔案，請確認資料夾包含 {joined}：{dataset_dir}"
#         )

#     # === 1️⃣ 讀取 RGB、Depth、內參 ===
#     color = cv2.cvtColor(cv2.imread(str(color_path)), cv2.COLOR_BGR2RGB)
#     depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
#     print(depth.min(), depth.max())
#     if depth.max() > 10:
#         depth /= 1000.0
#         print("ℹ️ 偵測到深度單位為 mm，已自動轉換為公尺。")
#     else:
#         print("ℹ️ 偵測到深度單位為公尺。")

#     meta = sio.loadmat(str(meta_path))
#     fx = float(meta["fx"].squeeze())
#     fy = float(meta["fy"].squeeze())
#     cx = float(meta["cx"].squeeze())
#     cy = float(meta["cy"].squeeze())

#     # === 2️⃣ 建立像素座標 ===
#     H, W = depth.shape
#     u, v = np.meshgrid(np.arange(W), np.arange(H))

#     # === 3️⃣ 反投影成 3D ===
#     Z = depth
#     X = (u - cx) * Z / fx
#     Y = (v - cy) * Z / fy

#     # === 4️⃣ 過濾無效深度 ===
#     mask = (Z > 0) & np.isfinite(Z)
#     if args.max_distance is not None:
#         mask &= Z <= args.max_distance
#         print(
#             f"ℹ️ 已套用最大距離 {args.max_distance:.2f} m，保留 {mask.sum()} / {mask.size} 個有效像素。"
#         )
#     if not (0 < args.focus_percentile <= 1.0):
#         raise ValueError("--focus-percentile 必須介於 0 與 1 之間。")

#     valid_indices = np.flatnonzero(mask)
#     if valid_indices.size == 0:
#         raise ValueError("❌ 沒有符合條件的深度像素可用於生成點雲。")

#     X_flat = X.flatten()
#     Y_flat = Y.flatten()
#     Z_flat = Z.flatten()
#     colors_flat = color.reshape(-1, 3)

#     depth_valid = Z_flat[valid_indices]
#     order = np.argsort(depth_valid)
#     sorted_indices = valid_indices[order]

#     focus_count = int(np.ceil(sorted_indices.size * args.focus_percentile))
#     if focus_count <= 0:
#         raise ValueError("❌ focus-percentile 太小，沒有點可保留。")
#     if focus_count < sorted_indices.size:
#         sorted_indices = sorted_indices[:focus_count]
#         print(
#             f"ℹ️ 依照 focus-percentile={args.focus_percentile:.2f} 保留最近的 {focus_count} / {valid_indices.size} 點。"
#         )

#     if args.target_points is not None and args.target_points < sorted_indices.size:
#         sampled_indices = sorted_indices[: args.target_points]
#         print(
#             f"ℹ️ 針對近距離採樣 {args.target_points} 點（原本 {sorted_indices.size} 點）。"
#         )
#     else:
#         sampled_indices = sorted_indices

#     X_sel = X_flat[sampled_indices]
#     Y_sel = Y_flat[sampled_indices]
#     Z_sel = Z_flat[sampled_indices]
#     colors_sampled = colors_flat[sampled_indices]
#     depth_sampled = Z_sel

#     depth_norm = (depth_sampled - depth_sampled.min()) / (
#         depth_sampled.max() - depth_sampled.min() + 1e-8
#     )
#     depth_brightness = 0.3 + 0.7 * (1.0 - depth_norm)
#     colors_for_plot = np.clip(
#         (colors_sampled.astype(np.float32) / 255.0) * depth_brightness[:, None],
#         0.0,
#         1.0,
#     )
#     print(
#         f"ℹ️ 最終點雲深度範圍：{depth_sampled.min():.3f} m → {depth_sampled.max():.3f} m，"
#         f"點數 {depth_sampled.size}。"
#     )
#     points = np.stack([X_sel, -Z_sel, Y_sel], axis=1)
#     print("ℹ️ 座標系統：X 向右、Y 向後（深度鏡像）、Z 向下（鏡像）。")
#     print("ℹ️ 圖形顏色 = 原始 RGB × 深度亮度（亮 = 近）。")

#     # === 5️⃣ 若要符合右手座標，可翻軸（Unity → GraspGen）
#     # Y = -Y
#     # Z = -Z

#     # === 6️⃣ 可視化 ===
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection="3d")
#     plot_slice = slice(None, None, 10)
#     ax.scatter(
#         points[plot_slice, 0],
#         points[plot_slice, 1],
#         points[plot_slice, 2],
#         c=colors_for_plot[plot_slice],
#         s=1,
#     )
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y (mirrored)")
#     ax.set_zlabel("Z (mirrored)")
#     ax.set_title("RGB-D → 3D Point Cloud")
#     ax.set_box_aspect([1, 1, 1])
#     plt.show()

#     # === 7️⃣ 輸出 GraspGen JSON ===
#     payload = {
#         "pc": points.tolist(),
#         "pc_color": colors_sampled.astype(np.uint8).tolist(),
#         "grasp_poses": [np.eye(4).tolist()],
#         "grasp_conf": [0.0],
#     }
#     with open("graspgen_input.json", "w") as f:
#         json.dump(payload, f)
#     print(f"✅ Saved {len(points)} points to graspgen_input.json")


# if __name__ == "__main__":
#     main()





import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def _extract_scalar(meta_dict: dict, key: str):
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


def _estimate_fake_scale(meta_dict: dict):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RGB-D data inside a dataset folder into a point cloud and export GraspGen input."
    )
    project_root = Path(__file__).resolve().parents[1]
    default_dataset = project_root / "dataset" / "example_data_apple"
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default=default_dataset,
        help=f"Path to the dataset directory (default: {default_dataset})",
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
    parser.add_argument(
        "--fake",
        action="store_true",
        help="針對生成深度圖啟用 FOV 對應的 Y 軸縮放（預設關閉）。",
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
def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = args.dataset.expanduser()
    if not dataset_dir.is_absolute():
        candidate = (Path.cwd() / dataset_dir).resolve()
        if candidate.exists():
            dataset_dir = candidate
        else:
            dataset_dir = (project_root / dataset_dir).resolve()
    else:
        dataset_dir = dataset_dir.resolve()

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

    # === 1️⃣ 讀取 RGB、Depth、內參 ===
    color = cv2.cvtColor(cv2.imread(str(color_path)), cv2.COLOR_BGR2RGB)
    
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    print(depth.min(), depth.max())
    if depth.max() > 10:
        depth /= 1000.0
        print("ℹ️ 偵測到深度單位為 mm，已自動轉換為公尺。")
    else:
        print("ℹ️ 偵測到深度單位為公尺。")

    meta = sio.loadmat(str(meta_path))
    fx = float(meta["fx"].squeeze())
    fy = float(meta["fy"].squeeze())
    cx = float(meta["cx"].squeeze())
    cy = float(meta["cy"].squeeze())

    depth_scale = 1.0
    if args.fake:
        try:
            depth_scale, used_fov, fov_key = _estimate_fake_scale(meta)
            print(
                f"ℹ️ fake 模式：使用 {fov_key}={used_fov:.3f}°，"
                f"Y 軸縮放 {depth_scale:.3f}"
            )
        except ValueError as exc:
            print(f"⚠️ fake 模式無法套用：{exc}")

    # === 2️⃣ 建立像素座標 ===
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # === 3️⃣ 反投影成 3D ===
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # === 4️⃣ 過濾無效深度 ===
    mask = (Z > 0) & np.isfinite(Z)
    if args.max_distance is not None:
        mask &= Z <= args.max_distance
        print(
            f"ℹ️ 已套用最大距離 {args.max_distance:.2f} m，保留 {mask.sum()} / {mask.size} 個有效像素。"
        )
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
    points = np.stack([X_sel, -depth_scale * Z_sel, Y_sel], axis=1)
    if args.fake:
        print(f"ℹ️ Y 軸已套用 FOV 縮放：{depth_scale:.3f}")
    print("ℹ️ 座標系統：X 向右、Y 向後（深度鏡像）、Z 向下（鏡像）。")
    print("ℹ️ 圖形顏色 = 原始 RGB × 深度亮度（亮 = 近）。")

    # === 5️⃣ 若要符合右手座標，可翻軸（Unity → GraspGen）
    # Y = -Y
    # Z = -Z

    # === 6️⃣ 可視化 ===
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
    ax.set_title("RGB-D → 3D Point Cloud")
    ax.set_box_aspect([1, 1, 1])
    plt.show()

if __name__ == "__main__":
    main()
