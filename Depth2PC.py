import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Depth data inside a dataset folder into a point cloud (no RGB) and export GraspGen input."
    )
    default_dataset = Path(__file__).resolve().parent / "example_data_apple"
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
        "--target-points",
        type=int,
        default=None,
        help="Optional number of points to sample from the near region for denser visualization.",
    )
    parser.add_argument(
        "--focus-percentile",
        type=float,
        default=1.0,
        help="Keep only the nearest X fraction of valid depths (0 < X ≤ 1).",
    )
    return parser.parse_args()


def main() -> None:
    def resolve_image_path(directory: Path, stem: str) -> Path:
        extensions = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
        for ext in extensions:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        supported = ", ".join(f"{stem}{ext}" for ext in extensions[:3])
        raise FileNotFoundError(
            f"❌ 找不到必要檔案，請確認資料夾包含 {supported}：{directory}"
        )

    args = parse_args()
    dataset_dir = args.dataset

    depth_path = resolve_image_path(dataset_dir, "depth")
    meta_path = dataset_dir / "meta.mat"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"❌ 找不到必要檔案，請確認資料夾包含 meta.mat：{dataset_dir}"
        )

    # === 1️⃣ 讀取 Depth、內參 ===
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    print(f"Depth range (raw): {depth.min():.2f} ~ {depth.max():.2f}")

    # 自動偵測單位（mm or m）
    if depth.max() > 10:  # 通常是毫米
        depth /= 1000.0
        print("ℹ️ 偵測到深度單位為 mm，已自動轉換為公尺。")
    else:
        print("ℹ️ 偵測到深度單位為公尺。")

    meta = sio.loadmat(str(meta_path))
    fx = float(meta["fx"].squeeze())
    fy = float(meta["fy"].squeeze())
    cx = float(meta["cx"].squeeze())
    cy = float(meta["cy"].squeeze())
    print(f"Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

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

    # 按距離排序，確保先處理最近的點
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size == 0:
        raise ValueError("❌ 沒有符合條件的深度像素可用於生成點雲。")

    depths_flat = Z.flatten()[valid_indices]
    order = np.argsort(depths_flat)
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

    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    X_sel = X_flat[sampled_indices]
    Y_sel = Y_flat[sampled_indices]
    Z_sel = Z_flat[sampled_indices]
    depth_values = Z_sel
    points = np.stack([X_sel, Z_sel, -Y_sel], axis=1)
    print(
        f"ℹ️ 最終點雲深度範圍：{depth_values.min():.3f} m → {depth_values.max():.3f} m，"
        f"點數 {depth_values.size}。"
    )
    print("ℹ️ 座標系統：X 向右、Y 向前（深度）、Z 向上。")

    # === 5️⃣ 若 Unity → 右手座標，翻軸（如需）===
    # Y = -Y
    # Z = -Z

    # === 6️⃣ 可視化（灰階點雲）===
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    depth_norm = (depth_values - depth_values.min()) / (
        depth_values.max() - depth_values.min() + 1e-8
    )
    depth_vis = 1.0 - depth_norm  # 讓亮度代表距離較近
    plot_slice = slice(None, None, 10)
    ax.scatter(
        points[plot_slice, 0],
        points[plot_slice, 1],
        points[plot_slice, 2],
        c=plt.cm.viridis(depth_vis[plot_slice]),
        s=1,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (forward, m)")
    ax.set_zlabel("Z (up, m)")
    ax.set_title("Depth → 3D Point Cloud (No RGB)")
    ax.set_box_aspect([1, 1, 1])
    plt.show()

    # === 7️⃣ 輸出 GraspGen JSON ===
    payload = {
        "pc": points.tolist(),
        "pc_color": [],  # 無顏色
        "grasp_poses": [np.eye(4).tolist()],
        "grasp_conf": [0.0],
    }
    output_path = dataset_dir / "graspgen_input.json"
    with open(output_path, "w") as f:
        json.dump(payload, f)
    print(f"✅ 已輸出點雲，共 {len(points)} 點 → {output_path}")


if __name__ == "__main__":
    main()


# import scipy.io as sio
# import cv2
# import numpy as np
# meta = sio.loadmat("dataset/example_data_apple/meta.mat")
# fx = float(meta["fx"].squeeze())
# fy = float(meta["fy"].squeeze())
# cx = float(meta["cx"].squeeze())
# cy = float(meta["cy"].squeeze())

# depth = cv2.imread("dataset/example_data_apple/depth.png", cv2.IMREAD_UNCHANGED)
# H, W = depth.shape

# print(f"depth size = {W}x{H}")
# print(f"meta.mat intrinsics:")
# print(f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

# # 檢查比例
# ratio_fx = fx / (W / (2 * np.tan(np.deg2rad(60) / 2)))  # 假設 Unity FOV=60
# print(f"fx 比例 (相對於 Unity FOV=60): {ratio_fx:.2f}")
