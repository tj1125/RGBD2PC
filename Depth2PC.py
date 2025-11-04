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
    X, Y, Z = X[mask], Y[mask], Z[mask]

    # === 5️⃣ 若 Unity → 右手座標，翻軸（如需）===
    # Y = -Y
    # Z = -Z

    points = np.stack([X, Y, Z], axis=1)

    # === 6️⃣ 可視化（灰階點雲）===
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    depth_vis = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
    ax.scatter(X[::10], Y[::10], Z[::10], c=plt.cm.viridis(depth_vis[::10]), s=1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
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
