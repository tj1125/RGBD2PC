import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RGB-D data inside a dataset folder into a point cloud and export GraspGen input."
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
    args = parse_args()
    dataset_dir = args.dataset

    color_path = dataset_dir / "color.png"
    depth_path = dataset_dir / "depth.png"
    meta_path = dataset_dir / "meta.mat"

    if not color_path.exists() or not depth_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"❌ 找不到 dataset 檔案，請確認資料夾包含 color.png、depth.png、meta.mat：{dataset_dir}"
        )

    # === 1️⃣ 讀取 RGB、Depth、內參 ===
    color = cv2.cvtColor(cv2.imread(str(color_path)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    print(depth.min(), depth.max())
    depth /= 1000.0  # 如果是毫米，改成公尺

    meta = sio.loadmat(str(meta_path))
    fx = float(meta["fx"].squeeze())
    fy = float(meta["fy"].squeeze())
    cx = float(meta["cx"].squeeze())
    cy = float(meta["cy"].squeeze())

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
    rgb = color.reshape(-1, 3)[mask.flatten()]

    # === 5️⃣ 若要符合右手座標，可翻軸（Unity → GraspGen）
    # Y = -Y
    # Z = -Z

    points = np.stack([X, Y, Z], axis=1)

    # === 6️⃣ 可視化 ===
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[::10], Y[::10], Z[::10], c=rgb[::10] / 255.0, s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("RGB-D → 3D Point Cloud")
    ax.set_box_aspect([1, 1, 1])
    plt.show()

    # === 7️⃣ 輸出 GraspGen JSON ===
    payload = {
        "pc": points.tolist(),
        "pc_color": rgb.tolist(),
        "grasp_poses": [np.eye(4).tolist()],
        "grasp_conf": [0.0],
    }
    with open("graspgen_input.json", "w") as f:
        json.dump(payload, f)
    print(f"✅ Saved {len(points)} points to graspgen_input.json")


if __name__ == "__main__":
    main()
