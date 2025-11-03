import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import scipy.io as sio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RGB-D data inside a dataset folder into an Open3D point cloud."
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

    # === 1. 讀取資料 ===
    color = cv2.cvtColor(cv2.imread(str(color_path)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)

    # 若深度是毫米，除以 1000 變成公尺
    print(depth.min(), depth.max())
    depth /= 1000.0

    # === 2. 讀取相機內參 ===
    meta = sio.loadmat(str(meta_path))
    fx = float(meta["fx"])
    fy = float(meta["fy"])
    cx = float(meta["cx"])
    cy = float(meta["cy"])
    H, W = depth.shape

    # === 3. 建立 Open3D 相機模型 ===
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(W, H, fx, fy, cx, cy)

    # === 4. 建立 RGB-D 圖像 ===
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(depth),
        depth_scale=1.0,        # 深度已是公尺
        depth_trunc=5.0,        # 超過5公尺捨棄
        convert_rgb_to_intensity=False
    )

    # === 5. 建立點雲 ===
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # === 6. 翻軸（Unity是左手系，Open3D是右手系） ===
    # 如發現點雲倒著或反方向，取消註解下面這行：
    # pcd.transform([[1, 0, 0, 0],
    #                [0,-1, 0, 0],
    #                [0, 0,-1, 0],
    #                [0, 0, 0, 1]])

    # === 7. 可視化 ===
    o3d.visualization.draw_geometries([pcd])

    # === 8. 輸出點雲檔（可用 MeshLab / CloudCompare 開） ===
    o3d.io.write_point_cloud("output.ply", pcd)
    print("✅ 已輸出 output.ply")


if __name__ == "__main__":
    main()
