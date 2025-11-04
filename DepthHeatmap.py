import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a depth image into a distance-based heatmap."
    )
    parser.add_argument(
        "depth_image",
        type=Path,
        help="Path to the input depth image (supports PNG/JPEG with 16-bit depth).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output image path. Default: <depth_image_stem>_heatmap.png",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=None,
        help="Minimum distance (in meters) for the colormap scale. Defaults to the minimum valid depth.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Maximum distance (in meters) for the colormap scale. Defaults to the maximum valid depth.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="inferno",
        choices=[
            "inferno",
            "plasma",
            "magma",
            "viridis",
            "turbo",
            "jet",
        ],
        help="OpenCV colormap to use for the heatmap.",
    )
    parser.add_argument(
        "--bright-far",
        action="store_true",
        help="Keep brighter colors for farther points (default keeps brighter colors for nearer points).",
    )
    return parser.parse_args()


COLORMAP_MAP = {
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
}


def load_depth_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"❌ 無法讀取深度圖：{path}")
    if image.ndim == 3:
        # 若輸入包含 RGB 頻道，轉為灰階再處理
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32)


def normalize_depth(
    depth: np.ndarray,
    min_distance: Optional[float],
    max_distance: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
    mask = (depth > 0) & np.isfinite(depth)
    if not np.any(mask):
        raise ValueError("❌ 深度圖沒有有效的深度值（>0）可用於產生熱度圖。")

    working_depth = depth.astype(np.float32)
    valid = working_depth[mask]

    # 若最大值大於 10，通常表示單位是毫米，轉換為公尺比較直覺
    converted = False
    if valid.max() > 10.0:
        working_depth = working_depth / 1000.0
        valid = working_depth[mask]
        converted = True

    min_val = valid.min() if min_distance is None else float(min_distance)
    max_val = valid.max() if max_distance is None else float(max_distance)

    if max_val <= min_val:
        raise ValueError(
            f"❌ max_distance ({max_val}) 必須大於 min_distance ({min_val})。"
        )

    normalized = np.zeros_like(working_depth, dtype=np.float32)
    normalized[mask] = (working_depth[mask] - min_val) / (max_val - min_val)
    np.clip(normalized, 0.0, 1.0, out=normalized)

    return normalized, mask, min_val, max_val, converted


def apply_colormap(normalized: np.ndarray, mask: np.ndarray, colormap: int) -> np.ndarray:
    uint8_img = (normalized * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(uint8_img, colormap)
    heatmap[~mask] = (0, 0, 0)
    return heatmap


def main() -> None:
    args = parse_args()
    depth_path = args.depth_image
    if not depth_path.exists():
        raise FileNotFoundError(f"❌ 找不到深度圖：{depth_path}")

    depth = load_depth_image(depth_path)
    normalized, mask, min_val, max_val, converted = normalize_depth(
        depth,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
    )

    if not args.bright_far:
        normalized = 1.0 - normalized

    heatmap = apply_colormap(normalized, mask, COLORMAP_MAP[args.colormap])

    output_path = (
        args.output
        if args.output is not None
        else depth_path.with_name(f"{depth_path.stem}_heatmap.png")
    )
    cv2.imwrite(str(output_path), heatmap)
    brightness_desc = "明亮 = 近（自動反轉）" if not args.bright_far else "明亮 = 遠（保留原始對應）"
    print(
        f"✅ 已輸出熱度圖：{output_path}\n"
        f"   深度範圍：{min_val:.3f} m → {max_val:.3f} m（無效像素已設為黑色）\n"
        f"   顏色：{brightness_desc}"
    )
    if converted:
        print("ℹ️ 偵測到輸入深度為毫米，已自動換算為公尺。")


if __name__ == "__main__":
    main()
