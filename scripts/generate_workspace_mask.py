#!/usr/bin/env python3
"""Create workspace masks for each dataset folder using a YOLO model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - ultralytics optional dependency
    raise SystemExit(
        "✖️ 需要安裝 ultralytics 套件，請先執行 `pip install ultralytics`。"
    ) from exc


COLOR_CANDIDATES = ("color", "rgb", "image")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use YOLO to predict workspace masks for dataset folders.")
    parser.add_argument(
        "-d",
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="資料集根目錄（會針對底下的每個子資料夾處理）",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=Path("models/yolo_model/best.pt"),
        help="YOLO 權重檔案 .pt 路徑",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device（如 'cpu', 'mps', 'cuda:0'）。預設由 YOLO 自動決定。",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="只保留指定類別 ID；若未設定則保留全部偵測結果。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若已存在 workspace_mask.png 仍重新產生。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出將會處理的資料夾，不實際寫檔。",
    )
    return parser.parse_args()


def find_color_image(folder: Path) -> Path | None:
    for stem in COLOR_CANDIDATES:
        for ext in IMAGE_EXTS:
            candidate = folder / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def collect_targets(root: Path) -> list[Path]:
    if root.is_file():
        return [root.parent]
    if root.is_dir():
        subdirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
        return subdirs or [root]
    raise FileNotFoundError(f"找不到資料夾：{root}")


def mask_from_result(result, image_shape, class_filter: list[int] | None) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    indices = range(len(result.boxes)) if result.boxes is not None else []
    if class_filter is not None and result.boxes is not None and result.boxes.cls is not None:
        classes = result.boxes.cls.int().cpu().tolist()
        indices = [i for i, cls_id in enumerate(classes) if cls_id in class_filter]

    if result.masks is not None and result.masks.data is not None:
        mask_data = result.masks.data.cpu().numpy()
        for idx in indices:
            if idx >= len(mask_data):
                continue
            seg = mask_data[idx]
            seg_resized = cv2.resize(seg, (width, height), interpolation=cv2.INTER_NEAREST)
            mask[seg_resized >= 0.5] = 255
        if mask.any():
            return mask

    if result.boxes is not None and result.boxes.xyxy is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        for idx in indices:
            if idx >= len(boxes):
                continue
            x1, y1, x2, y2 = boxes[idx].astype(int)
            x1 = int(np.clip(x1, 0, width - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
    return mask


def process_folder(folder: Path, model: YOLO, args: argparse.Namespace) -> None:
    color_path = find_color_image(folder)
    if color_path is None:
        print(f"⚠️  跳過 {folder}: 找不到 color 圖片。")
        return

    mask_path = folder / "workspace_mask.png"
    if mask_path.exists() and not args.overwrite:
        print(f"ℹ️  已存在 {mask_path}，使用 --overwrite 才會重新產生。")
        return

    image = cv2.imread(str(color_path))
    if image is None:
        print(f"⚠️  無法讀取影像：{color_path}")
        return

    print(f"▶️  推論 {color_path.relative_to(Path.cwd())}")
    results = model.predict(
        source=image,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )
    result = results[0]
    mask = mask_from_result(result, image.shape, args.classes)

    if not mask.any():
        print(f"⚠️  {folder} 沒有偵測到任何目標，生成全黑 mask。")

    if args.dry_run:
        print("🛈  dry-run 模式，未寫入檔案。")
        return

    cv2.imwrite(str(mask_path), mask)
    print(f"✅  已輸出 workspace mask → {mask_path.relative_to(Path.cwd())}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    model_path = args.model.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"找不到 YOLO 權重檔：{model_path}")

    targets = collect_targets(dataset_root)
    if not targets:
        print("❌ 沒有可處理的資料夾。")
        sys.exit(1)

    model = YOLO(str(model_path))
    for folder in targets:
        process_folder(folder, model, args)


if __name__ == "__main__":
    main()
