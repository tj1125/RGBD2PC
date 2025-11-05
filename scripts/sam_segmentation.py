#!/usr/bin/env python3
"""Generate a Segment Anything (SAM) mask for a dataset sample."""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Meta Segment Anything (SAM) on a dataset sample to create a refined object mask. "
            "The output mask is stored as 'segment_mask.png' inside the dataset folder by default."
        )
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default=Path("dataset/example_data_apple"),
        help="Path to a dataset directory containing color/depth images and optional workspace_mask.png.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional path to a workspace mask image. Defaults to workspace_mask.png inside the dataset.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to a SAM checkpoint file (e.g. sam_vit_h_4b8939.pth). "
            "Download checkpoints from the official Segment Anything repository."
        ),
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="auto",
        help="SAM model type (vit_h, vit_l, vit_b). Use 'auto' to infer from checkpoint filename.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the generated mask. Defaults to dataset/segment_mask.png.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device for SAM (e.g. 'auto', 'mps', 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=3,
        help="Maximum number of workspace mask components to feed into SAM boxes (default: 3).",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=500,
        help="Ignore workspace mask components smaller than this area in pixels (default: 500).",
    )
    return parser.parse_args()


def find_color_image(dataset_dir: Path) -> Path:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    preferred_names = [
        dataset_dir / "color.png",
        dataset_dir / "color.jpg",
        dataset_dir / "color.jpeg",
        dataset_dir / "color.bmp",
    ]
    for candidate in preferred_names:
        if candidate.exists():
            return candidate

    generic_matches = sorted(dataset_dir.glob("color.*"))
    if generic_matches:
        return generic_matches[0]

    raise FileNotFoundError(
        f"Unable to locate a color image inside {dataset_dir}. Expected files like color.png or color.jpg."
    )


def load_workspace_mask(dataset_dir: Path, mask_override: Optional[Path]) -> np.ndarray:
    if mask_override is None:
        mask_path = dataset_dir / "workspace_mask.png"
    else:
        mask_path = mask_override if mask_override.is_absolute() else dataset_dir / mask_override

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Workspace mask not found. Expected {mask_path}. "
            "Provide --mask if the file uses a different name."
        )

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask image at {mask_path}.")
    return mask


def extract_boxes(mask: np.ndarray, max_components: int, min_area: int) -> List[np.ndarray]:
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scored_boxes: List[Tuple[float, np.ndarray]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < max(min_area, 1):
            continue
        scored_boxes.append((area, np.array([x, y, x + w, y + h], dtype=np.float32)))

    scored_boxes.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in scored_boxes[: max_components or None]]


def resolve_device(requested: Optional[str]) -> str:
    if requested and requested.lower() != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    backend = getattr(torch, "backends", None)
    if backend is not None and getattr(backend, "mps", None) and backend.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_SAM_MODEL_HINTS: dict[str, Tuple[str, ...]] = {
    "vit_h": ("vit_h", "sam_vit_h"),
    "vit_l": ("vit_l", "sam_vit_l"),
    "vit_b": ("vit_b", "sam_vit_b"),
}


def infer_model_type(checkpoint_path: Path, requested: Optional[str]) -> str:
    if requested and requested.lower() != "auto":
        return requested.lower()

    name = checkpoint_path.name.lower()
    stem = checkpoint_path.stem.lower()

    for model_type, hints in _SAM_MODEL_HINTS.items():
        if any(hint in name or hint in stem for hint in hints):
            return model_type

    raise ValueError(
        "Unable to infer SAM model type automatically. "
        "Please specify --sam-model-type explicitly (vit_h, vit_l, or vit_b)."
    )


def build_sam_predictor(model_type: str, checkpoint_path: Path, device: str):
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "Segment Anything is not installed. Install it with "
            "`pip install git+https://github.com/facebookresearch/segment-anything.git` "
            "or follow the official installation instructions."
        ) from exc

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {checkpoint_path}. Download the weights and update --sam-checkpoint."
        )

    if model_type not in sam_model_registry:
        raise ValueError(
            f"Unknown SAM model type '{model_type}'. Expected one of {sorted(sam_model_registry.keys())}."
        )

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))

    import torch

    sam.to(device=device)
    return SamPredictor(sam)


def apply_sam_boxes(predictor, image_rgb: np.ndarray, boxes: Iterable[np.ndarray]) -> np.ndarray:
    predictor.set_image(image_rgb)

    mask_accumulator = np.zeros(image_rgb.shape[:2], dtype=bool)
    boxes = list(boxes)

    if not boxes:
        height, width = image_rgb.shape[:2]
        boxes = [np.array([0, 0, width - 1, height - 1], dtype=np.float32)]

    for box in boxes:
        masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box, multimask_output=True)
        if masks is None or len(masks) == 0:
            continue
        best_index = int(np.argmax(scores))
        mask_accumulator |= masks[best_index].astype(bool)

    return mask_accumulator.astype(np.uint8) * 255


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    output_path = args.output or dataset_dir / "segment_mask.png"

    try:
        color_image_path = find_color_image(dataset_dir)
        color_bgr = cv2.imread(str(color_image_path), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise ValueError(f"Failed to load color image at {color_image_path}.")

        mask = load_workspace_mask(dataset_dir, args.mask)
        boxes = extract_boxes(mask, args.max_components, args.min_component_area)

        if color_bgr.shape[:2] != mask.shape[:2]:
            raise ValueError(
                "Workspace mask dimensions do not match the color image. "
                f"Color image size: {color_bgr.shape[:2]}, mask size: {mask.shape[:2]}"
            )

        device = resolve_device(args.device)
        model_type = infer_model_type(args.sam_checkpoint, args.sam_model_type)
        print(f"Using SAM device: {device}")
        print(f"Using SAM model type: {model_type}")
        predictor = build_sam_predictor(model_type, args.sam_checkpoint, device)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        sam_mask = apply_sam_boxes(predictor, color_rgb, boxes)

        refined_mask = cv2.bitwise_and(sam_mask, mask)
        output_path = output_path.resolve()
        cv2.imwrite(str(output_path), refined_mask)
        print(f"SAM segmentation mask written to {output_path}")

    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
