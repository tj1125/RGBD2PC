# RGBD2PC

This repository contains utility scripts for transforming RGB-D or depth data into point clouds, heatmaps, or Open3D visualizations. Sample datasets are stored under `dataset/`.

## Requirements
- Python 3.9+
- Dependencies listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository Structure
- `scripts/`: Command-line tools. Run them from the project root (recommended) or from inside `scripts/`.
- `dataset/`: Sample object folders (`color.*`, `depth.*`, `meta.mat`, optional `workspace_mask.png`, etc.).
- `result_json/`: Point-cloud JSON files produced by `RGBD2PC_plt_masked.py`, named `[object]_graspgen.json`.
- `references/`: Placeholder for reference artifacts (currently empty).

## Tools

### 1. `scripts/RGBD2PC_plt_masked.py`
Creates a masked point cloud with Matplotlib and exports a GraspGen-style JSON file. *This is the only script that writes JSON.*

```bash
python scripts/RGBD2PC_plt_masked.py -d dataset/example_data_apple
```

Key options:
- `-d / --dataset`: dataset directory (default `dataset/example_data_apple`).
- `--mask`: workspace mask path (relative to the dataset or absolute). Defaults to `workspace_mask.png` inside the dataset if present.
- `-m / --max-distance`: clamp far depths.
- `--focus-percentile`: keep only the nearest fraction (0–1).
- `--target-points`: downsample to a fixed number of points.

The output JSON is written to `result_json/[object]_graspgen.json`, where the object name is derived from the dataset folder (e.g., `example_data_apple` → `apple_graspgen.json`).

### 2. `scripts/RGBD2PC_plt.py`
Visualizes an RGB-D point cloud with Matplotlib, without applying a workspace mask and without writing JSON.

```bash
python scripts/RGBD2PC_plt.py -d dataset/example_data_water
```

### 3. `scripts/RGBD2PC_o3d.py`
Generates an Open3D point cloud and saves `output.ply` for external tools (MeshLab, CloudCompare, etc.).

```bash
python scripts/RGBD2PC_o3d.py -d dataset/example_data_cup
```

### 4. `scripts/Depth2PC.py`
Builds a point cloud from depth-only data and renders a grayscale Matplotlib chart (no RGB, no JSON output).

```bash
python scripts/Depth2PC.py -d dataset/example_data_wine
```

### 5. `scripts/DepthHeatmap.py`
Converts a depth image into a heatmap. By default, the output filename appends `_heatmap` to the input stem.

```bash
python scripts/DepthHeatmap.py dataset/example_data_banana/depth.png
```

Useful flags:
- `--min-distance` / `--max-distance`: control the depth range mapped to colors (in meters).
- `--colormap`: choose the OpenCV colormap (default `inferno`).
- `--bright-far`: keep bright colors for far pixels (default keeps bright = near).

## Notes
- All scripts automatically detect whether depth is in millimeters or meters and convert to meters when necessary.
- When running outside the project root, pass absolute paths or adjust `--dataset` relative to your current working directory.
- Only `RGBD2PC_plt_masked.py` writes JSON; use the other scripts for visualization-only workflows.

## Sample Data
The `dataset/` directory ships with several ready-to-use examples. To experiment with your own recordings, keep filenames aligned with the expected convention (`color.*`, `depth.*`, `meta.mat`, optional `workspace_mask.png`) so the scripts can locate inputs correctly.
