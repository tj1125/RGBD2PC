# RGBD2PC

RGBD2PC is a toolbox for turning RGB‑D captures into point clouds, heatmaps, workspace masks and scene-level JSON packages. The scripts are purposely lightweight so you can mix and match them in data-prep pipelines or debugging sessions.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/RGBD2PC_plt_masked.py -d dataset/example_data_apple
```

Each dataset folder should contain:

```
dataset/<name>/
  color.png (or .jpg)
  depth.png
  meta.mat         # intrinsics
  workspace_mask.png (optional)
  segment_mask.png (optional)
```

## Key Scripts

| Script | Purpose | Highlights |
| --- | --- | --- |
| `RGBD2PC_plt_masked.py` | Produces a masked point cloud, Matplotlib preview, and `[object]_graspgen.json`. | Workspace mask support, focus-based filtering, target-point sampling. |
| `RGBD2PC_plt.py` | Visualizes raw RGB‑D clouds (no mask, no JSON). | Great for quick sanity checks. |
| `RGBD2PC_o3d.py` | Builds an Open3D cloud and saves `output.ply`. | Good for MeshLab/CloudCompare. |
| `Depth2PC.py` | Converts depth-only captures to grayscale point clouds. | Minimal dependencies. |
| `DepthHeatmap.py` | Turns depth maps into colored heatmaps. | Adjustable min/max distance and colormap. |
| `sam_segmentation.py` | Runs Meta’s Segment Anything to generate `segment_mask.png`. | Auto device resolution, SAM checkpoint inference. |
| `generate_workspace_mask.py` | Uses YOLO (Ultralytics) to create `workspace_mask.png` for all datasets. | Bring your own weights. |
| `export_scene_json.py` | Packages an entire scene (object cloud, full cloud, depth/color images, masks, grasp poses) into `result_scene_json/<timestamp>_<rand>.json`. | Optional `--fake` mode flattens background depths and rescales the depth axis based on FOV. |

## `export_scene_json.py` in Detail

```bash
python scripts/export_scene_json.py \
  --dataset dataset/Room_1_1 \
  --fake         # optional; enables synthetic depth tweaks
```

What you get:

- `object_info.pc` / `pc_color`: mask-filtered points plus RGB.
- `scene_info.full_pc`: up to 30k points, each stored as `[x, y, z, r, g, b, mask]`.
- `scene_info.img_color`, `scene_info.img_depth`, `scene_info.obj_mask`.
- `grasp_info.grasp_poses` + `grasp_conf`: default gripper pose + confidence placeholders.

`--fake` mode:

- Scales the depth axis using dataset FOV metadata.
- Fills mask-out pixels with a fitted plane so the background lies on a single surface.
- Copies the plane depths into both floating-point depth and raw image buffers.

## Typical Workflow

1. **Collect / copy data** into `dataset/<name>`.
2. **(Optional) Generate masks**  
   `python scripts/generate_workspace_mask.py -d dataset --model models/yolo/best.pt`
3. **Inspect with Matplotlib**  
   `python scripts/RGBD2PC_plt.py -d dataset/<name>`
4. **Export masked cloud**  
   `python scripts/RGBD2PC_plt_masked.py -d dataset/<name> --mask segment_mask.png`
5. **Package the whole scene**  
   `python scripts/export_scene_json.py -d dataset/<name> --fake`

## Notes & Tips

- All scripts detect millimeters vs meters automatically. Depth images >10k are assumed to be mm and will be divided by 1000.
- Run scripts from the repo root to keep relative paths simple.
- `RGBD2PC_plt_masked.py`, `RGBD2PC_plt.py`, `RGBD2PC_o3d.py`, and `export_scene_json.py` accept `--fake` to rescale the depth axis using FOV metadata.
- SAM- and YOLO-based scripts require extra dependencies (see `requirements.txt` comments).

## License / Credits

Use at your own risk. File an issue or PR if you extend the pipeline—pull requests that improve dataset tooling or camera-model handling are welcome. Happy point-clouding!
