# RGBD2PC

本專案提供數個將 RGB-D / 深度資料轉換為點雲、熱度圖或 Open3D 可視化的小工具腳本，資料預設放在 `dataset/` 目錄底下。

## 環境需求
- Python 3.9+
- 依賴套件請參考 `requirements.txt`

安裝：

```bash
pip install -r requirements.txt
```

## 專案結構
- `scripts/`：所有工具腳本，請從專案根目錄或 `scripts/` 目錄執行。
- `dataset/`：各物件的範例資料夾（`color`, `depth`, `meta.mat`, `workspace_mask.png` 等）。
- `result_json/`：`RGBD2PC_plt_masked.py` 生成的點雲 JSON 會存放於此，命名為 `[物品]_graspgen.json`。
- `references/`：可放置參考檔案範例（目前為空）。

## 工具腳本

### 1. `scripts/RGBD2PC_plt_masked.py`
以 Matplotlib 繪製點雲，可讀取 workspace mask 並輸出 GraspGen JSON。這是唯一會輸出 JSON 的腳本。

```bash
python scripts/RGBD2PC_plt_masked.py -d dataset/example_data_apple
```

參數：
- `-d / --dataset`：資料夾路徑（預設 `dataset/example_data_apple`）。
- `--mask`：指定遮罩路徑（相對或絕對）；若省略則會在資料夾內尋找 `workspace_mask.png`。
- `-m / --max-distance`：限定最大深度。
- `--focus-percentile`：僅保留最近的百分比點數（0~1）。
- `--target-points`：限制輸出點數量。

輸出會存到 `result_json/[物品]_graspgen.json`，物品名稱取自資料夾名稱的最後一段（例如 `example_data_apple` → `apple_graspgen.json`）。

### 2. `scripts/RGBD2PC_plt.py`
以 Matplotlib 繪製 RGB-D 點雲（不套用 workspace mask，也不寫出 JSON）。

```bash
python scripts/RGBD2PC_plt.py -d dataset/example_data_water
```

### 3. `scripts/RGBD2PC_o3d.py`
使用 Open3D 建立點雲，並輸出 `output.ply` 供第三方軟體檢視。

```bash
python scripts/RGBD2PC_o3d.py -d dataset/example_data_cup
```

### 4. `scripts/Depth2PC.py`
僅使用深度圖生成點雲，可視化灰階點雲（無 RGB），不輸出 JSON。

```bash
python scripts/Depth2PC.py -d dataset/example_data_wine
```

### 5. `scripts/DepthHeatmap.py`
將深度圖轉換為熱度圖，預設會在原始檔名後補上 `_heatmap.png`。

```bash
python scripts/DepthHeatmap.py dataset/example_data_banana/depth.png
```

常用參數：
- `--min-distance` / `--max-distance`：控制顏色對應的距離範圍。
- `--colormap`：指定 OpenCV colormap（預設 `inferno`）。
- `--bright-far`：保持亮色對應遠距離；預設亮色代表近距離。

## 注意事項
- 所有腳本都會自動辨識深度單位（毫米或公尺）並轉換。
- 若在專案根目錄以外位置執行腳本，可使用絕對路徑或 `--dataset` 的相對路徑（相對於當前工作目錄）。
- 只有 `RGBD2PC_plt_masked.py` 會輸出 JSON；若不需要 JSON 請執行其他腳本。

## 測試資料
`dataset/` 內提供多份範例資料，可直接使用上述指令測試。若需替換自己的資料，請留意檔案命名（`color.*`, `depth.*`, `meta.mat`, `workspace_mask.png` 等）需與腳本預設一致。

