# NYCU pytorch-retinanet

![img3](https://github.com/yhenon/pytorch-retinanet/blob/master/images/3.jpg)
![img5](https://github.com/yhenon/pytorch-retinanet/blob/master/images/5.jpg)

本專案基於 [yhenon/pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet)，實作 RetinaNet 物件偵測模型，詳細演算法請參閱論文 [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)（Lin et al.）。  
在原始版本的基礎上，本版本額外新增了 WSI（全切片影像）推論、Patch 批次推論、STAS_VGH 評估以及單張影像視覺化等功能。

---

## 效能結果

使用 Resnet-50 backbone、輸入解析度 600px，COCO 資料集可達 **33.5% mAP**（論文結果為 34.0%，差異來自使用 Adam 而非 SGD+weight decay）。

---

## 安裝

1. Clone 此 repo

2. 安裝系統套件：

```bash
apt-get install tk-dev python-tk
```

3. 安裝 Python 套件：

```bash
pip install pandas pycocotools opencv-python requests
pip install tifffile zarr scikit-image  # WSI 推論需要
```

---

## 訓練

使用 `train.py` 進行訓練，支援 COCO 與 CSV 兩種資料格式。

**COCO 格式：**

```bash
python train.py --dataset coco --coco_path ../coco --depth 50
```

**CSV 自定義格式：**

```bash
python train.py --dataset csv \
    --csv_train <訓練標注.csv> \
    --csv_classes <類別列表.csv> \
    --csv_val <驗證標注.csv>
```

> `--csv_val` 為選填，省略時不執行驗證。

---

## 預訓練模型

可從以下連結下載 PyTorch state dict：

- https://drive.google.com/open?id=1yLmjq3JtXi841yXWBxst0coAgR26MNBS

載入方式：

```python
retinanet = model.resnet50(num_classes=dataset_train.num_classes())
retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))
```

---

## 驗證

### COCO 驗證

```bash
python coco_validation.py \
    --coco_path ~/path/to/coco \
    --model_path /path/to/model/coco_resnet_50_map_0_335_state_dict.pt
```

預期輸出：

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.499
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.357
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.167
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.466
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.429
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.597
```

### CSV 驗證

```bash
python csv_validation.py \
    --csv_annotations_path path/to/annotations.csv \
    --model_path path/to/model.pt \
    --images_path path/to/images_dir \
    --class_list_path path/to/class_list.csv \
    --iou_threshold 0.5
```

輸出格式：

```
label_1 : (label_1_mAP)
Precision :  ...
Recall:  ...
```



---

## 視覺化

### visualize.py — 資料集批次視覺化

對驗證集批次顯示 bounding box。

**COCO 格式：**

```bash
python visualize.py --dataset coco --coco_path ../coco --model <path/to/model.pt>
```

**CSV 格式：**

```bash
python visualize.py --dataset csv \
    --csv_classes <path/to/class_list.csv> \
    --csv_val <path/to/val_annots.csv> \
    --model <path/to/model.pt>
```

---

### visualize_single_image.py — 單張影像視覺化

對指定資料夾中的所有影像逐一執行推論並顯示偵測結果。

```bash
python visualize_single_image.py \
    --image_dir <影像資料夾> \
    --model_path <path/to/model.pt> \
    --class_list <path/to/class_list.csv>
```

---

## 新增功能

### patch_infer.py — 預切 Patch 批次推論

對已預先切好的 patch 影像資料夾執行 RetinaNet 推論。  
每個 slide 的 patch 存放於獨立子資料夾，**檔名格式須為 `<x1>_<y1>_<x2>_<y2>.png`**（記錄該 patch 在 WSI 中的全域座標）。

**目錄結構：**

```
input_dir/
    <slide_name>/
        <x1>_<y1>_<x2>_<y2>.png
        ...
```

**執行範例：**

```bash
python patch_infer.py \
    --input_dir /workspace/with \
    --output_dir /workspace/with_predictions \
    --model_path /path/to/model.pt \
    --class_list /workspace/STAS_VGH/classes.csv \
    --score_threshold 0.3 \
    --target_label 0
```

**主要參數：**

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--input_dir` | Patch 根目錄，每個子資料夾代表一張 slide | `/workspace/with` |
| `--output_dir` | 輸出 CSV 結果的目錄 | `/workspace/with_predictions` |
| `--model_path` | 模型 `.pt` 路徑 | （必填） |
| `--class_list` | 類別對應 CSV（class_name,class_id） | `/workspace/STAS_VGH/classes.csv` |
| `--min_side` | 輸入影像最小邊長 | `704` |
| `--max_side` | 輸入影像最大邊長 | `1920` |
| `--score_threshold` | 偵測信心分數門檻 | `0.3` |
| `--target_label` | 只保留此類別標籤，`-1` 表示所有類別 | `0` |
| `--max_slides` | 只處理前 N 張 slide，`0` 表示全部 | `0` |
| `--device` | 指定 `cuda` 或 `cpu`，省略則自動偵測 | 自動 |
| `--disable_amp_norm` | 停用 AmpNorm（HarmoFL） | `False` |

**輸出檔案：**

- `predicted_tiles_manifest.csv` — 有偵測結果的 patch 清單
- `predicted_detections_global.csv` — 所有偵測框的全域座標
- `slide_summary.csv` — 每張 slide 的統計摘要

---

### wsi_infer_on_the_fly.py — WSI 即時推論

直接讀取 TIFF 金字塔（WSI）檔案，即時切 tile 並執行 RetinaNet 推論，無需預先切好 patch 存到磁碟。  
流程：讀取 WSI → 建立組織遮罩過濾背景 → 逐 tile 推論 → 儲存有偵測結果的 tile。

```bash
python wsi_infer_on_the_fly.py \
    --input_dir /workspace/no_stas_wsi \
    --output_dir /workspace/no_stas_wsi_predicted_tiles \
    --model_path /path/to/model.pt \
    --class_list /workspace/STAS_VGH/classes.csv \
    --target_mpp 0.5
```

**主要參數：**

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--input_dir` | 含有 TIFF WSI 的資料夾 | `/workspace/no_stas_wsi` |
| `--output_dir` | 輸出 tile 與 CSV 的資料夾 | `/workspace/no_stas_wsi_predicted_tiles` |
| `--model_path` | 模型 `.pt` 路徑 | （必填） |
| `--class_list` | 類別對應 CSV | `/workspace/STAS_VGH/classes.csv` |
| `--target_mpp` | 目標解析度（微米/像素） | `0.5` |
| `--tile_w` / `--tile_h` | Tile 寬度 / 高度（像素） | `1920` / `828` |
| `--stride_w` / `--stride_h` | Tile 滑動步長 | `1920` / `828` |
| `--min_side` / `--max_side` | 模型輸入縮放範圍 | `704` / `1920` |
| `--score_threshold` | 偵測信心分數門檻 | `0.3` |
| `--target_label` | 只保留此類別，`-1` 表示所有類別 | `0` |
| `--min_tissue_ratio` | Tile 需包含的最低組織比例（用於跳過背景） | `0.03` |
| `--tissue_mask_side` | 組織遮罩縮圖邊長 | `2048` |
| `--sat_threshold` | 組織遮罩彩度門檻 | `0.035` |
| `--intensity_threshold` | 組織遮罩亮度門檻 | `0.93` |
| `--max_slides` | 只處理前 N 張 WSI，`0` 表示全部 | `0` |
| `--max_tissue_tiles_per_slide` | 每張 WSI 最多處理的 tile 數，`0` 不限 | `0` |
| `--no_save_tiles` | 不儲存預測 tile 影像到磁碟 | `False` |
| `--disable_amp_norm` | 停用 AmpNorm | `False` |

---

### eval_stas_vgh_iouv5.py — STAS_VGH 評估

使用 `IoU_v5` 計算 AP、Precision/Recall，對指定的 RetinaNet checkpoint 在 STAS_VGH 測試集上進行完整評估。

```bash
python eval_stas_vgh_iouv5.py \
    --images_dir /workspace/STAS_VGH/Test_Images \
    --csv_gt /workspace/STAS_VGH/test_annotations_fixed.csv \
    --model_path /workspace/csv_retinanet_29.pt \
    --min_side 608 \
    --max_side 1024 \
    --conf_score 0.05 \
    --iou_threshold 0.5
```

**主要參數：**

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--images_dir` | 測試影像資料夾 | `/workspace/STAS_VGH/Test_Images` |
| `--csv_gt` | Ground truth 標注 CSV | `/workspace/STAS_VGH/test_annotations_fixed.csv` |
| `--model_path` | 模型 `.pt` 路徑 | `/workspace/csv_retinanet_29.pt` |
| `--min_side` | 輸入影像最小邊長 | `608` |
| `--max_side` | 輸入影像最大邊長 | `1024` |
| `--conf_score` | 評估用信心分數門檻 | `0.05` |
| `--iou_threshold` | IoU 判定門檻 | `0.5` |
| `--device` | 指定 `cuda` 或 `cpu` | 自動 |
| `--disable_amp_norm` | 停用 AmpNorm | `False` |
| `--out_pred_json` | 輸出預測 JSON 路徑 | `/root/pytorch-retinanet/stas_vgh_pred.json` |
| `--out_gt_json` | 輸出 GT JSON 路徑 | `/root/pytorch-retinanet/stas_vgh_gt.json` |

**輸出範例：**

```
[result] AP=0.8321  (best-F1 index=42, best-P=0.8500, best-R=0.8100)
```

---

### IoU_v5.py — PR 曲線計算工具

提供 `PR_func` 類別，可根據 Ground Truth 與預測結果計算 Precision-Recall 曲線及 mAP。  
此模組主要供 `eval_stas_vgh_iouv5.py` 呼叫，也可獨立使用：

```python
from IoU_v5 import get_precision_recall

ap, best_box, pr_curve, max_idx = get_precision_recall(
    gt_json,       # dict: {image_name: [[x1,y1,x2,y2], ...]}
    pred_json,     # dict: {image_name: [[x1,y1,x2,y2,score], ...]}
    classes=1,
    conf_score=0.05,
    iou_threshold=0.5,
)
precision, recall = pr_curve
```

`PR_func` 也支援繪製 PR 曲線：

```python
import pandas as pd
from IoU_v5 import PR_func

df = pd.DataFrame({'recall': recall, 'precision': precision})
pr = PR_func(df, class_names=['STAS'])
pr.plot_pr_curve(smooth=True)
print(pr.get_map(mode='smootharea'))
```

---

## 模型 Backbone

RetinaNet 使用 ResNet 作為 backbone，透過 `--depth` 設定深度。  
可選值：`18`、`34`、`50`、`101`、`152`。深度越大準確度越高，但速度較慢、記憶體需求也較大。

---

## CSV 資料集格式

### 標注格式

每行一個標注框，格式為：

```
path/to/image.jpg,x1,y1,x2,y2,class_name
```

無標注的負樣本影像：

```
path/to/image.jpg,,,,,
```

完整範例：

```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
```

### 類別對應格式

```
class_name,id
```

類別 ID 從 0 開始，不需加入背景類別。

範例：

```
cow,0
cat,1
bird,2
```

---

## 致謝

- 部分程式碼來自 [keras retinanet](https://github.com/fizyr/keras-retinanet)
- NMS 模組來自 [pytorch faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)

---

## 範例結果

![img1](https://github.com/yhenon/pytorch-retinanet/blob/master/images/1.jpg)
![img2](https://github.com/yhenon/pytorch-retinanet/blob/master/images/2.jpg)
![img4](https://github.com/yhenon/pytorch-retinanet/blob/master/images/4.jpg)
![img6](https://github.com/yhenon/pytorch-retinanet/blob/master/images/6.jpg)
![img7](https://github.com/yhenon/pytorch-retinanet/blob/master/images/7.jpg)
![img8](https://github.com/yhenon/pytorch-retinanet/blob/master/images/8.jpg)
