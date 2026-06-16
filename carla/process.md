# CARLA Sim-to-Real Pipeline 流程文件

**目標**：把 CARLA 語意標籤圖（label map）透過 pix2pixHD 轉成擬真道路影像。

Conda 環境：`carla_env`。所有指令從 `/home/itriu100/carla/` 執行。

---

## 架構總覽

```
RGB 圖像（NuRec / Cityscapes）
        │
        ▼
[Mask2Former] → 語意標籤（19 class trainId, 灰階 PNG）
        │
        ▼
 pix2pixHD 訓練
        │
        ▼
[pix2pixHD test] ← CARLA 錄製 RGB 的 Mask2Former 標籤
                   OR CARLA 語意感測器的 GT 標籤
        │
        ▼
   合成擬真圖 / 影片
```

---

## 資料來源

| 資料集 | 路徑 | 說明 |
|--------|------|------|
| NuRec 原始 | `datasets/nurec/` | 13 場景 × 594 幀 = 7,722 張 photorealistic |
| NuRec 擴充 | `datasets/nurec_raw/` | 38 場景 MP4 抽幀 ≈ 23,053 張 |
| Cityscapes | `datasets/training_semantic_v3/train_img/` | 2,975 張真實道路圖 |
| CARLA Town01 | `datasets/recorded/rgb/` + `semantic/` | 1,000 幀（RGB + 語意感測器） |
| CARLA Town03 | `datasets/recorded_Town03/rgb/` + `semantic/` | 1,000 幀（RGB + 語意感測器） |

---

## 模型版本對照

| 版本 | 訓練資料 | Label 來源 | 狀態 |
|------|----------|-----------|------|
| **v4** | NuRec 7,722 + Cityscapes 2,975 = 10,697 | NuRec: SegFormer / Cityscapes: 官方 trainId | 完成 |
| **v6** | NuRec 30,775 + Cityscapes 2,975 | 全部重跑 Mask2Former | 完成 |
| **v7** | 同 v6（fine-tune） | 同 v6 | 訓練中 |

---

## 流程一：V4

V4 使用較舊的 SegFormer 標籤（訓練資料在 v2/v3 dataset 已建好），主要流程是合併 dataset 後直接訓練。

### Step 1 — 組合 Dataset

```bash
conda run -n carla_env python3 prepare_v4_dataset.py
# 輸出：datasets/training_semantic_v4/
#   train_label/ : v2 (0~7721) + v3 (7722~10696) 的 symlink
#   train_img/   : 同上
#   test_label/  : 從 v2 複製（Town01 SegFormer 標籤）
#   test_Town03_label/ : 從 v2 複製（Town03 SegFormer + sky fix 標籤）
```

（若需要重新生成 SegFormer test 標籤，執行 `generate_v4_test_labels.py`，
輸出到 `datasets/v4_segformer_test/`，再手動複製進 v4 dataset。）

### Step 2 — 訓練

```bash
conda run -n carla_env bash train_semantic_v4.sh
# checkpoint: pix2pixHD/checkpoints/carla2real_semantic_v4/
# 設定: from scratch, lr=0.0002, 200 epochs, batchSize=2, scale_width_and_crop
```

監控：
```bash
tail -f pix2pixHD/checkpoints/carla2real_semantic_v4/loss_log.txt
```

### Step 3 — 測試推論

```bash
# Town03
conda run -n carla_env bash test_Town03_v4.sh
# 結果: pix2pixHD/results/carla2real_semantic_v4/test_Town03_latest/images/

# Town01
conda run -n carla_env bash test_Town01_v4.sh
# 結果: pix2pixHD/results/carla2real_semantic_v4/test_latest/images/
```

---

## 流程二：V6

V6 是完整重做的版本：所有訓練圖（NuRec + Cityscapes）都用 Mask2Former 重新生成標籤，Test 也換成 Mask2Former 標籤（或 CARLA GT）。

### Step 1 — 擴充 NuRec 原始幀（若 nurec_raw 未建立）

NuRec 原始 MP4 存在 `PhysicalAI-Autonomous-Vehicles-NuRec/sample_set/26.02_release/`。

```bash
# 從 38 個未使用場景的 MP4 抽幀
conda run -n carla_env python3 extract_nurec_mp4.py
# 輸出：datasets/nurec_raw/   （UUID_XXXXXX.png 格式）
```

### Step 2 — Mask2Former：NuRec 訓練標籤

```bash
conda run -n carla_env python3 generate_mask2former_labels.py
# 輸入：datasets/nurec/ + datasets/nurec_raw/
# 輸出：datasets/nurec_labels/   (灰階 19-class trainId PNG, 1024×512)
# 模型：facebook/mask2former-swin-large-cityscapes-semantic
# 已完成的會自動跳過（resume-safe）
```

### Step 3 — Mask2Former：Cityscapes 訓練標籤

```bash
conda run -n carla_env python3 generate_cityscapes_labels.py
# 輸入：datasets/training_semantic_v3/train_img/  (2975 張)
# 輸出：datasets/cityscapes_m2f_labels/
```

### Step 4 — Mask2Former：Test 標籤（CARLA RGB → Mask2Former）

```bash
conda run -n carla_env python3 generate_mask2former_labels.py --test-only
# 輸入：datasets/recorded_Town03/rgb/  +  datasets/recorded/rgb/
# 輸出：datasets/nurec_test_labels/test_Town03/
#        datasets/nurec_test_labels/test_Town01/
```

> 備選：使用 CARLA 語意感測器的 GT 標籤（更乾淨，不含 Mask2Former 誤差）：
> ```bash
> conda run -n carla_env python3 prepare_gt_test_label.py
> # 輸入：datasets/recorded_Town03/semantic/
> # 輸出：datasets/training_semantic_v6/test_Town03_gt_label/
> ```

### Step 5 — 組合 Dataset

```bash
conda run -n carla_env python3 prepare_v6_dataset.py
# 輸出：datasets/training_semantic_v6/
#   train_img/         : nurec + nurec_raw + Cityscapes (cs_ 前綴) 的 symlink
#   train_label/       : nurec_labels + cityscapes_m2f_labels 的 symlink
#   test_label/        : symlink → nurec_test_labels/test_Town01
#   test_Town03_label/ : symlink → nurec_test_labels/test_Town03
#   test_Town03_gt_label/ : CARLA GT 標籤（step 4 備選產生）
```

### Step 6 — 訓練

```bash
conda run -n carla_env bash train_semantic_v6.sh
# checkpoint: pix2pixHD/checkpoints/carla2real_semantic_v6/
# 設定: from scratch, lr=0.0002, 200 epochs, batchSize=4, scale_width_and_crop
```

監控：
```bash
tail -f pix2pixHD/checkpoints/carla2real_semantic_v6/loss_log.txt
```

自動接力腳本（等 Mask2Former 結束後接著跑）：
```bash
# 等 NuRec Mask2Former 結束再建 dataset + 訓練
bash after_mask2former.sh

# 等 extract_nurec_mp4.py 結束再跑 Mask2Former
bash wait_and_run_mask2former.sh

# Cityscapes Mask2Former → dataset → 訓練
bash run_cityscapes_then_train.sh
```

### Step 7 — 測試推論

```bash
# Mask2Former 標籤（一般比較用）
conda run -n carla_env bash test_Town03_v6.sh
conda run -n carla_env bash test_Town01_v6.sh
# 結果: pix2pixHD/results/carla2real_semantic_v6/test_Town03_latest/
#        pix2pixHD/results/carla2real_semantic_v6/test_latest/

# GT 標籤（上界評估）
conda run -n carla_env bash test_Town03_v6_gt.sh
conda run -n carla_env bash test_Town03_v6_gt_fullres.sh
# 結果: pix2pixHD/results/carla2real_semantic_v6/test_Town03_gt_latest/
```

---

## 流程三：V7（Fine-tune from V6）

V7 直接使用 v6 的 dataset，以 v6 checkpoint 為起點 fine-tune。

### Step 1 — 確認 V6 Checkpoint 存在

```bash
ls pix2pixHD/checkpoints/carla2real_semantic_v6/
# 需要有 latest_net_G.pth
```

### Step 2 — Fine-tune

```bash
conda run -n carla_env bash train_semantic_v7.sh
# checkpoint: pix2pixHD/checkpoints/carla2real_semantic_v7/
# 設定: load_pretrain=v6, lr=0.00005, lambda_feat=20, batchSize=1
#        scale_width（不裁切，全 1024 寬）, 50 epochs (30 niter + 20 decay)
```

監控：
```bash
tail -f pix2pixHD/checkpoints/carla2real_semantic_v7/loss_log.txt
```

### Step 3 — 測試推論

```bash
# Mask2Former 標籤
conda run -n carla_env bash test_Town03_v7.sh
conda run -n carla_env bash test_Town01_v7.sh
# 結果: pix2pixHD/results/carla2real_semantic_v7/test_Town03_latest/

# GT 標籤（上界）
conda run -n carla_env bash test_Town03_v7_gt.sh
# 結果: pix2pixHD/results/carla2real_semantic_v7/test_Town03_gt_latest/
```

---

## 流程四：Video 推論

### 任意 MP4 → 合成 MP4

```bash
conda run -n carla_env python3 run_video_inference.py \
  --input  datasets/test_mp4/your_video.mp4 \
  --output datasets/test_mp4/your_video_synthesized.mp4 \
  --model  v6
# 流程：MP4 → 抽幀 → Mask2Former → pix2pixHD → 合成 MP4
# 中間檔：datasets/test_mp4/<stem>_work/
```

### CARLA GT 語意 → 合成 MP4

```bash
conda run -n carla_env python3 make_carla_videos.py
# 輸入：datasets/recorded_Town01/ + recorded_Town03/ 的 semantic/
# 輸出：pix2pixHD/results/mp4/Town01_synthesized.mp4
#        pix2pixHD/results/mp4/Town03_synthesized.mp4
```

---

## 結果路徑整理

```
pix2pixHD/
├── checkpoints/
│   ├── carla2real_semantic_v4/       ← v4 weights
│   ├── carla2real_semantic_v6/       ← v6 weights
│   └── carla2real_semantic_v7/       ← v7 weights（訓練中）
└── results/
    ├── carla2real_semantic_v4/
    │   ├── test_Town03_latest/images/
    │   └── test_latest/images/
    ├── carla2real_semantic_v6/
    │   ├── test_Town03_latest/images/
    │   ├── test_latest/images/
    │   └── test_Town03_gt_latest/images/
    ├── carla2real_semantic_v7/
    │   ├── test_Town03_latest/images/
    │   ├── test_latest/images/
    │   └── test_Town03_gt_latest/images/
    └── mp4/
        ├── Town01_synthesized.mp4
        └── Town03_synthesized.mp4
```

---

## 注意事項

**Mask2Former 模型**：`facebook/mask2former-swin-large-cityscapes-semantic`
- 第一次執行會從 HuggingFace 下載（~2 GB），網路有 SSL 問題時需要 `REQUESTS_CA_BUNDLE` 或 `HF_HUB_DISABLE_PROGRESS_BARS`

**pix2pixHD label_nc 目錄規則**
- `label_nc > 0`：dataset 目錄需要 `train_label/`、`train_img/`、`test_label/`（或 `test_<phase>_label/`）
- `--phase test_Town03` → 讀取 `test_Town03_label/`

**V7 vs V6 差異**
- V6：`scale_width_and_crop`（512 高度裁切），batchSize=4
- V7：`scale_width`（不裁切，全 1024×512），batchSize=1，lambda_feat=20

**CARLA 語意圖顏色格式**
- CARLA 存的是 BGR-as-RGB（即 B 和 R channel 對調）
- `prepare_gt_test_label.py` 裡的 `CARLA_BGR_FILE_TO_ID` 對照表已處理此問題

**Cityscapes prefix**
- v6 dataset 中 Cityscapes 圖檔名有 `cs_` 前綴，避免與 NuRec UUID 命名衝突
