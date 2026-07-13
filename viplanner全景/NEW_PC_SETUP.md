# 新電腦交接檔：把「全景分割版 ViPlanner」重新跑起來

> **給 Claude Code 的指示**：這份文件是舊電腦（RTX 5090, driver 595, Ubuntu + miniconda）
> 交接過來的完整安裝與驗證手冊。請照第 0 節先盤點現況，缺什麼補什麼，依 GPU 型號
> 走 A 或 B 分支建環境，最後跑完第 5 節的三個驗證關卡才算完成。每一關卡都有
> 預期輸出，不符就停下來排查（第 7 節有常見錯誤對照）。

## 這個專案在做什麼（30 秒版）

ViPlanner 是一個學習式局部路徑規劃器：吃 **深度圖（公尺）＋ 34 類色碼語義圖 ＋ 目標點**，
吐 **路徑關鍵點 ＋ fear 碰撞風險分數**。「全景分割路線」= 語義圖不用模擬器 ground truth，
改由 **mmdetection 的 Mask2Former（COCO-panoptic）** 從 RGB 即時產生。兩個推論行程
（分割、規劃）刻意放在**兩個不同 conda env**，離線用檔案交接、上機器狗用 ROS topic 交接。
細節見同資料夾 `README.md` 與 repo 根目錄 `ARCHITECTURE.md`。

---

## 第 0 節：先盤點——這台新電腦上應該已經有什麼

以下檔案應該已從舊電腦 rsync 過來。**逐項確認，缺任何一項先停下來向使用者要**
（除了 `m2f_ckpt/` 可自行下載）：

```bash
cd ~/viplanner   # 或實際放置的位置，以下皆以此為準
ls viplanner/config/viplanner_sem_meta.py   # ① repo 本體（官方程式碼）
ls infer_single.py                           # ② 獨立推論腳本（非官方、未進 git，全景 pipeline 依賴它）
ls viplanner_models/model.pt viplanner_models/model.yaml  # ③ 訓練好的模型（272MB）
ls viplanner全景/panoptic_inference.py viplanner全景/*.yml # ④ 全景分割腳本 + 環境 yml
ls test_scene2_obstacle/depth.npy test_scene2_obstacle/sem_viplanner.png test_scene2_obstacle/rgb_debug.png
                                             # ⑤ 離線驗證資料（有 GT，不用裝 CARLA）
ls viplanner全景/m2f_ckpt/*.pth 2>/dev/null || echo "m2f_ckpt 未帶過來 → 第 3 節自行下載"
```

系統前提：

```bash
nvidia-smi            # 有輸出＝驅動 OK；記下 GPU 型號（決定第 2 節走哪個分支）與驅動版本
which conda || echo "缺 miniconda → 先裝 https://docs.conda.io/en/latest/miniconda.html"
```

驅動需求：viplanner env 的 torch 是 cu128 → **驅動 ≥ 570**。不足就先升級驅動再繼續。

---

## 第 1 節：建 viplanner 環境（兩種 GPU 都一樣）

```bash
cd ~/viplanner
conda env create -f viplanner全景/viplanner_env.yml
conda activate viplanner
pip install -e .[standard]        # yml 刻意不含本地套件 viplanner==0.1.0，必須手動裝
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 預期: 2.11.0+cu128 True <GPU名>
```

> torch 2.11+cu128 內含 sm_120 kernel，**RTX 50 系列在這個 env 沒有問題**，問題只出在下面的分割環境。

---

## 第 2 節：建 mask2former_env —— 依 GPU 型號分支

先判斷：

```bash
nvidia-smi --query-gpu=name --format=csv,noheader
```

- RTX **40 系列以前**（4090/4080/3090/A6000…，sm ≤ 8.9）→ 走 **分支 A**
- RTX **50 系列**（5090/5080/5070…，Blackwell，sm_120）→ 走 **分支 B**

### 分支 A：RTX 40 系列以前（簡單路線，直接用 yml）

```bash
conda env create -f viplanner全景/mask2former_env.yml
conda activate mask2former_env
python -c "import mmdet; from mmdet.evaluation import INSTANCE_OFFSET; print(mmdet.__version__, INSTANCE_OFFSET)"
# 預期: 3.3.0 1000
python -c "import torch; print(torch.cuda.is_available())"   # 預期: True
```

完成後跳到第 3 節。

### 分支 B：RTX 50 系列（cu121 的 torch 沒有 sm_120 kernel，yml 不能直接用）

**問題本質**：`mask2former_env.yml` pin 的 torch 2.1.2+cu121 最高只編到 sm_90，
在 50 系列上 GPU 推論會報 `no kernel image is available`。而 torch 換成 2.7+/cu128 之後，
openmmlab **沒有**對應的 mmcv 預編譯 wheel，mmcv 必須從原始碼編譯（需要 nvcc 12.8）。
另外 mmdet 3.3.0 的版本檢查上限是 `mmcv<2.2.0`，但舊 mmcv 2.1.0 的原始碼編不過新 torch，
所以要用 **mmcv 2.2.0 原始碼編譯 + 放寬 mmdet 的版本上限檢查**（社群標準解法）。

```bash
# B-1 建環境骨架（不用 yml）
conda create -n mask2former_env python=3.10 -y
conda activate mask2former_env

# B-2 torch cu128（含 sm_120）
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# B-3 CUDA 12.8 編譯工具鏈（nvcc，只進這個 env，不動系統）
conda install -c nvidia cuda-toolkit=12.8 -y
export CUDA_HOME=$CONDA_PREFIX
nvcc --version    # 預期: release 12.8

# B-4 mmcv 2.2.0 從原始碼編譯（10~20 分鐘；TORCH_CUDA_ARCH_LIST 指定 Blackwell）
pip install -U openmim mmengine "numpy<2" opencv-python matplotlib rospkg catkin-pkg pyyaml
TORCH_CUDA_ARCH_LIST="12.0" MMCV_WITH_OPS=1 FORCE_CUDA=1 \
    pip install mmcv==2.2.0 --no-binary mmcv --no-cache-dir -v

# B-5 mmdet 3.3.0 + 放寬它對 mmcv 的版本上限（3.3.0 寫死 <2.2.0，只差一個 patch 版本）
pip install mmdet==3.3.0
python - <<'EOF'
import mmdet, re, pathlib
init = pathlib.Path(mmdet.__file__).parent / "__init__.py"
text = init.read_text()
new = re.sub(r"mmcv_maximum_version = '[\d.]+'", "mmcv_maximum_version = '2.3.0'", text)
init.write_text(new)
print("已放寬 mmdet 的 mmcv 版本上限 → 2.3.0")
EOF

# B-6 驗證（三行都要過）
python -c "import mmcv; from mmcv.ops import MultiScaleDeformableAttention; print('mmcv ops OK', mmcv.__version__)"
python -c "import mmdet; from mmdet.evaluation import INSTANCE_OFFSET; print(mmdet.__version__, INSTANCE_OFFSET)"
python -c "import torch; x=torch.randn(8,device='cuda'); print('GPU kernel OK', (x*2).sum().item()!=0)"
```

> B-4 編譯若失敗，最常見原因是 gcc 太新/太舊——`conda install -c conda-forge gxx_linux-64=12 -y`
> 後重跑；仍失敗見第 7 節。**最後退路**：完全跳過 GPU，分割用 CPU 跑
> （`panoptic_inference.py --device cpu`，一張數秒），離線驗證夠用，即時上狗不行。

---

## 第 3 節：下載 Mask2Former 權重（若 m2f_ckpt/ 沒帶過來）

```bash
conda activate mask2former_env
cd ~/viplanner/viplanner全景
mim download mmdet --config mask2former_r50_8xb2-lsj-50e_coco-panoptic --dest ./m2f_ckpt
ls m2f_ckpt/   # 預期: 一個 .py config + 一個 .pth checkpoint（~300MB）
```

---

## 第 4 節：核對 run_pipeline.sh 頂部路徑

`viplanner全景/run_pipeline.sh` 的「依你的實際路徑修改這一段」區塊：
`SEG_ENV` / `VIP_ENV` 是否等於實際 env 名、`MODEL_DIR` 是否指向 `viplanner_models/`、
`M2F_CONFIG` 檔名是否和 `m2f_ckpt/` 裡實際下載到的 config 一致。

---

## 第 5 節：三個驗證關卡（依序，全過才算完成）

### 關卡 1：全景分割單獨跑通

```bash
conda activate mask2former_env
cd ~/viplanner/viplanner全景
python panoptic_inference.py \
    --input  ../test_scene2_obstacle/rgb_debug.png \
    --output /tmp/sem_predicted.png \
    --config     ./m2f_ckpt/<實際的>.py \
    --checkpoint ./m2f_ckpt/<實際的>.pth \
    --overlay
```
預期：印出「映射 …成功映射 N 類」、輸出 `/tmp/sem_predicted.png` 與 `_overlay.png`。
打開疊圖確認：路面/建築/車輛的顏色分區大致合理（顏色本身是 ViPlanner 色碼，不是 COCO 原色）。

### 關卡 2：ViPlanner 推論跑通（先用 GT 語義圖，隔離變因）

```bash
conda activate viplanner
cd ~/viplanner
python infer_single.py --model_dir ./viplanner_models \
    --depth ./test_scene2_obstacle/depth.npy \
    --semantic ./test_scene2_obstacle/sem_viplanner.png \
    --goal 13.0 0.0 0.0 --output /tmp/r_gt.png
```
預期：印出 `sem=True, rgb=False, knodes=5, in_channel=16, img_input_size=[360, 640]`、
軌跡 shape `[1, 51, 3]`、**fear ≈ 0.287**（舊電腦同資料的基準值，允許 ±0.02 浮動）。

### 關卡 3：整條 pipeline 串接 + 與 GT 比對

```bash
cd ~/viplanner/viplanner全景
./run_pipeline.sh ../test_scene2_obstacle/rgb_debug.png \
                  ../test_scene2_obstacle/depth.npy 13.0 0.0 0.0 ./out_test

conda run -n viplanner python compare_semantics.py \
    --gt ../test_scene2_obstacle/sem_viplanner.png \
    --pred ./out_test/sem_predicted.png \
    --output ./out_test/compare.png
```
預期：pipeline 產出 `out_test/{sem_predicted.png, result.png, traj.npy}`；
比對的**像素類別一致率大約 60~85%**（模型預測 vs 模擬器 GT 本來就有差，太低如 <40%
通常是色碼映射壞掉，見第 7 節）；`result.png` 軌跡走向應與關卡 2 的 `/tmp/r_gt.png` 相近。

**三關全過 = 離線全景 pipeline 完整復活。** 機器狗即時部署（三個 ROS 節點）另見
`README.md` 第 3 部分，前提是這三關先過。

---

## 第 6 節：（選配）機器狗部署額外需求

- ROS Noetic（`sudo apt install ros-noetic-desktop`），兩個 conda env 內 `pip install rospkg catkin-pkg pyyaml`（分支 A/B 都已含）。
- RealSense 驅動 `ros-noetic-realsense2-camera`，啟動時必加 `align_depth:=true`。
- 狗的驅動需提供 `/odom`、接受 `/cmd_vel`。
- 啟動腳本 `dog/run_dog.sh`，頂部的相機安裝參數（offset/pitch）**必須實際量測後修改**。

---

## 第 7 節：常見錯誤對照表

| 症狀 | 原因 | 解法 |
|---|---|---|
| `no kernel image is available for execution on the device` | 50 系列跑到 cu121 的 torch（走錯分支） | 走第 2 節分支 B |
| `ImportError: cannot import name 'INSTANCE_OFFSET' from mmdet.evaluation` | 裝到 mmdet 2.x | `pip install mmdet==3.3.0` |
| `MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.2.0` | mmdet 版本上限檢查 | 執行分支 B-5 的 patch 腳本 |
| mmcv 編譯中途 `nvcc fatal / unsupported gpu architecture 'compute_120'` | nvcc 不是 12.8 | `conda install -c nvidia cuda-toolkit=12.8`，確認 `export CUDA_HOME=$CONDA_PREFIX` 後重編 |
| mmcv 編譯 C++ 錯誤一大串 | gcc 版本不合 | `conda install -c conda-forge gxx_linux-64=12` 重編 |
| `pip install -e .[standard]` 抱怨 CUDA_HOME | viplanner env 找不到 CUDA toolkit | `export CUDA_HOME=$CONDA_PREFIX`（或跳過，推論其實用不到編譯步驟） |
| infer_single.py 匯入時 wandb / pkg_resources 炸 | numpy 2 vs 舊 wandb | infer_single.py 開頭已有 alias 補丁；若仍炸，`pip install "setuptools<81"` |
| 關卡 3 一致率 < 40%、疊圖顏色亂 | COCO→ViPlanner 映射表跟 checkpoint 類別順序不符 | 確認用的是 `mim download` 抓的 COCO-panoptic 版 config+權重（不是 instance/semantic 版）；換過 checkpoint 就重跑關卡 1 肉眼驗證 |
| `mim download` 找不到 config | mmdet 版本的 config 命名不同 | `mim search mmdet --model mask2former` 找出實際名稱（要 **coco-panoptic** 結尾的） |
| ROS 節點 `ModuleNotFoundError: rospy` | 沒 source ROS | 先 `source /opt/ros/noetic/setup.bash` 再 `conda run ...` |

---

## 附錄：舊電腦環境快照（比對基準）

- 舊機：RTX 5090（Blackwell, sm_120）、驅動 595.71.05、Ubuntu、miniconda。
- `viplanner` env：python 3.10.20、torch 2.11.0+cu128、numpy 2.2.6、open3d 0.17、pypose 0.9.5（完整清單=`viplanner_env.yml`）。
- 舊機的 mask2former_env **從未建立**、m2f_ckpt **從未下載**——全景分割這半邊在舊機也還沒實跑過，
  所以第 5 節關卡 1/3 沒有舊機基準值；只有關卡 2（fear≈0.287）是舊機實測過的數字。
- 舊機是 5090 → 若回到舊機建 mask2former_env，同樣要走分支 B。
