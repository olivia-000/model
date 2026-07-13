#!/usr/bin/env bash
# 離線全景分割 pipeline：RGB → (mask2former_env) 語義圖 → (viplanner env) 軌跡
#
# 兩個 conda env 之間用檔案系統當介面，不需要手動切換環境。
#
# 用法:
#   ./run_pipeline.sh <rgb圖> <depth.npy> <goal_x> <goal_y> <goal_z> [輸出資料夾]
# 範例:
#   ./run_pipeline.sh ../test_scene2_obstacle/rgb_debug.png \
#                     ../test_scene2_obstacle/depth.npy  13.0 0.0 0.0  ./out_scene2

set -euo pipefail

RGB_IMG=$1
DEPTH=$2
GOAL_X=$3
GOAL_Y=$4
GOAL_Z=$5
OUT_DIR=${6:-./pipeline_out}

# ===== 依你的實際路徑修改這一段 =====
SEG_ENV=mask2former_env                        # mmdetection 環境名
VIP_ENV=viplanner                              # viplanner 環境名
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
M2F_CONFIG="$SCRIPT_DIR/m2f_ckpt/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"
M2F_CKPT=$(ls "$SCRIPT_DIR"/m2f_ckpt/*.pth 2>/dev/null | head -1)
MODEL_DIR="$REPO_ROOT/viplanner_models"       # model.pt + model.yaml
# ===================================

if [ -z "${M2F_CKPT}" ]; then
    echo "[錯誤] 找不到 $SCRIPT_DIR/m2f_ckpt/*.pth"
    echo "先下載: conda run -n $SEG_ENV mim download mmdet --config mask2former_r50_8xb2-lsj-50e_coco-panoptic --dest $SCRIPT_DIR/m2f_ckpt"
    exit 1
fi

mkdir -p "$OUT_DIR"
SEM_PRED="$OUT_DIR/sem_predicted.png"

echo "===== 階段 1+2：全景分割（$SEG_ENV）====="
conda run -n "$SEG_ENV" python "$SCRIPT_DIR/panoptic_inference.py" \
    --input "$RGB_IMG" \
    --output "$SEM_PRED" \
    --config "$M2F_CONFIG" \
    --checkpoint "$M2F_CKPT" \
    --overlay

echo "===== 階段 3~6：ViPlanner 軌跡推論（$VIP_ENV）====="
conda run -n "$VIP_ENV" python "$REPO_ROOT/infer_single.py" \
    --model_dir "$MODEL_DIR" \
    --depth "$DEPTH" \
    --semantic "$SEM_PRED" \
    --goal "$GOAL_X" "$GOAL_Y" "$GOAL_Z" \
    --output "$OUT_DIR/result.png" \
    --save_traj "$OUT_DIR/traj.npy"

echo ""
echo "===== 完成 ====="
echo "  語義圖:   $SEM_PRED（+ 疊圖 _overlay.png）"
echo "  軌跡圖:   $OUT_DIR/result.png"
echo "  軌跡座標: $OUT_DIR/traj.npy"
