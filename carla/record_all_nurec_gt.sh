#!/bin/bash
# 批次錄製全部 13 個 NuRec 場景（已有 .usdz 且對應現有訓練資料）
# 輸出：datasets/nurec_gt/train_img/ + train_label/
# Usage: conda run -n carla_env bash record_all_nurec_gt.sh
# 前提：CARLA server 已在 localhost:2000 啟動，NuRec Docker 已啟動

USDZ_ROOT="/home/itriu100/PhysicalAI-Autonomous-Vehicles-NuRec/sample_set/26.02_release"
OUTPUT="/home/itriu100/carla/datasets/nurec_gt"
SCRIPT="/home/itriu100/carla/record_nurec_with_gt_label.py"
NUREC_DIR="/home/itriu100/carla/simulator/CARLA_0.9.16/PythonAPI/examples/nvidia/nurec"
MAX_FRAMES=600

export PYTHONPATH="$NUREC_DIR:$PYTHONPATH"

# 13 scenes matching our existing nurec dataset
SCENES=(
    "01d503d4-449b-46fc-8d78-9085e70d3554"
    "0245ff75-aa3f-46b7-ba87-16a7afb841af"
    "026d6a39-bd8f-4175-bc61-fe50ed0403a3"
    "04749bb9-9b37-495b-bed0-77f0e33ac7da"
    "05bb8212-63e1-40a8-b4fc-3142c0e94646"
    "05ecbbf5-d88c-41b7-97f2-f4ac2615a8c9"
    "05f35348-9de0-4f68-ac65-f31316dbb59e"
    "060131e7-ee72-477c-89da-083e0b446566"
    "065dcac9-ee67-4434-a835-c6b816c88e48"
    "07981e6a-22dd-4796-ad2f-1252037ecd28"
    "07ab4a5b-e934-4fff-b5d1-47efedadac47"
    "09a95ffa-37b7-486d-9404-ac8cf1c3e045"
    "0a18c5a4-9aca-4efd-b604-c75f3269c502"
)

echo "Starting NuRec GT recording for ${#SCENES[@]} scenes..."
echo "Output: $OUTPUT"
echo ""

for scene in "${SCENES[@]}"; do
    USDZ="$USDZ_ROOT/$scene/$scene.usdz"
    if [ ! -f "$USDZ" ]; then
        echo "[SKIP] $scene: .usdz not found"
        continue
    fi
    echo "========================================"
    echo "[SCENE] $scene"
    echo "========================================"
    python3 "$SCRIPT" \
        --host 127.0.0.1 \
        --usdz "$USDZ" \
        --output "$OUTPUT" \
        --max-frames "$MAX_FRAMES" \
        --fps 10
    echo ""
done

echo "All scenes done."
echo "Total images: $(ls $OUTPUT/train_img/*.png 2>/dev/null | wc -l)"
