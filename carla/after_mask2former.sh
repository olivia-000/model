#!/bin/bash
# 等 Mask2Former 的 Python 進程結束後，自動建 v6 dataset 並開始訓練
M2F_PID=163431
LOG=/home/itriu100/carla/mask2former.log
BASE=/home/itriu100/carla

echo "[$(date)] Waiting for Mask2Former PID $M2F_PID..."
while kill -0 $M2F_PID 2>/dev/null; do sleep 30; done
echo "[$(date)] Mask2Former done. Labels: $(ls $BASE/datasets/nurec_labels/ | wc -l)"

echo "[$(date)] Building v6 dataset..."
conda run -n carla_env python3 $BASE/prepare_v6_dataset.py
echo "[$(date)] Dataset built."

echo "[$(date)] Starting v6 training..."
conda run -n carla_env bash $BASE/train_semantic_v6.sh
echo "[$(date)] Training finished."
