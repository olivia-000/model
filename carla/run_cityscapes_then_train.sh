#!/bin/bash
LOG=/home/itriu100/carla/cityscapes_m2f.log
BASE=/home/itriu100/carla

echo "[$(date)] Running Mask2Former on Cityscapes..."
conda run -n carla_env python3 $BASE/generate_cityscapes_labels.py
echo "[$(date)] Cityscapes labels done: $(ls $BASE/datasets/cityscapes_m2f_labels/ | wc -l)"

echo "[$(date)] Rebuilding v6 dataset (NuRec + Cityscapes)..."
conda run -n carla_env python3 $BASE/prepare_v6_dataset.py
echo "[$(date)] Dataset built."

echo "[$(date)] Starting v6 training..."
conda run -n carla_env bash $BASE/train_semantic_v6.sh
echo "[$(date)] Training done."
