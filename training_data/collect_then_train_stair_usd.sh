#!/bin/bash
# Waits for the running stair_usd_scene data-collection run to finish (Isaac Sim stays open after
# run_walk's collection completes, so this stops the container itself once the log says so), drops
# fallen-out-of-scene samples from every batch, then fine-tunes (train_stair_usd.py) on all batches.
# Usage: COLLECT_LOG=<isaac log> [TRAIN_FILE_NAME=...] [TRAIN_ENVS=a,b,c] ./collect_then_train_stair_usd.sh
NAME="${TRAIN_FILE_NAME:-stair_usd_railing_v5}"
ENVS="${TRAIN_ENVS:-stair_usd_scene,stair_usd_scene_b}"
PY=/home/itri/miniconda3/envs/viplanner_train/bin/python
cd /home/itri/Desktop/viplanner2/training_data
while docker ps -q -f name=^isaac-sim$ | grep -q .; do
  if [ -n "$COLLECT_LOG" ] && grep -q "DATA COLLECTION complete" "$COLLECT_LOG"; then
    docker stop isaac-sim >/dev/null
  fi
  sleep 60
done
echo "[chain] collection finished $(date)"
for d in ${ENVS//,/ }; do $PY filter_fallen_samples.py "$d"; done
TRAIN_FILE_NAME="$NAME" TRAIN_ENVS="$ENVS" $PY train_stair_usd.py > logs/train_stair_usd_${NAME}.log 2>&1
echo "[chain] training exited with $? at $(date)"
