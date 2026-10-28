#!/bin/bash
# Ghost-dog test (VIPLANNER_GHOST=1: ViPlanner + guidance drive a gait-free kinematic base), then plots.
# Runs N episodes, stops Isaac Sim, writes ghost_logs/<name>/episode_NNN.{npz,png} + summary.txt.
# Usage: [GEO_V2=1] [RATES=15,5,10] tools/run_ghost_test.sh <name> [episodes=3]
# Waits first if an isaac-sim container is already running.
NAME="${1:?usage: run_ghost_test.sh <name> [episodes]}"; N="${2:-3}"
TD=/home/itri/Desktop/viplanner2/training_data
OUT=$TD/ghost_logs/$NAME; LOG=/home/itri/isaac-sim-docker/launch_ghost_${NAME}.log
while docker ps -q -f name=^isaac-sim$ | grep -q .; do sleep 60; done
mkdir -p "$OUT"
cd /home/itri/isaac-sim-docker
GHOST=1 GHOST_LOG="$NAME" nohup ./run_stair_usd.sh > "$LOG" 2>&1 &
echo "[ghost] started $(date), log $LOG"
# up to 26000 steps per episode; give each ~25 min before giving up
for i in $(seq $((N * 50))); do
  sleep 30
  [ "$(ls "$OUT"/episode_*.npz 2>/dev/null | wc -l)" -ge "$N" ] && break
  docker ps -q -f name=^isaac-sim$ | grep -q . || { echo "[ghost] Isaac Sim exited early"; break; }
done
docker stop isaac-sim >/dev/null 2>&1
grep -aE "episode [0-9]+: (reached|REACHED|stopped|FELL)|ghost: episode|Traceback|Crash detected" "$LOG" > "$OUT/run_log.txt"
/home/itri/miniconda3/envs/viplanner_train/bin/python $TD/tools/plot_ghost_paths.py "$OUT" > "$OUT/summary.txt" 2>&1
echo "[ghost] done $(date)"; cat "$OUT/summary.txt"
