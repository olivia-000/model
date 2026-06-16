#!/bin/bash
# Wait for extract_nurec_mp4.py to finish, then run Mask2Former
PID=153099
echo "[$(date)] Waiting for extraction PID $PID to finish..."
wait $PID 2>/dev/null || while kill -0 $PID 2>/dev/null; do sleep 10; done
echo "[$(date)] Extraction done. nurec_raw count: $(ls /home/itriu100/carla/datasets/nurec_raw/ | wc -l)"
echo "[$(date)] Starting Mask2Former..."
conda run -n carla_env python3 /home/itriu100/carla/generate_mask2former_labels.py
echo "[$(date)] Mask2Former done."
