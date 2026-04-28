#!/bin/bash

# 1. 載入 Conda 環境路徑 (請確認 miniconda3 的實際路徑)
source ~/miniconda3/etc/profile.d/conda.sh

# 2. 激活指定的環境
conda activate multi-task_detection

# 3. 進入程式目錄並執行
cd /home/itriu100/Firetruck_Project-AC_road_Inference_ROS/firetruck_detection
python main.py
