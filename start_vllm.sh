#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_service

echo "正在啟動 vLLM 服務器..."
echo "模型: Qwen2-VL-7B-Instruct-AWQ"
echo "地址: http://127.0.0.1:8000"
echo ""

vllm serve ~/models/Qwen2-VL-7B-Instruct-AWQ \
    --host 127.0.0.1 \
    --port 8000 \
    --served-model-name Qwen/Qwen2-VL-7B-Instruct-AWQ \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --dtype auto
