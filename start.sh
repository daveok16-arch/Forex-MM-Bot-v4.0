#!/bin/bash
export ONNXRUNTIME_DISABLE_GPU=1
export CUDA_VISIBLE_DEVICES=""
export ORT_DISABLE_GPU=1
echo "[Start] Starting scanner and Flask..."

# Start scanner in background
python3 /opt/render/project/src/bot_runner.py &
SCANNER_PID=$!
echo "[Start] Scanner PID: $SCANNER_PID"

# Start Flask
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2
