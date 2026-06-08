#!/bin/bash
export ONNXRUNTIME_DISABLE_GPU=1
export CUDA_VISIBLE_DEVICES=""
export ORT_DISABLE_GPU=1
echo "[Start] ONNX GPU disabled, starting Gunicorn..."
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2
