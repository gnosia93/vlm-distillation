# CUDA runtime base so the same image works on GPU nodes; falls back to CPU
# (gloo) automatically when no GPU is present. Swap for python:3.10-slim if
# you only ever run on CPU and want a smaller image.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 멀티노드/단일노드 multi-GPU 공용 학습 스크립트 (LOCAL_RANK 기준 통합).
COPY train.py .

# torchrun/env-based launch — Kubeflow injects MASTER_ADDR/PORT, RANK, WORLD_SIZE.
ENTRYPOINT ["python", "train.py"]
