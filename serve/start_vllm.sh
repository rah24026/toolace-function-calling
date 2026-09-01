#!/usr/bin/env bash
set -e

MODEL_DIR="${1:-outputs/qwen-toolace/fp8}"
SERVED_NAME="${2:-qwen25-7b-toolace}"
PORT="${PORT:-8000}"

vllm serve "${MODEL_DIR}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --max-num-seqs 32 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-chunked-prefill
