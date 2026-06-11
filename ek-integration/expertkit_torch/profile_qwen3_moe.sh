#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/Qwen/Qwen3-30B-A3B}"

INPUT_LEN="${INPUT_LEN:-64}"
OUTPUT_LEN="${OUTPUT_LEN:-8}"

WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-10}"

BS="${BS:-32}"
STEP="${STEP:-2}"

OUTPUT_PROFILE="profiles/expert-kit_bs_${BS}_in_${INPUT_LEN}_out_${OUTPUT_LEN}"


echo "============================================================================================="
echo "Model path: ${MODEL_PATH}"
echo "Input len: ${INPUT_LEN}"
echo "Output len: ${OUTPUT_LEN}"
echo "Warmup steps: ${WARMUP}"
echo "Iterations: ${ITERS}"
echo "Output profile: ${OUTPUT_PROFILE}"
echo "Batch size: ${BS}"

mkdir -p "$(dirname "${OUTPUT_PROFILE}")"
profile_args=(
    nsys profile 
    --trace=cuda,nvtx,osrt 
    # --gpu-metrics-devices=all 
    # --gpu-metrics-frequency=200 
    --force-overwrite=true
    -o "${OUTPUT_PROFILE}"

    uv run python3 -m expertkit_torch.models.qwen3_moe.benchmark
    --batch_size "${BS}" \
    --model_path "${MODEL_PATH}"
    --input_len "${INPUT_LEN}"
    --output_len "${OUTPUT_LEN}"
    --warmup "${WARMUP}"
    --num_iter "${ITERS}"
)
echo "Running latency profiling on [batch size = ${BS}]"

"${profile_args[@]}"

echo "Latency profiling on [batch size = ${BS}] finished"
echo "============================================================================================="
