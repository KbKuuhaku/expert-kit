#!/usr/bin/env bash
set -euo pipefail

SYSTEM="Expert-Kit"
MODE="${MODE:-expert-kit-baseline}"
RUN_DATE="$(date +%Y%m%d)"
RUN_ID="${RUN_ID:-baseline_${RUN_DATE}}"
MODEL_PATH="${MODEL_PATH:-/data/Qwen/Qwen3-30B-A3B}"

INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"

WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-30}"

MIN_BS="${MIN_BS:-1}"
MAX_BS="${MAX_BS:-1024}"
STEP="${STEP:-2}"


echo "============================================================================================="
echo "Run ID: ${RUN_ID}"
echo "Model path: ${MODEL_PATH}"
echo "Input len: ${INPUT_LEN}"
echo "Output len: ${OUTPUT_LEN}"
echo "Warmup steps: ${WARMUP}"
echo "Iterations: ${ITERS}"
echo "Mode: ${MODE}"
echo "Batch size: ${MIN_BS} -> ${MAX_BS}"

common_args=(
    uv run python3 -m expertkit_torch.models.qwen3_moe.benchmark
    --model_path "${MODEL_PATH}"
    --input_len "${INPUT_LEN}"
    --output_len "${OUTPUT_LEN}"
    --warmup "${WARMUP}"
    --num_iter "${ITERS}"
)

bs=${MIN_BS}
while [ $bs -le $MAX_BS ]; do
	echo "Running latency benchmark on [batch size = ${bs}]"

    output_json="logs/${SYSTEM}/${RUN_ID}/${MODE}/bs_${bs}_in_${INPUT_LEN}_out_${OUTPUT_LEN}.json"
    mkdir -p "$(dirname "${output_json}")"

    "${common_args[@]}" \
        --batch_size "${bs}" \
        --output_json "${output_json}"

	echo "Latency benchmark on [batch size = ${bs}] finished"
	
	bs=$((bs * STEP))
done

echo "============================================================================================="
