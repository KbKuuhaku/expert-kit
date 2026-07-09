#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PROFILE="profiles/expert-kit_worker"

echo "============================================================================================="
echo "Output profile: ${OUTPUT_PROFILE}"

mkdir -p "$(dirname "${OUTPUT_PROFILE}")"
profile_args=(
    nsys profile 
    --trace=cuda,nvtx,osrt 
    # --gpu-metrics-devices=all 
    # --gpu-metrics-frequency=200 
    --force-overwrite=true
    -o "${OUTPUT_PROFILE}"
    cargo run --release --bin ek-cli worker
)
"${profile_args[@]}"

echo "============================================================================================="
