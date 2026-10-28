#!/usr/bin/env bash
# Evaluate every preset center combination under $DATA_ROOT/{1..5}-shot.
# Each preset meta.json supplies the exact normal support images; no random
# few-shot sampling is performed.
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA_ROOT="${DATA_ROOT:-${GOEF_WORKSPACE:+${GOEF_WORKSPACE}/centers}}"
: "${DATA_ROOT:?set DATA_ROOT to the directory containing 1-shot .. 5-shot}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results}"
GPU_ID="${GPU_ID:-0}"
MODEL="${MODEL:-ViT-B-16-plus-240}"
IMAGE_SIZE="${IMAGE_SIZE:-240}"

if (( $# > 0 )); then
    SHOTS=("$@")
else
    SHOTS=(1 2 3 4 5)
fi

for shot in "${SHOTS[@]}"; do
    shot_dir="${DATA_ROOT}/${shot}-shot"
    result_dir="${RESULT_ROOT}/${shot}-shot"
    mkdir -p "${result_dir}"

    while IFS= read -r -d '' preset_dir; do
        preset_name="$(basename "${preset_dir}")"
        # Normalizes centers_01_02 to center_01_02 in the requested CSV name.
        result_name="${preset_name/centers_/center_}"

        for dataset in mvtec visa; do
            data_path="${preset_dir}/${dataset}"
            csv_path="${result_dir}/${result_name}_${dataset}.csv"
            run_dir="${result_dir}/${result_name}_${dataset}"

            [[ -f "${data_path}/meta.json" ]] || {
                echo "Skipping missing preset metadata: ${data_path}/meta.json" >&2
                continue
            }

            CUDA_VISIBLE_DEVICES="${GPU_ID}" python reproduce_WinCLIP.py \
                --dataset "${dataset}" \
                --data_path "${data_path}" \
                --save_path "${run_dir}" \
                --results_csv "${csv_path}" \
                --k_shot "${shot}" \
                --use_preset_train \
                --model "${MODEL}" \
                --image_size "${IMAGE_SIZE}"
        done
    done < <(find "${shot_dir}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done
