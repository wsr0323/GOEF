#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-${GOEF_WORKSPACE:+${GOEF_WORKSPACE}/centers}}"
: "${DATA_ROOT:?set DATA_ROOT to the directory containing 1-shot .. 5-shot}"
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results}"
MASK_ROOT="${MASK_ROOT:-${PROJECT_ROOT}/masks/fewshot}"
PYTHON_BIN="${PYTHON_BIN:-python}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

if (( $# > 0 )); then
    SHOTS=("$@")
else
    SHOTS=(1 2 3 4 5)
fi

for shot in "${SHOTS[@]}"; do
    shot_root="${DATA_ROOT}/${shot}-shot"
    result_dir="${RESULT_ROOT}/${shot}-shot"
    mkdir -p "${result_dir}"

    while IFS= read -r center_root; do
        center_name="$(basename "${center_root}")"
        for dataset in mvtec visa; do
            dataset_root="${center_root}/${dataset}"
            output_csv="${result_dir}/${center_name}_${dataset}.csv"

            if [[ ! -f "${dataset_root}/meta.json" ]]; then
                echo "Missing ${dataset_root}/meta.json" >&2
                exit 1
            fi
            if [[ "${SKIP_EXISTING}" == "1" && -f "${output_csv}" ]]; then
                echo "Skipping existing ${output_csv}"
                continue
            fi

            echo "Running ${shot}-shot ${center_name} ${dataset}"
            "${PYTHON_BIN}" test_univad.py \
                --dataset "${dataset}" \
                --data_path "${dataset_root}" \
                --k_shot "${shot}" \
                --image_size "${IMAGE_SIZE}" \
                --reference_from_meta \
                --mask_path "${MASK_ROOT}/${dataset}" \
                --output_csv "${output_csv}"
        done
    done < <(find "${shot_root}" -mindepth 1 -maxdepth 1 -type d | sort)
done
