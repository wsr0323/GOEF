#!/usr/bin/env bash

# Run every center combination under $DATA_ROOT/{1..5}-shot.
# Optional arguments select only specific shot counts, for example:
#   bash scripts/benchmark_center_few_shot.sh 1 2

set -Eeuo pipefail
shopt -s nullglob

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT="${DATA_ROOT:-${GOEF_WORKSPACE:+${GOEF_WORKSPACE}/centers}}"
: "${DATA_ROOT:?set DATA_ROOT to the directory containing 1-shot .. 5-shot}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results}"
WORK_ROOT="${RESULT_ROOT}/.center_few_shot_runs"
MODEL_CKPT="${MODEL_CKPT:-facebook/dinov2-with-registers-giant}"

if (( $# == 0 )); then
    SHOTS=(1 2 3 4 5)
else
    SHOTS=("$@")
fi

for shot in "${SHOTS[@]}"; do
    if [[ ! "$shot" =~ ^[1-5]$ ]]; then
        echo "Error: shot must be an integer from 1 to 5; got: $shot" >&2
        exit 2
    fi
done

if [[ ! -f "${PROJECT_ROOT}/main.py" ]]; then
    echo "Error: ${PROJECT_ROOT}/main.py does not exist" >&2
    exit 1
fi

if [[ ! -d "$MODEL_CKPT" ]]; then
    echo "Error: local model does not exist: ${MODEL_CKPT}" >&2
    exit 1
fi

# Allow an already activated environment; otherwise activate the project's env.
if [[ "${CONDA_DEFAULT_ENV:-}" != "subspacead" ]]; then
    CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
    if [[ ! -f "$CONDA_SH" ]]; then
        echo "Error: subspacead is not active and ${CONDA_SH} does not exist" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$CONDA_SH"
    conda activate subspacead
fi

cd "$PROJECT_ROOT"
mkdir -p "$WORK_ROOT"

declare -A DATASET_NAMES=(
    [mvtec]="mvtec_ad"
    [visa]="visa"
)

run_one() {
    local shot="$1"
    local combination_dir="$2"
    local data_key="$3"
    local combination dataset_name data_dir result_dir run_dir result_csv result_log
    local -a unified_csvs

    combination="$(basename "$combination_dir")"
    dataset_name="${DATASET_NAMES[$data_key]}"
    data_dir="${combination_dir}/${data_key}"
    result_dir="${RESULT_ROOT}/${shot}-shot"
    run_dir="${WORK_ROOT}/${shot}-shot/${combination}_${data_key}"
    result_csv="${result_dir}/${combination}_${data_key}.csv"
    result_log="${result_dir}/${combination}_${data_key}.log"

    if [[ ! -d "$data_dir" ]]; then
        echo "Error: dataset directory does not exist: ${data_dir}" >&2
        return 1
    fi

    mkdir -p "$result_dir" "$run_dir"
    echo "[$(date '+%F %T')] Starting ${shot}-shot / ${combination} / ${data_key}" \
        | tee "$result_log"

    python -u main.py \
        --dataset_name "$dataset_name" \
        --dataset_path "$data_dir" \
        --k_shot "$shot" \
        --image_res 518 \
        --layers="-12,-13,-14,-15,-16,-17,-18" \
        --model_ckpt "$MODEL_CKPT" \
        --local_files_only \
        --aug_count 30 \
        --pca_ev 0.99 \
        --agg_method mean \
        --seed 42 \
        --outdir "$run_dir" \
        2>&1 | tee -a "$result_log"

    unified_csvs=("${run_dir}"/*/unified_results.csv)
    if (( ${#unified_csvs[@]} == 0 )); then
        echo "Error: run completed without a unified_results.csv under ${run_dir}" \
            | tee -a "$result_log" >&2
        return 1
    fi
    if (( ${#unified_csvs[@]} > 1 )); then
        {
            echo "Error: multiple unified_results.csv files found under ${run_dir}:"
            printf '  %s\n' "${unified_csvs[@]}"
        } | tee -a "$result_log" >&2
        return 1
    fi
    if [[ ! -s "${unified_csvs[0]}" ]]; then
        echo "Error: unified result is empty: ${unified_csvs[0]}" \
            | tee -a "$result_log" >&2
        return 1
    fi

    cp "${unified_csvs[0]}" "$result_csv"
    echo "[$(date '+%F %T')] Completed: ${result_csv}" | tee -a "$result_log"
}

for shot in "${SHOTS[@]}"; do
    shot_data_root="${DATA_ROOT}/${shot}-shot"
    combination_dirs=("${shot_data_root}"/*)

    if (( ${#combination_dirs[@]} == 0 )); then
        echo "Error: no combination directories found under ${shot_data_root}" >&2
        exit 1
    fi

    for combination_dir in "${combination_dirs[@]}"; do
        [[ -d "$combination_dir" ]] || continue
        run_one "$shot" "$combination_dir" mvtec
        run_one "$shot" "$combination_dir" visa
    done
done

echo "All runs completed. Results are under ${RESULT_ROOT}/{1,2,3,4,5}-shot/"

#cd "$PROJECT_ROOT"
#  bash scripts/benchmark_center_few_shot.sh
#
#  也可以只运行指定 shot：
#
#  bash scripts/benchmark_center_few_shot.sh 1
#  bash scripts/benchmark_center_few_shot.sh 2 3 4 5
