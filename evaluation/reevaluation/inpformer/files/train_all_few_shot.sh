#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${GOEF_WORKSPACE:+${GOEF_WORKSPACE}/centers}}"
: "${DATA_ROOT:?set DATA_ROOT to the directory containing 1-shot .. 5-shot}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SHOTS="${SHOTS:-1 2 3 4 5}"
DATASETS="${DATASETS:-mvtec visa}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

dataset_argument() {
    case "$1" in
        mvtec) printf '%s\n' 'MVTec-AD' ;;
        visa) printf '%s\n' 'VisA' ;;
        *)
            printf 'Unsupported dataset: %s (use mvtec and/or visa)\n' "$1" >&2
            return 1
            ;;
    esac
}

run_training() {
    local shot="$1"
    local combination_dir="$2"
    local dataset_key="$3"
    local dataset_name run_name output_csv
    shift 3

    dataset_name="$(dataset_argument "$dataset_key")"
    run_name="$(basename "$combination_dir")_${dataset_key}"
    output_csv="${RESULTS_ROOT}/${shot}-shot/${run_name}.csv"

    if [[ "$SKIP_EXISTING" == "1" && -s "$output_csv" ]]; then
        printf '[skip] %s already exists\n' "$output_csv"
        return
    fi

    local -a command=("$PYTHON_BIN" "${PROJECT_DIR}/INP_Former_Few_Shot.py")
    command+=("$@")
    command+=(
        --dataset "$dataset_name"
        --data_path "${combination_dir}/${dataset_key}"
        --shot "$shot"
        --phase train
        --save_dir "${RESULTS_ROOT}/${shot}-shot/runs"
        --save_name "$run_name"
        --result_csv "$output_csv"
    )

    printf '[train] %s-shot %s %s\n' "$shot" "$(basename "$combination_dir")" "$dataset_key"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  '
        printf '%q ' "${command[@]}"
        printf '\n  CSV -> %s\n' "$output_csv"
        return
    fi

    "${command[@]}"

    if [[ ! -s "$output_csv" ]]; then
        printf 'Training finished but result CSV was not created: %s\n' "$output_csv" >&2
        return 1
    fi
    printf '[done] %s\n' "$output_csv"
}

main() {
    local shot dataset_key shot_dir combination_dir

    cd "$PROJECT_DIR"
    for shot in $SHOTS; do
        if [[ ! "$shot" =~ ^[1-5]$ ]]; then
            printf 'Invalid shot value: %s (expected 1 through 5)\n' "$shot" >&2
            return 1
        fi

        shot_dir="${DATA_ROOT}/${shot}-shot"
        if [[ ! -d "$shot_dir" ]]; then
            printf 'Shot directory does not exist: %s\n' "$shot_dir" >&2
            return 1
        fi
        mkdir -p "${RESULTS_ROOT}/${shot}-shot/runs"

        while IFS= read -r -d '' combination_dir; do
            for dataset_key in $DATASETS; do
                if [[ ! -d "${combination_dir}/${dataset_key}" ]]; then
                    printf 'Dataset directory does not exist: %s\n' "${combination_dir}/${dataset_key}" >&2
                    return 1
                fi
                run_training "$shot" "$combination_dir" "$dataset_key" "$@"
            done
        done < <(find "$shot_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
    done
}

main "$@"
