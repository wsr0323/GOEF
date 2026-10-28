#!/usr/bin/env bash
set -euo pipefail

# VisA 数据集路径（请根据实际位置修改）
VISA_PATH="${VISA_PATH:-${GOEF_DATA:-data}/VisA_pytorch/1cls/}"
MODEL_CKPT="${MODEL_CKPT:-facebook/dinov2-with-registers-giant}"
OUT_ROOT="${OUT_ROOT:-few_shot_results_each_train_1shot/visa_dinov2G}"
CONDA_ENV="${CONDA_ENV:-python310}"

IMAGE_RES="${IMAGE_RES:-512}"
LAYERS="${LAYERS:--12,-13,-14,-15,-16,-17,-18}"
AUG_COUNT="${AUG_COUNT:-30}"
PCA_EV="${PCA_EV:-0.99}"
AGG_METHOD="${AGG_METHOD:-mean}"
VIS_COUNT="${VIS_COUNT:-0}"
SKIP_DONE="${SKIP_DONE:-1}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SUBSPACEAD_HF_LOCAL_FILES_ONLY="${SUBSPACEAD_HF_LOCAL_FILES_ONLY:-1}"

if [[ ! -d "$VISA_PATH" ]]; then
    echo "VisA path does not exist: $VISA_PATH" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

echo "--- Starting per-train-image 1-shot experiments for VisA AD ---"
echo "Dataset: $VISA_PATH"
echo "Output:  $OUT_ROOT"

if [[ "$#" -gt 0 ]]; then
    category_names=("$@")
else
    category_names=()
    for category_dir in "$VISA_PATH"/*; do
        [[ -d "$category_dir" ]] || continue
        [[ "$(basename "$category_dir")" == "split_csv" ]] && continue
        category_names+=("$(basename "$category_dir")")
    done
fi

total_runs=0
completed_runs=0
skipped_runs=0

for category in "${category_names[@]}"; do
    train_dir="$VISA_PATH/$category/train/good"
    if [[ ! -d "$train_dir" ]]; then
        echo "--- Skipping $category: missing train/good directory ---"
        continue
    fi

    refs=()
    while IFS= read -r -d '' ref_path; do
        refs+=("$ref_path")
    done < <(find "$train_dir" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | sort -z)

    if [[ "${#refs[@]}" -eq 0 ]]; then
        echo "--- Skipping $category: no train/good images found ---"
        continue
    fi

    echo "--- Category: $category (${#refs[@]} reference images) ---"
    ref_idx=0
    for ref_path in "${refs[@]}"; do
        ref_idx=$((ref_idx + 1))
        total_runs=$((total_runs + 1))

        ref_file="$(basename "$ref_path")"
        ref_stem="${ref_file%.*}"
        run_id="$(printf "%03d_%s" "$ref_idx" "$ref_stem")"
        outdir="$OUT_ROOT/$category/$run_id"

        if [[ "$SKIP_DONE" == "1" && -d "$outdir" ]] \
            && find "$outdir" -name benchmark_results.csv -print -quit | grep -q .; then
            echo "--- Skipping done run: $category/$run_id ---"
            skipped_runs=$((skipped_runs + 1))
            continue
        fi

        echo "--- Running $category ref $ref_idx/${#refs[@]}: $ref_file ---"
        conda run -n "$CONDA_ENV" python -u main.py \
            --dataset_name visa \
            --dataset_path "$VISA_PATH" \
            --categories "$category" \
            --image_res "$IMAGE_RES" \
            --k_shot 1 \
            --k_shot_ref "$ref_path" \
            --layers="$LAYERS" \
            --model_ckpt "$MODEL_CKPT" \
            --aug_count "$AUG_COUNT" \
            --pca_ev "$PCA_EV" \
            --agg_method "$AGG_METHOD" \
            --vis_count "$VIS_COUNT" \
            --outdir "$outdir"

        completed_runs=$((completed_runs + 1))
    done
done

echo "--- All experiments complete ---"
echo "Total planned runs: $total_runs"
echo "Completed runs:     $completed_runs"
echo "Skipped runs:       $skipped_runs"
echo
echo "--- Merging per-run results into a single table ---"
if [[ -f "merge_shot_results.py" ]]; then
    conda run -n "$CONDA_ENV" python merge_shot_results.py --root "$OUT_ROOT" || true
else
    echo "WARNING: merge_shot_results.py not found in $(pwd)" >&2
fi
#./run_visa_each_train_1shot.sh
