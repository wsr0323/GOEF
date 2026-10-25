#!/usr/bin/env bash
# Step 4 - per-reference sweeps for the five methods with an affordable mu_g
# (Sec. III-D).  Every normal training image is used as the single 1-shot
# reference; each method writes one CSV row per image.
#
#   $RESULT_ROOT/<detector>/mvtec.csv
#   $RESULT_ROOT/<detector>/visa.csv
#
# All five sweeps resume: already evaluated references are skipped, so the
# script can be interrupted and restarted.
source "$(dirname -- "$0")/env.sh"

THIRD_PARTY=${THIRD_PARTY:-"${GOEF_WORKSPACE}/third_party"}
GPU=${GPU:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
mkdir -p "$RESULT_ROOT"/{patchcore,winclip,inpformer,subspacead,anomalydino}

run() {  # run <logfile> <command...>
  local log=$1; shift
  echo "   $*"
  "$@" 2>&1 | tee -a "$log"
}

# --- PatchCore ---------------------------------------------------------------
if [[ -d "${THIRD_PARTY}/patchcore-inspection" ]] && ! { [[ "$SKIP_EXISTING" == 1 && -s "${RESULT_ROOT}/patchcore/mvtec.csv" ]]; }; then
  cd "${THIRD_PARTY}/patchcore-inspection"
  for dataset in mvtec visa; do
    root=$([[ $dataset == mvtec ]] && echo "$MVTEC_ROOT" || echo "$VISA_ROOT")
    run "${RESULT_ROOT}/patchcore/${dataset}.log" \
      "${GOEF_PYTHON}" bin/run_one_shot_patchcore.py \
        --datasets "$dataset" --gpu "$GPU" \
        --mvtec-root "$MVTEC_ROOT" --visa-root "$VISA_ROOT" \
        --output-csv "${RESULT_ROOT}/patchcore/${dataset}.csv"
  done
fi

# --- WinCLIP -----------------------------------------------------------------
if [[ -d "${THIRD_PARTY}/Accurate-WinCLIP-pytorch" ]] && ! { [[ "$SKIP_EXISTING" == 1 && -s "${RESULT_ROOT}/winclip/mvtec.csv" ]]; }; then
  cd "${THIRD_PARTY}/Accurate-WinCLIP-pytorch"
  CUDA_VISIBLE_DEVICES="$GPU" run "${RESULT_ROOT}/winclip/mvtec.log" \
    "${GOEF_PYTHON}" run_one_category.py --datasets mvtec \
      --mvtec_root "$MVTEC_ROOT" --output_csv "${RESULT_ROOT}/winclip/mvtec.csv"
  CUDA_VISIBLE_DEVICES="$GPU" run "${RESULT_ROOT}/winclip/visa.log" \
    "${GOEF_PYTHON}" run_one_category.py --datasets visa \
      --visa_root "$VISA_ROOT" --output_csv "${RESULT_ROOT}/winclip/visa.csv"
fi

# --- INP-Former (trains one model per reference image) -----------------------
if [[ -d "${THIRD_PARTY}/INP-Former" ]] && ! { [[ "$SKIP_EXISTING" == 1 && -s "${RESULT_ROOT}/inpformer/results.csv" ]]; }; then
  cd "${THIRD_PARTY}/INP-Former"
  run "${RESULT_ROOT}/inpformer/results.log" \
    "${GOEF_PYTHON}" INP_Former_One_Image_Sweep.py \
      --datasets mvtec visa --device "cuda:${GPU}" \
      --mvtec-data-path "$MVTEC_ROOT" --visa-data-path "$VISA_ROOT" \
      --output-csv "${RESULT_ROOT}/inpformer/results.csv"
fi

# --- SubspaceAD --------------------------------------------------------------
if [[ -d "${THIRD_PARTY}/SubspaceAD" ]] && ! { [[ "$SKIP_EXISTING" == 1 && -s "${THIRD_PARTY}/SubspaceAD/few_shot_results_each_train_1shot/mvtec_dinov2G/all_runs_table.csv" ]]; }; then
  cd "${THIRD_PARTY}/SubspaceAD"
  MVTEC_PATH="${MVTEC_ROOT}/" VISA_PATH="${VISA_ROOT}/" \
    bash run_mvtec_each_train_1shot.sh
  MVTEC_PATH="${MVTEC_ROOT}/" VISA_PATH="${VISA_ROOT}/" \
    bash run_visa_each_train_1shot.sh
  cp few_shot_results_each_train_1shot/mvtec_dinov2G/all_runs_table.csv \
     "${RESULT_ROOT}/subspacead/mvtec.csv"
  cp few_shot_results_each_train_1shot/visa_dinov2G/all_runs_table.csv \
     "${RESULT_ROOT}/subspacead/visa.csv"
fi

# --- AnomalyDINO -------------------------------------------------------------
if [[ -d "${THIRD_PARTY}/AnomalyDINO" ]] && ! { [[ "$SKIP_EXISTING" == 1 && -s "${RESULT_ROOT}/anomalydino/summary.csv" ]]; }; then
  cd "${THIRD_PARTY}/AnomalyDINO"
  CUDA_VISIBLE_DEVICES="$GPU" run "${RESULT_ROOT}/anomalydino/mvtec.log" \
    "${GOEF_PYTHON}" run_mvtec_train_1shot_refs.py --data_root "$MVTEC_ROOT" \
      --output_root "${RESULT_ROOT}/anomalydino/mvtec"
  CUDA_VISIBLE_DEVICES="$GPU" run "${RESULT_ROOT}/anomalydino/visa.log" \
    "${GOEF_PYTHON}" run_visa_train_1shot_refs.py --data_root "$VISA_ROOT" \
      --output_root "${RESULT_ROOT}/anomalydino/visa"
  cp "${RESULT_ROOT}/anomalydino/mvtec/summary.csv" "${RESULT_ROOT}/anomalydino/mvtec.csv"
  cp "${RESULT_ROOT}/anomalydino/visa/summary.csv" "${RESULT_ROOT}/anomalydino/visa.csv"
fi

echo "per-reference sweeps finished; run scripts/06_aggregate_validation.sh next"
