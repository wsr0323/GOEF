#!/usr/bin/env bash
# Step 8 - copy the raw tables behind Table I and Table II into results/.
#
#   results/one_shot/<method>/<dataset>.csv     complete per-reference sweeps
#   results/centers_raw/<Method>/<shot>-shot/   raw per-centre evaluations
#   results/MANIFEST.csv                        source, rows and SHA-256
#
# Only files that exist are collected, so the script works when a subset of the
# methods has been run.
source "$(dirname -- "$0")/env.sh"

OUT="${OUTPUT_ROOT}/collected"
mkdir -p "$OUT"
args=(--output "$OUT")

add_one_shot() {  # add_one_shot <method> <dataset> <path>
  [[ -f "$3" ]] && args+=(--one_shot "$1=$2:$3")
}

add_one_shot patchcore MVTec "${VALIDATION_DIR}/patchcore-inspection/results/mvtec_one_shot.csv"
add_one_shot patchcore VisA  "${VALIDATION_DIR}/patchcore-inspection/results/visa_one_shot.csv"
add_one_shot winclip   MVTec "${VALIDATION_DIR}/Accurate-WinCLIP-pytorch/results/mvtec.csv"
add_one_shot winclip   VisA  "${VALIDATION_DIR}/Accurate-WinCLIP-pytorch/results/visa.csv"
add_one_shot anomalydino MVTec "${RESULT_ROOT}/anomalydino/mvtec.csv"
add_one_shot anomalydino VisA  "${RESULT_ROOT}/anomalydino/visa.csv"
add_one_shot subspacead MVTec "${VALIDATION_DIR}/SubspaceAD/few_shot_results_each_train_1shot/mvtec_dinov2G/all_runs_table.csv"
add_one_shot subspacead VisA  "${VALIDATION_DIR}/SubspaceAD/few_shot_results_each_train_1shot/visa_dinov2G/all_runs_table.csv"
[[ -f "${RESULT_ROOT}/inpformer/results.csv" ]] && \
  args+=(--one_shot_split "inpformer=${RESULT_ROOT}/inpformer/results.csv")

add_center() {  # add_center <name> <project dir>
  [[ -d "$2" ]] && args+=(--center "$1=$2")
}
add_center PatchCore    "${REEVALUATION_DIR}/patchcore-inspection"
add_center WinCLIP      "${REEVALUATION_DIR}/Accurate-WinCLIP-pytorch"
add_center AnomalyDINO  "${REEVALUATION_DIR}/AnomalyDINO"
add_center FoundAD      "${REEVALUATION_DIR}/FoundAD"
add_center INP-Former   "${REEVALUATION_DIR}/INPFormer"
add_center PromptAD     "${REEVALUATION_DIR}/PromptAD"
add_center SubspaceAD   "${REEVALUATION_DIR}/SubspaceAD"
add_center UniVAD       "${REEVALUATION_DIR}/UniVAD"

if [[ ${#args[@]} -le 2 ]]; then
  echo "ERROR: no raw result found; run stages 4 and 5 first" >&2
  exit 1
fi

goef collect "${args[@]}"
echo "collected raw tables under ${OUT}"
