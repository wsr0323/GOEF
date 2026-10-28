#!/usr/bin/env bash
# Step 7 - aggregate the per-center evaluations into Table II (Sec. IV).
#
#   $OUTPUT_ROOT/reevaluation/all_final_weighted_results.csv
#   $OUTPUT_ROOT/reevaluation/method_metric_rankings.csv
#   $OUTPUT_ROOT/reevaluation/metric_rank_frequency.csv
source "$(dirname -- "$0")/env.sh"

OUT="${OUTPUT_ROOT}/reevaluation"
mkdir -p "$OUT"

args=()
add() { [[ -d "$2" ]] && args+=(--method "$1=$2"); }
add patchcore         "${REEVALUATION_DIR}/patchcore-inspection"
add Accurate-WinCLIP  "${REEVALUATION_DIR}/Accurate-WinCLIP-pytorch"
add PromptAD          "${REEVALUATION_DIR}/PromptAD"
add AnomalyDINO       "${REEVALUATION_DIR}/AnomalyDINO"
add FoundAD           "${REEVALUATION_DIR}/FoundAD"
add INP-Former        "${REEVALUATION_DIR}/INPFormer"
add UniVAD            "${REEVALUATION_DIR}/UniVAD"
add SubspaceAD        "${REEVALUATION_DIR}/SubspaceAD"

if [[ ${#args[@]} -eq 0 ]]; then
  echo "ERROR: no method directory found under ${THIRD_PARTY}" >&2
  exit 1
fi

goef reevaluate "${args[@]}" --sampling_root "$SAMPLING_ROOT" \
  --shots $SHOTS --k "$K" --output "$OUT"
