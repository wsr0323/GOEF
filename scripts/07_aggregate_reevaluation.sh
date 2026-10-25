#!/usr/bin/env bash
# Step 7 - aggregate the per-center evaluations into Table II (Sec. IV).
#
#   $OUTPUT_ROOT/reevaluation/all_final_weighted_results.csv
#   $OUTPUT_ROOT/reevaluation/method_metric_rankings.csv
#   $OUTPUT_ROOT/reevaluation/metric_rank_frequency.csv
source "$(dirname -- "$0")/env.sh"

THIRD_PARTY=${THIRD_PARTY:-"${GOEF_WORKSPACE}/third_party"}
OUT="${OUTPUT_ROOT}/reevaluation"
mkdir -p "$OUT"

args=()
add() { [[ -d "$2" ]] && args+=(--method "$1=$2"); }
add patchcore         "${THIRD_PARTY}/patchcore-inspection"
add Accurate-WinCLIP  "${THIRD_PARTY}/Accurate-WinCLIP-pytorch"
add PromptAD          "${THIRD_PARTY}/PromptAD"
add AnomalyDINO       "${THIRD_PARTY}/AnomalyDINO"
add FoundAD           "${THIRD_PARTY}/FoundAD"
add INP-Former        "${THIRD_PARTY}/INP-Former"
add UniVAD            "${THIRD_PARTY}/UniVAD"
add SubspaceAD        "${THIRD_PARTY}/SubspaceAD"

if [[ ${#args[@]} -eq 0 ]]; then
  echo "ERROR: no method directory found under ${THIRD_PARTY}" >&2
  exit 1
fi

goef reevaluate "${args[@]}" --sampling_root "$SAMPLING_ROOT" \
  --shots $SHOTS --k "$K" --output "$OUT"
