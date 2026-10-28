#!/usr/bin/env bash
# Step 5 - evaluate all eight methods on the GOEF center datasets (Sec. IV).
#
# For every shot and every center dataset the corresponding driver writes
#   <project>/results/<shot>-shot/<center>_<dataset>.csv
#
# Pass a subset of shots to redo a single setting, e.g.
#   ./scripts/05_run_center_shots.sh 1
# Use ONLY=<method> to run a single method.
source "$(dirname -- "$0")/env.sh"

GPU=${GPU:-0}
ONLY=${ONLY:-""}
TARGET_SHOTS=${*:-$SHOTS}
export DATA_ROOT="${CENTER_ROOT}"

should_run() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

if should_run patchcore && [[ -d "${REEVALUATION_DIR}/patchcore-inspection" ]]; then
  echo "== PatchCore"; cd "${REEVALUATION_DIR}/patchcore-inspection"
  GPU="$GPU" SAVE_MODELS=0 ./train_all_shots.sh $TARGET_SHOTS
fi

if should_run winclip && [[ -d "${REEVALUATION_DIR}/Accurate-WinCLIP-pytorch" ]]; then
  echo "== WinCLIP"; cd "${REEVALUATION_DIR}/Accurate-WinCLIP-pytorch"
  DATA_ROOT="$DATA_ROOT" GPU_ID="$GPU" ./run_center_shots.sh $TARGET_SHOTS
fi

if should_run anomalydino && [[ -d "${REEVALUATION_DIR}/AnomalyDINO" ]]; then
  echo "== AnomalyDINO"; cd "${REEVALUATION_DIR}/AnomalyDINO"
  DATA_ROOT="$DATA_ROOT" SHOTS="$TARGET_SHOTS" SKIP_EXISTING=1 ./train_all_few_shot.sh
fi

if should_run inpformer && [[ -d "${REEVALUATION_DIR}/INPFormer" ]]; then
  echo "== INP-Former"; cd "${REEVALUATION_DIR}/INPFormer"
  DATA_ROOT="$DATA_ROOT" SHOTS="$TARGET_SHOTS" SKIP_EXISTING=1 ./train_all_few_shot.sh
fi

if should_run subspacead && [[ -d "${REEVALUATION_DIR}/SubspaceAD" ]]; then
  echo "== SubspaceAD"; cd "${REEVALUATION_DIR}/SubspaceAD"
  DATA_ROOT="$DATA_ROOT" bash scripts/benchmark_center_few_shot.sh $TARGET_SHOTS
fi

if should_run promptad && [[ -d "${REEVALUATION_DIR}/PromptAD" ]]; then
  echo "== PromptAD"; cd "${REEVALUATION_DIR}/PromptAD"
  "${GOEF_PYTHON}" run_center_shot_unified.py --data-base "$DATA_ROOT" \
    --shots $TARGET_SHOTS --gpu-id "$GPU"
fi

if should_run foundad && [[ -d "${REEVALUATION_DIR}/FoundAD" ]]; then
  echo "== FoundAD"; cd "${REEVALUATION_DIR}/FoundAD"
  "${GOEF_PYTHON}" batch_few_shot.py --data-root "$DATA_ROOT" --shots $TARGET_SHOTS \
    --device "cuda:${GPU}"
fi

if should_run univad && [[ -d "${REEVALUATION_DIR}/UniVAD" ]]; then
  echo "== UniVAD"; cd "${REEVALUATION_DIR}/UniVAD"
  # Component masks only depend on the 5-shot super-center (it contains every
  # reference used at lower shots), so they are prepared once.
  if [[ ! -d masks/fewshot ]]; then
    "${GOEF_PYTHON}" prepare_fewshot_masks.py \
      --source_root "${DATA_ROOT}/5-shot/centers_01_02_03_04_05" \
      --mask_root masks/fewshot --skip_existing
  fi
  DATA_ROOT="$DATA_ROOT" SKIP_EXISTING=1 ./run_fewshot_centers.sh $TARGET_SHOTS
fi

echo "center evaluations finished; run scripts/07_aggregate_reevaluation.sh next"
