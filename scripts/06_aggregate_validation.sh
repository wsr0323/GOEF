#!/usr/bin/env bash
# Step 6 - aggregate the per-reference sweeps into Table I (Sec. III-D).
#
#   $OUTPUT_ROOT/validation/table_validation.csv      eps_bar and r_bar per strategy
#   $OUTPUT_ROOT/validation/validation_details.csv    one row per comparison
source "$(dirname -- "$0")/env.sh"

OUT="${OUTPUT_ROOT}/validation"
mkdir -p "$OUT"
CONFIG="${OUT}/validation_config.yaml"

cat > "$CONFIG" <<YAML
sampling_root: ${SAMPLING_ROOT}
k: ${K}
seeds: [$(echo $SEEDS | tr ' ' ', ')]
datasets:
  MVTec:
    detectors:
      patchcore: ${RESULT_ROOT}/patchcore/mvtec.csv
      winclip: ${RESULT_ROOT}/winclip/mvtec.csv
      anomalydino: ${RESULT_ROOT}/anomalydino/mvtec.csv
      inpformer:
        path: ${RESULT_ROOT}/inpformer/results.csv
        filter: [MVTec, MVTec-AD]
      subspacead: ${RESULT_ROOT}/subspacead/mvtec.csv
  VisA:
    detectors:
      patchcore: ${RESULT_ROOT}/patchcore/visa.csv
      winclip: ${RESULT_ROOT}/winclip/visa.csv
      anomalydino: ${RESULT_ROOT}/anomalydino/visa.csv
      inpformer:
        path: ${RESULT_ROOT}/inpformer/results.csv
        filter: [VisA]
      subspacead: ${RESULT_ROOT}/subspacead/visa.csv
YAML

goef validate --config "$CONFIG" --output "$OUT"
