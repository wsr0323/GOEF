#!/usr/bin/env bash
# Step 1 - select GOEF representative images (Sec. III-C).
#
# Produces, for every strategy, dataset, and seed:
#   $SAMPLING_ROOT/<method>/<Dataset>_k5_seed<seed>/summary.csv
#   $SAMPLING_ROOT/<method>/<Dataset>_k5_seed<seed>/<category>/representative_mapping.csv
#
# `random` additionally writes the k=1 runs used as the SINGLE baseline.

source "$(dirname -- "$0")/env.sh"
require_file "$GOEF_DINOV3"

for method in $METHODS; do
  for dataset in MVTec VisA; do
    if [[ "$dataset" == "MVTec" ]]; then root="$MVTEC_ROOT"; else root="$VISA_ROOT"; fi
    require_dir "$root"
    for seed in $SEEDS; do
      out="${SAMPLING_ROOT}/${method}/${dataset}_k${K}_seed${seed}"
      extra=()
      [[ "$method" != "random" ]] && extra+=(--weights "$GOEF_DINOV3")
      echo "== ${method} ${dataset} k=${K} seed=${seed}"
      goef sample --method "$method" --dataset_root "$root" --out_dir "$out" \
        --k "$K" --seed "$seed" "${extra[@]}"
    done
    if [[ "$method" == "random" ]]; then
      for seed in $SEEDS; do
        goef sample --method random --dataset_root "$root" \
          --out_dir "${SAMPLING_ROOT}/random/${dataset}_k1_seed${seed}" --k 1 --seed "$seed"
      done
    fi
  done
done
