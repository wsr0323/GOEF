#!/usr/bin/env bash
# Step 3 - compute the globality weights of Eq. (21)-(22) for every shot.
#
# Output: $SAMPLING_ROOT/weights/{mvtec,visa}_k<shot>.csv
#         (long format: dataset, category, shot, evaluation_shot, centers, weight)
source "$(dirname -- "$0")/env.sh"

CLUSTER_SEED=${CLUSTER_SEED:-1234}
CLUSTER_METHOD=${CLUSTER_METHOD:-k_means}
mkdir -p "${SAMPLING_ROOT}/weights"

for dataset in MVTec VisA; do
  key=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
  summary="${SAMPLING_ROOT}/${CLUSTER_METHOD}/${dataset}_k${K}_seed${CLUSTER_SEED}/summary.csv"
  require_file "$summary"
  goef weights --summary "$summary" --dataset "$key" \
    --shot 1 --all_shots $SHOTS --output "${SAMPLING_ROOT}/weights/${key}_all.csv"
  for shot in $SHOTS; do
    goef weights --summary "$summary" --dataset "$key" --shot "$shot" \
      --output "${SAMPLING_ROOT}/weights/${key}_k${shot}.csv"
  done
done
