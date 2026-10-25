#!/usr/bin/env bash
# Step 2 - materialise the k-shot "center" datasets used in Sec. IV.
#
# The centers are built from the k-means, k=5, seed=1234 partition (the GOEF
# default after the empirical study).  Output:
#   $CENTER_ROOT/<shot>-shot/{mvtec,visa}/center(s)_XX[_YY]/...
source "$(dirname -- "$0")/env.sh"

CLUSTER_SEED=${CLUSTER_SEED:-1234}
CLUSTER_METHOD=${CLUSTER_METHOD:-k_means}
MATERIALIZE=${MATERIALIZE:-symlink}

for shot in $SHOTS; do
  goef build_centers \
    --mvtec_source "$MVTEC_ROOT" \
    --mvtec_clusters "${SAMPLING_ROOT}/${CLUSTER_METHOD}/MVTec_k${K}_seed${CLUSTER_SEED}" \
    --visa_source "$VISA_ROOT" \
    --visa_clusters "${SAMPLING_ROOT}/${CLUSTER_METHOD}/VisA_k${K}_seed${CLUSTER_SEED}" \
    --output "${CENTER_ROOT}/${shot}-shot" \
    --k "$K" --combination-size "$shot" --materialize "$MATERIALIZE"
done
