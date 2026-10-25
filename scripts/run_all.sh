#!/usr/bin/env bash
# End-to-end reproduction driver.
#
#   export GOEF_WORKSPACE=/data/goef-work
#   export GOEF_DATA=/data/datasets
#   export GOEF_DINOV3=/data/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
#   ./scripts/run_all.sh
#
# Each step is idempotent; comment out the stages you already have.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)

"${HERE}/install_methods.sh"
"${HERE}/01_select_representatives.sh"
"${HERE}/02_build_centers.sh"
"${HERE}/03_compute_weights.sh"
"${HERE}/04_run_one_shot_sweeps.sh"      # Sec. III-D (5 methods, all references)
"${HERE}/05_run_center_shots.sh"         # Sec. IV   (8 methods, 62 center datasets)
"${HERE}/06_aggregate_validation.sh"     # Table I
"${HERE}/07_aggregate_reevaluation.sh"   # Table II
