#!/usr/bin/env bash
# Shared environment for the GOEF pipeline.
#
#   export GOEF_WORKSPACE=/path/to/goef-work      # scratch space (default: ./work)
#   export GOEF_DATA=/path/to/datasets            # MVTec-AD / VisA root (default: $GOEF_WORKSPACE/data)
#   export GOEF_DINOV3=/path/to/dinov3_vitb16.pth
#   export GOEF_PYTHON=python                     # interpreter with goef installed
#
# Directory layout created below GOEF_WORKSPACE:
#   third_party/{validation,reevaluation}/<project>/  upstream checkouts + overlays
#   sampling/<method>/<Dataset>_k5_seed<seed>/     GOEF representative selections
#   centers/<shot>-shot/<center>/<dataset>/        materialised k-shot datasets
#   results/<detector>/                            per-reference one-image sweeps
#   results/centers/<shot>-shot/*.csv              per-center-dataset evaluations
#   outputs/validation, outputs/reevaluation       aggregated tables

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

GOEF_WORKSPACE=${GOEF_WORKSPACE:-"${REPO_ROOT}/work"}
GOEF_DATA=${GOEF_DATA:-"${GOEF_WORKSPACE}/data"}
GOEF_PYTHON=${GOEF_PYTHON:-python}
GOEF_DINOV3=${GOEF_DINOV3:-"${GOEF_WORKSPACE}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"}

THIRD_PARTY=${THIRD_PARTY:-"${REPO_ROOT}/third_party"}
VALIDATION_DIR=${VALIDATION_DIR:-"${THIRD_PARTY}/validation"}
REEVALUATION_DIR=${REEVALUATION_DIR:-"${THIRD_PARTY}/reevaluation"}

SAMPLING_ROOT=${SAMPLING_ROOT:-"${GOEF_WORKSPACE}/sampling"}
CENTER_ROOT=${CENTER_ROOT:-"${GOEF_WORKSPACE}/centers"}
RESULT_ROOT=${RESULT_ROOT:-"${GOEF_WORKSPACE}/results"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${GOEF_WORKSPACE}/outputs"}

MVTEC_ROOT=${MVTEC_ROOT:-"${GOEF_DATA}/mvtec_anomaly_detection"}
VISA_ROOT=${VISA_ROOT:-"${GOEF_DATA}/VisA_pytorch/1cls"}

# Paper protocol: 5 clusters / representatives, 5 seeds, n = 5.
K=${K:-5}
SEEDS=${SEEDS:-"0 1 42 1234 3407"}
METHODS=${METHODS:-"random coreset k_center k_means k_medoids"}
SHOTS=${SHOTS:-"1 2 3 4 5"}

export GOEF_WORKSPACE GOEF_DATA GOEF_PYTHON GOEF_DINOV3
export THIRD_PARTY VALIDATION_DIR REEVALUATION_DIR
export SAMPLING_ROOT CENTER_ROOT RESULT_ROOT OUTPUT_ROOT
export MVTEC_ROOT VISA_ROOT K SEEDS METHODS SHOTS
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

goef() { "${GOEF_PYTHON}" -m "goef.cli.$1" "${@:2}"; }

require_dir() {
  [[ -d "$1" ]] || { echo "ERROR: missing directory: $1" >&2; exit 1; }
}

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: missing file: $1" >&2; exit 1; }
}
