#!/usr/bin/env bash
# Clone the eight upstream FS-IAD projects and overlay the modified files in
# modifications/.  Upstream sources are never edited in place: the overlay is a
# copy of the files listed in modifications/<method>/README.md.
#
#   ./scripts/install_methods.sh                 # both experiments
#   ./scripts/install_methods.sh reevaluation    # only Sec. IV
#   ./scripts/install_methods.sh validation      # only Sec. III-D
#   METHOD=patchcore ./scripts/install_methods.sh reevaluation   # one project
#
# Destinations (override with VALIDATION_DIR / REEVALUATION_DIR):
#   <workspace>/third_party/validation/<project>     Sec. III-D per-reference sweeps
#   <workspace>/third_party/reevaluation/<project>   Sec. IV   per-centre evaluations
#
# PatchCore, WinCLIP, AnomalyDINO and SubspaceAD carry different changes in the
# two experiments (--train_image, specific_image_paths, ref_sample_names,
# --k_shot_ref), so they are installed into separate checkouts.
#
# Optional commit pins:
#   PIN_PATCHCORE=<sha> PIN_WINCLIP=<sha> ... ./scripts/install_methods.sh
set -euo pipefail

source "$(dirname -- "$0")/env.sh"

WANT=${1:-all}
PIN_PATCHCORE=${PIN_PATCHCORE:-}
PIN_WINCLIP=${PIN_WINCLIP:-}
PIN_ANOMALYDINO=${PIN_ANOMALYDINO:-}
PIN_INPFORMER=${PIN_INPFORMER:-}
PIN_SUBSPACEAD=${PIN_SUBSPACEAD:-}
PIN_PROMPTAD=${PIN_PROMPTAD:-}
PIN_FOUNDAD=${PIN_FOUNDAD:-}
PIN_UNIVAD=${PIN_UNIVAD:-}

install_one() {  # install_one <key> <dir> <url> <experiment> <pin>
  local key=$1 name=$2 url=$3 exp=$4 pin=${5:-}
  local src="${REPO_ROOT}/modifications/${key}/${exp}/files"
  local dest="${THIRD_PARTY}/${exp}/${name}"

  [[ -d "$src" ]] || return 0
  # METHOD accepts either the modifications/ key (patchcore) or the directory
  # name (patchcore-inspection).
  [[ -z "${METHOD:-}" || "${key}" == "${METHOD}" ]] || return 0

  if [[ -d "$dest/.git" ]]; then
    echo "== ${name} (${exp}): already cloned"
  elif [[ -d "$dest" ]]; then
    echo "== ${name} (${exp}): using existing directory ${dest}"
  else
    echo "== ${name} (${exp}): cloning ${url}"
    git clone --depth 1 "$url" "$dest"
  fi
  if [[ -n "$pin" ]]; then
    git -C "$dest" fetch --depth 1 origin "$pin" 2>/dev/null || true
    git -C "$dest" checkout --detach "$pin"
  fi

  cp -r "${src}/." "${dest}/"
  cp "${REPO_ROOT}/modifications/common/unified_evaluation.py" "${dest}/unified_evaluation.py"
  echo "   overlaid $(find "$src" -type f | wc -l) file(s) from modifications/${key}/${exp}/files"
}

install_all() {  # install_all <experiment>
  local exp=$1
  install_one patchcore    patchcore-inspection     https://github.com/amazon-science/patchcore-inspection.git "$exp" "$PIN_PATCHCORE"
  install_one winclip      Accurate-WinCLIP-pytorch https://github.com/zqhang/Accurate-WinCLIP-pytorch.git      "$exp" "$PIN_WINCLIP"
  install_one anomalydino  AnomalyDINO              https://github.com/dammsi/AnomalyDINO.git                  "$exp" "$PIN_ANOMALYDINO"
  install_one inpformer    INPFormer                https://github.com/luow23/INPFormer.git                    "$exp" "$PIN_INPFORMER"
  install_one subspacead   SubspaceAD               https://github.com/CLendering/SubspaceAD.git               "$exp" "$PIN_SUBSPACEAD"
  install_one promptad     PromptAD                 https://github.com/FuNz-0/PromptAD.git                    "$exp" "$PIN_PROMPTAD"
  install_one foundad      FoundAD                  https://github.com/ymxlzgy/FoundAD.git                    "$exp" "$PIN_FOUNDAD"
  install_one univad       UniVAD                   https://github.com/FantasticGNU/UniVAD.git                "$exp" "$PIN_UNIVAD"
}

case "$WANT" in
  all)             install_all validation; install_all reevaluation ;;
  validation)      install_all validation ;;
  reevaluation)    install_all reevaluation ;;
  *) echo "ERROR: expected 'all', 'validation' or 'reevaluation', got '$WANT'" >&2; exit 2 ;;
esac

echo
echo "Installed under ${THIRD_PARTY}/{validation,reevaluation}."
echo "Each checkout already contains unified_evaluation.py from modifications/common/."
