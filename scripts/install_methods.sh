#!/usr/bin/env bash
# Clone the eight upstream FS-IAD projects and overlay the GOEF adapters.
#
#   ./scripts/install_methods.sh            # clone missing repos, copy adapters
#   SUBSPACEAD_SRC=/path/to/SubspaceAD ./scripts/install_methods.sh
#
# Optional per-project pins (recommended for exact reproduction):
#   PIN_PATCHCORE=<commit> PIN_WINCLIP=<commit> ... ./scripts/install_methods.sh
#
# Upstream licences differ (Apache-2.0, MIT, Unlicense, CC BY-NC-SA 4.0); see
# third_party/README.md before redistributing.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
THIRD_PARTY=${THIRD_PARTY:-"${REPO_ROOT}/third_party"}
mkdir -p "$THIRD_PARTY"

clone() {  # clone <name> <url> <pin-var>
  local name=$1 url=$2 pin=${3:-}
  local dest="${THIRD_PARTY}/${name}"
  if [[ -d "$dest/.git" ]]; then
    echo "== ${name}: already cloned"
  elif [[ -d "$dest" ]]; then
    echo "== ${name}: using existing directory ${dest}"
  else
    echo "== ${name}: cloning ${url}"
    git clone --depth 1 "$url" "$dest"
  fi
  if [[ -n "$pin" ]]; then
    git -C "$dest" fetch --depth 1 origin "$pin" 2>/dev/null || true
    git -C "$dest" checkout --detach "$pin"
  fi
}

overlay() {  # overlay <adapter-dir> <target-dir>
  local src=$1 dst=$2
  local files="${src}/files"
  [[ -d "$files" ]] || { echo "ERROR: missing adapter files under $files" >&2; exit 1; }
  mkdir -p "$dst"
  cp -r "$files/." "$dst/"
  echo "   copied $(find "$files" -type f | wc -l) file(s) -> $dst"
}

overlay_metrics() {  # the shared metric module lands at the project root
  local dst=$1
  cp "${REPO_ROOT}/evaluation/common/unified_evaluation.py" "${dst}/unified_evaluation.py"
  echo "   copied unified_evaluation.py -> $dst"
}

# --- Sec. III-D per-reference sweeps + Sec. IV per-center evaluations --------
clone patchcore-inspection https://github.com/amazon-science/patchcore-inspection.git "${PIN_PATCHCORE:-}"
overlay "${REPO_ROOT}/evaluation/validation/patchcore"  "${THIRD_PARTY}/patchcore-inspection"
overlay "${REPO_ROOT}/evaluation/reevaluation/patchcore" "${THIRD_PARTY}/patchcore-inspection"
overlay_metrics "${THIRD_PARTY}/patchcore-inspection"

clone Accurate-WinCLIP-pytorch https://github.com/zqhang/Accurate-WinCLIP-pytorch.git "${PIN_WINCLIP:-}"
overlay "${REPO_ROOT}/evaluation/validation/winclip"   "${THIRD_PARTY}/Accurate-WinCLIP-pytorch"
overlay "${REPO_ROOT}/evaluation/reevaluation/winclip" "${THIRD_PARTY}/Accurate-WinCLIP-pytorch"
overlay_metrics "${THIRD_PARTY}/Accurate-WinCLIP-pytorch"

clone INP-Former https://github.com/luow23/INP-Former.git "${PIN_INPFORMER:-}"
overlay "${REPO_ROOT}/evaluation/validation/inpformer"   "${THIRD_PARTY}/INP-Former"
overlay "${REPO_ROOT}/evaluation/reevaluation/inpformer" "${THIRD_PARTY}/INP-Former"
overlay_metrics "${THIRD_PARTY}/INP-Former"

clone AnomalyDINO https://github.com/dammsi/AnomalyDINO.git "${PIN_ANOMALYDINO:-}"
overlay "${REPO_ROOT}/evaluation/validation/anomalydino"   "${THIRD_PARTY}/AnomalyDINO"
overlay "${REPO_ROOT}/evaluation/reevaluation/anomalydino" "${THIRD_PARTY}/AnomalyDINO"
overlay_metrics "${THIRD_PARTY}/AnomalyDINO"

clone PromptAD https://github.com/FuNz-0/PromptAD.git "${PIN_PROMPTAD:-}"
overlay "${REPO_ROOT}/evaluation/reevaluation/promptad" "${THIRD_PARTY}/PromptAD"
overlay_metrics "${THIRD_PARTY}/PromptAD"

clone FoundAD https://github.com/ymxlzgy/FoundAD.git "${PIN_FOUNDAD:-}"
overlay "${REPO_ROOT}/evaluation/reevaluation/foundad" "${THIRD_PARTY}/FoundAD"
overlay_metrics "${THIRD_PARTY}/FoundAD"

clone UniVAD https://github.com/FantasticGNU/UniVAD.git "${PIN_UNIVAD:-}"
overlay "${REPO_ROOT}/evaluation/reevaluation/univad" "${THIRD_PARTY}/UniVAD"
overlay_metrics "${THIRD_PARTY}/UniVAD"

# SubspaceAD has no public repository at the time of writing.
if [[ -n "${SUBSPACEAD_SRC:-}" ]]; then
  [[ -d "$SUBSPACEAD_SRC" ]] || { echo "ERROR: SUBSPACEAD_SRC is not a directory" >&2; exit 1; }
  overlay "${REPO_ROOT}/evaluation/validation/subspacead"   "${SUBSPACEAD_SRC}"
  overlay "${REPO_ROOT}/evaluation/reevaluation/subspacead" "${SUBSPACEAD_SRC}"
  overlay_metrics "${SUBSPACEAD_SRC}"
  echo "== SubspaceAD: adapters installed into ${SUBSPACEAD_SRC}"
else
  echo "== SubspaceAD: skipped (set SUBSPACEAD_SRC=/path/to/SubspaceAD to install)"
fi

echo
echo "Adapter files are copied into the upstream checkouts; upstream sources are"
echo "never modified in place by this repository (the overlays are copies of the"
echo "modified files used for the paper)."
