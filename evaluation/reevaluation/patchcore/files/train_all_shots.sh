#!/usr/bin/env bash
# Train every center combination in $DATA_ROOT/{1..5}-shot for MVTec and
# VisA. Per-combination CSVs are written directly to results/<shot>/.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
DATA_ROOT="${DATA_ROOT:-${GOEF_WORKSPACE:+${GOEF_WORKSPACE}/centers}}"
: "${DATA_ROOT:?set DATA_ROOT to the directory containing 1-shot .. 5-shot}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_MODELS="${SAVE_MODELS:-0}"
FAISS_ON_GPU="${FAISS_ON_GPU:-0}"
SHOTS=("$@")

if [[ ${#SHOTS[@]} -eq 0 ]]; then
  SHOTS=(1 2 3 4 5)
fi

run_dataset() {
  local shot_name="$1"
  local center_dir="$2"
  local dataset_name="$3"
  local data_dir="${center_dir}/${dataset_name}"
  local center_name result_name output_csv run_root
  center_name="$(basename "${center_dir}")"
  result_name="${center_name/#centers_/center_}"
  output_csv="${RESULTS_ROOT}/${shot_name}/${result_name}_${dataset_name}.csv"
  run_root="${RESULTS_ROOT}/${shot_name}/runs"

  mapfile -t categories < <(find "${data_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  if [[ ${#categories[@]} -eq 0 ]]; then
    echo "No categories found in ${data_dir}" >&2
    return 1
  fi

  flags=()
  faiss_args=()
  for category in "${categories[@]}"; do
    flags+=(-d "${category}")
  done
  if [[ "${FAISS_ON_GPU}" == "1" ]]; then
    faiss_args=(--faiss_on_gpu)
  fi

  mkdir -p "$(dirname "${output_csv}")" "${run_root}"
  echo "Training ${shot_name}/${center_name}/${dataset_name} -> ${output_csv}"

  command=(
    python "${PROJECT_ROOT}/bin/run_patchcore.py"
    --gpu "${GPU}" --seed "${SEED}"
    --log_project "runs" --log_group "${result_name}_${dataset_name}"
    "${RESULTS_ROOT}/${shot_name}"
    patch_core -b wideresnet50 -le layer2 -le layer3
    "${faiss_args[@]}" --pretrain_embed_dimension 1024 --target_embed_dimension 1024
    --anomaly_scorer_num_nn 1 --patchsize 3
    sampler -p 0.1 approx_greedy_coreset
    dataset --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}"
    --resize 256 --imagesize 224
    "${flags[@]}" "${dataset_name}" "${data_dir}"
  )
  if [[ "${SAVE_MODELS}" == "1" ]]; then
    command=(python "${PROJECT_ROOT}/bin/run_patchcore.py" --save_patchcore_model "${command[@]:2}")
  fi
  "${command[@]}"

  # create_storage_folder may append _N when a run already exists; select the
  # latest matching run and publish its summary under the requested flat name.
  latest_run="$(find "${run_root}" -mindepth 1 -maxdepth 1 -type d -name "${result_name}_${dataset_name}*" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  [[ -n "${latest_run}" && -f "${latest_run}/results.csv" ]] || {
    echo "Training completed without results.csv for ${result_name}_${dataset_name}" >&2
    return 1
  }
  cp "${latest_run}/results.csv" "${output_csv}"
  [[ -f "${latest_run}/unified_results.csv" ]] && cp "${latest_run}/unified_results.csv" "${output_csv%.csv}_unified.csv"
}

for shot in "${SHOTS[@]}"; do
  shot_name="${shot%-shot}-shot"
  shot_dir="${DATA_ROOT}/${shot_name}"
  [[ -d "${shot_dir}" ]] || { echo "Missing shot directory: ${shot_dir}" >&2; exit 1; }

  while IFS= read -r -d '' center_dir; do
    for dataset_name in mvtec visa; do
      [[ -d "${center_dir}/${dataset_name}" ]] || continue
      run_dataset "${shot_name}" "${center_dir}" "${dataset_name}"
    done
  done < <(find "${shot_dir}" -mindepth 1 -maxdepth 1 -type d -name 'center*' -print0 | sort -z)
done
