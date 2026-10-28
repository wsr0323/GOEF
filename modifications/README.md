# Modifications to the eight upstream FS-IAD projects

This folder contains **every file that differs from the upstream repositories**,
ready to be copied over a clean checkout.  Nothing else from the upstream
projects is redistributed here.

```
modifications/
├── common/unified_evaluation.py     # the shared four-metric protocol
├── <method>/
│   ├── README.md                    # files + run commands for this method
│   ├── validation/files/...         # Sec. III-D: per-reference 1-shot sweeps
│   └── reevaluation/files/...       # Sec. IV:   per-centre evaluations
```

Each `files/` directory mirrors the upstream repository layout, so applying a
modification is a single copy:

```bash
cp -r modifications/patchcore/reevaluation/files/. /path/to/patchcore-inspection/
# `unified_evaluation.py` is imported as a top-level module, so it lands in the
# repository root; everything else keeps its relative path.
```

`scripts/install_methods.sh` does this automatically for all eight projects.

> **Two variants cannot share one checkout.** For PatchCore, WinCLIP,
> AnomalyDINO and SubspaceAD the two experiments touch the same files with
> different changes (`--train_image`, `specific_image_paths`, `ref_sample_names`,
> `--k_shot_ref`).  Keep one checkout per experiment, e.g.
> `third_party/validation/...` and `third_party/reevaluation/...`.

## Upstream projects

| Method | Upstream | Licence | Experiments in this folder |
|---|---|---|---|
| PatchCore | <https://github.com/amazon-science/patchcore-inspection> | Apache-2.0 | validation + reevaluation |
| WinCLIP | <https://github.com/zqhang/Accurate-WinCLIP-pytorch> | MIT | validation + reevaluation |
| AnomalyDINO | <https://github.com/dammsi/AnomalyDINO> | Apache-2.0 | validation + reevaluation |
| INP-Former | <https://github.com/luow23/INPFormer> | MIT | validation + reevaluation |
| SubspaceAD | <https://github.com/CLendering/SubspaceAD> | not stated upstream — research use only | validation + reevaluation |
| PromptAD | <https://github.com/FuNz-0/PromptAD> | Unlicense (public domain) | reevaluation |
| FoundAD | <https://github.com/ymxlzgy/FoundAD> | not stated upstream — research use only | reevaluation |
| UniVAD | <https://github.com/FantasticGNU/UniVAD> | CC BY-NC-SA 4.0 (non-commercial) | reevaluation |

## Environment variables used by the commands

```bash
export GOEF_DATA=/path/to/datasets            # mvtec_anomaly_detection + VisA_pytorch/1cls
export CENTER_ROOT=/path/to/centers           # contains 1-shot/ ... 5-shot/ built by goef-build-centers
export RESULT_ROOT=/path/to/results           # per-reference sweep outputs
```

Every driver also accepts a subset of shots (`1 2 4`), so a single setting can be
reproduced without recomputing the others.  All sweeps are resumable.

## Method-by-method details

### PatchCore

* upstream: <https://github.com/amazon-science/patchcore-inspection>
* licence: Apache-2.0
* conda env used in the paper: `patchcore`

The sweep adds a `--train_image` option so that PatchCore can be trained on exactly one normal image.

| File | Type | Purpose |
|---|---|---|
| `validation/files/bin/run_one_shot_patchcore.py` | new | per-reference sweep driver; one PatchCore run per image in `train/good`, resumable by `object + reference_index` |
| `validation/files/ONE_SHOT_PATCHCORE.md` | new | usage notes for the sweep |
| `validation/files/bin/run_patchcore.py` | modified | adds `--train_image` and forwards it to the dataset |
| `validation/files/src/patchcore/datasets/mvtec.py` | modified | keeps only the requested image in `train/good` |
| `validation/files/src/patchcore/metrics.py` | modified | adds `compute_aupro()` (per-region overlap, FPR limit 0.30) |
| `validation/files/requirements.txt` | modified | pins of the environment used for the sweep |
| `reevaluation/files/train_all_shots.sh` | new | runs every centre dataset for the requested shots (`./train_all_shots.sh 1 2 4`) |
| `reevaluation/files/unified_evaluation.py` | new | shared four-metric protocol + inference benchmark |
| `reevaluation/files/UNIFIED_EVALUATION.md` | new | protocol description |
| `reevaluation/files/bin/run_patchcore.py` | modified | writes the unified result CSV with timing / GPU peaks |
| `reevaluation/files/bin/load_and_evaluate_patchcore.py` | modified | same unified output for the load-and-evaluate path |
| `reevaluation/files/src/patchcore/patchcore.py` | modified | `predict(..., benchmark=...)` hooks for the timer |
| `reevaluation/files/src/patchcore/common.py` | modified | clear error when GPU FAISS is requested but only faiss-cpu is installed |

**Sec. III-D — per-reference sweep**

```bash
cd /path/to/patchcore-inspection
conda activate patchcore
python bin/run_one_shot_patchcore.py --datasets mvtec --gpu 0 \
  --mvtec-root "$GOEF_DATA/mvtec_anomaly_detection" \
  --output-csv "$RESULT_ROOT/patchcore/mvtec.csv"
python bin/run_one_shot_patchcore.py --datasets visa  --gpu 0 \
  --visa-root "$GOEF_DATA/VisA_pytorch/1cls" \
  --output-csv "$RESULT_ROOT/patchcore/visa.csv"
```

Output: `$RESULT_ROOT/patchcore/<dataset>.csv` — one row per reference image (`object, ref_sample, Image AUROC, Image AUPR, Pixel AUROC, Pixel AUPRO, ...`).

**Sec. IV — per-centre evaluation**

```bash
cd /path/to/patchcore-inspection
conda activate patchcore
DATA_ROOT="$CENTER_ROOT" GPU=0 SAVE_MODELS=0 ./train_all_shots.sh 1 2 4
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv` plus `<...>_unified.csv` used by `goef-reevaluate`.

Details: [`patchcore/README.md`](patchcore/README.md)

### WinCLIP

* upstream: <https://github.com/zqhang/Accurate-WinCLIP-pytorch>
* licence: MIT
* conda env used in the paper: `winclip`

The sweep reuses one test-feature cache per category and feeds one support image at a time (`memory(..., specific_image_paths=...)`).

| File | Type | Purpose |
|---|---|---|
| `validation/files/run_one_category.py` | new | per-reference sweep driver (one class at a time, resumable) |
| `validation/files/reproduce_WinCLIP.py` | modified | adds `SingleClassEvaluator`, `evaluate_support()`, test-feature caching and `specific_image_paths=` support in `memory()` |
| `validation/files/few_shot.py` | modified | accepts `use_preset_train` and the evaluator hooks |
| `validation/files/data/mvtec.py` | modified | reads the curated `meta.json` training entries |
| `validation/files/data/visa.py` | modified | same for VisA 1cls |
| `validation/files/src/open_clip/modified_resnet.py` | modified | compatibility fix for the bundled open_clip |
| `validation/files/requirements.txt` | modified | environment pins |
| `reevaluation/files/run_center_shots.sh` | new | runs every centre dataset (`./run_center_shots.sh 1 2 4`) |
| `reevaluation/files/unified_evaluation.py` | new | shared four-metric protocol |
| `reevaluation/files/UNIFIED_EVALUATION.md` | new | protocol description |
| `reevaluation/files/reproduce_WinCLIP.py` | modified | writes the unified CSV and reports timing / GPU peaks |
| `reevaluation/files/dataset.py` | modified | `use_preset_train=True` takes exactly the images stored in the centre dataset |
| `reevaluation/files/few_shot.py` | modified | forwards `use_preset_train` |
| `reevaluation/files/src/training/params.py` | modified | seed 42 instead of 0 |

**Sec. III-D — per-reference sweep**

```bash
cd /path/to/Accurate-WinCLIP-pytorch
conda activate winclip
CUDA_VISIBLE_DEVICES=0 python run_one_category.py --datasets mvtec \
  --mvtec_root "$GOEF_DATA/mvtec_anomaly_detection" \
  --output_csv "$RESULT_ROOT/winclip/mvtec.csv"
CUDA_VISIBLE_DEVICES=0 python run_one_category.py --datasets visa \
  --visa_root "$GOEF_DATA/VisA_pytorch/1cls" \
  --output_csv "$RESULT_ROOT/winclip/visa.csv"
```

Output: `$RESULT_ROOT/winclip/<dataset>.csv` — one row per reference image.

**Sec. IV — per-centre evaluation**

```bash
cd /path/to/Accurate-WinCLIP-pytorch
conda activate winclip
DATA_ROOT="$CENTER_ROOT" GPU_ID=0 ./run_center_shots.sh 1 2 4
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv`.

Details: [`winclip/README.md`](winclip/README.md)

### AnomalyDINO

* upstream: <https://github.com/dammsi/AnomalyDINO>
* licence: Apache-2.0
* conda env used in the paper: `anomalydino`

AnomalyDINO is training-free; the sweep only rebuilds the memory bank for every reference image.

| File | Type | Purpose |
|---|---|---|
| `validation/files/run_mvtec_train_1shot_refs.py` | new | per-reference sweep over every MVTec training image |
| `validation/files/run_visa_train_1shot_refs.py` | new | same for VisA |
| `validation/files/run_anomalydino.py` | modified | accepts `ref_sample_names` and the per-reference output layout |
| `validation/files/src/detection.py` | modified | builds the memory bank from exactly one named reference |
| `validation/files/src/backbones.py` | modified | feature-extraction hooks used by the sweep |
| `validation/files/requirements.txt` | modified | environment pins and example commands |
| `reevaluation/files/train_all_few_shot.sh` | new | runs every centre dataset (`SHOTS="1 2 4" ./train_all_few_shot.sh`) |
| `reevaluation/files/unified_evaluation.py` | new | shared four-metric protocol |
| `reevaluation/files/UNIFIED_EVALUATION.md` | new | protocol description |
| `reevaluation/files/run_anomalydino.py` | modified | `--use_all_references`, unified CSV and result paths |
| `reevaluation/files/run_anomalydino_batched.py` | modified | batched variant aligned with the unified output |
| `reevaluation/files/src/detection.py` | modified | reference-count validation and unified metric call |
| `reevaluation/files/src/post_eval.py` | modified | delegates the four metrics to `unified_evaluation` |
| `reevaluation/files/src/backbones.py` | modified | device / resolution handling |

**Sec. III-D — per-reference sweep**

```bash
cd /path/to/AnomalyDINO
conda activate anomalydino
python run_mvtec_train_1shot_refs.py --data_root "$GOEF_DATA/mvtec_anomaly_detection" \
  --output_root "$RESULT_ROOT/anomalydino/mvtec"
python run_visa_train_1shot_refs.py  --data_root "$GOEF_DATA/VisA_pytorch/1cls" \
  --output_root "$RESULT_ROOT/anomalydino/visa"
```

Output: `<output_root>/summary.csv` — one row per reference image.

**Sec. IV — per-centre evaluation**

```bash
cd /path/to/AnomalyDINO
conda activate anomalydino
DATA_ROOT="$CENTER_ROOT" SHOTS="1 2 4" SKIP_EXISTING=1 ./train_all_few_shot.sh
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv`.

Details: [`anomalydino/README.md`](anomalydino/README.md)

### INP-Former

* upstream: <https://github.com/luow23/INPFormer>
* licence: MIT
* conda env used in the paper: `inpformer`

INP-Former trains one model per reference image / centre dataset (200 epochs), so both experiments are the expensive ones.

| File | Type | Purpose |
|---|---|---|
| `validation/files/INP_Former_One_Image_Sweep.py` | new | self-contained sweep: discovery, per-reference training + evaluation, atomic CSV ledger, error ledger |
| `reevaluation/files/train_all_few_shot.sh` | new | runs every centre dataset (`SHOTS="1 2 4" ./train_all_few_shot.sh`) |
| `reevaluation/files/unified_evaluation.py` | new | shared four-metric protocol |
| `reevaluation/files/UNIFIED_EVALUATION.md` | new | protocol description |
| `reevaluation/files/INP_Former_Few_Shot.py` | modified | `--shot`, `--result_csv`, unified output |
| `reevaluation/files/INP_Former_Single_Class.py` | modified | same unified output |
| `reevaluation/files/INP_Former_Multi_Class.py` | modified | same unified output |
| `reevaluation/files/INP_Former_Super_Multi_Class.py` | modified | same unified output |
| `reevaluation/files/INP_Former_Zero_Shot.py` | modified | same unified output |
| `reevaluation/files/utils.py` | modified | metric computation delegated to `unified_evaluation` |
| `reevaluation/files/convert_onnx.py` | modified | seed 42 |
| `reevaluation/files/dinov1/utils.py` | modified | default seed 42 |
| `reevaluation/files/dinov2/utils/utils.py` | modified | default seed 42 |
| `reevaluation/files/dinov2/configs/ssl_default_config.yaml` | modified | seed 42 |

**Sec. III-D — per-reference sweep**

```bash
cd /path/to/INPFormer
conda activate inpformer
python INP_Former_One_Image_Sweep.py --datasets mvtec visa --device cuda:0 \
  --mvtec-data-path "$GOEF_DATA/mvtec_anomaly_detection" \
  --visa-data-path  "$GOEF_DATA/VisA_pytorch/1cls" \
  --output-csv saved_results/one_image_sweep/results.csv
# optional: --categories bottle --start-index 0 --end-index 20 --max-runs 20
```

Output: `saved_results/one_image_sweep/results.csv` — one row per reference image (both datasets in one file, `dataset` column).

**Sec. IV — per-centre evaluation**

```bash
cd /path/to/INPFormer
conda activate inpformer
DATA_ROOT="$CENTER_ROOT" SHOTS="1 2 4" SKIP_EXISTING=1 ./train_all_few_shot.sh
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv`.

Details: [`inpformer/README.md`](inpformer/README.md)

### SubspaceAD

* upstream: <https://github.com/CLendering/SubspaceAD>
* licence: not stated upstream — research use only
* conda env used in the paper: `subspacead`

The modified `main.py` differs between the two experiments: the validation variant adds `--k_shot_ref`, the re-evaluation variant integrates the unified metrics. Use two checkouts, or copy the variant you need.

| File | Type | Purpose |
|---|---|---|
| `validation/files/run_mvtec_each_train_1shot.sh` | new | loops every MVTec training image, then merges the per-run tables |
| `validation/files/run_visa_each_train_1shot.sh` | new | same for VisA |
| `validation/files/merge_shot_results.py` | new | merges `<OUT_ROOT>/<category>/<run_id>/*/benchmark_results.csv` into `all_runs_table.csv` |
| `validation/files/main.py` | modified | adds `--k_shot_ref`, reference-index parsing, `ref_sample` / `Run` / `参考索引` columns, CPU AUPRO |
| `validation/files/src/subspacead/core/extractor.py` | modified | adds DINOv3 backbones through `transformers` |
| `validation/files/src/subspacead/core/patching.py` | modified | tolerates a missing saliency mask |
| `validation/files/src/subspacead/config.py` | modified | new backbone / layer options |
| `validation/files/src/subspacead/utils/viz.py` | modified | visualisation helper used by the sweep |
| `validation/files/requirements.txt` | modified | environment pins |
| `reevaluation/files/scripts/benchmark_center_few_shot.sh` | new | runs every centre dataset (`bash scripts/benchmark_center_few_shot.sh 1 2 4`) |
| `reevaluation/files/unified_evaluation.py` | new | shared four-metric protocol |
| `reevaluation/files/UNIFIED_EVALUATION.md` | new | protocol description |
| `reevaluation/files/main.py` | modified | unified CSV output, per-centre result publishing |
| `reevaluation/files/src/subspacead/config.py` | modified | resolution / layer options for the centre runs |
| `reevaluation/files/src/subspacead/core/extractor.py` | modified | layer selection fix for DINOv2-g |

**Sec. III-D — per-reference sweep**

```bash
cd /path/to/SubspaceAD
conda activate subspacead
MVTEC_PATH="$GOEF_DATA/mvtec_anomaly_detection/" bash run_mvtec_each_train_1shot.sh
VISA_PATH="$GOEF_DATA/VisA_pytorch/1cls/"      bash run_visa_each_train_1shot.sh
# each script finishes with merge_shot_results.py:
#   few_shot_results_each_train_1shot/mvtec_dinov2G/all_runs_table.csv
#   few_shot_results_each_train_1shot/visa_dinov2G/all_runs_table.csv
```

Output: `few_shot_results_each_train_1shot/<dataset>_dinov2G/all_runs_table.csv` — one row per reference image.

**Sec. IV — per-centre evaluation**

```bash
cd /path/to/SubspaceAD
conda activate subspacead
DATA_ROOT="$CENTER_ROOT" bash scripts/benchmark_center_few_shot.sh 1 2 4
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv`.

Details: [`subspacead/README.md`](subspacead/README.md)

### PromptAD

* upstream: <https://github.com/FuNz-0/PromptAD>
* licence: Unlicense (public domain)
* conda env used in the paper: `prompt_ad`

First run downloads the CLIP backbone. Each centre/dataset pair trains a classification and a segmentation model.

| File | Type | Purpose |
|---|---|---|
| `reevaluation/files/run_center_shot_unified.py` | new | orchestrates every centre/dataset: train_cls + train_seg + evaluate_unified |
| `reevaluation/files/run_center_shot_cls.py` | new | classification stage of the orchestrator |
| `reevaluation/files/evaluate_unified.py` | new | builds the unified CSV from the two training logs |
| `reevaluation/files/unified_evaluation.py` | new | shared four-metric protocol |
| `reevaluation/files/UNIFIED_EVALUATION.md` | new | protocol description |
| `reevaluation/files/train_cls.py` | modified | `--data-root`, `--csv-path`, `--k-shot`, GPU-peak logging |
| `reevaluation/files/train_seg.py` | modified | same for segmentation |
| `reevaluation/files/test_cls.py` | modified | unified metric call |
| `reevaluation/files/test_seg.py` | modified | unified metric call |
| `reevaluation/files/datasets/dataset.py` | modified | reads the centre `meta.json` |
| `reevaluation/files/datasets/mvtec.py` | modified | centre layout |
| `reevaluation/files/datasets/visa.py` | modified | centre layout |
| `reevaluation/files/datasets/__init__.py` | modified | dataset registry |
| `reevaluation/files/utils/csv_utils.py` | modified | per-class CSV rows used by the unified evaluation |
| `reevaluation/files/utils/training_utils.py` | modified | seed / device / logging helpers |

**Sec. IV — per-centre evaluation**

```bash
cd /path/to/PromptAD
conda activate prompt_ad
python run_center_shot_unified.py --data-base "$CENTER_ROOT" \
  --results-dir results --shots 1 2 4 --gpu-id 0 --batch-size 400
# --dry-run prints all commands without executing them
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv` (+ `_cls.csv`, `_seg.csv` training logs).

Details: [`promptad/README.md`](promptad/README.md)

### FoundAD

* upstream: <https://github.com/ymxlzgy/FoundAD>
* licence: not stated upstream — research use only
* conda env used in the paper: `foundad`

FoundAD trains a manifold projector per centre dataset; pre-trained projectors can be downloaded from the upstream README.

| File | Type | Purpose |
|---|---|---|
| `reevaluation/files/batch_few_shot.py` | new | runs every centre/dataset through `foundad/main.py` train + test and collects the unified CSV |
| `reevaluation/files/unified_evaluation.py` | new | shared four-metric protocol |
| `reevaluation/files/UNIFIED_EVALUATION.md` | new | protocol description |
| `reevaluation/files/foundad/src/AD.py` | modified | imports the shared metrics and injects the benchmark timer |
| `reevaluation/files/foundad/src/train.py` | modified | deterministic seed 42 (python / numpy / torch / cuda) |
| `reevaluation/files/foundad/src/foundad.py` | modified | exposes the per-image scores and maps |
| `reevaluation/files/foundad/src/datasets/dataset.py` | modified | `train_root` / `test_root` for the centre layout |
| `reevaluation/files/foundad/main.py` | modified | `diy_name`, checkpoint naming, test-mode paths |
| `reevaluation/files/foundad/configs/config.yaml` | modified | hydra defaults used by `batch_few_shot.py` |
| `reevaluation/files/foundad/configs/app/test.yaml` | modified | test-time output folder |
| `reevaluation/files/foundad/src/dinov2/utils/utils.py` | modified | default seed 42 |
| `reevaluation/files/foundad/src/dinov2/configs/ssl_default_config.yaml` | modified | seed 42 |

**Sec. IV — per-centre evaluation**

```bash
cd /path/to/FoundAD
conda activate foundad
python batch_few_shot.py --data-root "$CENTER_ROOT" --shots 1 2 4 --device cuda:0
# --combinations center_01  to redo a single centre,  --force to rerun finished jobs
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv` (plus `models/` checkpoints).

Details: [`foundad/README.md`](foundad/README.md)

### UniVAD

* upstream: <https://github.com/FantasticGNU/UniVAD>
* licence: CC BY-NC-SA 4.0 (non-commercial)
* conda env used in the paper: `univad`

UniVAD needs SAM-HQ, RAM and GroundingDINO checkpoints (`python download_sam_hq.py`). Component masks are built once from the 5-shot centre dataset, which contains every reference used at lower shots.

| File | Type | Purpose |
|---|---|---|
| `reevaluation/files/run_fewshot_centers.sh` | new | runs every centre dataset (`./run_fewshot_centers.sh 1 2 4`) |
| `reevaluation/files/prepare_fewshot_masks.py` | new | builds the component masks for all reference images |
| `reevaluation/files/download_sam_hq.py` | new | downloads the SAM-HQ checkpoint |
| `reevaluation/files/unified_evaluation.py` | new | shared four-metric protocol |
| `reevaluation/files/UNIFIED_EVALUATION.md` | new | protocol description |
| `reevaluation/files/UniVAD.py` | modified | reference cosine-similarity module and mask paths for external references |
| `reevaluation/files/test_univad.py` | modified | `--data_path`, `--reference_from_meta`, `--output_csv` |
| `reevaluation/files/test_univad_no_pixel.py` | modified | same interface for the classification-only run |

**Sec. IV — per-centre evaluation**

```bash
cd /path/to/UniVAD
conda activate univad
python download_sam_hq.py
python prepare_fewshot_masks.py \
  --source_root "$CENTER_ROOT/5-shot/centers_01_02_03_04_05" \
  --mask_root masks/fewshot --skip_existing
DATA_ROOT="$CENTER_ROOT" SKIP_EXISTING=1 ./run_fewshot_centers.sh 1 2 4
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv` (+ `.log`).

Details: [`univad/README.md`](univad/README.md)
