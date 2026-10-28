# SubspaceAD

Upstream: <https://github.com/CLendering/SubspaceAD>  
Licence: not stated upstream — research use only  
Conda env: `subspacead`

The modified `main.py` differs between the two experiments: the validation variant adds `--k_shot_ref`, the re-evaluation variant integrates the unified metrics. Use two checkouts, or copy the variant you need.

Changed files:

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

Apply (the shared metric module is always needed):

```bash
cp ../common/unified_evaluation.py /path/to/<repo>/
cp -r validation/files/. /path/to/<repo>/
cp -r reevaluation/files/. /path/to/<repo>/
```

## Sec. III-D — per-reference 1-shot sweep

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

## Sec. IV — per-centre evaluation

```bash
cd /path/to/SubspaceAD
conda activate subspacead
DATA_ROOT="$CENTER_ROOT" bash scripts/benchmark_center_few_shot.sh 1 2 4
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv`.

See [`../README.md`](../README.md) for the other methods and for the
shared `common/unified_evaluation.py`.
