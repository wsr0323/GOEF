# PatchCore

Upstream: <https://github.com/amazon-science/patchcore-inspection>  
Licence: Apache-2.0  
Conda env: `patchcore`

The sweep adds a `--train_image` option so that PatchCore can be trained on exactly one normal image.

Changed files:

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

Apply (the shared metric module is always needed):

```bash
cp ../common/unified_evaluation.py /path/to/<repo>/
cp -r validation/files/. /path/to/<repo>/
cp -r reevaluation/files/. /path/to/<repo>/
```

## Sec. III-D — per-reference 1-shot sweep

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

## Sec. IV — per-centre evaluation

```bash
cd /path/to/patchcore-inspection
conda activate patchcore
DATA_ROOT="$CENTER_ROOT" GPU=0 SAVE_MODELS=0 ./train_all_shots.sh 1 2 4
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv` plus `<...>_unified.csv` used by `goef-reevaluate`.

See [`../README.md`](../README.md) for the other methods and for the
shared `common/unified_evaluation.py`.
