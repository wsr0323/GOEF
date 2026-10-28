# AnomalyDINO

Upstream: <https://github.com/dammsi/AnomalyDINO>  
Licence: Apache-2.0  
Conda env: `anomalydino`

AnomalyDINO is training-free; the sweep only rebuilds the memory bank for every reference image.

Changed files:

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

Apply (the shared metric module is always needed):

```bash
cp ../common/unified_evaluation.py /path/to/<repo>/
cp -r validation/files/. /path/to/<repo>/
cp -r reevaluation/files/. /path/to/<repo>/
```

## Sec. III-D — per-reference 1-shot sweep

```bash
cd /path/to/AnomalyDINO
conda activate anomalydino
python run_mvtec_train_1shot_refs.py --data_root "$GOEF_DATA/mvtec_anomaly_detection" \
  --output_root "$RESULT_ROOT/anomalydino/mvtec"
python run_visa_train_1shot_refs.py  --data_root "$GOEF_DATA/VisA_pytorch/1cls" \
  --output_root "$RESULT_ROOT/anomalydino/visa"
```

Output: `<output_root>/summary.csv` — one row per reference image.

## Sec. IV — per-centre evaluation

```bash
cd /path/to/AnomalyDINO
conda activate anomalydino
DATA_ROOT="$CENTER_ROOT" SHOTS="1 2 4" SKIP_EXISTING=1 ./train_all_few_shot.sh
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv`.

See [`../README.md`](../README.md) for the other methods and for the
shared `common/unified_evaluation.py`.
