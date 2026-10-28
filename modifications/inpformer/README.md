# INP-Former

Upstream: <https://github.com/luow23/INPFormer>  
Licence: MIT  
Conda env: `inpformer`

INP-Former trains one model per reference image / centre dataset (200 epochs), so both experiments are the expensive ones.

Changed files:

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

Apply (the shared metric module is always needed):

```bash
cp ../common/unified_evaluation.py /path/to/<repo>/
cp -r validation/files/. /path/to/<repo>/
cp -r reevaluation/files/. /path/to/<repo>/
```

## Sec. III-D — per-reference 1-shot sweep

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

## Sec. IV — per-centre evaluation

```bash
cd /path/to/INPFormer
conda activate inpformer
DATA_ROOT="$CENTER_ROOT" SHOTS="1 2 4" SKIP_EXISTING=1 ./train_all_few_shot.sh
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv`.

See [`../README.md`](../README.md) for the other methods and for the
shared `common/unified_evaluation.py`.
