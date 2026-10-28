# WinCLIP

Upstream: <https://github.com/zqhang/Accurate-WinCLIP-pytorch>  
Licence: MIT  
Conda env: `winclip`

The sweep reuses one test-feature cache per category and feeds one support image at a time (`memory(..., specific_image_paths=...)`).

Changed files:

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

Apply (the shared metric module is always needed):

```bash
cp ../common/unified_evaluation.py /path/to/<repo>/
cp -r validation/files/. /path/to/<repo>/
cp -r reevaluation/files/. /path/to/<repo>/
```

## Sec. III-D — per-reference 1-shot sweep

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

## Sec. IV — per-centre evaluation

```bash
cd /path/to/Accurate-WinCLIP-pytorch
conda activate winclip
DATA_ROOT="$CENTER_ROOT" GPU_ID=0 ./run_center_shots.sh 1 2 4
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv`.

See [`../README.md`](../README.md) for the other methods and for the
shared `common/unified_evaluation.py`.
