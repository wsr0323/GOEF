# PromptAD

Upstream: <https://github.com/FuNz-0/PromptAD>  
Licence: Unlicense (public domain)  
Conda env: `prompt_ad`

First run downloads the CLIP backbone. Each centre/dataset pair trains a classification and a segmentation model.

Changed files:

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

Apply (the shared metric module is always needed):

```bash
cp ../common/unified_evaluation.py /path/to/<repo>/
cp -r reevaluation/files/. /path/to/<repo>/
```

## Sec. IV — per-centre evaluation

```bash
cd /path/to/PromptAD
conda activate prompt_ad
python run_center_shot_unified.py --data-base "$CENTER_ROOT" \
  --results-dir results --shots 1 2 4 --gpu-id 0 --batch-size 400
# --dry-run prints all commands without executing them
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv` (+ `_cls.csv`, `_seg.csv` training logs).

See [`../README.md`](../README.md) for the other methods and for the
shared `common/unified_evaluation.py`.
