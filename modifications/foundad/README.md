# FoundAD

Upstream: <https://github.com/ymxlzgy/FoundAD>  
Licence: not stated upstream — research use only  
Conda env: `foundad`

FoundAD trains a manifold projector per centre dataset; pre-trained projectors can be downloaded from the upstream README.

Changed files:

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

Apply (the shared metric module is always needed):

```bash
cp ../common/unified_evaluation.py /path/to/<repo>/
cp -r reevaluation/files/. /path/to/<repo>/
```

## Sec. IV — per-centre evaluation

```bash
cd /path/to/FoundAD
conda activate foundad
python batch_few_shot.py --data-root "$CENTER_ROOT" --shots 1 2 4 --device cuda:0
# --combinations center_01  to redo a single centre,  --force to rerun finished jobs
```

Output: `results/<shot>-shot/center(s)_<ids>_<dataset>.csv` (plus `models/` checkpoints).

See [`../README.md`](../README.md) for the other methods and for the
shared `common/unified_evaluation.py`.
