# UniVAD

Upstream: <https://github.com/FantasticGNU/UniVAD>  
Licence: CC BY-NC-SA 4.0 (non-commercial)  
Conda env: `univad`

UniVAD needs SAM-HQ, RAM and GroundingDINO checkpoints (`python download_sam_hq.py`). Component masks are built once from the 5-shot centre dataset, which contains every reference used at lower shots.

Changed files:

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

Apply (the shared metric module is always needed):

```bash
cp ../common/unified_evaluation.py /path/to/<repo>/
cp -r reevaluation/files/. /path/to/<repo>/
```

## Sec. IV — per-centre evaluation

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

See [`../README.md`](../README.md) for the other methods and for the
shared `common/unified_evaluation.py`.
