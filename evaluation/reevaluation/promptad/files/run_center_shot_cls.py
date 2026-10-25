"""Train image-level PromptAD on every materialized center split."""

import argparse
from pathlib import Path
import subprocess
import sys

DATASET_CLASSES = {
    'mvtec': [
        'carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable',
        'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush',
        'transistor', 'zipper',
    ],
    'visa': [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
        'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum',
    ],
}


def center_name(directory: Path) -> str:
    """Normalize centers_01_02 to the requested center_01_02 output prefix."""
    return directory.name.replace('centers_', 'center_', 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-base', type=Path, default=Path(os.environ.get('GOEF_CENTER_ROOT', 'data')))
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    for shot in args.shots:
        shot_dir = args.data_base / f'{shot}-shot'
        if not shot_dir.is_dir():
            raise FileNotFoundError(f'Missing shot directory: {shot_dir}')
        centers = sorted(path for path in shot_dir.iterdir()
                         if path.is_dir() and path.name.startswith('center'))
        if not centers:
            raise FileNotFoundError(f'No center directories found in: {shot_dir}')

        result_dir = args.results_dir / f'{shot}-shot'
        for center in centers:
            prefix = center_name(center)
            for dataset, classes in DATASET_CLASSES.items():
                data_root = center / dataset
                missing = [name for name in classes if not (data_root / name).is_dir()]
                if missing:
                    raise FileNotFoundError(f'{data_root} is missing: {", ".join(missing)}')

                csv_path = result_dir / f'{prefix}_{dataset}.csv'
                for class_name in classes:
                    command = [
                        sys.executable, 'train_cls.py',
                        '--dataset', dataset,
                        '--class_name', class_name,
                        '--k-shot', str(shot),
                        '--data-root', str(data_root),
                        '--root-dir', str(result_dir),
                        '--csv-path', str(csv_path),
                        '--run-name', f'{prefix}_{dataset}',
                        '--gpu-id', str(args.gpu_id),
                        '--Epoch', str(args.epoch),
                        '--batch-size', str(args.batch_size),
                    ]
                    print(' '.join(command), flush=True)
                    if not args.dry_run:
                        subprocess.run(command, cwd=project_root, check=True)


if __name__ == '__main__':
    main()
