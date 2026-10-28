"""Train CLS and SEG PromptAD branches, then write unified results per center."""

import argparse
from pathlib import Path
import subprocess
import sys

from run_center_shot_cls import DATASET_CLASSES, center_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-base', type=Path, default=Path(os.environ.get('GOEF_CENTER_ROOT', 'data')))
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--eval-batch-size', type=int, default=None,
                        help='Unified-evaluation batch size; defaults to --batch-size.')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def run(command, project_root, dry_run):
    print(' '.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=project_root, check=True)


def main():
    args = parse_args()
    eval_batch_size = args.eval_batch_size or args.batch_size
    project_root = Path(__file__).resolve().parent
    for shot in args.shots:
        shot_dir = args.data_base / f'{shot}-shot'
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

                run_name = f'{prefix}_{dataset}'
                training_csvs = []
                for task, trainer in (('cls', 'train_cls.py'), ('seg', 'train_seg.py')):
                    csv_path = result_dir / f'{prefix}_{dataset}_{task}.csv'
                    training_csvs.append(csv_path)
                    for class_name in classes:
                        run([
                            sys.executable, trainer,
                            '--dataset', dataset, '--class_name', class_name,
                            '--k-shot', str(shot), '--data-root', str(data_root),
                            '--root-dir', str(result_dir), '--csv-path', str(csv_path),
                            '--run-name', run_name, '--gpu-id', str(args.gpu_id),
                            '--Epoch', str(args.epoch), '--batch-size', str(args.batch_size),
                        ], project_root, args.dry_run)

                run([
                    sys.executable, 'evaluate_unified.py',
                    '--dataset', dataset, '--k-shot', str(shot),
                    '--data-root', str(data_root), '--root-dir', str(result_dir),
                    '--run-name', run_name, '--gpu-id', str(args.gpu_id),
                    '--batch-size', str(eval_batch_size), '--output',
                    str(result_dir / f'{prefix}_{dataset}.csv'),
                    '--training-csv', *(str(path) for path in training_csvs),
                ], project_root, args.dry_run)


if __name__ == '__main__':
    main()
