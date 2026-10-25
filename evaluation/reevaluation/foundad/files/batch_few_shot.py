#!/usr/bin/env python3
"""Train and evaluate every MVTec/VisA center combination for 1-5 shot."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


DATASETS = ("mvtec", "visa")
OVERALL_GPU_COLUMNS = {
    "GPU Peak Allocated (MiB)",
    "GPU Peak Reserved (MiB)",
}
TRAINING_GPU_COLUMNS = {
    "Training GPU Peak Allocated (MiB)",
    "Training GPU Peak Reserved (MiB)",
}
TESTING_GPU_COLUMNS = {
    "Testing GPU Peak Allocated (MiB)",
    "Testing GPU Peak Reserved (MiB)",
}
FULL_PROCESS_GPU_COLUMNS = (
    TRAINING_GPU_COLUMNS | TESTING_GPU_COLUMNS | OVERALL_GPU_COLUMNS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GOEF_CENTER_ROOT", "data")))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--shots", type=int, nargs="+", default=range(1, 6), choices=range(1, 6))
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument(
        "--combinations",
        nargs="+",
        default=None,
        help="Only run named center combinations, for example center_01.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Rerun jobs whose result CSV already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without training.")
    return parser.parse_args()


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def has_full_process_gpu_columns(result_csv: Path) -> bool:
    try:
        with result_csv.open(newline="") as handle:
            columns = set(next(csv.reader(handle)))
    except (OSError, StopIteration):
        return False
    return FULL_PROCESS_GPU_COLUMNS.issubset(columns)


def merge_training_gpu_peak(result_csv: Path, peak_path: Path) -> None:
    with peak_path.open() as handle:
        training_peak = json.load(handle)
    with result_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None or not OVERALL_GPU_COLUMNS.issubset(fieldnames):
        raise ValueError(f"Benchmark columns are missing from: {result_csv}")

    output_fieldnames = [
        column for column in fieldnames if column not in OVERALL_GPU_COLUMNS
    ]
    output_fieldnames.extend(
        [
            "Training GPU Peak Allocated (MiB)",
            "Training GPU Peak Reserved (MiB)",
            "Testing GPU Peak Allocated (MiB)",
            "Testing GPU Peak Reserved (MiB)",
            "GPU Peak Allocated (MiB)",
            "GPU Peak Reserved (MiB)",
        ]
    )
    for row in rows:
        training_allocated = float(training_peak["GPU Peak Allocated (MiB)"])
        training_reserved = float(training_peak["GPU Peak Reserved (MiB)"])
        testing_allocated = float(row["GPU Peak Allocated (MiB)"])
        testing_reserved = float(row["GPU Peak Reserved (MiB)"])
        row.update(
            {
                "Training GPU Peak Allocated (MiB)": training_allocated,
                "Training GPU Peak Reserved (MiB)": training_reserved,
                "Testing GPU Peak Allocated (MiB)": testing_allocated,
                "Testing GPU Peak Reserved (MiB)": testing_reserved,
                "GPU Peak Allocated (MiB)": max(training_allocated, testing_allocated),
                "GPU Peak Reserved (MiB)": max(training_reserved, testing_reserved),
            }
        )
    with result_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_jobs(
    data_root: Path,
    shots: list[int],
    datasets: list[str],
    combinations_to_use: list[str] | None = None,
):
    for shot in sorted(set(shots)):
        shot_root = data_root / f"{shot}-shot"
        if not shot_root.is_dir():
            raise FileNotFoundError(f"Shot directory does not exist: {shot_root}")
        combinations = sorted(path for path in shot_root.iterdir() if path.is_dir())
        if combinations_to_use is not None:
            requested = set(combinations_to_use)
            combinations = [path for path in combinations if path.name in requested]
        if not combinations:
            raise FileNotFoundError(
                f"No matching center combinations found in: {shot_root}"
            )
        for combination in combinations:
            for dataset in datasets:
                dataset_root = combination / dataset
                if not dataset_root.is_dir():
                    raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
                yield shot, combination.name, dataset, dataset_root


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    foundad_root = repo_root / "foundad"
    python = sys.executable
    jobs = list(
        discover_jobs(args.data_root, args.shots, args.datasets, args.combinations)
    )
    print(f"Discovered {len(jobs)} jobs.")

    for index, (shot, combination, dataset, dataset_root) in enumerate(jobs, 1):
        stem = f"{combination}_{dataset}"
        shot_results = args.results_root / f"{shot}-shot"
        result_csv = shot_results / f"{stem}.csv"
        model_dir = shot_results / "models"
        checkpoint = model_dir / f"{stem}.pth.tar"
        params_path = model_dir / f"{stem}.yaml"
        training_peak_path = model_dir / f"{stem}.gpu_peak.json"
        work_dir = shot_results / ".work" / stem
        eval_dir = work_dir / "eval"

        if result_csv.exists() and not args.force:
            if has_full_process_gpu_columns(result_csv) and training_peak_path.is_file():
                print(f"[{index}/{len(jobs)}] skip {stem}: {result_csv} exists")
                continue
            print(f"[{index}/{len(jobs)}] regenerate result without full-process GPU peak: {result_csv}")
        print(f"[{index}/{len(jobs)}] {shot}-shot {stem}")
        if args.dry_run:
            continue

        shot_results.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        if (
            not checkpoint.exists()
            or not params_path.exists()
            or not training_peak_path.exists()
            or args.force
        ):
            if work_dir.exists():
                shutil.rmtree(work_dir)
            train_command = [
                python,
                "main.py",
                "mode=train",
                "app=train_dinov3",
                f"devices=[{args.device}]",
                f"data.dataset={dataset}",
                f"data.data_name={stem}",
                f"data.train_root={dataset_root}",
                f"app.logging.folder={work_dir}",
                "optimization.checkpoint_every_steps=0",
                "optimization.final_checkpoint_name=final.pth.tar",
            ]
            if args.epochs is not None:
                train_command.append(f"optimization.epochs={args.epochs}")
            run(train_command, foundad_root)
            shutil.move(str(work_dir / "final.pth.tar"), checkpoint)
            shutil.move(str(work_dir / "params.yaml"), params_path)
            shutil.move(str(work_dir / "training_gpu_peak.json"), training_peak_path)

        eval_command = [
            python,
            "main.py",
            "mode=AD",
            "app=test",
            f"devices=[{args.device}]",
            f"data.dataset={dataset}",
            f"data.data_name={stem}",
            f"data.test_root={dataset_root}",
            f"app.ckpt_path={checkpoint}",
            f"app.params_path={params_path}",
            f"app.output_folder={eval_dir}",
            "testing.segmentation_vis=false",
        ]
        run(eval_command, foundad_root)
        generated_csv = eval_dir / "unified_results.csv"
        if not generated_csv.is_file():
            raise FileNotFoundError(f"Evaluation did not create: {generated_csv}")
        shutil.move(str(generated_csv), result_csv)
        merge_training_gpu_peak(result_csv, training_peak_path)
        shutil.rmtree(work_dir)
        print(f"Saved {result_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
