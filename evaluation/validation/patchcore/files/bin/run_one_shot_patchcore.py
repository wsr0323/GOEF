#!/usr/bin/env python3
"""Train/evaluate PatchCore once per train/good image, resuming from a CSV."""

import argparse
import os
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm


_DATA = os.environ.get("GOEF_DATA", "data")
ROOTS = {
    "mvtec": os.path.join(_DATA, "mvtec_anomaly_detection"),
    "visa": os.path.join(_DATA, "VisA_pytorch/1cls"),
}
COLUMNS = [
    "dataset", "object", "Run", "ref_sample", "reference_index",
    "Image AUROC", "Image AUPR", "Pixel AUROC", "Pixel AUPRO", "elapsed_seconds",
]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=ROOTS, default=list(ROOTS))
    parser.add_argument("--classes", nargs="+", help="Run only these categories.")
    parser.add_argument("--mvtec-root", default=ROOTS["mvtec"])
    parser.add_argument("--visa-root", default=ROOTS["visa"])
    parser.add_argument("--output-dir", default="one_shot_patchcore_results")
    parser.add_argument("--output-csv", help="CSV path (only with one selected dataset).")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--faiss-on-gpu", action="store_true", help="Use a faiss-gpu installation for nearest-neighbour search.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--coreset-percentage", type=float, default=0.1)
    parser.add_argument("--keep-run-results", action="store_true")
    return parser.parse_args()


def initialize_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=COLUMNS).writeheader()


def completed(path):
    with path.open(newline="", encoding="utf-8") as file:
        return {(row["object"], row["reference_index"]) for row in csv.DictReader(file)}


def append(path, row):
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=COLUMNS).writerow(row)


def get_categories(root, requested):
    available = sorted(p.name for p in root.iterdir() if (p / "train" / "good").is_dir())
    if requested is None:
        return available
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError("Unknown categories: {}".format(", ".join(sorted(unknown))))
    return requested


def patchcore_command(args, root, category, image_name, run_root):
    command = [
        sys.executable, "bin/run_patchcore.py", "--gpu", str(args.gpu), "--seed", str(args.seed),
        "--log_project", "one_shot", "--log_group", "patchcore", str(run_root),
        "patch_core", "-b", "wideresnet50", "-le", "layer2", "-le", "layer3",
        "--pretrain_embed_dimension", "1024", "--target_embed_dimension", "1024",
        "--anomaly_scorer_num_nn", "1", "--patchsize", "3", "sampler", "-p",
        str(args.coreset_percentage), "approx_greedy_coreset", "dataset", "--resize",
        str(args.resize), "--imagesize", str(args.imagesize), "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers), "--train_image", image_name, "-d", category,
        "mvtec", str(root),
    ]
    if args.faiss_on_gpu:
        command.insert(command.index("--pretrain_embed_dimension"), "--faiss_on_gpu")
    return command


def load_metrics(run_root):
    candidates = list(run_root.glob("one_shot/patchcore*/results.csv"))
    if not candidates:
        raise FileNotFoundError("No PatchCore result CSV in {}".format(run_root))
    with max(candidates, key=lambda p: p.stat().st_mtime).open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    return {
        "Image AUROC": row["instance_auroc"],
        "Image AUPR": row["image_aupr"],
        "Pixel AUROC": row["full_pixel_auroc"],
        "Pixel AUPRO": row["pixel_aupro"],
    }


def run_dataset(args, dataset, root, summary):
    initialize_csv(summary)
    done, failures = completed(summary), []
    raw_base = Path(args.output_dir).resolve() / "raw" / dataset
    for category in get_categories(root, args.classes):
        images = sorted(p for p in (root / category / "train" / "good").iterdir() if p.is_file())
        initial = sum((category, str(i)) in done for i in range(len(images)))
        with tqdm(total=len(images), initial=initial, desc="{} {}".format(dataset, category), unit="image") as bar:
            for index, image in enumerate(images):
                if (category, str(index)) in done:
                    continue
                run_root = raw_base / category / "{:06d}_{}".format(index, image.stem)
                started = time.perf_counter()
                process = subprocess.run(
                    patchcore_command(args, root, category, image.name, run_root),
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                )
                elapsed = time.perf_counter() - started
                if process.returncode:
                    error = process.stderr.strip().splitlines()
                    failures.append("{}/{}: {}".format(category, image.name, error[-1] if error else "unknown error"))
                    bar.update(1)
                    continue
                try:
                    row = {
                        "dataset": dataset, "object": category,
                        "Run": "{:03d}_{}".format(index + 1, image.stem),
                        "ref_sample": image.name, "reference_index": index,
                        "elapsed_seconds": "{:.3f}".format(elapsed),
                    }
                    row.update(load_metrics(run_root))
                    append(summary, row)
                    done.add((category, str(index)))
                    tqdm.write(
                        "[done] {}/{} | Image AUROC={} | Image AUPR={} | "
                        "Pixel AUPRO={} | Pixel AUROC={}".format(
                            category,
                            image.name,
                            row["Image AUROC"],
                            row["Image AUPR"],
                            row["Pixel AUPRO"],
                            row["Pixel AUROC"],
                        )
                    )
                    if not args.keep_run_results:
                        shutil.rmtree(run_root, ignore_errors=True)
                except Exception as error:
                    failures.append("{}/{}: {}".format(category, image.name, error))
                bar.update(1)
    return failures


def main():
    args = arguments()
    if args.output_csv and len(args.datasets) != 1:
        raise ValueError("--output-csv requires one dataset.")
    failures = []
    for dataset in args.datasets:
        root = Path(getattr(args, dataset + "_root")).resolve()
        if not root.is_dir():
            raise FileNotFoundError("Dataset root does not exist: {}".format(root))
        summary = Path(args.output_csv).resolve() if args.output_csv else Path(args.output_dir).resolve() / (dataset + "_one_shot.csv")
        failures.extend(run_dataset(args, dataset, root, summary))
        print("{} results: {}".format(dataset, summary))
    if failures:
        print("\nFailed rows were not written and will be retried next run:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
