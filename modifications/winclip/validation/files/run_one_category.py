import argparse
import csv
import json
import logging
import os
import sys
import time
from types import SimpleNamespace

import torch
from tqdm import tqdm

from reproduce_WinCLIP import SingleClassEvaluator, setup_seed


# Match the per-reference layout used by anomalydinovisa.csv.
CSV_COLUMNS = [
    "object", "ref_sample", "reference_index", "Image AUROC", "Image AUPR",
    "Pixel AUPRO", "Pixel AUROC",
]
DATASET_DEFAULT_ROOTS = {
    "mvtec": "/home/wsr/mvtec_anomaly_detection",
    "visa": "./data/visa",
}


def completed_references(csv_path):
    """Return references already recorded so an interrupted run can resume."""
    if not os.path.isfile(csv_path):
        return set()
    with open(csv_path, newline="", encoding="utf-8") as file:
        return {
            (row["object"], row["reference_index"])
            for row in csv.DictReader(file)
        }


def ensure_csv(csv_path):
    """Create the summary file immediately, even before the first result exists."""
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=CSV_COLUMNS).writeheader()


def create_logger(csv_path):
    """Write persistent progress records without interfering with tqdm bars."""
    logger = logging.getLogger(f"one_shot.{os.path.abspath(csv_path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.FileHandler(f"{csv_path}.log", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def log_progress(logger, message):
    logger.info(message)
    tqdm.write(message)


def append_result(csv_path, class_name, image_path, reference_index, metrics):
    row = {
        "object": class_name,
        "ref_sample": os.path.basename(image_path),
        "reference_index": reference_index,
        "Image AUROC": metrics["auroc_sp"],
        "Image AUPR": metrics["ap_sp"],
        "Pixel AUPRO": metrics["aupro"],
        "Pixel AUROC": metrics["auroc_px"],
    }
    write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def process_class(
    dataset_name, dataset_root, class_name, train_samples, csv_path, done,
    progress, logger, batch_size,
):
    """Keep one evaluator alive while every support image is tested in turn."""
    failures = []
    pending = [
        (reference_index, sample)
        for reference_index, sample in enumerate(train_samples)
        if (class_name, str(reference_index)) not in done
    ]
    if not pending:
        log_progress(logger, f"[{class_name}] all {len(train_samples)} references already exist in the CSV.")
        return failures

    evaluator_args = SimpleNamespace(
        dataset=dataset_name,
        data_path=dataset_root,
        class_name=class_name,
        model="ViT-B-16-plus-240",
        image_size=240,
        k_shot=1,
        few_shot_features=[3, 6, 9],
        batch_size=batch_size,
    )
    setup_seed(42)
    try:
        log_progress(logger, f"[{class_name}] loading model, text features, and test data once.")
        evaluator = SingleClassEvaluator(evaluator_args)
    except Exception as error:
        progress.update(len(pending))
        log_progress(logger, f"[{class_name}] evaluator initialization failed: {error}")
        return [f"{class_name} evaluator initialization failed: {error}"]
    try:
        for reference_index, sample in pending:
            image_path = os.path.join(dataset_root, sample["img_path"])
            try:
                if not os.path.isfile(image_path):
                    failures.append(f"Missing support image: {image_path}")
                    continue
                log_progress(
                    logger,
                    f"[{class_name} {reference_index + 1}/{len(train_samples)}] "
                    f"loading support {os.path.basename(image_path)}, then testing.",
                )
                started_at = time.perf_counter()
                metrics = evaluator.evaluate_support(image_path)
                append_result(csv_path, class_name, image_path, reference_index, metrics)
                done.add((class_name, str(reference_index)))
                elapsed = time.perf_counter() - started_at
                log_progress(
                    logger,
                    f"[{class_name} {reference_index + 1}/{len(train_samples)}] complete in {elapsed:.1f}s | "
                    f"Image AUROC={metrics['auroc_sp']:.4f}, Image AUPR={metrics['ap_sp']:.4f}, "
                    f"Pixel AUPRO={metrics['aupro']:.4f}, Pixel AUROC={metrics['auroc_px']:.4f}",
                )
            except Exception as error:
                failures.append(f"{class_name}/{os.path.basename(image_path)} failed: {error}")
                log_progress(logger, f"[{class_name} {reference_index + 1}/{len(train_samples)}] failed: {error}")
            finally:
                progress.update(1)
    finally:
        del evaluator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return failures


def resolve_root(args, dataset_name):
    if args.dataset_root:
        if len(args.datasets) != 1:
            raise ValueError("--dataset_root can only be used when one dataset is selected.")
        return args.dataset_root
    return getattr(args, f"{dataset_name}_root")


def output_path(args, dataset_name):
    if args.output_csv:
        if len(args.datasets) != 1:
            raise ValueError("--output_csv can only be used when one dataset is selected.")
        return args.output_csv
    return os.path.join(args.output_base, f"anomalydino{dataset_name}.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Run 1-shot cross-validation and write one CSV row per support image."
    )
    parser.add_argument("--datasets", nargs="+", choices=("mvtec", "visa"), default=["mvtec"])
    parser.add_argument("--classes", nargs="+", help="Optional category subset. Defaults to every category in meta.json.")
    parser.add_argument("--dataset_root", help="Compatibility option for a single selected dataset.")
    parser.add_argument("--mvtec_root", default=DATASET_DEFAULT_ROOTS["mvtec"])
    parser.add_argument("--visa_root", default=DATASET_DEFAULT_ROOTS["visa"])
    parser.add_argument("--output_base", default="./results")
    parser.add_argument("--output_csv", help="Output CSV path for a single selected dataset.")
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Test-feature batch size. Lower values reduce peak GPU memory (default: 32).",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1.")

    all_failures = []
    for dataset_name in args.datasets:
        dataset_root = resolve_root(args, dataset_name)
        meta_path = os.path.join(dataset_root, "meta.json")
        if not os.path.isfile(meta_path):
            print(f"Error: {meta_path} not found.", file=sys.stderr)
            all_failures.append(f"{dataset_name}: dataset metadata not found")
            continue

        with open(meta_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        available_classes = list(metadata["train"])
        classes = [name for name in (args.classes or available_classes) if name in available_classes]
        csv_path = output_path(args, dataset_name)
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        ensure_csv(csv_path)
        logger = create_logger(csv_path)
        done = completed_references(csv_path)
        log_progress(logger, f"Starting {dataset_name}: {len(done)} completed references found in {csv_path}.")
        for class_name in classes:
            train_samples = metadata["train"].get(class_name, [])
            completed_count = sum(
                (class_name, str(reference_index)) in done
                for reference_index in range(len(train_samples))
            )
            with tqdm(
                total=len(train_samples),
                initial=completed_count,
                desc=f"{dataset_name.upper()} {class_name}",
                unit="image",
            ) as progress:
                all_failures.extend(
                    process_class(
                        dataset_name, dataset_root, class_name, train_samples,
                        csv_path, done, progress, logger, args.batch_size,
                    )
                )

        print(f"\n{dataset_name.upper()} per-reference results: {csv_path}")
        log_progress(logger, f"Finished {dataset_name}: results saved to {csv_path}.")

    if all_failures:
        print("\nCompleted with failures:", file=sys.stderr)
        print("\n".join(all_failures), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0 python run_one_category.py \
#     --datasets mvtec \
#     --mvtec_root $GOEF_DATA/mvtec_anomaly_detection \
#     --output_csv results/mvtec.csv
