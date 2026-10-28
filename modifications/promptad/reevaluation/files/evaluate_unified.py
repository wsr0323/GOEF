"""Evaluate PromptAD classification and segmentation checkpoints together."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from unified_evaluation import (
    InferenceBenchmark,
    compute_metrics,
    format_result,
    make_result_row,
    write_results_csv,
)
def str2bool(value):
    return value.lower() in ("yes", "true", "t", "1")


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def checkpoint_path(args, task, category):
    path = (
        Path(args.root_dir)
        / args.dataset
        / f"k_{args.k_shot}"
    )
    if args.run_name:
        path /= args.run_name
    return path / "checkpoint" / f"{task}-Seed_{args.seed}-{category}-check_point.pt"


def build_model(args, category, task, device):
    from PromptAD import PromptAD

    checkpoint = checkpoint_path(args, task, category)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {task} checkpoint: {checkpoint}")

    model = PromptAD(
        out_size_h=args.resolution,
        out_size_w=args.resolution,
        device=str(device),
        backbone=args.backbone,
        pretrained_dataset=args.pretrained_dataset,
        n_ctx=args.n_ctx,
        n_pro=args.cls_n_pro if task == "CLS" else args.seg_n_pro,
        n_ctx_ab=args.n_ctx_ab,
        n_pro_ab=args.n_pro_ab,
        class_name=category,
        k_shot=args.k_shot,
        img_resize=args.img_resize,
        img_cropsize=args.img_cropsize,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval_mode()
    return model


def load_training_peak_memory(paths, dataset, category):
    """Return the maximum per-category training peak across CLS and SEG CSVs."""
    import csv

    result = {'train_peak_allocated_mib': 0.0, 'train_peak_reserved_mib': 0.0}
    row_name = f'{dataset}-{category}'
    for path in paths:
        with Path(path).open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row.get('') != row_name:
                    continue
                for key in result:
                    value = row.get(key, '')
                    if value:
                        result[key] = max(result[key], float(value))
                break
    return result


def evaluate_category(args, category, device, training_csvs):
    import cv2

    from datasets import get_dataloader_from_args

    loader_args = vars(args).copy()
    loader_args["class_name"] = category
    dataloader, _ = get_dataloader_from_args(
        phase="test", perturbed=False, **loader_args
    )
    cls_model = build_model(args, category, "CLS", device)
    seg_model = build_model(args, category, "SEG", device)

    labels = []
    image_scores = []
    masks = []
    anomaly_maps = []
    benchmark = InferenceBenchmark(device)

    with torch.inference_mode():
        for data, mask, label, _name, _img_type in dataloader:
            batch_size = len(data)
            benchmark.start_batch()
            images = torch.stack(
                [cls_model.transform(Image.fromarray(item.numpy())) for item in data]
            ).to(device)
            cls_scores, cls_maps = cls_model(images, "cls")
            seg_maps = seg_model(images, "seg")

            cls_scores = np.asarray(cls_scores, dtype=np.float64)
            cls_map_scores = np.asarray(cls_maps, dtype=np.float64).reshape(
                batch_size, -1
            ).max(axis=1)
            denominator = cls_scores + cls_map_scores
            combined_scores = np.divide(
                cls_scores * cls_map_scores,
                denominator,
                out=np.zeros_like(cls_scores),
                where=denominator != 0,
            )
            image_scores.extend(combined_scores.tolist())
            anomaly_maps.extend(np.asarray(seg_maps, dtype=np.float64))
            benchmark.stop_batch(batch_size)

            labels.extend(label.cpu().numpy().tolist())
            for item in mask.cpu().numpy():
                resized = cv2.resize(
                    (item > 0).astype(np.uint8),
                    (args.resolution, args.resolution),
                    interpolation=cv2.INTER_NEAREST,
                )
                masks.append(resized)

    metrics = compute_metrics(labels, image_scores, masks, anomaly_maps)
    row = make_result_row(category, metrics, benchmark.result())
    row.update(load_training_peak_memory(training_csvs, args.dataset, category))
    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified PromptAD classification and segmentation evaluation"
    )
    parser.add_argument("--dataset", choices=["mvtec", "visa"], default="mvtec")
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--root-dir", default="./result")
    parser.add_argument("--output", default=None)
    parser.add_argument("--data-root", default='')
    parser.add_argument("--run-name", default='')
    parser.add_argument("--training-csv", nargs='*', default=[],
                        help="CLS/SEG training CSVs used to append training peak memory.")
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", type=str2bool, default=False)
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument(
        "--backbone",
        choices=["ViT-B-16-plus-240", "ViT-B-16"],
        default="ViT-B-16-plus-240",
    )
    parser.add_argument("--pretrained-dataset", default="laion400m_e32")
    parser.add_argument("--n-ctx", type=int, default=4)
    parser.add_argument("--n-ctx-ab", type=int, default=1)
    parser.add_argument("--cls-n-pro", type=int, default=3)
    parser.add_argument("--seg-n-pro", type=int, default=1)
    parser.add_argument("--n-pro-ab", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    from datasets import dataset_classes

    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    setup_seed(args.seed)

    available = dataset_classes[args.dataset]
    categories = args.classes or available
    unknown = sorted(set(categories) - set(available))
    if unknown:
        raise ValueError(f"Unknown {args.dataset} categories: {unknown}")

    device = torch.device("cpu" if args.use_cpu else "cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested, but CUDA is not available")

    rows = []
    for category in categories:
        row = evaluate_category(args, category, device, args.training_csv)
        rows.append(row)
        print(format_result(row))

    output = args.output or str(
        Path(args.root_dir)
        / args.dataset
        / f"k_{args.k_shot}"
        / "csv"
        / f"Seed_{args.seed}-unified_results.csv"
    )
    result_path = write_results_csv(output, rows)
    print(f"Unified benchmark saved to {result_path}")


if __name__ == "__main__":
    main()
