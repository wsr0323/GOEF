from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.init import trunc_normal_
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset import MVTecDataset, get_data_transforms
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Aggregation_Block, Mlp, Prototype_Block
from optimizers import StableAdamW
from utils import (
    WarmCosineScheduler,
    augmentation,
    evaluation_batch,
    global_cosine_hm_adaptive,
    setup_seed,
)


LOGGER = logging.getLogger("one_image_sweep")

_DATA = os.environ.get("GOEF_DATA", "data")
DATASETS = {
    "mvtec": {
        "name": "MVTec-AD",
        "default_path": os.path.join(_DATA, "mvtec_anomaly_detection"),
    },
    "visa": {
        "name": "VisA",
        "default_path": os.path.join(_DATA, "VisA_pytorch/1cls"),
    },
}

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".pgm",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}

RESULT_FIELDS = [
    "dataset",
    "object",
    "Run",
    "ref_sample",
    "参考索引",
    "ref_path",
    "Image AUROC",
    "Image AUPR",
    "Image F1",
    "Pixel AUROC",
    "Pixel AUPR",
    "Pixel F1",
    "Pixel AUPRO",
    "config_id",
    "encoder",
    "input_size",
    "crop_size",
    "INP_num",
    "total_epochs",
    "batch_size",
    "seed",
    "completed_at",
]

ERROR_FIELDS = [
    "dataset",
    "object",
    "Run",
    "ref_sample",
    "参考索引",
    "ref_path",
    "config_id",
    "error",
    "failed_at",
]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    root: Path


@dataclass(frozen=True)
class Experiment:
    dataset: DatasetSpec
    category: str
    reference_index: int
    reference_count: int
    reference_path: Path

    @property
    def relative_path(self) -> str:
        return self.reference_path.relative_to(self.dataset.root).as_posix()

    @property
    def run_name(self) -> str:
        width = max(3, len(str(self.reference_count)))
        return f"{self.reference_index + 1:0{width}d}_{self.reference_path.stem}"


def natural_key(value: str) -> List[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def discover_categories(root: Path) -> List[str]:
    return sorted(
        [
            path.name
            for path in root.iterdir()
            if path.is_dir() and (path / "train" / "good").is_dir() and (path / "test").is_dir()
        ],
        key=natural_key,
    )


def discover_reference_images(root: Path, category: str) -> List[Path]:
    train_dir = root / category / "train" / "good"
    images = [
        path
        for path in train_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda path: natural_key(path.name))


def build_experiments(args: argparse.Namespace, specs: Sequence[DatasetSpec]) -> List[Experiment]:
    requested_categories = set(args.categories or [])
    found_categories = set()
    experiments: List[Experiment] = []

    for spec in specs:
        categories = discover_categories(spec.root)
        if requested_categories:
            categories = [category for category in categories if category in requested_categories]
            found_categories.update(categories)

        for category in categories:
            references = discover_reference_images(spec.root, category)
            start = args.start_index
            end = len(references) if args.end_index is None else min(args.end_index, len(references))
            for index in range(start, end):
                experiments.append(
                    Experiment(
                        dataset=spec,
                        category=category,
                        reference_index=index,
                        reference_count=len(references),
                        reference_path=references[index],
                    )
                )

    missing_categories = requested_categories - found_categories
    if missing_categories:
        missing = ", ".join(sorted(missing_categories, key=natural_key))
        raise ValueError(f"Categories not found in the selected datasets: {missing}")

    return experiments


def config_values(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "encoder": args.encoder,
        "input_size": args.input_size,
        "crop_size": args.crop_size,
        "INP_num": args.INP_num,
        "total_epochs": args.total_epochs,
        "batch_size": args.batch_size,
        "drop_last": args.drop_last,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "final_learning_rate": args.final_learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_iters": args.warmup_iters,
        "max_ratio": args.max_ratio,
        "eval_resize": args.eval_resize,
    }


def make_config_id(args: argparse.Namespace) -> str:
    payload = json.dumps(config_values(args), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def read_csv_rows(path: Path, expected_fields: Sequence[str]) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(expected_fields):
            raise ValueError(
                f"Unexpected CSV header in {path}. Use a new --output-csv path or restore the expected header."
            )
        return list(reader)


def write_csv_atomic(path: Path, rows: Iterable[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def result_key(config_id: str, experiment: Experiment) -> Tuple[str, str, str, str]:
    return config_id, experiment.dataset.name, experiment.category, experiment.relative_path


def row_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return row["config_id"], row["dataset"], row["object"], row["ref_path"]


def create_model(
    args: argparse.Namespace, encoder: nn.Module, device: torch.device
) -> Tuple[INP_Former, nn.ModuleList]:
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    if "small" in args.encoder:
        embed_dim, num_heads = 384, 6
    elif "base" in args.encoder:
        embed_dim, num_heads = 768, 12
    elif "large" in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Encoder name must contain small, base, or large")

    bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)])
    inp = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim))])
    extractor = nn.ModuleList(
        [
            Aggregation_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
        ]
    )
    decoder = nn.ModuleList(
        [
            Prototype_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
            for _ in range(8)
        ]
    )

    model = INP_Former(
        encoder=encoder,
        bottleneck=bottleneck,
        aggregation=extractor,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        prototype_token=inp,
    ).to(device)

    trainable = nn.ModuleList([bottleneck, decoder, extractor, inp])
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    return model, trainable


def create_train_loader(
    args: argparse.Namespace, reference_path: Path, data_transform
) -> DataLoader:
    with Image.open(reference_path) as image:
        image_tensor = data_transform(image.convert("RGB"))

    augmented_images = augmentation(image_tensor)
    if args.drop_last and len(augmented_images) < args.batch_size:
        raise ValueError(
            f"batch_size={args.batch_size} exceeds the {len(augmented_images)} augmented samples "
            "while drop_last is enabled"
        )

    dataset = TensorDataset(
        augmented_images,
        torch.zeros(len(augmented_images), dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # The augmented samples already live in memory. Worker processes would
        # be restarted on every epoch and add substantial overhead here.
        num_workers=0,
        drop_last=args.drop_last,
        pin_memory=True,
    )


def create_test_loader(
    args: argparse.Namespace, experiment: Experiment, data_transform, gt_transform
) -> DataLoader:
    dataset = MVTecDataset(
        root=str(experiment.dataset.root / experiment.category),
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def train_and_evaluate(
    args: argparse.Namespace,
    experiment: Experiment,
    encoder: nn.Module,
    device: torch.device,
    data_transform,
    test_loader: DataLoader,
) -> List[float]:
    # Resetting the seed here makes resumed runs identical to uninterrupted runs.
    setup_seed(args.seed)
    train_loader = create_train_loader(args, experiment.reference_path, data_transform)
    model, trainable = create_model(args, encoder, device)

    optimizer = StableAdamW(
        [{"params": trainable.parameters()}],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        amsgrad=True,
        eps=1e-10,
    )
    total_iters = args.total_epochs * len(train_loader)
    warmup_iters = min(args.warmup_iters, max(total_iters - 1, 0))
    scheduler = WarmCosineScheduler(
        optimizer,
        base_value=args.learning_rate,
        final_value=args.final_learning_rate,
        total_iters=total_iters,
        warmup_iters=warmup_iters,
    )

    for epoch in range(args.total_epochs):
        model.train()
        losses = []
        batches = tqdm(
            train_loader,
            desc=f"train {experiment.dataset.key}/{experiment.category}/{experiment.reference_path.name}",
            ncols=100,
            leave=False,
            disable=not args.progress,
        )
        for image, _ in batches:
            image = image.to(device, non_blocking=True)
            encoded, decoded, gather_loss = model(image)
            loss = global_cosine_hm_adaptive(encoded, decoded, y=3) + 0.2 * gather_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch + 1 == args.total_epochs:
            LOGGER.info(
                "  epoch %d/%d loss=%.6f",
                epoch + 1,
                args.total_epochs,
                float(np.mean(losses)),
            )

    metrics = evaluation_batch(
        model,
        test_loader,
        device,
        max_ratio=args.max_ratio,
        resize_mask=args.eval_resize,
    )
    return [float(value) for value in metrics]


def make_result_row(
    args: argparse.Namespace,
    config_id: str,
    experiment: Experiment,
    metrics: Sequence[float],
) -> Dict[str, object]:
    (
        image_auroc,
        image_aupr,
        image_f1,
        pixel_auroc,
        pixel_aupr,
        pixel_f1,
        pixel_aupro,
    ) = metrics
    return {
        "dataset": experiment.dataset.name,
        "object": experiment.category,
        "Run": experiment.run_name,
        "ref_sample": experiment.reference_path.name,
        "参考索引": experiment.reference_index,
        "ref_path": experiment.relative_path,
        "Image AUROC": f"{image_auroc:.6f}",
        "Image AUPR": f"{image_aupr:.6f}",
        "Image F1": f"{image_f1:.6f}",
        "Pixel AUROC": f"{pixel_auroc:.6f}",
        "Pixel AUPR": f"{pixel_aupr:.6f}",
        "Pixel F1": f"{pixel_f1:.6f}",
        "Pixel AUPRO": f"{pixel_aupro:.6f}",
        "config_id": config_id,
        "encoder": args.encoder,
        "input_size": args.input_size,
        "crop_size": args.crop_size,
        "INP_num": args.INP_num,
        "total_epochs": args.total_epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def make_error_row(config_id: str, experiment: Experiment, error: Exception) -> Dict[str, object]:
    return {
        "dataset": experiment.dataset.name,
        "object": experiment.category,
        "Run": experiment.run_name,
        "ref_sample": experiment.reference_path.name,
        "参考索引": experiment.reference_index,
        "ref_path": experiment.relative_path,
        "config_id": config_id,
        "error": f"{type(error).__name__}: {error}",
        "failed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate INP-Former independently with every normal training image."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=["mvtec", "visa"],
        help="Datasets to sweep (default: both).",
    )
    parser.add_argument(
        "--mvtec-data-path",
        type=Path,
        default=Path(DATASETS["mvtec"]["default_path"]),
    )
    parser.add_argument(
        "--visa-data-path",
        type=Path,
        default=Path(DATASETS["visa"]["default_path"]),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Only run these categories, for example: --categories bottle cable.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Inclusive index per category.")
    parser.add_argument("--end-index", type=int, help="Exclusive index per category.")
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Limit the number of pending experiments in this invocation.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("saved_results/one_image_sweep/results.csv"),
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run entries already present in the CSV again and replace their rows.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record an error and continue with the next image instead of stopping.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List the experiment counts without loading the model or training.",
    )

    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--input-size", "--input_size", dest="input_size", type=int, default=448)
    parser.add_argument("--crop-size", "--crop_size", dest="crop_size", type=int, default=392)
    parser.add_argument("--INP-num", "--INP_num", dest="INP_num", type=int, default=6)
    parser.add_argument("--total-epochs", "--total_epochs", dest="total_epochs", type=int, default=200)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--final-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--max-ratio", type=float, default=0.01)
    parser.add_argument("--eval-resize", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=20)
    drop_last_group = parser.add_mutually_exclusive_group()
    drop_last_group.add_argument(
        "--drop-last",
        dest="drop_last",
        action="store_true",
        help="Match the original few-shot DataLoader behavior (default).",
    )
    drop_last_group.add_argument(
        "--no-drop-last",
        dest="drop_last",
        action="store_false",
        help="Keep an incomplete final training batch.",
    )
    parser.set_defaults(drop_last=True)

    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Show per-batch progress bars (default).",
    )
    progress_group.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Hide per-batch progress bars.",
    )
    parser.set_defaults(progress=True)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < args.start_index:
        parser.error("--end-index must be greater than or equal to --start-index")
    for name in ("total_epochs", "batch_size", "log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.warmup_iters < 0:
        parser.error("--num-workers and --warmup-iters must be non-negative")
    if args.max_runs is not None and args.max_runs <= 0:
        parser.error("--max-runs must be positive")
    return args


def setup_logging(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_csv.with_suffix(".log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def main() -> int:
    args = parse_args()
    setup_logging(args.output_csv)

    specs = []
    for key in args.datasets:
        root = args.mvtec_data_path if key == "mvtec" else args.visa_data_path
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {root}")
        specs.append(DatasetSpec(key=key, name=DATASETS[key]["name"], root=root))

    experiments = build_experiments(args, specs)
    config_id = make_config_id(args)
    result_rows = read_csv_rows(args.output_csv, RESULT_FIELDS)
    completed = {row_key(row) for row in result_rows}

    pending = [
        experiment
        for experiment in experiments
        if args.rerun_completed or result_key(config_id, experiment) not in completed
    ]
    if args.max_runs is not None:
        pending = pending[: args.max_runs]

    LOGGER.info(
        "config=%s selected=%d completed=%d pending_this_run=%d output=%s",
        config_id,
        len(experiments),
        len(experiments) - sum(result_key(config_id, item) not in completed for item in experiments),
        len(pending),
        args.output_csv.resolve(),
    )
    for spec in specs:
        count = sum(experiment.dataset == spec for experiment in experiments)
        LOGGER.info("  %s: %d selected reference images", spec.name, count)

    if args.list_only or not pending:
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "This project's evaluation_batch uses the CUDA-only adeval accumulator; use a CUDA device."
        )
    torch.cuda.set_device(device)
    setup_seed(args.seed)

    LOGGER.info("Loading frozen encoder %s on %s", args.encoder, device)
    encoder = vit_encoder.load(args.encoder).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    error_csv = args.output_csv.with_name(f"{args.output_csv.stem}_errors.csv")
    error_rows = read_csv_rows(error_csv, ERROR_FIELDS)
    current_test_key = None
    test_loader = None

    try:
        for run_index, experiment in enumerate(pending, start=1):
            LOGGER.info(
                "[%d/%d] %s/%s index=%d image=%s run=%s",
                run_index,
                len(pending),
                experiment.dataset.name,
                experiment.category,
                experiment.reference_index,
                experiment.reference_path.name,
                experiment.run_name,
            )

            test_key = (experiment.dataset.name, experiment.category)
            if test_key != current_test_key:
                test_loader = create_test_loader(args, experiment, data_transform, gt_transform)
                current_test_key = test_key

            try:
                metrics = train_and_evaluate(
                    args,
                    experiment,
                    encoder,
                    device,
                    data_transform,
                    test_loader,
                )
                new_row = make_result_row(args, config_id, experiment, metrics)
                key = result_key(config_id, experiment)
                result_rows = [row for row in result_rows if row_key(row) != key]
                result_rows.append(new_row)
                write_csv_atomic(args.output_csv, result_rows, RESULT_FIELDS)
                completed.add(key)
                LOGGER.info(
                    "  result I-AUROC=%.4f I-AUPR=%.4f P-AUROC=%.4f P-AUPRO=%.4f (saved)",
                    metrics[0],
                    metrics[1],
                    metrics[3],
                    metrics[6],
                )
            except Exception as error:
                LOGGER.exception("Experiment failed: %s", experiment.relative_path)
                error_rows.append(make_error_row(config_id, experiment, error))
                write_csv_atomic(error_csv, error_rows, ERROR_FIELDS)
                if not args.continue_on_error:
                    raise
            finally:
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        LOGGER.warning(
            "Interrupted. Completed rows are already saved; the unfinished image will be retried next time."
        )
        return 130

    LOGGER.info("Sweep invocation finished: %d successful rows for config %s", len(completed), config_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 完整运行：
#
  # cd <INP-Former checkout>
  #
  # conda run -n inpformer python INP_Former_One_Image_Sweep.py \
  #   --datasets mvtec visa \
  #   --output-csv saved_results/one_image_sweep/results.csv
#
#   中断后直接执行同一条命令即可续跑。
#
#   建议先进行一次小规模测试：
#
  # conda run -n inpformer python INP_Former_One_Image_Sweep.py \
  #   --datasets mvtec \
  #   --categories bottle \
  #   --start-index 0 \
  #   --end-index 1 \
  #   --total-epochs 1 \
  #   --output-csv /tmp/inpformer_smoke.csv
#
#   仅查看任务数量，不训练：
#
#   conda run -n inpformer python INP_Former_One_Image_Sweep.py --list-only
