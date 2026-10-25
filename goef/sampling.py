"""Per-category representative selection and artifact writing.

This is the stage that turns a normal-image training set into the few reference
images used by GOEF.  Every run writes a self-describing directory::

    <out_dir>/
        run_config.json
        summary.csv                     # method, category, cluster_id, count, representative_image
        <category>/
            selected_samples.csv        # sample_id, filename, full_path
            representative_mapping.csv  # cluster_id, original_filename, original_path, saved_path
            representatives/            # one copy per representative (optional)
            clusters.csv                # path, filename, cluster   (clustering methods only)
            clusters_preview.jpg        # optional visual sanity check
            clustered_images/           # optional, only with --copy-images

``summary.csv`` is the only file required by the later stages of GOEF: the
``count`` column is the cluster size ``N_j`` of Eq. (2) and the
``representative_image`` column identifies ``I_bar_j``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .features import (
    extract_embeddings,
    list_images,
    load_encoder,
    load_image,
    make_transform,
    resolve_device,
)
from .selection import METHODS, choose_k, cluster


# --------------------------------------------------------------------------- #
# Small CSV helpers (BOM keeps the files readable in Excel)
# --------------------------------------------------------------------------- #
def write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _unique_destination(source: Path, destination_dir: Path) -> Path:
    destination = destination_dir / source.name
    if not destination.exists():
        return destination
    suffix = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    return destination_dir / f"{source.stem}_{suffix}{source.suffix}"


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def copy_representatives(
    image_paths: list[Path],
    representative_indices: np.ndarray,
    out_dir: Path,
    copy_images: bool = True,
):
    """Record (and optionally copy) the representative image of every cluster."""
    representative_dir = out_dir / "representatives"
    rows = []
    for cluster_id, image_index in enumerate(representative_indices):
        source = image_paths[int(image_index)]
        destination = representative_dir / f"cluster_{cluster_id:02d}_{source.name}"
        if copy_images:
            representative_dir.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                suffix = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
                destination = representative_dir / (
                    f"cluster_{cluster_id:02d}_{source.stem}_{suffix}{source.suffix}"
                )
            shutil.copy2(source, destination)
        rows.append((cluster_id, source.name, str(source), str(destination)))
    write_csv(
        out_dir / "representative_mapping.csv",
        ("cluster_id", "original_filename", "original_path", "saved_path"),
        rows,
    )
    write_csv(
        out_dir / "selected_samples.csv",
        ("sample_id", "filename", "full_path"),
        ((cluster_id, name, source) for cluster_id, name, source, _ in rows),
    )
    return rows


def copy_clusters(image_paths: list[Path], labels: np.ndarray, out_dir: Path) -> None:
    cluster_root = out_dir / "clustered_images"
    for source, label in zip(image_paths, labels):
        destination_dir = cluster_root / f"cluster_{int(label):02d}"
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, _unique_destination(source, destination_dir))


def _get_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_preview(
    image_paths: list[Path],
    labels: np.ndarray,
    out_path: Path,
    thumb_size: int = 160,
    cols: int = 6,
) -> None:
    """Render one row per cluster so the partition can be inspected by eye."""
    padding, header_height = 12, 42
    groups: dict[int, list[Path]] = {cluster_id: [] for cluster_id in sorted(set(map(int, labels)))}
    for path, label in zip(image_paths, labels):
        groups[int(label)].append(path)
    width = cols * thumb_size + (cols + 1) * padding
    height = sum(
        header_height + padding + math.ceil(len(items) / cols) * (thumb_size + padding)
        for items in groups.values()
    )
    canvas = Image.new("RGB", (width, max(height, 200)), (245, 245, 245))
    draw, y = ImageDraw.Draw(canvas), 0
    title_font, name_font = _get_font(20), _get_font(11)
    for cluster_id, items in groups.items():
        draw.rectangle((0, y, width, y + header_height), fill=(220, 220, 220))
        draw.text(
            (padding, y + 8),
            f"Cluster {cluster_id:02d}  ({len(items)})",
            fill=(0, 0, 0),
            font=title_font,
        )
        y += header_height + padding
        for index, path in enumerate(items):
            x = padding + (index % cols) * (thumb_size + padding)
            tile_y = y + (index // cols) * (thumb_size + padding)
            image = load_image(path)
            image.thumbnail((thumb_size, thumb_size))
            tile = Image.new("RGB", (thumb_size, thumb_size), "white")
            tile.paste(
                image,
                ((thumb_size - image.width) // 2, (thumb_size - image.height) // 2),
            )
            canvas.paste(tile, (x, tile_y))
            draw.rectangle(
                (x, tile_y + thumb_size - 20, x + thumb_size, tile_y + thumb_size),
                fill="white",
            )
            name = path.name if len(path.name) <= 22 else path.name[:19] + "..."
            draw.text((x + 3, tile_y + thumb_size - 17), name, fill="black", font=name_font)
        y += math.ceil(len(items) / cols) * (thumb_size + padding)
    canvas.save(out_path, quality=92)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def validate_k(k: int, sample_count: int, max_k: int) -> int:
    requested = max_k if k <= 0 else k
    if requested <= 0:
        raise ValueError("k or max_k must be positive")
    return min(requested, sample_count)


def process_category(
    image_dir: Path,
    out_dir: Path,
    args,
    model=None,
    transform=None,
    copy_images: bool = False,
):
    """Select representatives for one category; return summary rows."""
    image_paths = list_images(image_dir, recursive=not args.no_recursive)
    if not image_paths:
        print(f"WARNING: no image under {image_dir}, skipped")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    k = validate_k(args.k, len(image_paths), args.max_k)
    print(f"\n{image_dir}: {len(image_paths)} images -> {out_dir}")

    if args.method == "random":
        representative_indices = np.asarray(
            random.Random(args.seed).sample(range(len(image_paths)), k), dtype=int
        )
        rows = copy_representatives(image_paths, representative_indices, out_dir, copy_images)
        return [(index, 1, name) for index, name, _, _ in rows]

    features = extract_embeddings(model, image_paths, transform, args.device, args.batch_size)

    if args.method == "coreset":
        # A pure sampler: no partition, hence no clustering artifacts.
        from .selection import run_coreset

        representative_indices = run_coreset(features, k, args.seed, args.coreset_dim)
        rows = copy_representatives(image_paths, representative_indices, out_dir, copy_images)
        return [(index, 1, name) for index, name, _, _ in rows]

    if args.k <= 0:
        k = choose_k(features, args.method, args.max_k, args.seed, args.coreset_dim)
    labels, representative_indices = cluster(
        features, args.method, k, args.seed, args.coreset_dim
    )
    representative_rows = copy_representatives(
        image_paths, representative_indices, out_dir, copy_images
    )
    write_csv(
        out_dir / "clusters.csv",
        ("path", "filename", "cluster"),
        sorted(
            ((str(path), path.name, int(label)) for path, label in zip(image_paths, labels)),
            key=lambda row: (row[2], row[1]),
        ),
    )
    if args.preview:
        make_preview(image_paths, labels, out_dir / "clusters_preview.jpg", args.thumb_size, args.cols)
    if copy_images:
        copy_clusters(image_paths, labels, out_dir)

    representatives = {cluster_id: name for cluster_id, name, _, _ in representative_rows}
    return [
        (cluster_id, int(np.sum(labels == cluster_id)), representatives[cluster_id])
        for cluster_id in range(k)
    ]


def parse_categories(value: str | None):
    return None if not value else {item.strip() for item in value.split(",") if item.strip()}


def run(args) -> Path:
    """Execute one selection run and return its output directory."""
    if args.batch_size <= 0 or args.cols <= 0 or args.thumb_size <= 0:
        raise ValueError("batch_size, cols and thumb_size must be positive")
    if args.coreset_dim < 0:
        raise ValueError("coreset_dim must be non-negative")
    if args.method != "random" and args.weights is None:
        raise ValueError(f"{args.method} requires --weights (the DINOv3 checkpoint)")
    for path in (args.image_dir, args.dataset_root, args.weights):
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"path does not exist: {path}")

    model = transform = None
    if args.method != "random":
        args.device = resolve_device(args.device)
        model = load_encoder(Path(args.weights), args.model_name, args.device)
        transform = make_transform(args.image_size)
    elif args.device == "auto":
        args.device = "unused"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[tuple] = []

    if args.dataset_root:
        allowed = parse_categories(args.categories)
        dataset_root = Path(args.dataset_root)
        categories = sorted(
            path
            for path in dataset_root.iterdir()
            if path.is_dir() and (allowed is None or path.name in allowed)
        )
        if not categories:
            raise RuntimeError(f"no category under {dataset_root}")
        for category_dir in categories:
            image_dir = category_dir / args.sub_path
            if not image_dir.exists():
                print(f"WARNING: skip {category_dir.name}, missing {image_dir}")
                continue
            for cluster_id, count, representative in process_category(
                image_dir,
                out_dir / category_dir.name,
                args,
                model,
                transform,
                copy_images=args.copy_images,
            ):
                summary.append((args.method, category_dir.name, cluster_id, count, representative))
    else:
        image_dir = Path(args.image_dir)
        for cluster_id, count, representative in process_category(
            image_dir,
            out_dir,
            args,
            model,
            transform,
            copy_images=args.copy_images,
        ):
            summary.append((args.method, image_dir.name, cluster_id, count, representative))

    if not summary:
        raise RuntimeError("no representative was selected")
    write_csv(
        out_dir / "summary.csv",
        ("method", "category", "cluster_id", "count", "representative_image"),
        summary,
    )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    print(f"\ndone: {out_dir}")
    return out_dir
