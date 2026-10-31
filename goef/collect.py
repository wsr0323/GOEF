"""Collect the raw result tables that back Table I and Table II.

The repository ships the raw numbers so that every reported cell can be audited
without re-running any method:

``results/one_shot/<method>/<dataset>.csv``
    the complete per-reference 1-shot sweep (one row per normal training image)
    of the five methods whose ``mu_g`` is affordable;
``results/centers_raw/<method>/<shot>-shot/<center>_<dataset>.csv``
    the 62 per-centre evaluation files of all eight re-evaluated methods;
``results/MANIFEST.csv``
    one row per collected file with its source, row count and SHA-256.

Files are copied verbatim; only the file names are normalised.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

METRICS = ("Image AUROC", "Image AUPR", "Pixel AUROC", "Pixel AUPRO")
ONE_SHOT_COLUMNS = ("object", "ref_sample", *METRICS)


@dataclass(frozen=True)
class ManifestRow:
    path: str
    kind: str
    method: str
    dataset: str
    shot: str
    source: str
    rows: int
    bytes: int
    sha256: str


MANIFEST_COLUMNS = (
    "path",
    "kind",
    "method",
    "dataset",
    "shot",
    "source",
    "rows",
    "bytes",
    "sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def _record(dest: Path, root: Path, kind: str, method: str, dataset: str, shot: str, source: Path) -> ManifestRow:
    return ManifestRow(
        path=str(dest.relative_to(root)),
        kind=kind,
        method=method,
        dataset=dataset,
        shot=shot,
        source=str(source),
        rows=_row_count(dest),
        bytes=dest.stat().st_size,
        sha256=_sha256(dest),
    )


def _validate_one_shot(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    missing = set(ONE_SHOT_COLUMNS).difference(header)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def collect_one_shot(
    output: Path,
    sources: Mapping[str, Mapping[str, Path]],
    datasets: Sequence[str] | None = None,
) -> list[ManifestRow]:
    """Copy ``method -> dataset -> csv`` into ``<output>/one_shot/<method>/``."""
    root = output
    rows: list[ManifestRow] = []
    for method in sorted(sources):
        for dataset in sorted(sources[method]):
            if datasets and dataset not in datasets:
                continue
            source = Path(sources[method][dataset])
            if not source.is_file():
                raise FileNotFoundError(f"{method}/{dataset}: {source}")
            _validate_one_shot(source)
            dest = root / "one_shot" / method / f"{dataset}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            rows.append(_record(dest, root, "one_shot", method, dataset, "1", source))
    return rows


def split_combined_one_shot(
    source: Path,
    output: Path,
    method: str,
    dataset_column: str = "dataset",
    aliases: Mapping[str, str] | None = None,
) -> list[ManifestRow]:
    """Split a CSV that holds several benchmarks (INP-Former) per dataset."""
    aliases = {key.casefold(): value for key, value in (aliases or {}).items()}
    root = output
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if dataset_column not in header:
            raise ValueError(f"{source}: missing {dataset_column!r} column")
        buckets: dict[str, list[dict[str, str]]] = {}
        for row in reader:
            raw = (row.get(dataset_column) or "").strip()
            key = aliases.get(raw.casefold(), raw)
            buckets.setdefault(key, []).append(row)
    rows = []
    for dataset, items in buckets.items():
        dest = root / "one_shot" / method / f"{dataset}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(items)
        rows.append(_record(dest, root, "one_shot", method, dataset, "1", source))
    return rows


def collect_centers(
    output: Path,
    methods: Mapping[str, Path],
    shots: Sequence[int] = (1, 2, 3, 4, 5),
) -> list[ManifestRow]:
    """Copy every per-centre result CSV of every method."""
    root = output
    rows: list[ManifestRow] = []
    for method in sorted(methods):
        base = Path(methods[method])
        results = base / "results" if (base / "results").is_dir() else base
        found = 0
        for shot in shots:
            shot_dir = results / f"{shot}-shot"
            if not shot_dir.is_dir():
                continue
            for source in sorted(shot_dir.glob("*.csv")):
                name = source.name
                if name.endswith(("_cls.csv", "_seg.csv")):
                    continue
                # keep the unified schema when the project writes both
                if name.endswith("_unified.csv"):
                    pass
                elif (shot_dir / f"{source.stem}_unified.csv").is_file():
                    continue
                dataset = "mvtec" if name.endswith("_mvtec.csv") or "_mvtec_" in name else "visa"
                dest = root / "centers_raw" / method / f"{shot}-shot" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                rows.append(_record(dest, root, "center", method, dataset, str(shot), source))
                found += 1
        if not found:
            print(f"WARNING: no centre result found for {method} under {results}")
    return rows


def write_manifest(path: Path, rows: Iterable[ManifestRow]) -> Path:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in MANIFEST_COLUMNS})
    return path
