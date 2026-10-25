"""Materialise the k-shot "center" datasets used by Sec. IV.

GOEF's k-shot evaluation needs one dataset per *combination* of cluster
representatives.  For ``n = 5`` clusters per category this gives

===========  =========================================  ==================
shot         directory names                            combinations
===========  =========================================  ==================
1            ``center_01`` .. ``center_05``             5
2            ``centers_01_02`` .. ``centers_04_05``     10
3            ``centers_01_02_03`` .. ``centers_03_04_05``  10
4            ``centers_01_02_03_04`` .. ``centers_02_03_04_05`` 5
5            ``centers_01_02_03_04_05``                 1
===========  =========================================  ==================

Every output directory is a drop-in MVTec/VisA style dataset: the ``train``
split of each category holds exactly the selected representative(s), while the
complete ``test`` split and all ground-truth masks are preserved.  A
``meta.json`` is written in the format consumed by the eight evaluated FS-IAD
methods, together with ``representatives.csv`` and ``dataset_info.json`` for
provenance.
"""

from __future__ import annotations

import csv
import filecmp
import itertools
import json
import os
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .features import IMAGE_EXTENSIONS

META_FIELDS = ("img_path", "mask_path", "cls_name", "specie_name", "anomaly")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_root: Path
    cluster_root: Path


@dataclass(frozen=True)
class CenterPlan:
    spec: DatasetSpec
    center_numbers: tuple[int, ...]
    cluster_ids: tuple[int, ...]
    meta: dict[str, dict[str, list[dict[str, Any]]]]
    representatives: dict[str, dict[int, str]]
    representative_artifacts: dict[str, dict[int, Path]]

    @property
    def directory_name(self) -> str:
        prefix = "center" if len(self.center_numbers) == 1 else "centers"
        numbers = "_".join(f"{number:02d}" for number in self.center_numbers)
        return f"{prefix}_{numbers}"


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def resolve_meta_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_representative_mappings(
    spec: DatasetSpec,
    k: int,
    check_content: bool = True,
) -> tuple[dict[str, dict[int, str]], dict[str, dict[int, Path]]]:
    """Validate every category's cluster partition and representative images."""
    if not spec.source_root.is_dir():
        raise FileNotFoundError(f"{spec.name}: dataset root not found: {spec.source_root}")
    if not spec.cluster_root.is_dir():
        raise FileNotFoundError(f"{spec.name}: selection root not found: {spec.cluster_root}")

    category_dirs = sorted(
        path
        for path in spec.cluster_root.iterdir()
        if path.is_dir() and (path / "representative_mapping.csv").is_file()
    )
    if not category_dirs:
        raise ValueError(f"{spec.name}: no <category>/representative_mapping.csv under {spec.cluster_root}")

    mappings: dict[str, dict[int, str]] = {}
    artifacts: dict[str, dict[int, Path]] = {}
    expected_ids = set(range(k))

    for category_dir in category_dirs:
        category = category_dir.name
        rows: dict[int, str] = {}
        mapping_file = category_dir / "representative_mapping.csv"
        with mapping_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"cluster_id", "original_filename"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{mapping_file}: expected columns {sorted(required)}")
            for row in reader:
                cluster_id = int(row["cluster_id"])
                filename = row["original_filename"].strip()
                if cluster_id in rows:
                    raise ValueError(f"{mapping_file}: duplicate cluster_id {cluster_id}")
                if Path(filename).name != filename:
                    raise ValueError(f"{mapping_file}: expected a basename, got {filename}")
                rows[cluster_id] = filename
        if set(rows) != expected_ids:
            raise ValueError(
                f"{mapping_file}: cluster ids are {sorted(rows)}, expected {sorted(expected_ids)}"
            )

        category_artifacts: dict[int, Path] = {}
        for cluster_id, filename in rows.items():
            source_image = spec.source_root / category / "train" / "good" / filename
            if not source_image.is_file():
                raise FileNotFoundError(
                    f"{spec.name}/{category}: representative source missing: {source_image}"
                )
            artifact = category_dir / "representatives" / f"cluster_{cluster_id:02d}_{filename}"
            if artifact.is_file():
                if check_content and not filecmp.cmp(source_image, artifact, shallow=False):
                    raise ValueError(
                        f"{spec.name}/{category}: representative differs from source: {artifact}"
                    )
                category_artifacts[cluster_id] = artifact
            else:
                # `goef-sample` only copies representatives with --copy_images;
                # the mapping alone is enough to build the centre datasets.
                category_artifacts[cluster_id] = source_image

        mappings[category] = rows
        artifacts[category] = category_artifacts
    return mappings, artifacts


def find_mask(image: Path, mask_dir: Path) -> Path:
    by_stem = {path.stem: path for path in image_files(mask_dir)}
    for stem in (image.stem, f"{image.stem}_mask"):
        if stem in by_stem:
            return by_stem[stem]
    raise FileNotFoundError(f"no mask for test image {image} under {mask_dir}")


def build_meta_from_tree(source_root: Path, categories: Iterable[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Build the standard train/test meta schema when the dataset has none."""
    meta: dict[str, dict[str, list[dict[str, Any]]]] = {"train": {}, "test": {}}
    for category in sorted(categories):
        category_root = source_root / category
        train_images = image_files(category_root / "train" / "good")
        if not train_images:
            raise ValueError(f"no training image for {category}")
        meta["train"][category] = [
            {
                "img_path": relative_path(image, source_root),
                "mask_path": "",
                "cls_name": category,
                "specie_name": "good",
                "anomaly": 0,
            }
            for image in train_images
        ]

        test_root = category_root / "test"
        if not test_root.is_dir():
            raise FileNotFoundError(f"missing test directory: {test_root}")
        entries = []
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            anomaly = int(defect_dir.name.lower() not in {"good", "normal"})
            for image in image_files(defect_dir):
                mask_path = ""
                if anomaly:
                    mask = find_mask(image, category_root / "ground_truth" / defect_dir.name)
                    mask_path = relative_path(mask, source_root)
                entries.append(
                    {
                        "img_path": relative_path(image, source_root),
                        "mask_path": mask_path,
                        "cls_name": category,
                        "specie_name": defect_dir.name,
                        "anomaly": anomaly,
                    }
                )
        if not entries:
            raise ValueError(f"no test image for {category}")
        meta["test"][category] = entries
    return meta


def load_or_build_meta(source_root: Path, categories: Iterable[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    meta_path = source_root / "meta.json"
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8-sig") as handle:
            meta = json.load(handle)
    else:
        meta = build_meta_from_tree(source_root, categories)
    if not isinstance(meta, dict) or not {"train", "test"}.issubset(meta):
        raise ValueError(f"invalid meta schema in {meta_path}")
    return meta


def validate_meta_entry(entry: dict[str, Any], source_root: Path, split: str, category: str) -> None:
    missing = set(META_FIELDS).difference(entry)
    if missing:
        raise ValueError(f"{category}/{split}: meta entry lacks {sorted(missing)}")
    if not resolve_meta_path(source_root, str(entry["img_path"])).is_file():
        raise FileNotFoundError(f"meta image does not exist: {entry['img_path']}")
    mask_value = str(entry["mask_path"])
    if mask_value and not resolve_meta_path(source_root, mask_value).is_file():
        raise FileNotFoundError(f"meta mask does not exist: {mask_value}")
    if int(entry["anomaly"]) and not mask_value:
        raise ValueError(f"anomalous sample has no mask: {entry['img_path']}")


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #
def build_center_plans(
    spec: DatasetSpec,
    mappings: dict[str, dict[int, str]],
    artifacts: dict[str, dict[int, Path]],
    k: int,
    combination_size: int,
) -> list[CenterPlan]:
    categories = sorted(mappings)
    full_meta = load_or_build_meta(spec.source_root, categories)
    for category in categories:
        if category not in full_meta["train"] or category not in full_meta["test"]:
            raise ValueError(f"{spec.name}: {category!r} is absent from the dataset meta")
        for split in ("train", "test"):
            for entry in full_meta[split][category]:
                validate_meta_entry(entry, spec.source_root, split, category)

    plans = []
    for cluster_ids in itertools.combinations(range(k), combination_size):
        selected_train: dict[str, list[dict[str, Any]]] = {}
        representatives: dict[str, dict[int, str]] = {}
        representative_artifacts: dict[str, dict[int, Path]] = {}
        for category in categories:
            selected_train[category] = []
            representatives[category] = {}
            representative_artifacts[category] = {}
            for cluster_id in cluster_ids:
                filename = mappings[category][cluster_id]
                source_image = (spec.source_root / category / "train" / "good" / filename).resolve()
                matches = [
                    entry
                    for entry in full_meta["train"][category]
                    if resolve_meta_path(spec.source_root, str(entry["img_path"])).resolve() == source_image
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"{spec.name}/{category}: expected one meta entry for {source_image}, "
                        f"found {len(matches)}"
                    )
                selected = deepcopy(matches[0])
                if int(selected["anomaly"]) != 0:
                    raise ValueError(f"representative is marked anomalous: {selected}")
                selected_train[category].append(selected)
                representatives[category][cluster_id] = str(selected["img_path"])
                representative_artifacts[category][cluster_id] = artifacts[category][cluster_id]

        plans.append(
            CenterPlan(
                spec=spec,
                center_numbers=tuple(cluster_id + 1 for cluster_id in cluster_ids),
                cluster_ids=cluster_ids,
                meta={
                    "train": selected_train,
                    "test": {category: deepcopy(full_meta["test"][category]) for category in categories},
                },
                representatives=representatives,
                representative_artifacts=representative_artifacts,
            )
        )
    return plans


def iter_referenced_paths(meta) -> Iterable[str]:
    seen: set[str] = set()
    for split in ("train", "test"):
        for entries in meta[split].values():
            for entry in entries:
                for field in ("img_path", "mask_path"):
                    value = str(entry[field])
                    if value and value not in seen:
                        seen.add(value)
                        yield value


def materialize_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as error:
            raise OSError(
                f"cannot hard-link {source} -> {destination}; use --materialize symlink or copy"
            ) from error
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unsupported materialization mode: {mode}")


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_representatives_csv(path: Path, plan: CenterPlan) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("category", "center_number", "cluster_id", "img_path", "representative_artifact"),
        )
        writer.writeheader()
        for category in sorted(plan.representatives):
            for cluster_id in plan.cluster_ids:
                writer.writerow(
                    {
                        "category": category,
                        "center_number": cluster_id + 1,
                        "cluster_id": cluster_id,
                        "img_path": plan.representatives[category][cluster_id],
                        "representative_artifact": str(plan.representative_artifacts[category][cluster_id]),
                    }
                )


def count_entries(meta, split: str) -> int:
    return sum(len(entries) for entries in meta[split].values())


def write_center(plan: CenterPlan, output_root: Path, mode: str) -> Path:
    dataset_output = output_root / plan.spec.name
    dataset_output.mkdir(parents=True, exist_ok=True)
    destination = dataset_output / plan.directory_name
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=dataset_output))
    try:
        if mode != "meta":
            for value in iter_referenced_paths(plan.meta):
                if Path(value).is_absolute():
                    raise ValueError(f"materialised datasets require relative meta paths: {value}")
                materialize_file(resolve_meta_path(plan.spec.source_root, value), temporary / value, mode)
        write_json(temporary / "meta.json", plan.meta)
        write_representatives_csv(temporary / "representatives.csv", plan)
        info: dict[str, Any] = {
            "dataset": plan.spec.name,
            "combination_size": len(plan.cluster_ids),
            "center_numbers": list(plan.center_numbers),
            "cluster_ids": list(plan.cluster_ids),
            "materialization": mode,
            "data_root": str(plan.spec.source_root.resolve() if mode == "meta" else destination.resolve()),
            "source_root": str(plan.spec.source_root.resolve()),
            "cluster_root": str(plan.spec.cluster_root.resolve()),
            "category_count": len(plan.representatives),
            "train_sample_count": count_entries(plan.meta, "train"),
            "test_sample_count": count_entries(plan.meta, "test"),
        }
        if len(plan.cluster_ids) == 1:
            info["center_number"] = plan.center_numbers[0]
            info["cluster_id"] = plan.cluster_ids[0]
        write_json(temporary / "dataset_info.json", info)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def ensure_destinations_available(plans: Iterable[CenterPlan], output_root: Path, overwrite: bool) -> None:
    destinations = [output_root / plan.spec.name / plan.directory_name for plan in plans]
    existing = [path for path in destinations if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(f"output directories already exist (use --overwrite):\n  {joined}")
    if overwrite:
        for path in existing:
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)


def build(
    specs: Iterable[DatasetSpec],
    output_root: Path,
    k: int = 5,
    combination_size: int = 1,
    materialize: str = "meta",
    overwrite: bool = False,
    check_representative_content: bool = True,
) -> list[Path]:
    """Build every center dataset; returns the written directories."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not 1 <= combination_size <= k:
        raise ValueError("combination_size must be between 1 and k")

    all_plans: list[CenterPlan] = []
    for spec in specs:
        mappings, artifacts = read_representative_mappings(
            spec, k, check_content=check_representative_content
        )
        plans = build_center_plans(spec, mappings, artifacts, k, combination_size)
        all_plans.extend(plans)
        print(
            f"validated {spec.name}: {len(mappings)} categories, "
            f"{k} centers/category, {len(plans)} combinations of {combination_size}"
        )

    ensure_destinations_available(all_plans, output_root, overwrite)
    written = []
    for plan in all_plans:
        destination = write_center(plan, output_root, materialize)
        written.append(destination)
        print(
            f"wrote {destination}: train={count_entries(plan.meta, 'train')}, "
            f"test={count_entries(plan.meta, 'test')}, mode={materialize}"
        )
    return written
