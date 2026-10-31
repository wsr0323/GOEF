"""Raw-result collection and manifest writing."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from goef.collect import collect_centers, collect_one_shot, split_combined_one_shot, write_manifest

HEADER = ["object", "ref_sample", "Image AUROC", "Image AUPR", "Pixel AUROC", "Pixel AUPRO"]


def write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def test_collect_one_shot_keeps_rows_and_manifest(tmp_path: Path):
    source = write(tmp_path / "src" / "mvtec.csv", [["bottle", "a.png", 0.9, 0.9, 0.9, 0.9]])
    rows = collect_one_shot(tmp_path / "out", {"winclip": {"MVTec": source}})
    assert (tmp_path / "out" / "one_shot" / "winclip" / "MVTec.csv").is_file()
    assert rows[0].rows == 1
    assert rows[0].sha256
    manifest = write_manifest(tmp_path / "out" / "MANIFEST.csv", rows)
    with manifest.open() as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_collect_one_shot_rejects_a_bad_schema(tmp_path: Path):
    source = tmp_path / "bad.csv"
    source.write_text("object,ref_sample\nbottle,a.png\n")
    with pytest.raises(ValueError):
        collect_one_shot(tmp_path / "out", {"x": {"MVTec": source}})


def test_split_combined_csv_by_dataset(tmp_path: Path):
    source = tmp_path / "combined.csv"
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", *HEADER])
        writer.writerow(["MVTec-AD", "bottle", "a.png", 0.9, 0.9, 0.9, 0.9])
        writer.writerow(["VisA", "candle", "b.png", 0.8, 0.8, 0.8, 0.8])
    rows = split_combined_one_shot(
        source, tmp_path / "out", "inpformer", aliases={"MVTec-AD": "mvtec", "VisA": "visa"}
    )
    assert {row.dataset for row in rows} == {"mvtec", "visa"}
    assert (tmp_path / "out" / "one_shot" / "inpformer" / "visa.csv").is_file()


def test_collect_centers_prefers_the_unified_csv(tmp_path: Path):
    results = tmp_path / "proj" / "results"
    base = results / "1-shot" / "center_01_mvtec.csv"
    base.parent.mkdir(parents=True)
    base.write_text("Category,Image AUROC,Image AUPR,Pixel AUROC,Pixel AUPRO\nbottle,1,1,1,1\n")
    unified = results / "1-shot" / "center_01_mvtec_unified.csv"
    unified.write_text(
        "Category,Image AUROC,Image AUPR,Pixel AUROC,Pixel AUPRO,Images\nbottle,1,1,1,1,10\n"
    )
    rows = collect_centers(tmp_path / "out", {"PatchCore": tmp_path / "proj"}, shots=(1,))
    assert [row.path.split("/")[-1] for row in rows] == ["center_01_mvtec_unified.csv"]
    assert rows[0].dataset == "mvtec"
