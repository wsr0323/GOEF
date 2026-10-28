#!/usr/bin/env python3
"""Resumable, range-verified downloader for the SAM-HQ ViT-H checkpoint."""

import argparse
import hashlib
import subprocess
from pathlib import Path


URL = "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
TOTAL_BYTES = 2_570_940_653
SHA256 = "a7ac14a085326d9fa6199c8c698c4f0e7280afdbb974d2c4660ec60877b45e35"
CHUNK_BYTES = 64 * 1024 * 1024


def download_chunk(target, state_dir, index):
    start = index * CHUNK_BYTES
    end = min(start + CHUNK_BYTES, TOTAL_BYTES) - 1
    expected_size = end - start + 1
    marker = state_dir / f"{index:03d}.done"
    if marker.exists():
        return False

    chunk_path = state_dir / f"{index:03d}.chunk"
    subprocess.run(
        [
            "curl",
            "-fsSL",
            "--retry",
            "10",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "-r",
            f"{start}-{end}",
            "-o",
            str(chunk_path),
            URL,
        ],
        check=True,
    )
    if chunk_path.stat().st_size != expected_size:
        raise RuntimeError(
            f"chunk {index} has {chunk_path.stat().st_size} bytes, expected {expected_size}"
        )
    with chunk_path.open("rb") as source, target.open("r+b") as destination:
        destination.seek(start)
        while block := source.read(8 * 1024 * 1024):
            destination.write(block)
    chunk_path.unlink()
    marker.touch()
    print(f"downloaded chunk {index}: bytes {start}-{end}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="pretrained_ckpts/sam_hq_vit_h.pth")
    parser.add_argument("--max-chunks", type=int, default=3)
    args = parser.parse_args()

    output = Path(args.output)
    partial = output.with_suffix(output.suffix + ".partial")
    state_dir = output.parent / (output.name + ".chunks")
    state_dir.mkdir(parents=True, exist_ok=True)
    if not partial.exists():
        with partial.open("wb") as file:
            file.truncate(TOTAL_BYTES)
    if partial.stat().st_size != TOTAL_BYTES:
        raise RuntimeError(f"partial file has unexpected size: {partial.stat().st_size}")

    chunks = (TOTAL_BYTES + CHUNK_BYTES - 1) // CHUNK_BYTES
    completed = 0
    for index in range(chunks):
        if completed >= args.max_chunks:
            break
        if download_chunk(partial, state_dir, index):
            completed += 1

    complete = sum((state_dir / f"{index:03d}.done").exists() for index in range(chunks))
    print(f"completed chunks: {complete}/{chunks}")
    if complete != chunks:
        return

    digest = hashlib.sha256()
    with partial.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise RuntimeError("SHA-256 mismatch; delete the .partial file and .chunks directory")
    partial.replace(output)
    for marker in state_dir.glob("*.done"):
        marker.unlink()
    state_dir.rmdir()
    print(f"verified and saved {output}")


if __name__ == "__main__":
    main()
