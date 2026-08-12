"""Downloads the full LOCO archive once (single sequential connection), then locally
selects only pallet and forklift images, capped at a target count, for a fast POC."""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parent.parent

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/tum-fml/loco/main/rgb"
ANNOTATION_FILES = [
    "loco-all-v1.json",
    "loco-sub1-v1-val.json",
    "loco-sub2-v1-train.json",
    "loco-sub3-v1-train.json",
    "loco-sub4-v1-val.json",
    "loco-sub5-v1-train.json",
]
IMAGES_ZIP_URL = "https://go.mytum.de/239870"

_last_percent = [-1]

def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    percent = int(block_num * block_size * 100 / total_size)
    if percent != _last_percent[0] and percent % 5 == 0:
        _last_percent[0] = percent
        mb_done = block_num * block_size / 1_000_000
        mb_total = total_size / 1_000_000
        print(f"  {percent}%  ({mb_done:.0f} MB / {mb_total:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "loco_raw")
    args = parser.parse_args()

    ann_dir = args.out / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    print("=== Downloading annotations ===")
    for fname in ANNOTATION_FILES:
        dest = ann_dir / fname
        if not dest.exists():
            urlretrieve(f"{GITHUB_RAW_BASE}/{fname}", dest)
    print("done")

    zip_path = args.out / "loco_images.zip"
    if not zip_path.exists():
        print("\n=== Downloading full images archive, ONE connection, this is the slow-but-reliable step ===")
        start = time.time()
        urlretrieve(IMAGES_ZIP_URL, zip_path, reporthook=_progress)
        print(f"Downloaded in {time.time()-start:.0f}s")
    else:
        print("\nZip already downloaded, skipping")

    extract_dir = args.out / "images"
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print("Already extracted, skipping")
    else:
        print("\n=== Extracting all images (local disk only, no network) ===")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        print("done")

    print(f"\nRaw data ready in {args.out}")


if __name__ == "__main__":
    main()
