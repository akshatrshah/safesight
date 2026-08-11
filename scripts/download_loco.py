"""
Download a SUBSET of the LOCO (Logistics Objects in Context) dataset.

LOCO is a real, published, public-domain dataset built by the Technical
University of Munich specifically for logistics object detection —
forklifts, pallets, pallet trucks, small load carriers, stillages —
collected across 5 distinct real warehouse environments.
Citation: Mayershofer, C., Holm, D.-M., Molter, B., Fottner, J.
"LOCO: Logistics Objects in Context", IEEE ICMLA 2020.
License: public domain (CC0-equivalent) — see the repo's LICENSE file.

WHY THIS DOWNLOADS A SUBSET, NOT THE WHOLE ARCHIVE
--------------------------------------------------------
The full annotated-image set is several GB — too much for a lightweight
laptop to download and store just for a first fine-tuning pass. Instead
of downloading the whole zip, this script uses `remotezip` to read
SPECIFIC FILES directly out of the remote zip archive over HTTP Range
requests — the same technique tools like `aws s3 cp --range` use. Only
the bytes for the images you actually ask for get transferred, plus a
tiny read of the zip's central directory (a KB-scale index of what's in
the archive) — not the whole multi-GB file.

`--images-per-subset N` controls how many images get pulled from EACH
of the 5 warehouse subsets (so you get variety across environments, not
just a chunk of one warehouse). Default is a small number suited to a
lightweight first pass — increase it later once you've confirmed the
pipeline works end-to-end.

FALLBACK
------------
If the remote server doesn't support Range requests for some reason
(rare for a standard file host, but possible), `remotezip` will raise a
clear error rather than silently downloading everything. In that case,
fall back to downloading the full zip manually from the link in
https://github.com/tum-fml/loco and extracting only the images you need
yourself.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from urllib.request import urlretrieve

from remotezip import RemoteZip

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
SUBSET_FILES = ANNOTATION_FILES[1:]  # exclude the combined "all" file

# Official annotated-images download, linked from https://github.com/tum-fml/loco
IMAGES_ZIP_URL = "https://go.mytum.de/239870"


def download_annotations(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ANNOTATION_FILES:
        dest = out_dir / fname
        if dest.exists():
            print(f"  {fname} already downloaded, skipping")
            continue
        url = f"{GITHUB_RAW_BASE}/{fname}"
        print(f"  downloading {fname} ...")
        urlretrieve(url, dest)
    print(f"Annotations saved to {out_dir}")


def pick_sample_filenames(annotations_dir: Path, subset_file: str, n: int, seed: int) -> list[str]:
    with open(annotations_dir / subset_file) as f:
        data = json.load(f)
    all_filenames = [img["file_name"] for img in data["images"]]
    rng = random.Random(seed)
    return rng.sample(all_filenames, min(n, len(all_filenames)))


def download_image_subset(annotations_dir: Path, out_images_dir: Path, images_per_subset: int, seed: int) -> None:
    out_images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to remote archive to read its file index (this reads only the archive's "
          f"central directory, not the full multi-GB file)...")
    with RemoteZip(IMAGES_ZIP_URL) as zf:
        all_zip_names = zf.namelist()
        # Build basename -> full-path-in-zip index, since the zip's internal
        # folder structure may not exactly match the "file_name" field alone.
        basename_index = {Path(n).name: n for n in all_zip_names if n.lower().endswith(".jpg")}
        print(f"  archive contains {len(basename_index)} image files total")

        total_fetched = 0
        for subset_file in SUBSET_FILES:
            sample_names = pick_sample_filenames(annotations_dir, subset_file, images_per_subset, seed)
            print(f"\n{subset_file}: sampling {len(sample_names)} images...")

            for fname in sample_names:
                zip_path = basename_index.get(fname)
                if zip_path is None:
                    print(f"  WARNING: {fname} not found in remote archive index, skipping")
                    continue

                dest = out_images_dir / fname
                if dest.exists():
                    continue

                data = zf.read(zip_path)   # fetches ONLY this file's bytes via a range request
                dest.write_bytes(data)
                total_fetched += 1

        print(f"\nFetched {total_fetched} images total (subset only, not the full archive).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "loco_raw")
    parser.add_argument("--images-per-subset", type=int, default=150,
                         help="How many images to sample from EACH of the 5 warehouse subsets (default: 150, ~750 total)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Downloading LOCO annotations (small — from GitHub) ===")
    download_annotations(args.out / "annotations")

    print(f"\n=== Downloading a {args.images_per_subset}-per-subset image SAMPLE (not the full archive) ===")
    download_image_subset(args.out / "annotations", args.out / "images", args.images_per_subset, args.seed)

    print(f"\nDone. Raw LOCO subset is in {args.out}")
    print("Next: python scripts/convert_loco_to_yolo.py")


if __name__ == "__main__":
    main()