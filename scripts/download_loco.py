"""Downloads a small sample of the real LOCO warehouse dataset instead of the whole multi-GB archive.

LOCO (Logistics Objects in Context) is a public domain dataset from the Technical
University of Munich: forklifts, pallets, pallet trucks, small load carriers, and
stillages, across 5 real warehouse environments.
Citation: Mayershofer, C., Holm, D.-M., Molter, B., Fottner, J.,
"LOCO: Logistics Objects in Context", IEEE ICMLA 2020.

I use remotezip to read specific files straight out of the remote archive over
HTTP range requests, so I only pull the bytes I actually need instead of the
whole multi-GB zip, that's what keeps this workable on a lightweight laptop.

--images-per-subset controls how many images I pull from EACH of the 5 warehouse
subsets, so I get variety across environments instead of one lopsided chunk.

I prioritize images containing forklift by default, since that's the rarest
class in the dataset (598 of 151,428 annotations) and random sampling alone
barely picks any up.
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
SUBSET_FILES = ANNOTATION_FILES[1:]

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


def pick_sample_filenames(annotations_dir: Path, subset_file: str, n: int, seed: int, prioritize_class: str | None = "forklift") -> list[str]:
    with open(annotations_dir / subset_file) as f:
        data = json.load(f)

    all_filenames = [img["file_name"] for img in data["images"]]
    rng = random.Random(seed)

    if prioritize_class is None:
        return rng.sample(all_filenames, min(n, len(all_filenames)))

    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    priority_filenames = sorted(set(
        image_id_to_filename[ann["image_id"]]
        for ann in data["annotations"]
        if cat_names.get(ann["category_id"]) == prioritize_class
    ))

    if len(priority_filenames) >= n:
        return rng.sample(priority_filenames, n)

    remaining_budget = n - len(priority_filenames)
    remaining_pool = [f for f in all_filenames if f not in set(priority_filenames)]
    fill = rng.sample(remaining_pool, min(remaining_budget, len(remaining_pool)))

    return priority_filenames + fill


def download_image_subset(annotations_dir: Path, out_images_dir: Path, images_per_subset: int, seed: int, prioritize_class: str | None) -> None:
    out_images_dir.mkdir(parents=True, exist_ok=True)

    print("Connecting to the remote archive to read its file index (just the central directory, not the whole zip)...")
    with RemoteZip(IMAGES_ZIP_URL) as zf:
        all_zip_names = zf.namelist()
        basename_index = {Path(n).name: n for n in all_zip_names if n.lower().endswith(".jpg")}
        print(f"  archive contains {len(basename_index)} image files total")

        total_fetched = 0
        for subset_file in SUBSET_FILES:
            sample_names = pick_sample_filenames(annotations_dir, subset_file, images_per_subset, seed, prioritize_class)
            print(f"\n{subset_file}: sampling {len(sample_names)} images"
                  f"{f' (prioritizing images with {prioritize_class})' if prioritize_class else ''}...")

            for fname in sample_names:
                zip_path = basename_index.get(fname)
                if zip_path is None:
                    print(f"  WARNING: {fname} not found in remote archive index, skipping")
                    continue

                dest = out_images_dir / fname
                if dest.exists():
                    continue

                data = zf.read(zip_path)
                dest.write_bytes(data)
                total_fetched += 1

        print(f"\nFetched {total_fetched} images total (subset only, not the full archive).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "loco_raw")
    parser.add_argument("--images-per-subset", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Downloading LOCO annotations (small, from GitHub) ===")
    download_annotations(args.out / "annotations")

    print(f"\n=== Downloading a {args.images_per_subset}-per-subset image sample (not the full archive) ===")
    download_image_subset(args.out / "annotations", args.out / "images", args.images_per_subset, args.seed, prioritize_class="forklift")

    print(f"\nDone. Raw LOCO subset is in {args.out}")
    print("Next: python scripts/convert_loco_to_yolo.py")


if __name__ == "__main__":
    main()
