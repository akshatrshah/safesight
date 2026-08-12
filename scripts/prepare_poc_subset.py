"""Filters the already-downloaded LOCO data down to pallet + forklift only,
capped near 1000 images, forklift-containing images guaranteed included
first since it's the rare class. Converts straight to YOLO format.
Builds a filename index first since images sit in deeply nested subfolders."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TRAIN_SUBSETS = ["loco-sub2-v1-train.json", "loco-sub3-v1-train.json", "loco-sub5-v1-train.json"]
VAL_SUBSETS = ["loco-sub1-v1-val.json", "loco-sub4-v1-val.json"]
TARGET_CLASSES = {"forklift", "pallet"}
CLASS_TO_YOLO_ID = {"forklift": 0, "pallet": 1}


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    return ((x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h)


def build_filename_index(images_root: Path) -> dict[str, Path]:
    print(f"Indexing images under {images_root} (one-time scan, nested folders)...")
    index = {}
    for path in images_root.rglob("*.jpg"):
        index[path.name] = path
    print(f"  found {len(index)} image files")
    return index


def select_and_convert(subset_files, annotations_dir, filename_index, out_images, out_labels, cap):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    forklift_images = {}
    pallet_only_images = {}
    all_images_meta = {}
    all_anns_by_image = {}

    for subset_file in subset_files:
        with open(annotations_dir / subset_file) as f:
            data = json.load(f)
        cat_names = {c["id"]: c["name"] for c in data["categories"]}
        for img in data["images"]:
            all_images_meta[img["file_name"]] = img
        img_id_to_name = {img["id"]: img["file_name"] for img in data["images"]}
        for ann in data["annotations"]:
            cname = cat_names.get(ann["category_id"])
            if cname not in TARGET_CLASSES:
                continue
            fname = img_id_to_name[ann["image_id"]]
            all_anns_by_image.setdefault(fname, []).append((cname, ann["bbox"]))
            if cname == "forklift":
                forklift_images[fname] = True
            elif cname == "pallet" and fname not in forklift_images:
                pallet_only_images[fname] = True

    selected = list(forklift_images.keys())
    remaining = max(0, cap - len(selected))
    selected += sorted(pallet_only_images.keys())[:remaining]

    print(f"  forklift images: {len(forklift_images)}, pallet-only available: {len(pallet_only_images)}, selected: {len(selected)}")

    copied, missing = 0, 0
    for fname in selected:
        src = filename_index.get(fname)
        if src is None:
            missing += 1
            continue
        shutil.copy(src, out_images / fname)

        img_meta = all_images_meta[fname]
        lines = []
        for cname, bbox in all_anns_by_image.get(fname, []):
            x_c, y_c, w, h = coco_bbox_to_yolo(bbox, img_meta["width"], img_meta["height"])
            lines.append(f"{CLASS_TO_YOLO_ID[cname]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        (out_labels / (Path(fname).stem + ".txt")).write_text("\n".join(lines))
        copied += 1

    print(f"  copied {copied}, missing {missing}")
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "datasets" / "loco_raw")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "loco_poc")
    parser.add_argument("--train-cap", type=int, default=800)
    parser.add_argument("--val-cap", type=int, default=200)
    args = parser.parse_args()

    ann_dir = args.raw / "annotations"
    img_dir = args.raw / "images"

    filename_index = build_filename_index(img_dir)

    print("=== Train split ===")
    select_and_convert(TRAIN_SUBSETS, ann_dir, filename_index, args.out / "train" / "images", args.out / "train" / "labels", args.train_cap)

    print("=== Val split ===")
    select_and_convert(VAL_SUBSETS, ann_dir, filename_index, args.out / "valid" / "images", args.out / "valid" / "labels", args.val_cap)

    data_yaml = args.out / "data.yaml"
    data_yaml.write_text("train: train/images\nval: valid/images\nnc: 2\nnames: ['forklift', 'pallet']\n")
    print(f"\nWrote {data_yaml}")
    print(f"Train with:\n  python scripts/finetune_detector.py --data {data_yaml} --epochs 40 --imgsz 640")


if __name__ == "__main__":
    main()
