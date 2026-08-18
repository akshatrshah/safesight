"""Builds a forklift-only, single-class training set, every image, no pallet dilution.

My last training run used 800 images total, but only 348 of them
actually contained a forklift, the rest were pallet-only images filling
out the budget. That means well over half of every epoch, the model saw
zero forklift signal at all. LOCO only has 449 forklift images total
across the whole dataset (there's no "more data" to go get), so the real
fix isn't volume, it's concentration: use every single forklift image,
nothing else, so every training image actually teaches the model
something about forklifts.

Single class also removes the pallet-vs-forklift competition entirely,
the model's whole capacity goes toward forklift-vs-background instead of
splitting attention across two classes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TRAIN_SUBSETS = ["loco-sub2-v1-train.json", "loco-sub3-v1-train.json", "loco-sub5-v1-train.json"]
VAL_SUBSETS = ["loco-sub1-v1-val.json", "loco-sub4-v1-val.json"]


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    return ((x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h)


def build_filename_index(images_root: Path) -> dict[str, Path]:
    print(f"Indexing images under {images_root} (nested folders, one-time scan)...")
    index = {}
    for path in images_root.rglob("*.jpg"):
        index[path.name] = path
    print(f"  found {len(index)} image files")
    return index


def convert_forklift_only(subset_files, annotations_dir, filename_index, out_images, out_labels):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    copied, missing, total_boxes = 0, 0, 0

    for subset_file in subset_files:
        with open(annotations_dir / subset_file) as f:
            data = json.load(f)
        cat_names = {c["id"]: c["name"] for c in data["categories"]}
        img_id_to_meta = {img["id"]: img for img in data["images"]}

        # Only keep images that have at least one forklift annotation,
        # and only keep forklift boxes within them, everything else in
        # that image (pallet, stillage, whatever) gets ignored entirely.
        forklift_anns_by_image: dict[int, list] = {}
        for ann in data["annotations"]:
            if cat_names.get(ann["category_id"]) == "forklift":
                forklift_anns_by_image.setdefault(ann["image_id"], []).append(ann)

        for image_id, anns in forklift_anns_by_image.items():
            img_meta = img_id_to_meta[image_id]
            fname = img_meta["file_name"]
            src = filename_index.get(fname)
            if src is None:
                missing += 1
                continue

            shutil.copy(src, out_images / fname)
            lines = []
            for ann in anns:
                x_c, y_c, w, h = coco_bbox_to_yolo(ann["bbox"], img_meta["width"], img_meta["height"])
                lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")  # class 0, only class, forklift
                total_boxes += 1
            (out_labels / (Path(fname).stem + ".txt")).write_text("\n".join(lines))
            copied += 1

    return copied, missing, total_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "datasets" / "loco_raw")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "loco_forklift_only")
    args = parser.parse_args()

    ann_dir = args.raw / "annotations"
    img_dir = args.raw / "images"

    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations not found at {ann_dir}, run download_loco_full.py first")

    filename_index = build_filename_index(img_dir)

    print("\n=== Train split, forklift only ===")
    train_copied, train_missing, train_boxes = convert_forklift_only(
        TRAIN_SUBSETS, ann_dir, filename_index, args.out / "train" / "images", args.out / "train" / "labels"
    )
    print(f"  {train_copied} images (every one contains a forklift), {train_boxes} forklift boxes, {train_missing} missing")

    print("\n=== Val split, forklift only ===")
    val_copied, val_missing, val_boxes = convert_forklift_only(
        VAL_SUBSETS, ann_dir, filename_index, args.out / "valid" / "images", args.out / "valid" / "labels"
    )
    print(f"  {val_copied} images (every one contains a forklift), {val_boxes} forklift boxes, {val_missing} missing")

    data_yaml = args.out / "data.yaml"
    data_yaml.write_text("train: train/images\nval: valid/images\nnc: 1\nnames: ['forklift']\n")
    print(f"\nWrote {data_yaml}")
    print(f"Train with:\n  python scripts/finetune_detector.py --data {data_yaml} --base-model yolov8s.pt --epochs 60 --imgsz 640")


if __name__ == "__main__":
    main()
