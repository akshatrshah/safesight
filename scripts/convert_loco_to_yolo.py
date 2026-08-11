"""
Convert LOCO's COCO-format annotations into the YOLOv8 format
`finetune_detector.py` expects (per-image .txt label files + data.yaml).

WHY WE USE LOCO'S OWN TRAIN/VAL SPLIT, NOT A RANDOM ONE
--------------------------------------------------------------
LOCO ships 5 separate subsets (sub1-sub5), each recorded in a DIFFERENT
real warehouse / recording session. The dataset authors already
designated sub2, sub3, sub5 as train and sub1, sub4 as val. This is
exactly the right way to split: by session/environment, not by
individual frame — using a random frame-level split here would let
near-duplicate frames from the same short recording end up in both
train and val, leaking information and inflating validation metrics
artificially (the same data-leakage issue covered in FOUNDATIONS.md).
We use the authors' split as-is rather than re-splitting ourselves.

COCO BBOX FORMAT -> YOLO BBOX FORMAT
------------------------------------------
COCO format:  [x_top_left, y_top_left, width, height]   (absolute pixels)
YOLO format:  [x_center, y_center, width, height]        (all normalized 0-1)

Conversion:
    x_center = (x_top_left + width / 2) / image_width
    y_center = (y_top_left + height / 2) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TRAIN_SUBSETS = ["loco-sub2-v1-train.json", "loco-sub3-v1-train.json", "loco-sub5-v1-train.json"]
VAL_SUBSETS = ["loco-sub1-v1-val.json", "loco-sub4-v1-val.json"]


def coco_bbox_to_yolo(bbox: list[float], img_width: int, img_height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    return (x_center, y_center, w / img_width, h / img_height)


def find_image_file(file_name: str, search_root: Path, index: dict[str, Path]) -> Path | None:
    """Look up an image by filename in a prebuilt index (built once via os.walk, not re-searched per file)."""
    return index.get(file_name)


def build_filename_index(search_root: Path) -> dict[str, Path]:
    print(f"Indexing image files under {search_root} (one-time scan)...")
    index: dict[str, Path] = {}
    for path in search_root.rglob("*.jpg"):
        index[path.name] = path
    print(f"  found {len(index)} image files")
    return index


def convert_subset(
    annotation_files: list[str],
    annotations_dir: Path,
    images_search_root: Path,
    filename_index: dict[str, Path],
    out_images_dir: Path,
    out_labels_dir: Path,
    category_id_to_yolo_id: dict[int, int],
) -> tuple[int, int, int]:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    copied, missing, total_annotations = 0, 0, 0

    for ann_file in annotation_files:
        with open(annotations_dir / ann_file) as f:
            data = json.load(f)

        # Group annotations by image_id for fast lookup.
        anns_by_image: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        for img in data["images"]:
            src_path = find_image_file(img["file_name"], images_search_root, filename_index)
            if src_path is None:
                missing += 1
                continue

            dest_image_path = out_images_dir / img["file_name"]
            if not dest_image_path.exists():
                shutil.copy(src_path, dest_image_path)

            label_lines = []
            for ann in anns_by_image.get(img["id"], []):
                if ann["category_id"] not in category_id_to_yolo_id:
                    continue
                yolo_class_id = category_id_to_yolo_id[ann["category_id"]]
                x_c, y_c, w, h = coco_bbox_to_yolo(ann["bbox"], img["width"], img["height"])
                label_lines.append(f"{yolo_class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                total_annotations += 1

            label_path = out_labels_dir / (Path(img["file_name"]).stem + ".txt")
            label_path.write_text("\n".join(label_lines))
            copied += 1

    return copied, missing, total_annotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "datasets" / "loco_raw",
                         help="Directory produced by download_loco.py")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "loco_yolo",
                         help="Output directory in YOLOv8 format")
    args = parser.parse_args()

    annotations_dir = args.raw / "annotations"
    images_search_root = args.raw / "images"

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations not found at {annotations_dir} — run scripts/download_loco.py first")

    # Load the full annotation file just to get the category list (consistent across all subsets).
    with open(annotations_dir / "loco-all-v1.json") as f:
        all_data = json.load(f)

    categories = sorted(all_data["categories"], key=lambda c: c["id"])
    category_id_to_yolo_id = {c["id"]: i for i, c in enumerate(categories)}
    class_names = [c["name"] for c in categories]
    print(f"Classes (COCO id -> YOLO id -> name): "
          + ", ".join(f"{cid}->{yid}:{name}" for cid, (yid, name) in
                       zip(category_id_to_yolo_id.keys(), zip(category_id_to_yolo_id.values(), class_names))))

    filename_index = build_filename_index(images_search_root)

    print("\n=== Converting TRAIN split ===")
    train_copied, train_missing, train_anns = convert_subset(
        TRAIN_SUBSETS, annotations_dir, images_search_root, filename_index,
        args.out / "train" / "images", args.out / "train" / "labels", category_id_to_yolo_id,
    )
    print(f"  {train_copied} images converted, {train_missing} missing, {train_anns} annotations")

    print("\n=== Converting VAL split ===")
    val_copied, val_missing, val_anns = convert_subset(
        VAL_SUBSETS, annotations_dir, images_search_root, filename_index,
        args.out / "valid" / "images", args.out / "valid" / "labels", category_id_to_yolo_id,
    )
    print(f"  {val_copied} images converted, {val_missing} missing, {val_anns} annotations")

    data_yaml = args.out / "data.yaml"
    data_yaml.write_text(
        f"train: train/images\n"
        f"val: valid/images\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    print(f"\nWrote {data_yaml}")

    if train_missing or val_missing:
        print(f"\nWARNING: {train_missing + val_missing} images referenced in annotations were not found "
              f"under {images_search_root}. Check that download_loco.py's image download/extraction completed fully.")

    print(f"\nDone. Fine-tune with:\n  python scripts/finetune_detector.py --data {data_yaml} --epochs 50")


if __name__ == "__main__":
    main()