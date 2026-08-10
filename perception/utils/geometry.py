"""
Core geometry primitives used everywhere in perception: IoU and NMS.

WHY THIS FILE EXISTS
---------------------
YOLO (and every other detector) doesn't output one clean box per object.
It outputs a raw prediction grid, and after filtering by confidence, you
are usually left with MANY overlapping boxes around the same real object
(e.g. 5 slightly different boxes all around the same forklift). Two ideas
fix that:

1. IoU (Intersection over Union): a single number from 0.0 to 1.0 that
   measures how much two boxes overlap. 0.0 = no overlap at all,
   1.0 = identical boxes. This is the currency used everywhere else:
   evaluation metrics (mAP), tracking (matching boxes across frames),
   and NMS all reduce to "how similar are these two boxes".

2. NMS (Non-Max Suppression): the algorithm that takes the messy pile of
   overlapping boxes and collapses them down to one box per real object,
   by using IoU to decide which boxes are "the same detection" and
   keeping only the most confident one.

Ultralytics' YOLO wrapper already does NMS internally and correctly — you
will NOT use this hand-rolled version in production. It exists so you can
open this file, step through it, and *actually* understand the mechanism
well enough to explain it in an interview, instead of treating it as a
black box inside someone else's library.
"""

from __future__ import annotations


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union between two boxes.

    Each box is (x1, y1, x2, y2) — top-left and bottom-right corners,
    in the same coordinate space (e.g. pixel coordinates of one frame).

    Intuition:
        - Find the overlapping rectangle between the two boxes.
        - IoU = (area of overlap) / (area of union)
        - If the boxes don't overlap at all, IoU = 0.0
        - If the boxes are identical, IoU = 1.0
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Coordinates of the intersection rectangle.
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # If the intersection rectangle is invalid (no overlap), width/height go negative.
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def non_max_suppression(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    iou_threshold: float = 0.45,
) -> list[int]:
    """
    Classic greedy NMS. Returns the INDICES (into `boxes`/`scores`) of the
    boxes to keep, highest-confidence-first.

    Algorithm (this is the whole idea, nothing more):
        1. Sort all boxes by confidence score, descending.
        2. Take the highest-confidence box, keep it, remove it from the list.
        3. Remove every remaining box whose IoU with the box you just kept
           is above `iou_threshold` (i.e. "this is basically the same
           detection, just a slightly different box for it").
        4. Repeat with whatever boxes are left.

    Threshold tradeoff (important for interviews):
        - Lower iou_threshold -> more aggressive suppression -> fewer boxes
          survive -> risk of merging two genuinely separate, close-together
          objects (e.g. two workers standing next to each other) into one.
        - Higher iou_threshold -> less aggressive -> more duplicate boxes
          survive for the same object.
        - 0.45 is a common default; the right value is domain-dependent.
    """
    if not boxes:
        return []

    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []

    while order:
        current = order.pop(0)
        keep.append(current)
        order = [i for i in order if iou(boxes[current], boxes[i]) <= iou_threshold]

    return keep
