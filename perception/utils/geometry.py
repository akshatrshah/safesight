"""IoU and NMS, written by hand so I actually understand the mechanics, not just import them."""

from __future__ import annotations


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Overlap between two boxes: intersection area over union area. 1.0 = identical, 0.0 = no overlap."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

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
    Collapses overlapping duplicate boxes down to one per object. Keeps the
    highest-confidence box, drops anything overlapping it past the threshold, repeats.
    I don't actually call this in production, Ultralytics' own NMS handles that,
    this version exists so I have the algorithm proven and understood myself.
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
