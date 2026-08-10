from perception.utils.geometry import iou, non_max_suppression


def test_iou_identical_boxes_is_one():
    box = (10, 10, 50, 50)
    assert iou(box, box) == 1.0


def test_iou_no_overlap_is_zero():
    box_a = (0, 0, 10, 10)
    box_b = (100, 100, 110, 110)
    assert iou(box_a, box_b) == 0.0


def test_iou_partial_overlap_known_value():
    # Box A: (0,0)-(10,10), area 100
    # Box B: (5,5)-(15,15), area 100
    # Intersection: (5,5)-(10,10) = 5x5 = 25
    # Union = 100 + 100 - 25 = 175
    # IoU = 25 / 175
    box_a = (0, 0, 10, 10)
    box_b = (5, 5, 15, 15)
    expected = 25 / 175
    assert abs(iou(box_a, box_b) - expected) < 1e-9


def test_iou_touching_edges_is_zero_area_overlap():
    # Boxes touch at a single edge line -> zero-area intersection -> IoU 0
    box_a = (0, 0, 10, 10)
    box_b = (10, 0, 20, 10)
    assert iou(box_a, box_b) == 0.0


def test_nms_keeps_highest_confidence_and_suppresses_duplicate():
    # Two near-identical boxes around the "same" object, one low-conf duplicate,
    # plus one genuinely separate box that must survive.
    boxes = [
        (0, 0, 10, 10),    # duplicate, lower confidence -> suppressed
        (1, 1, 11, 11),    # highest confidence -> kept
        (100, 100, 120, 120),  # separate object -> kept
    ]
    scores = [0.6, 0.9, 0.8]

    kept = non_max_suppression(boxes, scores, iou_threshold=0.5)

    assert 1 in kept   # highest-confidence duplicate survives
    assert 0 not in kept  # lower-confidence duplicate suppressed
    assert 2 in kept   # separate object always survives


def test_nms_empty_input_returns_empty():
    assert non_max_suppression([], []) == []


def test_nms_no_overlap_keeps_all():
    boxes = [(0, 0, 10, 10), (50, 50, 60, 60), (200, 200, 210, 210)]
    scores = [0.5, 0.6, 0.7]
    kept = non_max_suppression(boxes, scores, iou_threshold=0.45)
    assert set(kept) == {0, 1, 2}
