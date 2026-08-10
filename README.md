# SafeSight: Real-Time Warehouse Perception and Risk Prediction System

> A Voxel-inspired research and engineering project exploring production computer vision for workplace safety. This project does not reproduce Voxel's proprietary technology, and does not claim any knowledge of Voxel's internal architecture — it demonstrates hands-on engineering with the perception problems (object detection, tracking, pose, activity recognition, risk prediction) described in Voxel's public Software Engineer, Perception job description.

## Problem Statement

Warehouses use security cameras for safety monitoring, but raw video isn't actionable on its own — someone has to watch it, or something has to interpret it. This project builds a perception pipeline that takes monocular camera footage and progressively extracts structure from it: first "what objects are here" (detection), then "where did they go" (tracking), and — in later milestones — "what are they doing" (pose + activity) and "is this becoming dangerous" (risk prediction). Each stage is built and evaluated independently before being combined, so every claim in this README is backed by a number, not an impression.

## Current Status: Milestone 1 — Detection Baseline + Evaluation Harness

This repo currently implements the **first milestone** of a larger planned system (full roadmap below). It is not yet a warehouse-specific or forklift-aware system — this milestone proves the detection + evaluation pipeline works correctly end-to-end, using pretrained COCO weights, before fine-tuning or adding tracking/pose/risk layers on top.

### What's implemented
- A YOLO detector wrapper (`perception/detection/detector.py`) with a stable, library-independent output schema
- Hand-implemented IoU and NMS (`perception/utils/geometry.py`) — implemented from scratch specifically to demonstrate understanding of the mechanism, even though production inference uses Ultralytics' built-in (and better-tested) NMS
- A visual inference script (`scripts/run_detection.py`)
- A latency/FPS benchmark comparing model sizes (`scripts/benchmark_latency.py`)
- A real evaluation harness computing precision, recall, mAP@50, mAP@50:95 (`perception/detection/evaluate.py`)
- Unit tests for both the geometry primitives and the detector wrapper (`tests/`)

## Results (Milestone 1)

**Detection latency**, measured on this repo's dev hardware (CPU, no GPU), single image, 30 timed frames after 5 warmup frames:

| Model | Avg Latency (ms) | FPS |
|---|---|---|
| yolov8n.pt | 130.29 | 7.68 |
| yolov8m.pt | 674.92 | 1.48 |

**Detection accuracy**, evaluated with Ultralytics' validation pipeline on the COCO8 sample set (a small placeholder dataset used only to prove the evaluation harness is wired correctly — see note below):

| Model | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---|---|---|---|
| yolov8n.pt | 0.621 | 0.833 | 0.888 | 0.629 |
| yolov8m.pt | 0.811 | 0.853 | 0.928 | 0.740 |

**Reading these results:** yolov8m is meaningfully more accurate on every metric but roughly 5x slower — a real accuracy/latency tradeoff, not a free upgrade. Which model is "right" depends on the deployment target (edge device vs. GPU server), which is exactly the kind of tradeoff analysis this project is meant to build fluency in.

> **Dataset note:** COCO8 is Ultralytics' built-in 8-image sample set, used here purely to confirm the evaluation harness computes correct precision/recall/mAP end-to-end. It is not a warehouse dataset and these numbers should not be read as "how well this detects forklift hazards" — that requires the warehouse/forklift dataset planned for Milestone 2. Numbers are re-run and replaced with real fine-tuned/domain results as later milestones land.

All numbers above are actually measured by running the scripts in this repo — none are estimated or fabricated.

## Setup

```bash
git clone <this-repo>
cd safesight
pip install -r requirements.txt
```

Model weights (`yolov8n.pt`, `yolov8m.pt`) download automatically on first use — they are not committed to the repo (see `.gitignore`).

## Usage

**Run detection on a folder of images:**
```bash
python scripts/run_detection.py --source sample_data --out outputs/annotated
```
Saves annotated images (bounding boxes + class + confidence) to `outputs/annotated/`.

**Benchmark latency across model sizes:**
```bash
python scripts/benchmark_latency.py --source sample_data/bus.jpg
```
Writes latency/FPS results to `experiments/detection_latency_results.json`.

**Run the accuracy evaluation (precision/recall/mAP):**
```bash
python perception/detection/evaluate.py
```
Writes results to `experiments/detection_eval_results.json`.

**Run tests:**
```bash
pytest tests/ -v
```

Target classes, confidence threshold, and which models to benchmark are all configured in `configs/detection.yaml` — nothing is hard-coded in the scripts.

## Core Concepts (for readers new to computer vision)

**Object detection** answers "what objects are in this image, and where?" for every frame independently — it has no concept of time or identity across frames (that comes later, in tracking).

**Bounding box** — a rectangle `(x1, y1, x2, y2)` marking where an object is in pixel coordinates.

**IoU (Intersection over Union)** — a 0–1 score measuring how much two boxes overlap: `overlap area / combined area`. It's the shared currency behind evaluation, tracking, and duplicate-box suppression. See the fully-commented implementation in `perception/utils/geometry.py`.

**NMS (Non-Max Suppression)** — a detector's raw output has many overlapping candidate boxes around the same real object. NMS keeps the highest-confidence box and discards others that overlap it above an IoU threshold, collapsing duplicates into one clean detection per object.

**Confidence score** — the model's own estimate of how sure it is that a detected box contains a real object of the predicted class.

**Precision** — of everything the model flagged, what fraction was actually correct? Low precision = false alarms.

**Recall** — of everything that was actually there, what fraction did the model find? Low recall = missed real hazards — the more dangerous failure mode for a safety system.

**mAP@50 / mAP@50:95** — "mean Average Precision," the standard detector accuracy metric. `@50` counts a prediction correct if it overlaps the true object by IoU ≥ 0.5 (forgiving). `@50:95` averages across stricter thresholds from 0.5 to 0.95, punishing loosely-fitted boxes — always report both, since a model can look great on `@50` and much weaker on `@50:95`.

**YOLO ("You Only Look Once")** — a single-stage detector: one forward pass over the whole image predicts all candidate boxes and classes directly, rather than first proposing regions and classifying each separately (the two-stage approach older detectors like Faster R-CNN use). Single-stage is faster; two-stage is often slightly more accurate on small/crowded objects — a tradeoff worth being able to discuss.

## Project Structure

```
safesight/
├── perception/
│   ├── detection/
│   │   ├── detector.py     # YOLO wrapper, stable output schema
│   │   └── evaluate.py     # precision/recall/mAP evaluation harness
│   └── utils/
│       └── geometry.py     # IoU + NMS, implemented from scratch for understanding
├── scripts/
│   ├── run_detection.py    # visual inference + annotated output
│   └── benchmark_latency.py
├── tests/
│   ├── test_geometry.py
│   └── test_detector.py
├── configs/
│   └── detection.yaml      # thresholds, target classes, models to compare
└── sample_data/             # sample image(s) for smoke-testing
```

## Roadmap

This is Milestone 1 of a larger planned system. Full context and later milestones are tracked separately; the near-term next steps are:

- **Milestone 2** — fine-tune detection on a warehouse/forklift dataset, repeat the accuracy/latency comparison on real domain data
- **Milestone 3** — multi-object tracking (ByteTrack), persistent IDs across frames, IDF1/MOTA evaluation
- **Milestone 4** — human pose estimation
- **Milestone 5** — temporal activity recognition (MLP → LSTM/GRU → Transformer ablation)
- Later milestones add spatial zone reasoning, risk prediction (heuristic + learned), an event engine, Kafka streaming, a Go backend, Redis/PostgreSQL, edge inference optimization (ONNX/TensorRT), Docker/Kubernetes deployment, and a monitoring dashboard.

## Limitations (Milestone 1)

- Evaluated on a small placeholder dataset (COCO8), not warehouse-specific data yet
- "Forklift" is not a native COCO class; `truck` is used as a proxy class until fine-tuning
- Latency numbers reflect this repo's dev hardware (CPU) and will differ on GPU/edge hardware
- No tracking, pose, or temporal reasoning yet — this milestone is detection-only by design
