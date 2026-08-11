# SafeSight: Real-Time Warehouse Perception and Risk Prediction System

> A Voxel-inspired research and engineering project exploring production computer vision for workplace safety. This project does not reproduce Voxel's proprietary technology, and does not claim any knowledge of Voxel's internal architecture — it demonstrates hands-on engineering with the perception problems (object detection, tracking, pose, activity recognition, risk prediction, anomaly detection) described in Voxel's public Software Engineer, Perception job description.

## Problem Statement

Warehouses use security cameras for safety monitoring, but raw video isn't actionable on its own. This project builds a perception pipeline that takes monocular camera footage and progressively extracts structure from it: **what** objects are here (detection) → **which** object is which, over time (tracking) → **how** a person is positioned (pose) → **what** they're doing (activity recognition) → **where** in the facility this is happening (spatial zones) → **is this becoming dangerous** (risk prediction) → **is anything statistically unusual** (anomaly detection). Every stage is built and evaluated independently, and every number in this README is measured by actually running the code in this repo — none are estimated or fabricated.

## Current Status: Full Perception Stack Implemented

This repo currently implements the entire **perception layer** (Python/PyTorch/OpenCV) end-to-end. The backend/event-streaming layer (Kafka, Go, Redis, PostgreSQL) is intentionally deferred — this phase is scoped specifically to demonstrate deep, hands-on computer vision ability, which is the gap this project exists to close.

| Stage | What it answers | Status |
|---|---|---|
| Object detection | What objects are in this frame? | **Implemented** |
| Multi-object tracking | Which object is which, across frames? | **Implemented** |
| Pose estimation | How is this person's body positioned? | **Implemented** |
| Temporal activity recognition | What are they doing, over time? | **Implemented** |
| Spatial reasoning (zones) | Where in the facility is this happening? | **Implemented** |
| Risk prediction | Is this interaction becoming dangerous? | **Implemented** |
| Anomaly detection | Is something unusual happening? | **Implemented** |
| Event engine, Kafka, Go backend, Redis, PostgreSQL | Production event infrastructure | Deferred — next phase |
| Fine-tuned forklift detector | Real (not COCO-proxy) vehicle detection | Deferred — needs a labeled forklift dataset |

## Setup

```bash
git clone <this-repo>
cd safesight
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

## Usage

**Run the full integration pipeline** (tracking → pose → zones → risk, wired together on a real image):
```bash
python scripts/run_pipeline_demo.py --source sample_data/bus.jpg
```

**Run all three research experiments** (heuristic vs learned risk, baseline vs LSTM activity recognition, rule-based vs ML anomaly detection):
```bash
python scripts/run_experiments.py
```

**Run detection specifically:**
```bash
python scripts/run_detection.py --source sample_data --out outputs/annotated
python scripts/benchmark_latency.py --source sample_data/bus.jpg
python perception/detection/evaluate.py
```

**Run tests:**
```bash
pytest tests/ -v
```
42 tests currently pass, covering geometry math, detection schema, tracking velocity/history, pose joint-angle math, zone geometry, risk TTC calculations, temporal model shapes, and anomaly detection logic.

## Results

### Detection (Milestone 1 — real COCO8 evaluation data)
| Model | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---|---|---|---|
| yolov8n.pt | 0.621 | 0.833 | 0.888 | 0.629 |
| yolov8m.pt | 0.811 | 0.853 | 0.928 | 0.740 |

> Evaluated on Ultralytics' COCO8 sample set — proves the evaluation harness is correct, not a warehouse-specific accuracy claim (see Limitations).

### Risk prediction: heuristic vs learned (synthetic interaction data)
| Model | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|
| Heuristic (hand-written formula) | 0.55 | 0.40 | 0.35 |
| Learned (gradient-boosted trees) | 0.99 | 1.00 | 0.99 |

**Honest reading of this result:** the learned model dramatically outperforms the heuristic here — but this is on synthetic data where the heuristic's hand-picked thresholds (150px distance, 3s TTC) were never actually tuned against the synthetic labeling rule's real boundaries. This is a genuine, useful finding: it demonstrates *why* an untuned heuristic can lose badly to a learned model, and that heuristic thresholds need calibration against real outcome data before a fair comparison is possible. It is not evidence that heuristics are inherently worse — it's evidence that this particular heuristic's thresholds need tuning.

### Activity recognition: baseline MLP vs LSTM (synthetic pose-sequence data)
| Model | Accuracy |
|---|---|
| Baseline (hand-engineered features + MLP) | 1.00 |
| LSTM (raw sequence) | 1.00 |

**Honest reading:** both models saturate at 100% because the synthetic activity patterns (standing/walking/bending/falling) were generated with cleanly distinct, separable motion signatures — this does not demonstrate the LSTM's advantage over hand-engineered features, since the task is too easy to show a gap. Real, messier pose-sequence data (Milestone 4/5's real dataset) is needed to produce a meaningful comparison here — this result currently only proves the training/evaluation pipeline itself is wired correctly end-to-end.

### Anomaly detection: rule-based vs Isolation Forest (synthetic activity data)
| Detector | Precision | Recall | False Positives |
|---|---|---|---|
| Rule-based | 1.00 | 1.00 | 0 |
| Isolation Forest | 0.59 | 1.00 | 7 |

**Honest reading:** the rule-based detector wins on precision here specifically because the synthetic anomalies were generated to match its exact thresholds — an advantage that doesn't necessarily hold on real data with anomalies the rules weren't designed for. Isolation Forest catches everything (perfect recall) but at the cost of more false positives, since it flags any statistical outlier rather than only the specific patterns the rules were written for. This tradeoff (rule precision on known patterns vs. ML recall on unknown patterns) is the expected, real tradeoff between these two approaches.

## Core Concepts (for readers new to computer vision)

**Detection** — a trained neural network looks at one frame and outputs bounding boxes with class labels and confidence scores. See `perception/detection/detector.py` and `perception/utils/geometry.py` for IoU/NMS mechanics.

**Tracking** — detection has no memory across frames; tracking adds persistent IDs by predicting where each object should be next (via a Kalman filter) and matching new detections to that prediction using IoU. **ByteTrack** specifically recovers tracks through brief occlusion by keeping low-confidence detections as a second-pass fallback instead of discarding them.

**Pose estimation** — for each tracked person, locate 17 body keypoints (COCO format: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles). **Joint angles** (e.g. hip-knee-ankle) turn raw keypoint positions into geometric signals like "how bent is this knee."

**Temporal activity recognition** — a single frame of pose data is ambiguous (bent knees could mean lifting, falling, or tying a shoe); activity recognition looks at a **window** of consecutive frames to distinguish them. Compares hand-engineered summary features + MLP against a raw-sequence LSTM.

**Spatial reasoning** — named zones defined as **polygons** in pixel coordinates; whether a point is inside a zone is a classic **point-in-polygon** geometry problem (ray casting), no ML involved.

**Risk prediction** — combines distance, closing speed, and **time-to-collision** (TTC — if both objects kept their current velocity, how many seconds until they'd meet) into a risk score, computed two ways: a hand-written heuristic formula, and a gradient-boosted-trees model trained on labeled examples.

**Anomaly detection** — catches statistically unusual activity that risk prediction wasn't explicitly designed to catch, via hand-written rules (e.g. "flag prolonged inactivity") or an unsupervised **Isolation Forest** model trained only on examples of normal behavior.

## Project Structure

```
safesight/
├── perception/
│   ├── detection/       # YOLO wrapper + precision/recall/mAP evaluation
│   ├── tracking/         # ByteTrack wrapper + position history/velocity
│   ├── pose/              # YOLOv8-pose wrapper + joint angle math
│   ├── spatial/            # Zone polygons + point-in-polygon checks
│   ├── risk/                # Heuristic + learned (gradient-boosted trees) risk models
│   ├── temporal/              # Synthetic activity data + baseline MLP + LSTM
│   ├── anomaly/                 # Rule-based + Isolation Forest anomaly detection
│   └── utils/                     # IoU + NMS, implemented from scratch
├── scripts/
│   ├── run_detection.py            # Visual detection sanity check
│   ├── benchmark_latency.py         # Model-size latency/FPS comparison
│   ├── run_pipeline_demo.py          # Full stack wired together, end to end
│   └── run_experiments.py             # All 3 model-comparison experiments
├── tests/                                # 42 tests across every module
├── configs/
│   └── detection.yaml                     # Thresholds, target classes, models
└── sample_data/                             # Sample image for smoke-testing
```

## Roadmap

**Done (this phase):** detection, tracking, pose, temporal activity recognition, spatial zones, risk prediction (heuristic + learned), anomaly detection (rule-based + ML).

**Next (deferred by design, not forgotten):**
- Fine-tune detection on a real, labeled forklift dataset — currently `truck` is used as a COCO proxy class, and cross-module demos (e.g. `run_pipeline_demo.py`) use a second tracked person as an explicit stand-in for a vehicle, clearly labeled as such.
- Replace every synthetic dataset (risk interactions, activity sequences, anomaly activity logs) with real labeled data, and re-run every experiment above with real numbers.
- Event engine, Kafka streaming, Go backend, Redis, PostgreSQL.
- Edge inference optimization (ONNX/TensorRT), Docker/Kubernetes deployment, AWS, monitoring dashboard.

## Limitations

- **Synthetic data, clearly labeled throughout:** risk model, activity recognition, and anomaly detection are all trained/evaluated on synthetic placeholder data (see each module's docstring) — proving the pipelines are wired correctly end-to-end, not making real-world accuracy claims yet.
- **No forklift-specific detector yet** — `truck` is used as the closest available COCO proxy class; a real fine-tuned detector needs a labeled forklift dataset (Milestone 2, deferred).
- **Monocular camera limitation:** all distance/velocity figures are in pixel units from a single 2D view, not real-world meters — converting to real-world distance requires camera calibration, which isn't implemented yet.
- **No backend/streaming infrastructure yet** — this phase is deliberately scoped to the perception layer only.
