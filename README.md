# SafeSight: Real-Time Warehouse Perception and Risk Prediction System

> A Voxel-inspired research and engineering project exploring production computer vision for workplace safety. This project does not reproduce Voxel's proprietary technology, and does not claim any knowledge of Voxel's internal architecture — it demonstrates hands-on engineering with the perception problems (object detection, tracking, pose, activity recognition, risk prediction, anomaly detection) described in Voxel's public Software Engineer, Perception job description.

## What This Project Is

Warehouses use security cameras for safety monitoring, but raw video isn't actionable on its own. This project builds a perception pipeline that takes monocular camera footage and progressively extracts structure from it: **what** objects are here (detection) → **which** object is which, over time (tracking) → **how** a person is positioned (pose) → **what** they're doing (temporal activity recognition) → **where** in the facility this is happening (spatial zones) → **is this becoming dangerous** (risk prediction) → **is anything statistically unusual** (anomaly detection).

Every stage is built and evaluated independently. Every number in this README is measured by actually running the code in this repo — none are estimated or fabricated. Where results are on synthetic/placeholder data rather than real labeled data, that's stated explicitly.

## Project Status

| Stage | What it answers | Status |
|---|---|---|
| Object detection | What objects are in this frame? | **Implemented**, pretrained COCO weights |
| Object detection (fine-tuned) | What objects are in this frame, on real warehouse data? | **In progress** — fine-tuned on a LOCO subset; forklift class needs a class-imbalance fix (see Fine-Tuning Results below) |
| Multi-object tracking | Which object is which, across frames? | **Implemented** |
| Pose estimation | How is this person's body positioned? | **Implemented** |
| Temporal activity recognition | What are they doing, over time? | **Implemented** (synthetic training data) |
| Spatial reasoning (zones) | Where in the facility is this happening? | **Implemented** |
| Risk prediction | Is this interaction becoming dangerous? | **Implemented** (heuristic + learned, synthetic training data) |
| Anomaly detection | Is something unusual happening? | **Implemented** (rule-based + ML, synthetic training data) |
| Event engine, Kafka, Go backend, Redis, PostgreSQL | Production event infrastructure | Deferred by design — this project is scoped to the perception layer |

## Setup

```bash
git clone <this-repo>
cd safesight
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

`setup.sh` creates a virtual environment and installs everything in `requirements.txt`. Run it once; after that just `source venv/bin/activate` in new terminal sessions.

## Repository Structure

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
│   ├── run_experiments.py             # Risk/activity/anomaly model comparisons
│   ├── download_loco.py                # Downloads a subset of the real LOCO dataset
│   ├── convert_loco_to_yolo.py          # Converts LOCO's COCO annotations to YOLO format
│   └── finetune_detector.py              # Fine-tunes YOLO on real labeled data
├── tests/                                  # 42 tests across every perception module
├── configs/
│   └── detection.yaml                       # Thresholds, target classes, models
├── sample_data/                                  # Sample image for smoke-testing
├── conftest.py                                    # Makes `perception` importable by pytest
└── setup.sh                                        # Environment bootstrap
```

## Running Everything

### Tests
```bash
pytest tests/ -v
```
Expect `42 passed`. Covers geometry math (IoU/NMS against hand-calculated values), detection schema, tracking velocity/history, pose joint-angle math, zone geometry, risk TTC calculations, temporal model shapes, and anomaly detection logic.

### Detection (pretrained COCO weights)
```bash
python scripts/run_detection.py --source sample_data --out outputs/annotated
```
Expect: annotated images saved to `outputs/annotated/`, boxes drawn around detected people/vehicles with class + confidence labels.

```bash
python scripts/benchmark_latency.py --source sample_data/bus.jpg
```
Expect: a latency (ms) and FPS table comparing `yolov8n` vs `yolov8m`.

```bash
python perception/detection/evaluate.py
```
Expect: precision/recall/mAP@50/mAP@50:95 on Ultralytics' COCO8 sample set (see Results below).

### Full integrated pipeline
```bash
python scripts/run_pipeline_demo.py --source sample_data/bus.jpg
```
Expect: tracking IDs assigned to detected people, pose estimated, zone membership checked, pairwise interaction risk scores printed. Output also saved to `experiments/pipeline_demo_output.json`.

### Research experiments (risk / activity / anomaly model comparisons)
```bash
python scripts/run_experiments.py
```
Expect: three comparisons printed and saved to `experiments/perception_experiments_results.json` — heuristic vs. learned risk model, baseline MLP vs. LSTM activity recognition, rule-based vs. Isolation Forest anomaly detection. All three currently run on synthetic data (see Results below for what that does and doesn't prove).

### Fine-tuning on real data (LOCO dataset)
```bash
python scripts/download_loco.py --images-per-subset 150
python scripts/convert_loco_to_yolo.py
python scripts/finetune_detector.py --data datasets/loco_yolo/data.yaml --epochs 15 --imgsz 416 --device mps
```
See Results below for what came out of a first real run. `--device mps` forces Apple Silicon GPU training (much faster than CPU on a Mac); use `--device 0` for an NVIDIA GPU, or omit the flag to let Ultralytics auto-detect.

## Results

### Detection — pretrained COCO weights (Milestone 1)
Evaluated on Ultralytics' COCO8 sample set — proves the evaluation harness is correct, not a warehouse-specific accuracy claim.

| Model | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---|---|---|---|
| yolov8n.pt | 0.621 | 0.833 | 0.888 | 0.629 |
| yolov8m.pt | 0.811 | 0.853 | 0.928 | 0.740 |

### Detection — fine-tuned on real LOCO warehouse data
Dataset: [LOCO (Logistics Objects in Context)](https://github.com/tum-fml/loco), Technical University of Munich — public domain, real logistics environments. Citation: Mayershofer, C., Holm, D.-M., Molter, B., Fottner, J., "LOCO: Logistics Objects in Context", IEEE ICMLA 2020.

First real fine-tuning run: 750 images (150 randomly sampled per warehouse subset, using LOCO's own official train/val split), 15 epochs, 416px, on Apple M3 (MPS).

| Class | Instances (val) | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---|---|---|---|---|
| pallet | 7,475 | 0.336 | 0.376 | 0.277 | 0.099 |
| small_load_carrier | 1,300 | 0.266 | 0.329 | 0.216 | 0.074 |
| stillage | 259 | 0.330 | 0.135 | 0.115 | 0.051 |
| pallet_truck | 136 | 0.459 | 0.175 | 0.197 | 0.085 |
| **forklift** | **12** | **0.054** | **0.083** | **0.017** | **0.010** |
| **Overall** | 9,182 | 0.289 | 0.220 | 0.165 | 0.064 |

**Honest reading — this is a real, diagnosed class-imbalance problem, not a training bug.** forklift is the rarest class in the entire LOCO dataset: 598 of 151,428 total annotations (0.39%), present in only 449 of 5,097 total images (~8.8%). Random sampling of 750 images pulled almost none of them (12 forklift instances landed in the 300-image validation split), so the model had almost no signal to learn forklift from — while `pallet` (79.5% of all annotations) learned a real, if modest, signal in just 15 epochs.

**Fix identified, not yet re-run:** `download_loco.py`'s sampling was updated to guarantee every forklift-containing image gets included first, with the remaining budget filled randomly — rather than leaving forklift representation to chance. Re-running the download + fine-tune with this fix is the next concrete step before treating forklift detection numbers as meaningful.

### Risk prediction: heuristic vs. learned (synthetic interaction data)
| Model | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|
| Heuristic (hand-written formula) | 0.55 | 0.40 | 0.35 |
| Learned (gradient-boosted trees) | 0.99 | 1.00 | 0.99 |

Honest reading: the learned model wins here largely because the heuristic's thresholds (150px, 3s TTC) were never tuned against the synthetic label boundaries — a real finding about the cost of an uncalibrated heuristic, not proof learned models are categorically better.

### Activity recognition: baseline MLP vs. LSTM (synthetic pose-sequence data)
| Model | Accuracy |
|---|---|
| Baseline (hand-engineered features + MLP) | 1.00 |
| LSTM (raw sequence) | 1.00 |

Honest reading: both saturate at 100% because the synthetic activity patterns were generated with cleanly separable motion signatures — this proves the training pipeline works end-to-end, not that the two approaches perform identically on real, messier data.

### Anomaly detection: rule-based vs. Isolation Forest (synthetic activity data)
| Detector | Precision | Recall | False Positives |
|---|---|---|---|
| Rule-based | 1.00 | 1.00 | 0 |
| Isolation Forest | 0.59 | 1.00 | 7 |

Honest reading: rule-based wins on precision here because the synthetic anomalies were generated to match its exact thresholds. The real tradeoff this demonstrates: rules catch known patterns precisely; Isolation Forest catches unknown patterns at the cost of more false positives.

## Core Concepts (for readers new to computer vision)

Brief summary of the key ideas behind each stage:

**Detection** — a trained neural network looks at one frame and outputs bounding boxes with class labels and confidence scores (via sigmoid on raw logits). **IoU** (intersection over union) measures box overlap; **NMS** removes duplicate overlapping boxes.

**Tracking** — detection has no memory across frames; tracking adds persistent IDs by predicting where each object should be next (Kalman filter) and matching new detections to that prediction via IoU. **ByteTrack** specifically recovers tracks through brief occlusion.

**Pose estimation** — 17 body keypoints (COCO format) per tracked person. **Joint angles** (via vector dot product) turn raw keypoint positions into geometric signals like "how bent is this knee."

**Temporal activity recognition** — a single frame is ambiguous (bent knees could mean lifting, falling, or tying a shoe); a 20-frame window resolves it. Compares hand-engineered summary features + MLP against a raw-sequence LSTM.

**Spatial reasoning** — named zones as polygons; point-in-polygon via ray casting, no ML involved.

**Risk prediction** — distance, closing speed, and **time-to-collision** (TTC) combined into a score, two ways: hand-written heuristic and gradient-boosted trees learned from labeled examples.

**Anomaly detection** — hand-written rules, or an unsupervised **Isolation Forest** trained only on examples of normal behavior.

## Limitations

- **Fine-tuned forklift detection is not yet reliable** — real class-imbalance issue, diagnosed and understood, fix identified but not yet re-validated (see Results above).
- **Risk, activity recognition, and anomaly detection are trained/evaluated on synthetic placeholder data** — proves the pipelines are correctly wired end-to-end, not real-world accuracy claims. Every module's docstring states this explicitly.
- **Monocular camera limitation** — all distance/velocity figures are in pixel units from a single 2D view, not real-world meters. Real-world distance requires camera calibration, not yet implemented.
- **No backend/streaming infrastructure** — Kafka, Go, Redis, PostgreSQL are deliberately out of scope for this phase; this project is scoped to demonstrate perception depth specifically.

## Roadmap

**Done:** detection (pretrained + first real fine-tuning pass), tracking, pose, temporal activity recognition, spatial zones, risk prediction (heuristic + learned), anomaly detection (rule-based + ML), lightweight real-data fine-tuning pipeline (LOCO).

**Next:**
1. Re-run LOCO fine-tuning with class-balanced sampling, validate forklift detection numbers are meaningful.
2. Replace synthetic risk/activity/anomaly training data with real labeled/logged data, re-run all experiments.
3. Edge inference optimization (ONNX/TensorRT comparison), Docker/Kubernetes deployment, monitoring dashboard.
4. Event engine, Kafka streaming, Go backend, Redis, PostgreSQL.