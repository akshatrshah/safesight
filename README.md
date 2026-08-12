# SafeSight: Real-Time Warehouse Perception and Risk Prediction System

## Why I Built This

I wanted to go deep on computer vision and perception engineering instead of just reading about it. Object detection, tracking, pose estimation, temporal modeling, risk scoring: I wanted to actually build each of these from the ground up, understand the math behind them, and be able to explain every design decision I made, not just call a library function and hope it works.

So I built SafeSight: a full perception pipeline for a real, hard problem. Take ordinary security camera footage from a warehouse, no special sensors, no depth cameras, just a single 2D camera, and figure out whether a worker and a piece of moving equipment are heading toward a dangerous interaction, before it becomes an accident.

This project is general purpose. The pipeline architecture, the individual modules, and the engineering decisions here apply to any monocular video perception problem: retail loss prevention, construction site safety, sports analytics, traffic monitoring. Warehouse safety is just the concrete problem I picked to ground the work in something real.

Every number in this README comes from actually running the code in this repository. Where a result is based on synthetic or placeholder data instead of real labeled data, I say so explicitly. I would rather be upfront about what is proven and what is not than overstate what this system can currently do.

## What SafeSight Does

The pipeline takes video and progressively builds understanding, one layer at a time:

1. **Detection**: what objects are in this frame, and how confident is the model.
2. **Tracking**: which object is which, across frames, so the system has memory instead of treating every frame as brand new.
3. **Pose estimation**: how is a tracked person's body positioned right now.
4. **Temporal activity recognition**: what is that person doing, based on how their position changes over roughly the last second, not just one frozen frame.
5. **Spatial reasoning**: which zone of the facility is this happening in.
6. **Risk prediction**: is this interaction between a person and a moving object becoming dangerous, scored two different ways so I could compare them honestly.
7. **Anomaly detection**: is something statistically unusual happening, even if I never wrote an explicit rule for it.

I built and tested each stage independently before wiring them together, so I could always tell which part of the system was responsible for a given result.

## Project Status

| Stage | What it answers | Status |
|---|---|---|
| Object detection | What objects are in this frame? | Implemented, using pretrained COCO weights |
| Object detection, fine tuned | What objects are in this frame, on real target domain data? | Implemented. Fine tuned on real forklift and pallet imagery, run alongside pretrained person detection via a multi model tracker, confirmed working together on a real photo, see Results below. |
| Multi object tracking | Which object is which, across frames? | Implemented |
| Pose estimation | How is this person's body positioned? | Implemented |
| Temporal activity recognition | What are they doing, over time? | Implemented, trained on synthetic data |
| Spatial reasoning, zones | Where in the facility is this happening? | Implemented |
| Risk prediction | Is this interaction becoming dangerous? | Implemented, both a heuristic model and a learned model, trained on synthetic data |
| Anomaly detection | Is something unusual happening? | Implemented, both rule based and machine learning based, trained on synthetic data |
| Event streaming, backend, database | Production infrastructure to serve this at scale | Deliberately out of scope for this phase. I focused my time on the perception layer specifically. |

## Setup

```bash
git clone <this-repo>
cd safesight
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

`setup.sh` creates a virtual environment and installs everything in `requirements.txt`. Run it once. After that, just `source venv/bin/activate` in new terminal sessions.

## Repository Structure

```
safesight/
├── perception/
│   ├── detection/       YOLO wrapper, plus precision, recall, and mAP evaluation
│   ├── tracking/         ByteTrack wrapper, plus position history and velocity
│   ├── pose/              Pose estimation wrapper, plus joint angle math
│   ├── spatial/            Zone polygons and point in polygon checks
│   ├── risk/                Heuristic risk model and a learned (gradient boosted trees) risk model
│   ├── temporal/              Synthetic activity data, a baseline MLP, and an LSTM
│   ├── anomaly/                 Rule based and Isolation Forest anomaly detection
│   └── utils/                     IoU and NMS, implemented from scratch
├── scripts/
│   ├── run_detection.py            Visual detection sanity check
│   ├── benchmark_latency.py         Model size latency and FPS comparison
│   ├── run_pipeline_demo.py          The full stack wired together, end to end
│   ├── run_experiments.py             Risk, activity, and anomaly model comparisons
│   ├── download_loco.py                Downloads a subset of the LOCO dataset, all 5 classes
│   ├── convert_loco_to_yolo.py          Converts those annotations into YOLO format, all 5 classes
│   ├── download_loco_full.py            Downloads the full LOCO archive in one connection, faster and more reliable on a fast network
│   ├── prepare_poc_subset.py            Filters the full archive down to forklift and pallet only, what I actually used for my real results
│   └── finetune_detector.py              Fine tunes the detector on real labeled data
├── tests/                                  42 tests across every perception module
├── configs/
│   └── detection.yaml                       Thresholds, target classes, and model choice
├── sample_data/                               A sample image for smoke testing
├── conftest.py                                 Makes the perception package importable by pytest
└── setup.sh                                     Environment bootstrap script
```

## Running Everything

### Tests

```bash
pytest tests/ -v
```

I expect this to show 42 passed. The tests cover geometry math checked against hand calculated values, detection output schema, tracking velocity and history, pose joint angle math, zone geometry, risk time to collision calculations, temporal model shapes, and anomaly detection logic.

### Detection, pretrained weights

```bash
python scripts/run_detection.py --source sample_data --out outputs/annotated
```

This saves annotated images to `outputs/annotated/`, with boxes drawn around detected people and vehicles, labeled with class and confidence.

```bash
python scripts/benchmark_latency.py --source sample_data/bus.jpg
```

This prints a latency and FPS comparison between the small and medium YOLO model sizes.

```bash
python perception/detection/evaluate.py
```

This prints precision, recall, mAP at 50, and mAP at 50 through 95 on a small sample evaluation set. See Results below for what I actually measured.

### The full integrated pipeline

```bash
python scripts/run_pipeline_demo.py --source sample_data/bus.jpg
```

This runs tracking, pose estimation, zone membership checks, and pairwise interaction risk scoring together on one real image, and saves the output to `experiments/pipeline_demo_output.json`.

### Research experiments

```bash
python scripts/run_experiments.py
```

This runs and saves three comparisons: heuristic versus learned risk scoring, a hand engineered baseline versus an LSTM for activity recognition, and rule based versus Isolation Forest anomaly detection. All three currently run on synthetic data I generated myself. See Results below for exactly what that does and does not prove.

### Fine tuning on real data

This is the actual sequence I used to produce the real results below, downloading the full LOCO archive once over a single connection rather than many small requests, then filtering locally to just the two classes I need:

```bash
python scripts/download_loco_full.py
python scripts/prepare_poc_subset.py
python scripts/finetune_detector.py --data datasets/loco_poc/data.yaml --epochs 15 --imgsz 640
```

`download_loco_full.py` pulls the full LOCO image archive (about 770 MB) and annotation files from the Technical University of Munich, public domain. `prepare_poc_subset.py` filters that down to only images containing forklift or pallet, guaranteeing every forklift image gets included first since it is the rare class, and converts everything to YOLO format, capped at 1,000 images by default. Add `--device mps` for an Apple Silicon GPU, or `--device 0` for an NVIDIA GPU.

An older, slower path also still works, `download_loco.py` plus `convert_loco_to_yolo.py`, which keeps all 5 LOCO classes instead of just 2, useful if I want to extend beyond forklift and pallet later, but noticeably slower to download since it fetches images one at a time instead of as one archive.

## Results

A note before the numbers below: I trained and evaluated everything here on a small fraction of the full available data, by choice, given the compute and time I had available for this phase. If any accuracy number below looks low, that is very likely why, not a flaw in the pipeline itself.

### Detection, pretrained COCO weights

Evaluated on a small standard sample set. This confirms my evaluation code is correct. It is not a claim about how well this detects warehouse specific objects.

| Model | Precision | Recall | mAP at 50 | mAP at 50 through 95 |
|---|---|---|---|---|
| yolov8n | 0.621 | 0.833 | 0.888 | 0.629 |
| yolov8m | 0.811 | 0.853 | 0.928 | 0.740 |

### Detection, fine tuned on real logistics data, first attempt

My first pass trained on 750 randomly sampled images across all 5 LOCO object classes, and forklift, the rarest class in the dataset, ended up with only 12 validation instances by chance, essentially unusable, 0.017 mAP at 50. I traced this to the actual root cause: forklift makes up only 0.39 percent of all annotations in the dataset. Random sampling alone was never going to give me enough forklift examples to learn from.

### Detection, fine tuned on real logistics data, second attempt

I fixed this two ways: narrowed the dataset down to just forklift and pallet, the two classes this project actually needs, and rewrote my sampling to guarantee every forklift-containing image gets included first before filling the rest randomly. I trained on 1,000 images (800 train, 200 validation) this way, for 15 epochs at 640 pixels, on a 16-core CPU machine (NC State's VCL), about 46 minutes of training time.

| Class | Instances in validation | Precision | Recall | mAP at 50 | mAP at 50 through 95 |
|---|---|---|---|---|---|
| forklift | 124 | 0.440 | 0.403 | 0.356 | 0.154 |
| pallet | 3,746 | 0.480 | 0.397 | 0.380 | 0.137 |
| overall | 3,870 | 0.460 | 0.400 | 0.368 | 0.145 |

Forklift went from 0.017 to 0.356 mAP at 50, a direct result of fixing the sampling, not a fluke, forklift now performs essentially on par with pallet. This is still below the roughly 65 percent mAP at 50 a properly tuned, full-scale run on this dataset has achieved in published research, so I am treating this as a solid proof of concept, not a finished model, but it is a real, working, honestly measured result on real warehouse imagery, not a placeholder.

### Risk prediction: heuristic versus learned

Tested on synthetic interaction data I generated myself.

| Model | Precision, macro average | Recall, macro average | F1, macro average |
|---|---|---|---|
| Heuristic, a hand written formula | 0.55 | 0.40 | 0.35 |
| Learned, gradient boosted trees | 0.99 | 1.00 | 0.99 |

The learned model wins by a wide margin here, but mostly because I never actually tuned my heuristic's thresholds against the real boundaries of my synthetic labels. That is a genuinely useful thing to have learned: an uncalibrated heuristic can lose badly to a learned model, and that is a fact about calibration, not proof that learned models are always better.

### Activity recognition: a hand engineered baseline versus an LSTM

Tested on synthetic pose sequence data I generated myself.

| Model | Accuracy |
|---|---|
| Baseline, hand engineered features plus a small neural network | 1.00 |
| LSTM, learns directly from the raw sequence | 1.00 |

Both models hit 100 percent, which taught me something I did not expect going in: my synthetic activity patterns were too cleanly separable to actually stress test the difference between the two approaches. This experiment proves my training pipeline works correctly end to end. It does not prove the two modeling approaches perform the same on real, messier data.

### Anomaly detection: rule based versus Isolation Forest

Tested on synthetic activity data I generated myself.

| Detector | Precision | Recall | False positives |
|---|---|---|---|
| Rule based | 1.00 | 1.00 | 0 |
| Isolation Forest | 0.59 | 1.00 | 7 |

Rule based wins on precision here mostly because I generated the synthetic anomalies to match its exact thresholds, which gave it something of a home field advantage. The real tradeoff this demonstrates is genuine though: rules catch known patterns precisely, while Isolation Forest can catch patterns nobody thought to write a rule for, at the cost of more false positives.

## What I Learned, Concept by Concept

- **Detection**: a network outputs boxes plus a sigmoid confidence score. IoU measures box overlap, non max suppression uses it to collapse duplicate boxes down to one per object.
- **Tracking**: detection alone has no memory across frames. I used ByteTrack, which predicts each object's next position with a Kalman filter, matches new detections to that prediction with IoU, and specifically keeps low confidence detections around for a second pass to survive brief occlusion.
- **Pose estimation**: 17 body keypoints per tracked person, with joint angles computed from basic vector math, the dot product formula for the angle between two vectors.
- **Temporal activity recognition**: one frame is ambiguous, bent knees could mean lifting, falling, or tying a shoe. I compared hand engineered summary features against an LSTM that learns directly from the raw sequence.
- **Spatial reasoning**: named zones as polygons, checked with ray casting, no machine learning needed for this one.
- **Risk prediction**: distance, closing speed, and time to collision combined into a score, built two ways on purpose, a hand written interpretable formula and a gradient boosted trees model, so I could compare them honestly.
- **Anomaly detection**: catches unusual activity my risk model was never explicitly designed for, built both as hand written rules and as an unsupervised Isolation Forest.

## Limitations

I want to be direct about exactly where this project currently stands.

- My fine tuned model still only knows forklift and pallet on its own, since fine tuning on a 2-class dataset replaced the model's detection head. I fixed the practical impact of this though, by running the pretrained person detector and my fine tuned forklift detector side by side on the same frame, merged with unique IDs. I confirmed this works on a real photo with both people and a forklift in it, correctly tracked both, correctly computed zone membership, and correctly produced a differentiated risk score based on real distance. One honest caveat, that specific test used a single still image fed through the tracker multiple times to build up velocity history, so the risk score there was driven by distance and zone only, not real motion, time to collision only engages with genuinely moving video.
- My risk prediction, activity recognition, and anomaly detection modules are all trained and evaluated on synthetic data that I generated myself, not real labeled data. Every module says this explicitly in its own code. These experiments prove my pipelines are correctly built end to end. They do not yet prove real world accuracy.
- This entire system currently uses a single 2D camera with no depth information. All of my distance and velocity numbers are in pixel units, not real world meters. Converting to real world distance would require camera calibration, which I have not implemented.
- I deliberately did not build the production backend, streaming, or database infrastructure for this phase. I chose to put my time into the perception layer specifically.

## What Is Next

1. Test the multi model pipeline on real video instead of a repeated still frame, so time to collision actually engages with genuine motion instead of just distance and zone.
2. Scale up the fine tuning run with more images and more epochs now that the sampling and class-selection approach is proven to work.
3. Replace every synthetic dataset in this project with real labeled or logged data, and re run every experiment.
4. Look into edge inference optimization, comparing PyTorch against ONNX Runtime and TensorRT, and containerize the system with Docker.
5. Build the event streaming and backend layer I deliberately left out of this phase.
