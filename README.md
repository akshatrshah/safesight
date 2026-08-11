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
| Object detection, fine tuned | What objects are in this frame, on real target domain data? | In progress. Fine tuned on a real logistics dataset. The rarest class needs a class imbalance fix I have already identified but not yet re run. |
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
│   ├── download_loco.py                Downloads a subset of a real, public logistics dataset
│   ├── convert_loco_to_yolo.py          Converts those annotations into YOLO format
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

```bash
python scripts/download_loco.py --images-per-subset 150
python scripts/convert_loco_to_yolo.py
python scripts/finetune_detector.py --data datasets/loco_yolo/data.yaml --epochs 15 --imgsz 416 --device mps
```

This downloads a small sample of a real, publicly available logistics dataset (I used LOCO, built by the Technical University of Munich, which is public domain), converts its annotations into YOLO format, and fine tunes a pretrained detector on it. `--device mps` uses an Apple Silicon GPU if you have one, which is dramatically faster than CPU. Use `--device 0` for an NVIDIA GPU, or leave the flag off to let it auto detect.

## Results

### Detection, pretrained COCO weights

Evaluated on a small standard sample set. This confirms my evaluation code is correct. It is not a claim about how well this detects warehouse specific objects.

| Model | Precision | Recall | mAP at 50 | mAP at 50 through 95 |
|---|---|---|---|---|
| yolov8n | 0.621 | 0.833 | 0.888 | 0.629 |
| yolov8m | 0.811 | 0.853 | 0.928 | 0.740 |

### Detection, fine tuned on real logistics data

I fine tuned on 750 real images (150 randomly sampled from each of 5 real warehouse recording sessions in the LOCO dataset), for 15 epochs, at 416 pixels, on an Apple M3 using its GPU.

| Class | Instances in validation | Precision | Recall | mAP at 50 | mAP at 50 through 95 |
|---|---|---|---|---|---|
| pallet | 7,475 | 0.336 | 0.376 | 0.277 | 0.099 |
| small load carrier | 1,300 | 0.266 | 0.329 | 0.216 | 0.074 |
| stillage | 259 | 0.330 | 0.135 | 0.115 | 0.051 |
| pallet truck | 136 | 0.459 | 0.175 | 0.197 | 0.085 |
| forklift | 12 | 0.054 | 0.083 | 0.017 | 0.010 |
| overall | 9,182 | 0.289 | 0.220 | 0.165 | 0.064 |

Here is what I actually found when I looked into why forklift performed so much worse than everything else. It is not a training bug. Forklift is genuinely the rarest object in the entire dataset: only 598 of 151,428 total annotations, about 0.39 percent, appearing in only about 9 percent of all images. When I randomly sampled 750 images, chance alone gave me almost no forklift examples to learn from, only 12 in my whole validation set. Meanwhile pallet, which makes up almost 80 percent of all annotations, actually learned a real signal in just 15 epochs.

I have already rewritten my sampling logic to fix this: it now guarantees every image containing a forklift gets included first, before filling the rest of the sample randomly. I have not re run training with the fix yet. That is my clear next step before I would treat forklift detection numbers as meaningful.

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

**Detection.** A trained neural network looks at one frame and outputs bounding boxes, each with a class label and a confidence score. That confidence score comes from a sigmoid function applied to the network's raw output. Two ideas make the raw output usable: IoU, which measures how much two boxes overlap, and non max suppression, which uses IoU to collapse a messy pile of overlapping candidate boxes down to one clean box per real object.

**Tracking.** Detection by itself has no memory. Every frame is evaluated completely independently, so there is no way to know if the person in frame two is the same person from frame one. Tracking fixes this by predicting where each object should be next, based on its recent motion, using a Kalman filter, and matching new detections against that prediction using IoU. I used ByteTrack specifically because it keeps low confidence detections around for a second matching pass, which helps recover a person's identity through brief occlusion, like walking behind a shelf.

**Pose estimation.** For each tracked person, I locate 17 body keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles. I compute joint angles from these using basic vector math, the dot product formula for the angle between two vectors, which turns raw keypoint positions into a real geometric signal, like how bent someone's knee currently is.

**Temporal activity recognition.** A single frame is genuinely ambiguous. Bent knees could mean someone is lifting something, falling, or just tying their shoe. The only way to tell these apart is to look at a window of frames, roughly the last second, and see how the position actually changed over time. I built and compared two approaches: hand engineered summary features fed into a small neural network, and an LSTM that learns directly from the raw sequence and carries its own internal memory forward frame by frame.

**Spatial reasoning.** I define named zones as polygons in pixel coordinates, and check whether a tracked object's position falls inside one using ray casting, a classic, exactly solvable geometry algorithm. No machine learning involved here at all, because this particular question does not need it.

**Risk prediction.** I combine distance, closing speed, and time to collision, which estimates how many seconds until two objects would meet if they kept their current velocity, into a single risk score. I built this two different ways on purpose: a hand written, fully interpretable formula, and a gradient boosted trees model trained on labeled examples, specifically so I could compare an interpretable approach against a learned one honestly.

**Anomaly detection.** This catches statistically unusual activity that my risk model was never explicitly designed to catch. I built this two ways as well: hand written rules, and an unsupervised Isolation Forest that trains only on examples of normal behavior and flags outliers based on how easily they can be statistically isolated from everything else.

## Limitations

I want to be direct about exactly where this project currently stands.

- My fine tuned forklift detection is not reliable yet. I have a real, diagnosed class imbalance problem and a fix identified, but I have not re run training with that fix.
- My risk prediction, activity recognition, and anomaly detection modules are all trained and evaluated on synthetic data that I generated myself, not real labeled data. Every module says this explicitly in its own code. These experiments prove my pipelines are correctly built end to end. They do not yet prove real world accuracy.
- This entire system currently uses a single 2D camera with no depth information. All of my distance and velocity numbers are in pixel units, not real world meters. Converting to real world distance would require camera calibration, which I have not implemented.
- I deliberately did not build the production backend, streaming, or database infrastructure for this phase. I chose to put my time into the perception layer specifically.

## What Is Next

1. Re run fine tuning with my class balanced sampling fix, and confirm forklift detection numbers are actually meaningful.
2. Replace every synthetic dataset in this project with real labeled or logged data, and re run every experiment.
3. Look into edge inference optimization, comparing PyTorch against ONNX Runtime and TensorRT, and containerize the system with Docker.
4. Build the event streaming and backend layer I deliberately left out of this phase.