# SafeSight: Real-Time Warehouse Perception and Risk Prediction System

## Why I Built This

I wanted to go deep on computer vision and perception engineering instead of just reading about it, so I built a full monocular video perception pipeline: detection, tracking, pose estimation, temporal modeling, and risk scoring, each built and tested by hand, not just called from a library.

The concrete problem: take ordinary security camera footage from a warehouse, no depth sensors, one 2D camera, and figure out whether a worker and a moving vehicle are heading toward a dangerous interaction before it happens. The architecture generalizes to any monocular video perception problem, retail loss prevention, construction safety, sports analytics, warehouse safety is just the concrete case I grounded it in.

Every number below comes from actually running this code. Where a result is based on synthetic data instead of real labeled data, I say so directly.

## What SafeSight Does

1. **Detection**: what objects are in this frame, and how confident is the model.
2. **Tracking**: which object is which, across frames.
3. **Pose estimation**: how a tracked person's body is positioned.
4. **Temporal activity recognition**: what someone is doing, based on how their position changes over roughly the last second.
5. **Spatial reasoning**: which zone of the facility this is happening in.
6. **Risk prediction**: is a person-vehicle interaction becoming dangerous, scored two ways so I could compare them.
7. **Anomaly detection**: is something statistically unusual happening, even without an explicit rule for it.

Each stage was built and tested independently before wiring them together.

## Project Status

| Stage | Status |
|---|---|
| Object detection | Implemented, pretrained COCO weights |
| Object detection, fine tuned | Implemented. Fine tuned on real forklift/pallet imagery, running alongside pretrained person detection via a multi model tracker. See Results. |
| Multi object tracking | Implemented |
| Pose estimation | Implemented |
| Temporal activity recognition | Implemented, synthetic training data |
| Spatial reasoning, zones | Implemented |
| Risk prediction | Implemented, heuristic and learned models, synthetic training data |
| Anomaly detection | Implemented, rule based and Isolation Forest, synthetic training data |
| Backend API, database, live event streaming | Implemented. FastAPI + SQLite, job queue, Server-Sent Events. See Backend and Dashboard. |
| Dashboard | Implemented. Streamlit client of the backend API. |

## Setup

```bash
git clone <this-repo>
cd safesight
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

## Repository Structure

```
safesight/
├── perception/
│   ├── detection/       YOLO wrapper, precision/recall/mAP evaluation
│   ├── tracking/         ByteTrack wrapper, position history, velocity, multi model merging
│   ├── pose/              Pose estimation, joint angle math
│   ├── spatial/            Zone polygons, point in polygon checks
│   ├── risk/                Heuristic and learned (gradient boosted trees) risk models
│   ├── temporal/              Synthetic activity data, baseline MLP, LSTM
│   ├── anomaly/                 Rule based and Isolation Forest anomaly detection
│   └── utils/                     IoU and NMS, implemented from scratch
├── scripts/
│   ├── run_detection.py            Visual detection sanity check
│   ├── benchmark_latency.py         Model size latency/FPS comparison
│   ├── run_pipeline_demo.py          Full stack wired together, end to end
│   ├── run_experiments.py             Risk/activity/anomaly model comparisons
│   ├── download_loco_full.py            Downloads the LOCO dataset archive
│   ├── prepare_poc_subset.py             Filters to forklift + pallet, what I used for real results
│   └── finetune_detector.py               Fine tunes the detector on real labeled data
├── backend/
│   ├── main.py                             FastAPI app: upload, jobs, timeline, alerts, SSE streams
│   ├── worker.py                            Background job runner around VideoProcessor
│   ├── db.py                                 SQLite storage: jobs, per-frame stats, interaction events
│   ├── events.py                              In-process event bus feeding the live streams
│   └── config.py                               Settings, all overridable by environment variable
├── dashboard/app.py                            Streamlit dashboard, talks to the backend over HTTP
├── app.py                                       Original standalone Streamlit page (runs the pipeline inline, no backend)
├── tests/                                   Tests across every module, including the backend
├── configs/detection.yaml                     Thresholds, target classes, model choice
└── sample_data/                                 Sample image for smoke testing
```

## Running It

```bash
pytest tests/ -v                                                    # 42 tests
python scripts/run_detection.py --source sample_data --out outputs/annotated
python scripts/run_pipeline_demo.py --source sample_data/bus.jpg    # full stack, one image
python scripts/run_experiments.py                                    # risk/activity/anomaly comparisons

# Fine tuning on real data, what I actually ran to produce the results below:
python scripts/download_loco_full.py
python scripts/prepare_poc_subset.py
python scripts/finetune_detector.py --data datasets/loco_poc/data.yaml --epochs 15 --imgsz 640
```

`download_loco_full.py` pulls the full LOCO archive (Technical University of Munich, public domain, ~770MB) in one connection. `prepare_poc_subset.py` filters to forklift and pallet only, guaranteeing forklift images get included first since it's the rare class. Add `--device mps` (Apple Silicon) or `--device 0` (NVIDIA) to the fine-tuning command for GPU training.

## Backend and Dashboard

The backend turns the pipeline from a script into a service. You upload a video, it queues a job, one background worker runs the real pipeline, and everything lands in SQLite: job status, per-frame risk and object counts, and every person-vehicle interaction. While a job runs, per-frame progress and MEDIUM/HIGH risk alerts stream out over Server-Sent Events.

```bash
uvicorn backend.main:create_app --factory --port 8000    # API, docs at http://localhost:8000/docs
streamlit run dashboard/app.py                            # dashboard, in a second terminal
```

| Endpoint | Purpose |
|---|---|
| `POST /api/jobs` | Upload a video (multipart: `file`, `frame_skip`, `max_frames`, `use_vehicle_model`), returns 202 and the queued job |
| `GET /api/jobs`, `GET /api/jobs/{id}` | List jobs, or one job with progress, summary, and detection breakdown |
| `GET /api/jobs/{id}/timeline` | Per-frame max risk and object counts |
| `GET /api/jobs/{id}/events?min_risk=` | Stored person-vehicle interactions |
| `GET /api/jobs/{id}/video` | Annotated output video |
| `GET /api/jobs/{id}/stream` | Live SSE: `frame`, `alert`, then `job_done` or `job_failed` |
| `GET /api/alerts?min_risk=` | MEDIUM/HIGH interactions across all jobs |
| `GET /api/stream` | Live SSE of every event from every job |
| `GET /api/stats`, `GET /api/health`, `DELETE /api/jobs/{id}` | Overview counts, liveness, cleanup |

Configuration, all environment variables: `SAFESIGHT_DATA_DIR` (default `outputs/backend`), `SAFESIGHT_PERSON_MODEL`, `SAFESIGHT_VEHICLE_MODEL` (path to the fine tuned weights, defaults to `model_name` in `configs/detection.yaml`), `SAFESIGHT_CONFIDENCE`, `SAFESIGHT_MAX_UPLOAD_MB`, and `SAFESIGHT_API_URL` for the dashboard.

Design choices worth knowing: jobs run one at a time because the models are memory and compute heavy, a fresh processor is built per job so tracker state never leaks between videos, and jobs left `running` by a crashed server are marked failed on the next startup instead of hanging forever. Backend tests use a scripted fake processor so they don't need any model weights.

## Results

I trained and evaluated everything below on a deliberately small slice of the available data, given the time and compute I had for this phase. Numbers below that look modest reflect that choice, not a flaw in the pipeline.

### Detection, pretrained COCO weights

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| yolov8n | 0.621 | 0.833 | 0.888 | 0.629 |
| yolov8m | 0.811 | 0.853 | 0.928 | 0.740 |

### Detection, fine tuned on real logistics data (LOCO dataset)

My first pass trained on 750 images across all 5 LOCO classes. Forklift, the rarest class (0.39% of annotations), ended up with only 12 validation instances by chance, mAP@50 of 0.017, essentially unusable. I fixed it two ways: narrowed to just forklift and pallet, and rewrote sampling to guarantee forklift images get included first. Retrained on 1,000 images (800 train, 200 val), 15 epochs at 640px, ~46 minutes on a 16-core CPU machine:

| Class | Val instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| forklift | 124 | 0.440 | 0.403 | 0.356 | 0.154 |
| pallet | 3,746 | 0.480 | 0.397 | 0.380 | 0.137 |
| overall | 3,870 | 0.460 | 0.400 | 0.368 | 0.145 |

Forklift went from 0.017 to 0.356 mAP@50, a direct result of the sampling fix, now on par with pallet. Still below the ~65% mAP@50 a full-scale tuned run on this dataset has achieved in published research, so I treat this as a solid proof of concept, not a finished model, but a real, honestly measured result on real warehouse imagery.

### Risk prediction: heuristic vs. learned (synthetic data)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Heuristic formula | 0.55 | 0.40 | 0.35 |
| Learned (gradient boosted trees) | 0.99 | 1.00 | 0.99 |

The learned model wins mostly because I never tuned the heuristic's thresholds against the real label boundaries, a real finding about calibration, not proof learned models are always better.

### Activity recognition: hand engineered baseline vs. LSTM (synthetic data)

| Model | Accuracy |
|---|---|
| Baseline (hand features + small network) | 1.00 |
| LSTM (raw sequence) | 1.00 |

Both hit 100%, which tells me the synthetic patterns were too cleanly separable to stress-test the difference, this proves the training pipeline works end to end, not that both approaches perform equally on real data.

### Anomaly detection: rule based vs. Isolation Forest (synthetic data)

| Detector | Precision | Recall | False positives |
|---|---|---|---|
| Rule based | 1.00 | 1.00 | 0 |
| Isolation Forest | 0.59 | 1.00 | 7 |

Rule based wins on precision mostly because I generated the synthetic anomalies to match its exact thresholds. The real tradeoff: rules catch known patterns precisely, Isolation Forest catches unknown patterns at the cost of more false positives.

## Limitations

- My fine tuned model alone only detects forklift and pallet, since fine tuning on those two classes replaced the original detection head. I fixed this by running it alongside the pretrained person detector in a merged multi model tracker, confirmed working on real photos with both people and a forklift present, correctly tracked, zoned, and risk scored. Not yet tested on real video with genuine motion, only on repeated still frames, so time to collision hasn't been exercised with real velocity yet.
- Risk prediction, activity recognition, and anomaly detection are trained and evaluated on synthetic data I generated myself, not real labeled data. This proves the pipelines are correctly built end to end, not real-world accuracy.
- Single 2D camera, no depth. All distance and velocity figures are in pixel units, not real-world meters, converting would require camera calibration, which I haven't implemented.
- The backend is a single process with SQLite and an in-process event bus, fine for one machine, not horizontally scalable. There is no authentication, and CORS only allows localhost. Multiple workers, Postgres, and a real message broker (Redis/Kafka) would be the next step up.
- The backend and dashboard have only been exercised on uploaded video files, not a live camera (RTSP) feed, and the MEDIUM/HIGH alert path has only been verified against scripted data, not a real forklift-plus-person video.

## What Is Next

1. Test the multi model pipeline on real video, so time to collision engages with genuine motion.
2. Scale up fine tuning with more images and epochs, now that the sampling approach is proven.
3. Replace synthetic risk/activity/anomaly data with real labeled data, re-run every experiment.
4. Edge inference optimization (ONNX/TensorRT), containerize with Docker.
5. ~~Build the event streaming and backend layer.~~ Done, see Backend and Dashboard. Follow-ups: live RTSP camera ingestion, authentication, and swapping SQLite/in-process events for Postgres and a broker.