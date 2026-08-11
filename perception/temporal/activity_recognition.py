"""
Temporal activity recognition: classify what a tracked person is doing
(standing, walking, bending, falling) from a SEQUENCE of recent frames,
not a single frame.

WHY A SEQUENCE, NOT ONE FRAME
----------------------------------
A single frame showing "knees bent, torso lowered" is ambiguous — it
could be the middle of a squat, a fall, or someone tying their shoe.
The only way to tell them apart is to look at how that position changed
over the last second or two. Every model below takes a WINDOW of
consecutive frames (default 20) as input, not one frame.

TWO MODELS COMPARED, DELIBERATELY SIMPLEST-FIRST
------------------------------------------------------
1. BaselineClassifier: hand-engineered summary features (e.g. average
   hip-height change, average knee angle, max vertical velocity) fed
   into a small feedforward network (MLP). Fast, interpretable, but
   limited to whatever patterns we thought to compute by hand.
2. LSTMClassifier: consumes the raw per-frame feature sequence directly
   and lets a recurrent network (which carries a hidden state forward
   frame-by-frame) learn what temporal patterns matter, instead of us
   hand-designing them.

Per-frame input features (4 numbers, kept deliberately simple):
    [hip_height, knee_angle_degrees, vertical_velocity, horizontal_velocity]
In a full system these would come from perception/pose/pose_estimator.py
keypoints across real tracked people; here the SAME 4-number schema is
used for both real and synthetic data so the models transfer directly
once real labeled pose sequences are available.

TRAINING DATA NOTE
-----------------------
`generate_synthetic_dataset()` produces SYNTHETIC sequences with
hand-coded per-activity motion signatures (e.g. "falling" = sudden hip
height drop). This is a placeholder standing in for real labeled
warehouse activity clips (Milestone 4/5's real dataset) — it exists so
the training/evaluation pipeline itself is proven correct end-to-end
before real data is available, exactly like the COCO8 placeholder used
for detection evaluation. Do not present these accuracy numbers as
"real-world activity recognition accuracy" in the README — they measure
whether the models can learn the synthetic generating pattern, which is
a much easier task than the real one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

WINDOW_SIZE = 20   # frames per sequence (~0.6s at 30fps)
NUM_FEATURES = 4   # hip_height, knee_angle, vertical_velocity, horizontal_velocity


class Activity(str, Enum):
    STANDING = "standing"
    WALKING = "walking"
    BENDING = "bending"
    FALLING = "falling"


ACTIVITIES = list(Activity)


def generate_synthetic_dataset(n_sequences_per_class: int = 150, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y): X has shape (N, WINDOW_SIZE, NUM_FEATURES), y has
    shape (N,) with integer class labels (index into ACTIVITIES).

    Each activity gets a distinct, hand-coded synthetic motion pattern
    plus noise — see module docstring for why this is a placeholder.
    """
    rng = np.random.default_rng(seed)
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for class_idx, activity in enumerate(ACTIVITIES):
        for _ in range(n_sequences_per_class):
            t = np.arange(WINDOW_SIZE)
            noise = rng.normal(0, 0.05, size=(WINDOW_SIZE, NUM_FEATURES))

            if activity == Activity.STANDING:
                hip_height = np.full(WINDOW_SIZE, 1.0)
                knee_angle = np.full(WINDOW_SIZE, 175.0) / 180.0  # normalize roughly to [0,1]
                v_vel = np.zeros(WINDOW_SIZE)
                h_vel = np.zeros(WINDOW_SIZE)

            elif activity == Activity.WALKING:
                hip_height = np.full(WINDOW_SIZE, 1.0) + 0.03 * np.sin(t * 0.8)  # slight bob
                knee_angle = (150 + 25 * np.sin(t * 0.8)) / 180.0
                v_vel = np.zeros(WINDOW_SIZE)
                h_vel = np.full(WINDOW_SIZE, 0.6)  # steady horizontal motion

            elif activity == Activity.BENDING:
                # Smooth, gradual hip height decrease then partial recovery — controlled motion.
                hip_height = 1.0 - 0.4 * np.sin(np.pi * t / WINDOW_SIZE)
                knee_angle = (170 - 60 * np.sin(np.pi * t / WINDOW_SIZE)) / 180.0
                v_vel = -0.4 * np.cos(np.pi * t / WINDOW_SIZE) * 0.1
                h_vel = np.zeros(WINDOW_SIZE)

            else:  # FALLING
                # Sudden, sharp hip height drop partway through the window — fast, not gradual.
                drop_point = WINDOW_SIZE // 2
                hip_height = np.where(t < drop_point, 1.0, 0.2)
                knee_angle = np.where(t < drop_point, 175.0, 90.0) / 180.0
                v_vel = np.where(t == drop_point, -3.0, 0.0)  # sharp velocity spike at the fall instant
                h_vel = rng.normal(0, 0.1, size=WINDOW_SIZE)

            sequence = np.stack([hip_height, knee_angle, v_vel, h_vel], axis=1) + noise
            sequences.append(sequence)
            labels.append(class_idx)

    X = np.stack(sequences).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    # Shuffle so classes aren't grouped in order.
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


class BaselineClassifier(nn.Module):
    """
    Hand-engineered features (computed from the raw sequence, see
    `extract_hand_features`) fed into a small MLP. This does NOT see
    the raw per-frame sequence — only our own summary statistics of it.
    """

    def __init__(self, n_hand_features: int = 6, n_classes: int = len(ACTIVITIES)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_hand_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def extract_hand_features(sequence: np.ndarray) -> np.ndarray:
    """
    Turn one (WINDOW_SIZE, NUM_FEATURES) sequence into a small set of
    hand-picked summary numbers — the "manual feature engineering" the
    baseline model relies on instead of learning temporal patterns
    itself.
    """
    hip_height = sequence[:, 0]
    knee_angle = sequence[:, 1]
    v_vel = sequence[:, 2]
    h_vel = sequence[:, 3]

    return np.array([
        hip_height.mean(),
        hip_height.min() - hip_height.max(),   # total hip-height range (drop signal)
        knee_angle.mean(),
        np.abs(v_vel).max(),                    # sharpest vertical velocity spike
        h_vel.mean(),
        np.abs(np.diff(hip_height)).max(),      # sharpest single-frame hip-height jump
    ], dtype=np.float32)


class LSTMClassifier(nn.Module):
    """
    Consumes the raw (WINDOW_SIZE, NUM_FEATURES) sequence directly. An
    LSTM processes the sequence one timestep at a time, carrying a
    hidden state forward — the network itself learns what temporal
    patterns (e.g. "sudden drop" vs "gradual drop") distinguish classes,
    rather than us hand-designing summary features.
    """

    def __init__(self, n_features: int = NUM_FEATURES, hidden_size: int = 32, n_classes: int = len(ACTIVITIES)) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: (batch, WINDOW_SIZE, NUM_FEATURES)
        _, (hidden, _) = self.lstm(x)   # hidden: (1, batch, hidden_size) — final timestep's hidden state
        return self.classifier(hidden.squeeze(0))


def _train_torch_model(model: nn.Module, X_train, y_train, epochs: int = 30, lr: float = 0.01) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()


def _evaluate_torch_model(model: nn.Module, X_test, y_test) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        preds = logits.argmax(dim=1).numpy()
    return float((preds == y_test).mean())


def compare_models(n_sequences_per_class: int = 150, seed: int = 42) -> dict:
    """
    Trains and evaluates both models on the same synthetic train/test
    split, returns accuracy for each. This is Milestone 5's "which
    approach gives the best accuracy/latency tradeoff" comparison — run
    on synthetic data (see module docstring) until real labeled activity
    clips are available.
    """
    X, y = generate_synthetic_dataset(n_sequences_per_class=n_sequences_per_class, seed=seed)
    n = len(X)
    split = int(n * 0.7)
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- Baseline: hand-engineered features + MLP ---
    X_train_hand = np.stack([extract_hand_features(s) for s in X_train_raw])
    X_test_hand = np.stack([extract_hand_features(s) for s in X_test_raw])

    baseline = BaselineClassifier(n_hand_features=X_train_hand.shape[1])
    _train_torch_model(baseline, X_train_hand, y_train)
    baseline_acc = _evaluate_torch_model(baseline, X_test_hand, y_test)

    # --- LSTM: raw sequence ---
    lstm = LSTMClassifier()
    _train_torch_model(lstm, X_train_raw, y_train)
    lstm_acc = _evaluate_torch_model(lstm, X_test_raw, y_test)

    return {
        "dataset": "synthetic (see module docstring — placeholder, not real activity clips)",
        "n_train": int(split),
        "n_test": int(n - split),
        "classes": [a.value for a in ACTIVITIES],
        "baseline_mlp_accuracy": round(baseline_acc, 4),
        "lstm_accuracy": round(lstm_acc, 4),
    }
