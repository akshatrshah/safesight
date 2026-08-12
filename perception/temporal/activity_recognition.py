"""Activity recognition from a window of frames, not one frame. Compares hand-engineered features against an LSTM."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

WINDOW_SIZE = 20
NUM_FEATURES = 4


class Activity(str, Enum):
    STANDING = "standing"
    WALKING = "walking"
    BENDING = "bending"
    FALLING = "falling"


ACTIVITIES = list(Activity)


def generate_synthetic_dataset(n_sequences_per_class: int = 150, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic sequences with a hand-coded motion signature per activity, a stand-in until I have real labeled clips."""
    rng = np.random.default_rng(seed)
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for class_idx, activity in enumerate(ACTIVITIES):
        for _ in range(n_sequences_per_class):
            t = np.arange(WINDOW_SIZE)
            noise = rng.normal(0, 0.05, size=(WINDOW_SIZE, NUM_FEATURES))

            if activity == Activity.STANDING:
                hip_height = np.full(WINDOW_SIZE, 1.0)
                knee_angle = np.full(WINDOW_SIZE, 175.0) / 180.0
                v_vel = np.zeros(WINDOW_SIZE)
                h_vel = np.zeros(WINDOW_SIZE)

            elif activity == Activity.WALKING:
                hip_height = np.full(WINDOW_SIZE, 1.0) + 0.03 * np.sin(t * 0.8)
                knee_angle = (150 + 25 * np.sin(t * 0.8)) / 180.0
                v_vel = np.zeros(WINDOW_SIZE)
                h_vel = np.full(WINDOW_SIZE, 0.6)

            elif activity == Activity.BENDING:
                hip_height = 1.0 - 0.4 * np.sin(np.pi * t / WINDOW_SIZE)
                knee_angle = (170 - 60 * np.sin(np.pi * t / WINDOW_SIZE)) / 180.0
                v_vel = -0.4 * np.cos(np.pi * t / WINDOW_SIZE) * 0.1
                h_vel = np.zeros(WINDOW_SIZE)

            else:  # FALLING
                drop_point = WINDOW_SIZE // 2
                hip_height = np.where(t < drop_point, 1.0, 0.2)
                knee_angle = np.where(t < drop_point, 175.0, 90.0) / 180.0
                v_vel = np.where(t == drop_point, -3.0, 0.0)
                h_vel = rng.normal(0, 0.1, size=WINDOW_SIZE)

            sequence = np.stack([hip_height, knee_angle, v_vel, h_vel], axis=1) + noise
            sequences.append(sequence)
            labels.append(class_idx)

    X = np.stack(sequences).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    perm = rng.permutation(len(X))
    return X[perm], y[perm]


class BaselineClassifier(nn.Module):
    """Hand-picked summary features fed into a small MLP. Never sees the raw sequence."""

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
    """Turns one raw sequence into 6 hand-picked summary numbers, the features the baseline relies on."""
    hip_height = sequence[:, 0]
    knee_angle = sequence[:, 1]
    v_vel = sequence[:, 2]
    h_vel = sequence[:, 3]

    return np.array([
        hip_height.mean(),
        hip_height.min() - hip_height.max(),
        knee_angle.mean(),
        np.abs(v_vel).max(),
        h_vel.mean(),
        np.abs(np.diff(hip_height)).max(),
    ], dtype=np.float32)


class LSTMClassifier(nn.Module):
    """Consumes the raw sequence directly, learns its own temporal features instead of me hand-designing them."""

    def __init__(self, n_features: int = NUM_FEATURES, hidden_size: int = 32, n_classes: int = len(ACTIVITIES)) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
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
    """Trains and evaluates both models on the same synthetic split, my baseline-vs-LSTM comparison."""
    X, y = generate_synthetic_dataset(n_sequences_per_class=n_sequences_per_class, seed=seed)
    n = len(X)
    split = int(n * 0.7)
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_hand = np.stack([extract_hand_features(s) for s in X_train_raw])
    X_test_hand = np.stack([extract_hand_features(s) for s in X_test_raw])

    baseline = BaselineClassifier(n_hand_features=X_train_hand.shape[1])
    _train_torch_model(baseline, X_train_hand, y_train)
    baseline_acc = _evaluate_torch_model(baseline, X_test_hand, y_test)

    lstm = LSTMClassifier()
    _train_torch_model(lstm, X_train_raw, y_train)
    lstm_acc = _evaluate_torch_model(lstm, X_test_raw, y_test)

    return {
        "dataset": "synthetic, placeholder, not real activity clips",
        "n_train": int(split),
        "n_test": int(n - split),
        "classes": [a.value for a in ACTIVITIES],
        "baseline_mlp_accuracy": round(baseline_acc, 4),
        "lstm_accuracy": round(lstm_acc, 4),
    }
