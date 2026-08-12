"""Rule-based and Isolation Forest anomaly detection, catching what the risk model isn't explicitly built for."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class ActivitySample:
    seconds_since_last_movement: float
    velocity_magnitude: float
    velocity_change: float

    def to_vector(self) -> list[float]:
        return [self.seconds_since_last_movement, self.velocity_magnitude, self.velocity_change]


class RuleBasedAnomalyDetector:
    """Plain hand-written thresholds, every one a config value, none buried in logic."""

    def __init__(
        self,
        prolonged_inactivity_seconds: float = 120.0,
        sudden_velocity_change_threshold: float = 150.0,
    ) -> None:
        self.prolonged_inactivity_seconds = prolonged_inactivity_seconds
        self.sudden_velocity_change_threshold = sudden_velocity_change_threshold

    def is_anomalous(self, sample: ActivitySample) -> tuple[bool, str | None]:
        if sample.seconds_since_last_movement > self.prolonged_inactivity_seconds:
            return True, "prolonged_inactivity"
        if abs(sample.velocity_change) > self.sudden_velocity_change_threshold:
            return True, "sudden_movement_change"
        return False, None


class IsolationForestAnomalyDetector:
    """Trains only on normal behavior, no anomaly labels needed at all."""

    def __init__(self, contamination: float = 0.05) -> None:
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self._is_fitted = False

    def fit(self, normal_samples: list[ActivitySample]) -> None:
        X = np.array([s.to_vector() for s in normal_samples])
        self.model.fit(X)
        self._is_fitted = True

    def is_anomalous(self, sample: ActivitySample) -> bool:
        if not self._is_fitted:
            raise RuntimeError("IsolationForestAnomalyDetector.fit() must be called before is_anomalous()")
        prediction = self.model.predict([sample.to_vector()])[0]
        return bool(prediction == -1)


def generate_normal_activity_data(n_samples: int = 300, seed: int = 42) -> list[ActivitySample]:
    """Synthetic normal warehouse activity, people moving around regularly, no long stalls."""
    rng = random.Random(seed)
    samples = []
    for _ in range(n_samples):
        samples.append(
            ActivitySample(
                seconds_since_last_movement=rng.uniform(0, 15),
                velocity_magnitude=rng.uniform(10, 80),
                velocity_change=rng.uniform(-20, 20),
            )
        )
    return samples


def generate_test_data_with_anomalies(n_normal: int = 90, n_anomalous: int = 10, seed: int = 7) -> tuple[list[ActivitySample], list[bool]]:
    rng = random.Random(seed)
    samples: list[ActivitySample] = []
    labels: list[bool] = []

    for _ in range(n_normal):
        samples.append(
            ActivitySample(
                seconds_since_last_movement=rng.uniform(0, 15),
                velocity_magnitude=rng.uniform(10, 80),
                velocity_change=rng.uniform(-20, 20),
            )
        )
        labels.append(False)

    for _ in range(n_anomalous):
        if rng.random() < 0.5:
            samples.append(ActivitySample(seconds_since_last_movement=rng.uniform(180, 400), velocity_magnitude=0.0, velocity_change=0.0))
        else:
            samples.append(ActivitySample(seconds_since_last_movement=rng.uniform(0, 5), velocity_magnitude=rng.uniform(100, 200), velocity_change=rng.uniform(200, 400)))
        labels.append(True)

    combined = list(zip(samples, labels))
    rng.shuffle(combined)
    samples, labels = zip(*combined)
    return list(samples), list(labels)


def compare_detectors(seed: int = 42) -> dict:
    """Both detectors on the same synthetic mixed test set, my rules-vs-Isolation-Forest comparison."""
    normal_training_data = generate_normal_activity_data(seed=seed)
    test_samples, true_labels = generate_test_data_with_anomalies(seed=seed + 1)

    rule_based = RuleBasedAnomalyDetector()
    rule_preds = [rule_based.is_anomalous(s)[0] for s in test_samples]

    iso_forest = IsolationForestAnomalyDetector()
    iso_forest.fit(normal_training_data)
    iso_preds = [iso_forest.is_anomalous(s) for s in test_samples]

    def metrics(preds: list[bool]) -> dict:
        tp = sum(1 for p, t in zip(preds, true_labels) if p and t)
        fp = sum(1 for p, t in zip(preds, true_labels) if p and not t)
        fn = sum(1 for p, t in zip(preds, true_labels) if not p and t)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "true_positives": tp, "false_positives": fp, "false_negatives": fn}

    return {
        "dataset": "synthetic, placeholder, not real logged activity",
        "n_test_samples": len(test_samples),
        "n_true_anomalies": sum(true_labels),
        "rule_based": metrics(rule_preds),
        "isolation_forest": metrics(iso_preds),
    }
