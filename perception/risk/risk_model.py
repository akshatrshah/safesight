"""Risk scoring: a hand-written heuristic and a learned model, sharing the same features so I can compare them fairly."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class InteractionFeatures:
    distance: float
    closing_speed: float
    time_to_collision: float | None
    person_in_forklift_zone: bool

    def to_vector(self) -> list[float]:
        ttc = self.time_to_collision if self.time_to_collision is not None else 999.0
        return [self.distance, self.closing_speed, ttc, float(self.person_in_forklift_zone)]


def compute_interaction_features(
    forklift_position: tuple[float, float],
    forklift_velocity: tuple[float, float],
    person_position: tuple[float, float],
    person_velocity: tuple[float, float],
    person_in_forklift_zone: bool = False,
) -> InteractionFeatures:
    """Distance, closing speed, and time-to-collision between a forklift and a person, from position + velocity."""
    rel_x = person_position[0] - forklift_position[0]
    rel_y = person_position[1] - forklift_position[1]
    distance = math.hypot(rel_x, rel_y)

    rel_vx = person_velocity[0] - forklift_velocity[0]
    rel_vy = person_velocity[1] - forklift_velocity[1]

    if distance == 0:
        closing_speed = 0.0
    else:
        closing_speed = -(rel_x * rel_vx + rel_y * rel_vy) / distance

    ttc = None
    if closing_speed > 0:
        ttc = distance / closing_speed

    return InteractionFeatures(
        distance=distance,
        closing_speed=closing_speed,
        time_to_collision=ttc,
        person_in_forklift_zone=person_in_forklift_zone,
    )


class HeuristicRiskModel:
    """Hand-written weights, not tuned against real data yet, that gap is a known limitation."""

    def __init__(
        self,
        ttc_danger_threshold_seconds: float = 3.0,
        distance_danger_threshold_pixels: float = 150.0,
    ) -> None:
        self.ttc_danger_threshold = ttc_danger_threshold_seconds
        self.distance_danger_threshold = distance_danger_threshold_pixels

    def score(self, features: InteractionFeatures) -> float:
        score = 0.0

        if features.distance < self.distance_danger_threshold:
            score += 0.4 * (1 - features.distance / self.distance_danger_threshold)

        if features.time_to_collision is not None and features.time_to_collision < self.ttc_danger_threshold:
            score += 0.4 * (1 - features.time_to_collision / self.ttc_danger_threshold)

        if features.person_in_forklift_zone:
            score += 0.2

        return min(1.0, max(0.0, score))

    def classify(self, features: InteractionFeatures) -> RiskLevel:
        score = self.score(features)
        if score >= 0.7:
            return RiskLevel.HIGH
        if score >= 0.35:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class LearnedRiskModel:
    """Same input features as the heuristic, gradient boosted trees instead of a hand-written formula."""

    def __init__(self) -> None:
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        self._is_fitted = False

    def fit(self, features_list: list[InteractionFeatures], labels: list[RiskLevel]) -> None:
        X = [f.to_vector() for f in features_list]
        y = [label.value for label in labels]
        self.model.fit(X, y)
        self._is_fitted = True

    def classify(self, features: InteractionFeatures) -> RiskLevel:
        if not self._is_fitted:
            raise RuntimeError("LearnedRiskModel.fit() must be called before classify()")
        prediction = self.model.predict([features.to_vector()])[0]
        return RiskLevel(prediction)


def generate_synthetic_training_data(n_samples: int = 500, seed: int = 42) -> tuple[list[InteractionFeatures], list[RiskLevel]]:
    """Synthetic labeled interactions, placeholder until I have real labeled outcome data."""
    rng = random.Random(seed)
    features_list: list[InteractionFeatures] = []
    labels: list[RiskLevel] = []

    for _ in range(n_samples):
        distance = rng.uniform(10, 400)
        closing_speed = rng.uniform(-50, 100)
        ttc = distance / closing_speed if closing_speed > 0 else None
        in_zone = rng.random() < 0.3

        features = InteractionFeatures(distance, closing_speed, ttc, in_zone)
        features_list.append(features)

        danger_signal = 0
        if distance < 100:
            danger_signal += 1
        if ttc is not None and ttc < 3:
            danger_signal += 1
        if in_zone:
            danger_signal += 1

        if danger_signal >= 2:
            label = RiskLevel.HIGH
        elif danger_signal == 1:
            label = RiskLevel.MEDIUM
        else:
            label = RiskLevel.LOW
        labels.append(label)

    return features_list, labels


def compare_models(n_samples: int = 500, seed: int = 42) -> dict:
    """Trains and evaluates both models on the same synthetic split. Real numbers, synthetic data, said plainly."""
    features_list, labels = generate_synthetic_training_data(n_samples=n_samples, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        features_list, labels, test_size=0.3, random_state=seed, stratify=[l.value for l in labels]
    )

    heuristic = HeuristicRiskModel()
    heuristic_preds = [heuristic.classify(f) for f in X_test]

    learned = LearnedRiskModel()
    learned.fit(X_train, y_train)
    learned_preds = [learned.classify(f) for f in X_test]

    y_test_values = [l.value for l in y_test]
    heuristic_pred_values = [p.value for p in heuristic_preds]
    learned_pred_values = [p.value for p in learned_preds]

    def metrics(preds: list[str]) -> dict:
        return {
            "precision_macro": round(precision_score(y_test_values, preds, average="macro", zero_division=0), 4),
            "recall_macro": round(recall_score(y_test_values, preds, average="macro", zero_division=0), 4),
            "f1_macro": round(f1_score(y_test_values, preds, average="macro", zero_division=0), 4),
        }

    return {
        "dataset": "synthetic, placeholder, not real interaction data",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "heuristic": metrics(heuristic_pred_values),
        "learned": metrics(learned_pred_values),
    }
