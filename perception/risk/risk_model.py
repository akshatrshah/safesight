"""
Worker-forklift risk prediction: the centerpiece of the project.

TWO SEPARATE APPROACHES, DELIBERATELY KEPT DISTINCT
--------------------------------------------------------
1. HeuristicRiskModel: a hand-written formula combining distance,
   closing speed, and time-to-collision into a score. Fully
   interpretable — you can always point to exactly which factor drove a
   given score.
2. LearnedRiskModel: the SAME input features, but the combining formula
   is learned from labeled examples (gradient-boosted trees) instead of
   hand-written. Potentially more accurate on patterns a human wouldn't
   think to encode, at the cost of interpretability.

Both consume the same `InteractionFeatures` — this is intentional, so
comparing them (Milestone 8 / Experiment 2) is an apples-to-apples
comparison of MODELING approach, not a comparison confounded by
different input data.

TIME-TO-COLLISION (TTC), EXPLAINED
---------------------------------------
If both objects kept moving at their CURRENT velocity in a straight
line, how many seconds until they'd occupy the same position? Computed
as: (distance between them) / (rate at which that distance is
shrinking). If they're moving apart, or moving in parallel without
closing the gap, TTC is undefined (returned as None) — there's no
meaningful "collision" to estimate.

IMPORTANT MONOCULAR CAMERA CAVEAT
--------------------------------------
All distances/velocities here are in PIXEL units, from a single 2D
camera view — not real-world meters. Converting pixel distance to real
distance requires camera calibration (knowing camera height/angle, or
using a known real-world reference size in the frame), which is a
separate, harder problem this module does not attempt to solve. Treat
these as relative/comparative risk signals, not literal physical
distances, unless calibration is added later.

TRAINING DATA NOTE
-----------------------
`generate_synthetic_training_data()` below produces SYNTHETIC labeled
interactions purely so LearnedRiskModel has something concrete to train
and be evaluated on end-to-end. It is explicitly a placeholder — real
labeled interaction outcomes (from real warehouse footage) would replace
it before any real accuracy claim is made. This mirrors the same honesty
standard applied to the COCO8 placeholder in the detection evaluation.
"""

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
    """The shared input both risk models consume."""

    distance: float                  # pixels, straight-line
    closing_speed: float             # pixels/second, positive = closing the gap
    time_to_collision: float | None  # seconds, None if not on a collision course
    person_in_forklift_zone: bool    # is the person inside a forklift-lane zone

    def to_vector(self) -> list[float]:
        """Flatten to a fixed-length numeric feature vector for the learned model."""
        # Represent "no TTC" (moving apart) as a large number rather than
        # None, since the model needs a numeric input — a large TTC means
        # "far from colliding," which is the correct signal either way.
        ttc = self.time_to_collision if self.time_to_collision is not None else 999.0
        return [self.distance, self.closing_speed, ttc, float(self.person_in_forklift_zone)]


def compute_interaction_features(
    forklift_position: tuple[float, float],
    forklift_velocity: tuple[float, float],
    person_position: tuple[float, float],
    person_velocity: tuple[float, float],
    person_in_forklift_zone: bool = False,
) -> InteractionFeatures:
    """
    Compute distance, closing speed, and time-to-collision between one
    forklift and one person, given their current position + velocity
    (velocity typically comes from TrackHistory.velocity()).
    """
    rel_x = person_position[0] - forklift_position[0]
    rel_y = person_position[1] - forklift_position[1]
    distance = math.hypot(rel_x, rel_y)

    rel_vx = person_velocity[0] - forklift_velocity[0]
    rel_vy = person_velocity[1] - forklift_velocity[1]

    if distance == 0:
        closing_speed = 0.0
    else:
        # Negative of the rate of change of distance: positive value
        # means the gap is shrinking (closing), negative means widening.
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
    """
    Hand-written formula. Weights below are illustrative starting
    points, not tuned against real data yet — that tuning is exactly
    the kind of thing Experiment 2 (heuristic vs learned) should surface
    as a limitation of the heuristic approach.
    """

    def __init__(
        self,
        ttc_danger_threshold_seconds: float = 3.0,
        distance_danger_threshold_pixels: float = 150.0,
    ) -> None:
        self.ttc_danger_threshold = ttc_danger_threshold_seconds
        self.distance_danger_threshold = distance_danger_threshold_pixels

    def score(self, features: InteractionFeatures) -> float:
        """Returns a continuous 0.0-1.0 risk score."""
        score = 0.0

        # Closer than the danger threshold contributes up to 0.4.
        if features.distance < self.distance_danger_threshold:
            score += 0.4 * (1 - features.distance / self.distance_danger_threshold)

        # A low, POSITIVE time-to-collision (i.e. genuinely on a collision
        # course, and soon) contributes up to 0.4.
        if features.time_to_collision is not None and features.time_to_collision < self.ttc_danger_threshold:
            score += 0.4 * (1 - features.time_to_collision / self.ttc_danger_threshold)

        # Being in the forklift's lane at all contributes a flat 0.2.
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
    """
    Gradient-boosted trees trained on labeled InteractionFeatures ->
    risk label. Chosen over a deep net here because with a small
    labeled interaction dataset (the realistic case for a project like
    this), tree-based models typically generalize better than a neural
    network would.
    """

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
    """
    Generates SYNTHETIC labeled interactions for demonstrating the
    learned-vs-heuristic comparison end-to-end. See module docstring —
    this is a placeholder for real labeled outcome data.

    Label logic intentionally mirrors real-world intuition (close +
    fast-closing + in-lane = high risk) but with randomized noise, so
    the learned model has a genuine pattern to discover rather than a
    trivial lookup.
    """
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

        # Ground-truth labeling rule used ONLY to generate synthetic data —
        # not used anywhere in the actual models above.
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
    """
    Trains/evaluates both models on the same synthetic train/test split
    and returns precision/recall/F1 for each, per Experiment 2 in the
    project spec. Real numbers, computed from an actual held-out split —
    just on synthetic data, clearly labeled as such in the output.
    """
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
        "dataset": "synthetic (see module docstring — placeholder, not real interaction data)",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "heuristic": metrics(heuristic_pred_values),
        "learned": metrics(learned_pred_values),
    }
