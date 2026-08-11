"""
Runs the three "compare two modeling approaches" experiments defined
across the perception modules, and saves all results to one file.

This is the single script to run for Milestone 5 / 8's research
questions: heuristic vs learned risk, baseline vs LSTM activity
recognition, rule-based vs ML-based anomaly detection.

All datasets used here are SYNTHETIC placeholders — see the docstring
in each module (perception/risk/risk_model.py,
perception/temporal/activity_recognition.py,
perception/anomaly/anomaly_detector.py) for why, and swap in real
labeled data before treating these numbers as real-world claims.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perception.risk.risk_model import compare_models as compare_risk_models
from perception.temporal.activity_recognition import compare_models as compare_activity_models
from perception.anomaly.anomaly_detector import compare_detectors as compare_anomaly_detectors


def main() -> None:
    print("=== Experiment: Heuristic vs Learned Risk Model ===")
    risk_results = compare_risk_models()
    print(json.dumps(risk_results, indent=2))

    print("\n=== Experiment: Baseline MLP vs LSTM Activity Recognition ===")
    activity_results = compare_activity_models()
    print(json.dumps(activity_results, indent=2))

    print("\n=== Experiment: Rule-Based vs Isolation Forest Anomaly Detection ===")
    anomaly_results = compare_anomaly_detectors()
    print(json.dumps(anomaly_results, indent=2))

    out_path = REPO_ROOT / "experiments" / "perception_experiments_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "risk_model_comparison": risk_results,
                "activity_recognition_comparison": activity_results,
                "anomaly_detection_comparison": anomaly_results,
            },
            f,
            indent=2,
        )
    print(f"\nAll results written to {out_path}")


if __name__ == "__main__":
    main()
