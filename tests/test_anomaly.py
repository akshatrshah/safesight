from perception.anomaly.anomaly_detector import (
    ActivitySample,
    RuleBasedAnomalyDetector,
    IsolationForestAnomalyDetector,
    generate_normal_activity_data,
)


def test_rule_based_flags_prolonged_inactivity():
    detector = RuleBasedAnomalyDetector(prolonged_inactivity_seconds=120.0)
    sample = ActivitySample(seconds_since_last_movement=200.0, velocity_magnitude=0.0, velocity_change=0.0)
    is_anomaly, reason = detector.is_anomalous(sample)
    assert is_anomaly is True
    assert reason == "prolonged_inactivity"


def test_rule_based_flags_sudden_velocity_change():
    detector = RuleBasedAnomalyDetector(sudden_velocity_change_threshold=150.0)
    sample = ActivitySample(seconds_since_last_movement=1.0, velocity_magnitude=100.0, velocity_change=300.0)
    is_anomaly, reason = detector.is_anomalous(sample)
    assert is_anomaly is True
    assert reason == "sudden_movement_change"


def test_rule_based_does_not_flag_normal_activity():
    detector = RuleBasedAnomalyDetector()
    sample = ActivitySample(seconds_since_last_movement=5.0, velocity_magnitude=40.0, velocity_change=5.0)
    is_anomaly, reason = detector.is_anomalous(sample)
    assert is_anomaly is False
    assert reason is None


def test_isolation_forest_requires_fit_before_use():
    detector = IsolationForestAnomalyDetector()
    sample = ActivitySample(seconds_since_last_movement=5.0, velocity_magnitude=40.0, velocity_change=5.0)
    try:
        detector.is_anomalous(sample)
        assert False, "expected RuntimeError before fit() is called"
    except RuntimeError:
        pass


def test_isolation_forest_flags_extreme_outlier_after_fit():
    detector = IsolationForestAnomalyDetector(contamination=0.05)
    detector.fit(generate_normal_activity_data(n_samples=200))

    extreme_outlier = ActivitySample(seconds_since_last_movement=999.0, velocity_magnitude=0.0, velocity_change=0.0)
    assert detector.is_anomalous(extreme_outlier) is True
